"""Station data -> Gridded Daily data with Barnes interpolation in projected LAEA space.

Barnes analysis is performed on a coarse ~20 km LAEA grid, then bilinearly
resampled to the target 1 km LAI projected grid (from lai_aggregate.py output).

All distance calculations use true projected distances in km — no flat-Earth
approximation needed.
"""
import os
import numpy as np
import pandas as pd
from ghcnd_station_analysis import parse_stations
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer


# ── Barnes parameters ─────────────────────────────────────────────────────────
D_BAR_KM  = 19.0   # average station spacing (km) from domain analysis
E         = 0.5    # smoothing parameter
KAPPA_KM2 = (1.33 * D_BAR_KM) ** 2 / (4 * E)
GAMMA     = 0.3    # error correction factor for second pass (0.2-0.5 typical)
N_PASSES  = 2

MIN_STATIONS  = 3
MAX_DIST_KM   = 150.0    # search radius in km
COARSE_RES_M  = 20_000   # Barnes coarse grid spacing in metres

VARIABLES = ["TMAX", "TMIN"]


# ── Grid helpers ──────────────────────────────────────────────────────────────

def load_lai_grid(zarr_path):
    """Load the 1-km projected grid from a LAI zarr produced by lai_aggregate.py.

    Returns
    -------
    x_m  : 1-D easting  coordinates in metres  (the 'lon' dim)
    y_m  : 1-D northing coordinates in metres  (the 'lat' dim)
    crs  : projection string stored in zarr attrs
    """
    ds = xr.open_zarr(zarr_path)
    x_m = ds['lon'].values   # projected easting  (metres)
    y_m = ds['lat'].values   # projected northing (metres)
    crs = ds.attrs['crs']
    ds.close()
    return x_m, y_m, crs


def build_coarse_grid(x_m, y_m, coarse_res_m=COARSE_RES_M):
    """Build a coarse 1-D x/y grid at coarse_res_m spacing over the fine domain."""
    x_c = np.arange(x_m[0], x_m[-1] + coarse_res_m * 0.5, coarse_res_m)
    y_c = np.arange(y_m[0], y_m[-1] + coarse_res_m * 0.5, coarse_res_m)
    return x_c, y_c


def project_stations(stations_df, transformer):
    """Project station lat/lon (degrees) to LAEA x_km / y_km.

    Parameters
    ----------
    stations_df : DataFrame with columns station_id, lat, lon
    transformer : pyproj.Transformer (EPSG:4326 -> projected CRS, always_xy=True)

    Returns
    -------
    Copy of stations_df with added x_km, y_km columns.
    """
    sx, sy = transformer.transform(
        stations_df['lon'].values,
        stations_df['lat'].values,
    )
    df = stations_df.copy()
    df['x_km'] = sx / 1000.0
    df['y_km'] = sy / 1000.0
    return df


# ── Barnes analysis in projected km space ────────────────────────────────────

def barnes_analysis_proj(sxy_km, sval, grid_x2d_km, grid_y2d_km):
    """Barnes analysis using projected km distances.

    Parameters
    ----------
    sxy_km      : (N, 2) station [x_km, y_km]
    sval        : (N,)   observed values (NaN allowed)
    grid_x2d_km : (ny_c, nx_c) coarse grid x in km
    grid_y2d_km : (ny_c, nx_c) coarse grid y in km

    Returns
    -------
    (ny_c, nx_c) Barnes analysis field
    """
    valid = np.isfinite(sval)
    sxy_km, sval = sxy_km[valid], sval[valid]
    if len(sval) < MIN_STATIONS:
        return np.full(grid_x2d_km.shape, np.nan)

    gxy = np.column_stack([grid_x2d_km.ravel(), grid_y2d_km.ravel()])

    tree  = cKDTree(sxy_km)
    gnbrs = tree.query_ball_point(gxy, r=MAX_DIST_KM)

    # ── Pass 1 ──
    g0 = np.full(len(gxy), np.nan)
    for gi, nb in enumerate(gnbrs):
        if len(nb) < MIN_STATIONS:
            continue
        nb = np.array(nb)
        d  = np.linalg.norm(gxy[gi] - sxy_km[nb], axis=1)
        w  = np.exp(-d**2 / KAPPA_KM2)
        g0[gi] = np.dot(w, sval[nb]) / w.sum()

    if N_PASSES < 2:
        return g0.reshape(grid_x2d_km.shape)

    # ── First guess at station locations ──
    snbrs = tree.query_ball_point(sxy_km, r=MAX_DIST_KM)
    g0s   = np.full(len(sval), np.nan)
    for si, nb in enumerate(snbrs):
        nb = np.array(nb)
        d  = np.linalg.norm(sxy_km[si] - sxy_km[nb], axis=1)
        w  = np.exp(-d**2 / KAPPA_KM2)
        if w.sum() > 0:
            g0s[si] = np.dot(w, sval[nb]) / w.sum()

    # ── Pass 2: residual correction ──
    resid  = np.where(np.isfinite(g0s), sval - g0s, 0.0)
    kappa2 = GAMMA * KAPPA_KM2
    g1     = g0.copy()
    for gi, nb in enumerate(gnbrs):
        if len(nb) < MIN_STATIONS or not np.isfinite(g0[gi]):
            continue
        nb = np.array(nb)
        d  = np.linalg.norm(gxy[gi] - sxy_km[nb], axis=1)
        w  = np.exp(-d**2 / kappa2)
        if w.sum() > 0:
            g1[gi] += np.dot(w, resid[nb]) / w.sum()

    return g1.reshape(grid_x2d_km.shape)


def bilinear_to_fine(coarse, x_c_km, y_c_km, x_f_km, y_f_km):
    """Bilinear resample from the coarse Barnes grid to the fine 1-km projected grid.

    Parameters
    ----------
    coarse          : (ny_c, nx_c) Barnes result
    x_c_km, y_c_km : 1-D coarse grid axes in km
    x_f_km, y_f_km : 1-D fine   grid axes in km  (from LAI zarr / 1000)

    Returns
    -------
    (ny_f, nx_f) interpolated field
    """
    fn = RegularGridInterpolator(
        (y_c_km, x_c_km),
        coarse,
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    yy, xx = np.meshgrid(y_f_km, x_f_km, indexing='ij')
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    return fn(pts).reshape(len(y_f_km), len(x_f_km))


# ── Per-day interpolation ─────────────────────────────────────────────────────

def interpolate_barnes_proj(stations_df, var_df, x_c_km, y_c_km, x_f_km, y_f_km):
    """Interpolate one day's station values onto the fine 1-km projected grid.

    Parameters
    ----------
    stations_df         : DataFrame with station_id, lat, lon, x_km, y_km
    var_df              : DataFrame with station_id, date, <varname> (single date)
    x_c_km / y_c_km    : 1-D coarse Barnes grid axes in km
    x_f_km / y_f_km    : 1-D fine LAI grid axes in km

    Returns
    -------
    fine_grid : np.ndarray (ny_f, nx_f)
    used      : DataFrame (station_id, lat, lon) of stations used this day
    """
    val_col = [c for c in var_df.columns if c not in ("station_id", "date")][0]
    merged  = var_df[["station_id", val_col]].merge(
        stations_df[["station_id", "lat", "lon", "x_km", "y_km"]],
        on="station_id", how="inner",
    )

    sxy_km = merged[["x_km", "y_km"]].values
    sval   = merged[val_col].values

    xx_c, yy_c = np.meshgrid(x_c_km, y_c_km)           # (ny_c, nx_c)
    coarse = barnes_analysis_proj(sxy_km, sval, xx_c, yy_c)
    fine   = bilinear_to_fine(coarse, x_c_km, y_c_km, x_f_km, y_f_km)

    return fine, merged[["station_id", "lat", "lon"]]


# ── Yearly processing ─────────────────────────────────────────────────────────

def process_yearly_data(
    year: int,
    lai_zarr_path: str,
    station_data_dir: str,
    station_inventory_path: str,
    out_path: str,
    append: bool = False,
    coarse_res_m: float = COARSE_RES_M,
):
    """Interpolate station TMAX/TMIN onto the 1-km LAI grid for one year.

    Writes tmax and tmin (float32, zlib-compressed) to the zarr at out_path.
    Set append=True to extend an existing zarr along the time dimension;
    set append=False (default) to create a new zarr (overwrites if present).
    """
    print(f"Processing {year}...")

    # Load 1-km LAI projected grid
    x_m, y_m, crs = load_lai_grid(lai_zarr_path)
    x_f_km = x_m / 1000.0
    y_f_km = y_m / 1000.0

    # Build coarse Barnes grid
    x_c_m, y_c_m = build_coarse_grid(x_m, y_m, coarse_res_m)
    x_c_km = x_c_m / 1000.0
    y_c_km = y_c_m / 1000.0

    fine_res_m = float(x_m[1] - x_m[0]) if len(x_m) > 1 else 1000.0
    print(f"  CRS        : {crs}")
    print(f"  Fine   grid: {len(y_f_km)} y × {len(x_f_km)} x  @ {fine_res_m:.0f} m")
    print(f"  Coarse grid: {len(y_c_km)} y × {len(x_c_km)} x  @ {coarse_res_m:.0f} m")

    # Project station inventory into LAEA km
    with open(station_inventory_path) as f:
        text = f.read()
    stations_raw = parse_stations(text)
    transformer  = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    stations_df  = project_stations(stations_raw, transformer)

    results = {}
    for var in VARIABLES:
        var_path = os.path.join(station_data_dir, f"{var}_{year}.txt")
        if not os.path.exists(var_path):
            print(f"  Warning: {var_path} not found, skipping {var}.")
            continue

        df = pd.read_csv(var_path, dtype={"station_id": str})
        df["date"] = pd.to_datetime(df["date"])
        print(f"  Loaded {var}: {len(df):,} records")
        print(f"  Interpolating {var}...")

        dates = sorted(df["date"].unique())
        grids = []
        daily_stations = []
        for date in dates:
            day_df = df[df["date"] == date]
            grid, used = interpolate_barnes_proj(
                stations_df, day_df, x_c_km, y_c_km, x_f_km, y_f_km,
            )
            grids.append(grid)
            daily_stations.append(used)

        results[var] = (dates, np.stack(grids, axis=0), daily_stations)

    # ── Build and save xarray Dataset ─────────────────────────────────────────
    if not results:
        return results

    data_vars = {}
    for var, (_, grids_arr, _daily_stations) in results.items():
        data_vars[var.lower()] = xr.Variable(
            ("time", "y", "x"),
            grids_arr,
            attrs={"units": "degC", "long_name": var},
        )

    first_dates = next(iter(results.values()))[0]
    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", first_dates),
            "y":    ("y",    y_m),   # projected northing in metres
            "x":    ("x",    x_m),   # projected easting  in metres
        },
        attrs={
            "crs":                  crs,
            "barnes_d_bar_km":      D_BAR_KM,
            "barnes_E":             E,
            "barnes_kappa_km2":     KAPPA_KM2,
            "barnes_gamma":         GAMMA,
            "barnes_n_passes":      N_PASSES,
            "barnes_max_dist_km":   MAX_DIST_KM,
            "barnes_min_stations":  MIN_STATIONS,
            "barnes_coarse_res_m":  coarse_res_m,
        },
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    encoding = {v: {"dtype": "float32", "compressors": {"name": "gzip", "configuration": {"level": 4}}}
                for v in ["tmax", "tmin"] if v in ds}
    if append:
        ds.to_zarr(out_path, append_dim="time")
    else:
        ds.to_zarr(out_path, mode="w", encoding=encoding)
    print(f"  Saved -> {out_path}")

    return results


if __name__ == "__main__":
    LAI_ZARR_PATH     = "/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km.zarr"
    STATION_DATA_DIR  = "/path/to/data-root/Data/station_data_2026/TX_domain"
    STATION_INVENTORY = "/path/to/data-root/Data/station_data_2026/ghcnd-stations.txt"
    OUTPUT_DIR        = "/path/to/data-root/Data/station_data_2026/TX_domain_interpolated/1km"
    YEARS             = range(2002, 2026)
    coarse_km         = int(COARSE_RES_M / 1000)
    OUT_PATH          = os.path.join(OUTPUT_DIR, f"tmax_tmin_1km_barnes{coarse_km}km.zarr")

    for i, year in enumerate(YEARS):
        process_yearly_data(
            year=year,
            lai_zarr_path=LAI_ZARR_PATH,
            station_data_dir=STATION_DATA_DIR,
            station_inventory_path=STATION_INVENTORY,
            out_path=OUT_PATH,
            append=(i > 0),
        )

    print("\nDone.")
