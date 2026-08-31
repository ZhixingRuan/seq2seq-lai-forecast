"""GHCND station data -> Gridded Daily data with IDW interpolation in projected LAEA space.

IDW is performed on a coarse grid (default 10 km), then bilinearly resampled to the
target 1-km LAI projected grid (from lai_aggregate.py output).
All distance calculations use true projected distances in km — no flat-Earth approximation.
"""
import os
import numpy as np
import pandas as pd
from ghcnd_station_analysis import parse_stations
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from pyproj import Transformer


# ── IDW parameters ────────────────────────────────────────────────────────────
MAX_DIST_PRCP = 50.0   # km
MIN_STATIONS  = 3
IDW_POWER     = 2
COARSE_RES_M  = 10_000  # IDW coarse grid spacing in metres

VARIABLES = ["PRCP"]


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


def build_coarse_grid(x_m, y_m, coarse_res_m=COARSE_RES_M):
    """Build a coarse 1-D x/y grid at coarse_res_m spacing over the fine domain."""
    x_c = np.arange(x_m[0], x_m[-1] + coarse_res_m * 0.5, coarse_res_m)
    y_c = np.arange(y_m[0], y_m[-1] + coarse_res_m * 0.5, coarse_res_m)
    return x_c, y_c


def bilinear_to_fine(coarse, x_c_km, y_c_km, x_f_km, y_f_km):
    """Bilinear resample from the coarse IDW grid to the fine 1-km projected grid.

    Parameters
    ----------
    coarse          : (ny_c, nx_c) IDW result
    x_c_km, y_c_km : 1-D coarse grid axes in km
    x_f_km, y_f_km : 1-D fine   grid axes in km

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


# ── IDW in projected km space ─────────────────────────────────────────────────

def idw_precipitation_proj(sxy_km, sval, grid_x2d_km, grid_y2d_km,
                            drizzle_threshold=0.1):
    """IDW precipitation interpolation using projected km distances.

    Parameters
    ----------
    sxy_km      : (N, 2) station [x_km, y_km]
    sval        : (N,)   observed precipitation values (NaN allowed)
    grid_x2d_km : (ny_c, nx_c) coarse grid x in km
    grid_y2d_km : (ny_c, nx_c) coarse grid y in km

    Returns
    -------
    (ny_c, nx_c) IDW field
    """
    valid = np.isfinite(sval) & (sval >= 0)
    sxy_km, sval = sxy_km[valid], sval[valid]
    if len(sval) < MIN_STATIONS:
        return np.full(grid_x2d_km.shape, np.nan)

    sval_log = np.log1p(sval)
    gxy = np.column_stack([grid_x2d_km.ravel(), grid_y2d_km.ravel()])

    tree = cKDTree(sxy_km)
    nbrs = tree.query_ball_point(gxy, r=MAX_DIST_PRCP)

    out_log = np.full(len(gxy), np.nan)
    for gi, nb in enumerate(nbrs):
        if len(nb) < MIN_STATIONS:
            continue
        nb = np.array(nb)
        d  = np.linalg.norm(gxy[gi] - sxy_km[nb], axis=1)
        if np.any(d == 0):
            out_log[gi] = sval_log[nb[d == 0][0]]
            continue
        w = 1.0 / d**IDW_POWER
        out_log[gi] = np.dot(w, sval_log[nb]) / w.sum()

    out = np.where(np.isfinite(out_log), np.expm1(out_log), np.nan)
    out = np.where(np.isfinite(out) & (out < drizzle_threshold), 0.0, out)
    return out.reshape(grid_x2d_km.shape)


# ── Per-day interpolation ─────────────────────────────────────────────────────

def interpolate_idw_proj(stations_df, var_df, x_c_km, y_c_km, x_f_km, y_f_km):
    """Interpolate one day's station values onto the fine 1-km projected grid.

    Parameters
    ----------
    stations_df         : DataFrame with station_id, lat, lon, x_km, y_km
    var_df              : DataFrame with station_id, date, <varname> (single date)
    x_c_km / y_c_km    : 1-D coarse IDW grid axes in km
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
    coarse = idw_precipitation_proj(sxy_km, sval, xx_c, yy_c)
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
    """Interpolate station PRCP onto the 1-km LAI grid for one year.

    IDW is run on a coarse grid at coarse_res_m spacing, then bilinearly
    resampled to the 1-km fine grid.

    Writes prcp (float32, zlib-compressed) to the zarr at out_path.
    Set append=True to extend an existing zarr along the time dimension;
    set append=False (default) to create a new zarr (overwrites if present).
    """
    print(f"Processing {year}...")

    # Load 1-km LAI projected grid
    x_m, y_m, crs = load_lai_grid(lai_zarr_path)
    x_f_km = x_m / 1000.0
    y_f_km = y_m / 1000.0

    # Build coarse IDW grid
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
            grid, used = interpolate_idw_proj(stations_df, day_df, x_c_km, y_c_km, x_f_km, y_f_km)
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
            attrs={"units": "mm", "long_name": var},
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
            "idw_power":            IDW_POWER,
            "idw_max_dist_km":      MAX_DIST_PRCP,
            "idw_min_stations":     MIN_STATIONS,
            "idw_coarse_res_m":     coarse_res_m,
        },
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    encoding = {"prcp": {"dtype": "float32", "compressors": {"name": "gzip", "configuration": {"level": 4}}}}
    encoding = {k: v for k, v in encoding.items() if k in ds}
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
    OUT_PATH          = os.path.join(OUTPUT_DIR, f"prcp_1km_idw{coarse_km}km.zarr")

    for i, year in enumerate(YEARS):
        process_yearly_data(
            year=year,
            lai_zarr_path=LAI_ZARR_PATH,
            station_data_dir=STATION_DATA_DIR,
            station_inventory_path=STATION_INVENTORY,
            out_path=OUT_PATH,
            append=(i > 0),
            coarse_res_m=COARSE_RES_M,
        )

    print("\nDone.")
