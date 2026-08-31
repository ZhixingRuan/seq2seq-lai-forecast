"""Aggregate MODIS landcover 500m to a Cartesian projected grid using majority resampling.

Resolution is specified in metres (e.g. 1000 = 1 km, 10000 = 10 km).
The output grid uses Lambert Azimuthal Equal-Area (LAEA) centred on the
domain, so grid cells are equal-area and distances are in metres.
Output coordinates are x / y in metres (easting / northing).

Output is written to a single Zarr store. All time steps (years) in the
input dataset are processed sequentially and appended along the time axis.
"""
import numpy as np
import xarray as xr
from pyresample import create_area_def
from pyresample.bucket import BucketResampler
from pyproj import Transformer
import dask.array as da
import pandas as pd
import os


# ── Private helpers ────────────────────────────────────────────────────────────

def _majority_resample(resampler, data_flat, n_y, n_x):
    """Return the majority (mode) landcover class for each target bucket."""
    idxs = resampler.idxs.compute()

    valid = (idxs >= 0) & (idxs < n_y * n_x)
    if data_flat.dtype.kind == 'f':
        valid &= np.isfinite(data_flat)

    src_tgt = idxs[valid]
    src_val = data_flat[valid]

    result = np.full(n_y * n_x, fill_value=np.nan, dtype=float)
    if src_tgt.size == 0:
        return result.reshape(n_y, n_x)

    df = pd.DataFrame({'tgt': src_tgt, 'val': src_val})
    majority = df.groupby('tgt')['val'].agg(lambda x: x.value_counts().index[0])
    result[majority.index.to_numpy()] = majority.to_numpy()
    return result.reshape(n_y, n_x)


def _build_cartesian_grid(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    resolution_m: float,
    proj_crs: str = None,
    lat_min: float = None,
    lat_max: float = None,
    lon_min: float = None,
    lon_max: float = None,
):
    """Build a projected Cartesian grid at *resolution_m* metres.

    Returns x_coords, y_coords, proj_crs (all in metres / proj string).
    """
    if lat_min is None: lat_min = float(src_lats.min())
    if lat_max is None: lat_max = float(src_lats.max())
    if lon_min is None: lon_min = float(src_lons.min())
    if lon_max is None: lon_max = float(src_lons.max())

    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    if proj_crs is None:
        proj_crs = (f"+proj=laea +lat_0={center_lat:.4f} "
                    f"+lon_0={center_lon:.4f} +datum=WGS84 +units=m")

    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    corners_lon = [lon_min, lon_max, lon_min, lon_max]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    x_c, y_c = (np.array(v) for v in transformer.transform(corners_lon, corners_lat))

    x_min = np.floor(x_c.min() / resolution_m) * resolution_m
    x_max = np.ceil (x_c.max() / resolution_m) * resolution_m
    y_min = np.floor(y_c.min() / resolution_m) * resolution_m
    y_max = np.ceil (y_c.max() / resolution_m) * resolution_m

    x_coords = np.arange(x_min, x_max + resolution_m * 0.5, resolution_m)
    y_coords = np.arange(y_min, y_max + resolution_m * 0.5, resolution_m)
    return x_coords, y_coords, proj_crs


def _build_resampler(src_lats_flat, src_lons_flat, x_coords, y_coords, proj_crs, resolution_m):
    """Construct the BucketResampler and back-projected lat/lon grids."""
    n_y, n_x = len(y_coords), len(x_coords)

    target_def = create_area_def(
        f'target_{int(resolution_m)}m', proj_crs,
        width=n_x, height=n_y,
        area_extent=[
            x_coords[0]  - resolution_m / 2, y_coords[0]  - resolution_m / 2,
            x_coords[-1] + resolution_m / 2, y_coords[-1] + resolution_m / 2,
        ],
    )
    resampler = BucketResampler(
        target_def,
        da.from_array(src_lons_flat, chunks=src_lons_flat.shape),
        da.from_array(src_lats_flat, chunks=src_lats_flat.shape),
    )

    xx, yy = np.meshgrid(x_coords, y_coords)
    inv_tr = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
    grid_lons, grid_lats = inv_tr.transform(xx.ravel(), yy.ravel())
    grid_lons = np.array(grid_lons).reshape(n_y, n_x)
    grid_lats = np.array(grid_lats).reshape(n_y, n_x)

    return resampler, grid_lons, grid_lats


# ── Public helpers ─────────────────────────────────────────────────────────────

def load_lai_grid(zarr_path: str):
    """Load the projected grid from a LAI zarr produced by lai_aggregate.py.

    Returns
    -------
    x_m  : 1-D easting  coordinates in metres  (the 'lon' dim)
    y_m  : 1-D northing coordinates in metres  (the 'lat' dim)
    crs  : projection string stored in zarr attrs
    """
    ds = xr.open_zarr(zarr_path)
    x_m = ds['lon'].values
    y_m = ds['lat'].values
    crs = ds.attrs['crs']
    ds.close()
    return x_m, y_m, crs


# ── Main function ──────────────────────────────────────────────────────────────

def aggregate_landcover(
    ds_lc: xr.Dataset,
    output_path: str,
    resolution_m: float = None,
    lc_var_name: str = 'LC_Type1',
    lat_var_name: str = 'lat',
    lon_var_name: str = 'lon',
    lai_zarr_path: str = None,
    proj_crs: str = None,
    lat_min: float = None,
    lat_max: float = None,
    lon_min: float = None,
    lon_max: float = None,
):
    """Aggregate MODIS 500 m landcover to a projected grid using majority resampling.

    All time steps (years) in ds_lc are processed sequentially and saved into
    a single Zarr store at output_path.

    Parameters
    ----------
    ds_lc         : xr.Dataset  MODIS landcover dataset with a 'time' dimension
                                (can contain multiple years)
    output_path   : str         destination zarr store path (e.g. 'lc_1km.zarr')
    resolution_m  : float       grid cell size in metres; inferred from lai_zarr_path
                                when not provided
    lc_var_name   : str         variable name in ds_lc  (default 'LC_Type1')
    lat_var_name  : str         latitude  coordinate name in ds_lc (default 'lat')
    lon_var_name  : str         longitude coordinate name in ds_lc (default 'lon')
    lai_zarr_path : str or None LAI zarr whose grid / CRS defines the output grid;
                                when set, resolution_m / proj_crs / bounds are ignored
    proj_crs      : str or None proj4 string; used only when lai_zarr_path is None
    lat/lon_min/max : float     optional bounding box; used only when lai_zarr_path is None
    """

    # ── Source coordinates ─────────────────────────────────────────────────────
    src_lats = ds_lc[lat_var_name].values
    src_lons = ds_lc[lon_var_name].values
    if src_lats.ndim == 1:
        src_lons_2d, src_lats_2d = np.meshgrid(src_lons, src_lats)
    else:
        src_lats_2d, src_lons_2d = src_lats, src_lons
    src_lats_flat = src_lats_2d.flatten()
    src_lons_flat = src_lons_2d.flatten()

    # ── Target grid ───────────────────────────────────────────────────────────
    if lai_zarr_path is not None:
        x_coords, y_coords, proj_crs = load_lai_grid(lai_zarr_path)
        resolution_m = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else resolution_m
    else:
        if resolution_m is None:
            raise ValueError("Provide either lai_zarr_path or resolution_m.")
        x_coords, y_coords, proj_crs = _build_cartesian_grid(
            src_lats_flat, src_lons_flat, resolution_m, proj_crs,
            lat_min, lat_max, lon_min, lon_max,
        )
    n_y, n_x = len(y_coords), len(x_coords)

    resampler, grid_lons, grid_lats = _build_resampler(
        src_lats_flat, src_lons_flat, x_coords, y_coords, proj_crs, resolution_m,
    )

    print(f"Projection : {proj_crs}")
    print(f"Target grid: {n_y} y × {n_x} x  @ {resolution_m:.0f} m/pixel")
    print(f"  x: {x_coords[0]:.0f} → {x_coords[-1]:.0f} m")
    print(f"  y: {y_coords[0]:.0f} → {y_coords[-1]:.0f} m")
    print(f"  lon: {grid_lons.min():.4f} → {grid_lons.max():.4f}°")
    print(f"  lat: {grid_lats.min():.4f} → {grid_lats.max():.4f}°")

    ds_attrs = {'crs': proj_crs, 'resolution_m': resolution_m}

    # Ensure 'time' is always a proper dimension
    if 'time' not in ds_lc.dims:
        ds_lc = ds_lc.expand_dims('time')
    time_steps = ds_lc['time'].values
    n_times = len(time_steps)
    print(f"Processing {n_times} time step(s): {time_steps}")

    # ── Loop over time steps ──────────────────────────────────────────────────
    for i, t in enumerate(time_steps):
        print(f"  [{i+1}/{n_times}] time={t} ...", end=' ', flush=True)

        lc_flat = ds_lc[lc_var_name].sel(time=t).values.flatten().astype(float)
        result  = _majority_resample(resampler, lc_flat, n_y, n_x)
        result  = result[::-1]   # flip N→S → S→N to match ascending y_coords

        ds_out = xr.Dataset(
            {'LC_aggregated': xr.DataArray(
                result[np.newaxis, ...],
                dims=['time', 'lat', 'lon'],
                coords={'time': [t], 'lat': y_coords, 'lon': x_coords},
                attrs={'long_name': 'Majority landcover class',
                       'source_var': lc_var_name,
                       'resolution_m': resolution_m},
            )},
            attrs=ds_attrs,
        )

        if i == 0:
            # First write: create the store and include static geo arrays
            ds_out['latitude'] = xr.DataArray(
                grid_lats, dims=['lat', 'lon'],
                coords={'lat': y_coords, 'lon': x_coords},
                attrs={'units': 'degrees_north'},
            )
            ds_out['longitude'] = xr.DataArray(
                grid_lons, dims=['lat', 'lon'],
                coords={'lat': y_coords, 'lon': x_coords},
                attrs={'units': 'degrees_east'},
            )
            ds_out.to_zarr(output_path, mode='w')
        else:
            # Subsequent writes: append along time, exclude static geo arrays
            ds_out.to_zarr(output_path, mode='a', append_dim='time')

        print("done")

    print(f'Saved {n_times} time step(s) -> {output_path}')


if __name__ == "__main__":
    LC_NC     = '/path/to/data-root/Data/Land_cover/MCD12Q1.061_500m_aid0001_PFT_2002-2025.nc'
    LAI_ZARR  = '/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km.zarr'
    OUT_ZARR  = '/path/to/data-root/Data/lc_tx_1km_PFT.zarr'

    # Load all years from the landcover file
    ds_lc = xr.open_dataset(LC_NC)

    aggregate_landcover(
        ds_lc,
        output_path   = OUT_ZARR,
        lc_var_name   = 'LC_Type5',
        lai_zarr_path = LAI_ZARR,
    )
