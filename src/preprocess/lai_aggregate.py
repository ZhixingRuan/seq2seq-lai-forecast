"""Aggregate MODIS_LAI 500m to a Cartesian projected grid.

Resolution is specified in metres (e.g. 1000 = 1 km, 10000 = 10 km).
The output grid uses Lambert Azimuthal Equal-Area (LAEA) centred on the
domain, so grid cells are equal-area and distances are in metres.
Output coordinates are x / y in metres (easting / northing).
"""
import numpy as np
import xarray as xr
from pyresample import create_area_def
from pyresample.bucket import BucketResampler
from pyproj import Transformer
import dask.array as da
import pandas as pd
import os


# FparLai_QC: values below are "Significant clouds WERE present"
mask_out_qc = [8, 10, 40, 42, 73, 75, 105, 107]
# FparExtra_QC: values below are "0 LAND AggrQC (3,5) values {001}" in LandSea Pass-Thru
mask_in_extra_qc = [0, 4, 8, 12, 32, 36, 48, 52, 64, 68, 72, 76,
                    128, 132, 136, 140, 160, 164, 176, 180,
                    192, 196, 200, 204]


def _lai_qc_mask(lai_da, qc_da, extra_qc_da):
    mask       = ~qc_da.isin(mask_out_qc)
    mask_extra = extra_qc_da.isin(mask_in_extra_qc)
    return lai_da.where(mask & mask_extra)


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

    Parameters
    ----------
    src_lats, src_lons : 1-D source coordinate arrays (degrees)
    resolution_m       : grid cell size in metres (e.g. 1000, 10000)
    proj_crs           : optional proj4/WKT string; defaults to LAEA
                         centred on the domain
    lat/lon_min/max    : optional bounding box in degrees; defaults to
                         full extent of src_lats / src_lons

    Returns
    -------
    x_coords  : np.ndarray  easting  coordinates of grid centres (m)
    y_coords  : np.ndarray  northing coordinates of grid centres (m)
    proj_crs  : str         projection string used
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

    # Project the four bounding-box corners to metres
    transformer  = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
    corners_lon  = [lon_min, lon_max, lon_min, lon_max]
    corners_lat  = [lat_min, lat_min, lat_max, lat_max]
    x_c, y_c    = (np.array(v) for v in transformer.transform(corners_lon, corners_lat))

    # Snap to resolution multiples so grid edges are clean
    x_min = np.floor(x_c.min() / resolution_m) * resolution_m
    x_max = np.ceil (x_c.max() / resolution_m) * resolution_m
    y_min = np.floor(y_c.min() / resolution_m) * resolution_m
    y_max = np.ceil (y_c.max() / resolution_m) * resolution_m

    x_coords = np.arange(x_min, x_max + resolution_m * 0.5, resolution_m)
    y_coords = np.arange(y_min, y_max + resolution_m * 0.5, resolution_m)

    return x_coords, y_coords, proj_crs


def aggregate_lai(
    ds_lai: xr.Dataset,
    output_path: str,
    resolution_m: float,
    first_zarr: bool = True,
    lat_var_name: str = 'lat',
    lon_var_name: str = 'lon',
    proj_crs: str = None,
    lat_min: float = None,
    lat_max: float = None,
    lon_min: float = None,
    lon_max: float = None,
):
    """Aggregate MODIS 500 m LAI to a Cartesian projected grid.

    Parameters
    ----------
    ds_lai       : xr.Dataset  source MODIS dataset (lat/lon in degrees)
    output_path  : str         output zarr path
    resolution_m : float       target cell size in metres
                               (e.g. 500, 1000, 5000, 10000)
    first_zarr   : bool        True = create new zarr; False = append
    proj_crs     : str or None custom proj4 string; None = LAEA on domain
    lat/lon_min/max : optional bounding box in degrees

    Output zarr dimensions: (time, y, x)
    Output coordinates   : x [m easting], y [m northing]
    Projection stored in dataset attributes as 'crs'.
    """
    if first_zarr and os.path.exists(output_path):
        raise FileExistsError(
            f'{output_path} already exists. Delete it first or choose a different path.'
        )

    # ── Source coordinates ─────────────────────────────────────────────────────
    src_lats = ds_lai[lat_var_name].values
    src_lons = ds_lai[lon_var_name].values
    if src_lats.ndim == 1:
        src_lons_2d, src_lats_2d = np.meshgrid(src_lons, src_lats)
    else:
        src_lats_2d, src_lons_2d = src_lats, src_lons
    src_lats_flat = src_lats_2d.flatten()
    src_lons_flat = src_lons_2d.flatten()

    # ── Build projected target grid ────────────────────────────────────────────
    x_coords, y_coords, proj_crs = _build_cartesian_grid(
        src_lats_flat, src_lons_flat, resolution_m, proj_crs,
        lat_min, lat_max, lon_min, lon_max,
    )
    n_y, n_x = len(y_coords), len(x_coords)

    target_def = create_area_def(
        f'target_{int(resolution_m)}m',
        proj_crs,
        width  = n_x,
        height = n_y,
        area_extent=[
            x_coords[0]  - resolution_m / 2,
            y_coords[0]  - resolution_m / 2,
            x_coords[-1] + resolution_m / 2,
            y_coords[-1] + resolution_m / 2,
        ]
    )

    src_lons_dask = da.from_array(src_lons_flat, chunks=src_lons_flat.shape)
    src_lats_dask = da.from_array(src_lats_flat, chunks=src_lats_flat.shape)
    resampler = BucketResampler(target_def, src_lons_dask, src_lats_dask)

    # Back-project every grid-centre (x, y) → (lon, lat) once before the loop.
    # x_coords and y_coords are 1-D; meshgrid gives the 2-D (n_y, n_x) arrays.
    xx, yy = np.meshgrid(x_coords, y_coords)   # (n_y, n_x)
    inv_transformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
    grid_lons, grid_lats = inv_transformer.transform(xx.ravel(), yy.ravel())
    grid_lons = np.array(grid_lons).reshape(n_y, n_x)
    grid_lats = np.array(grid_lats).reshape(n_y, n_x)

    print(f"Projection : {proj_crs}")
    print(f"Target grid: {n_y} y × {n_x} x  @ {resolution_m:.0f} m/pixel")
    print(f"  x: {x_coords[0]:.0f} → {x_coords[-1]:.0f} m")
    print(f"  y: {y_coords[0]:.0f} → {y_coords[-1]:.0f} m")
    print(f"  lon: {grid_lons.min():.4f} → {grid_lons.max():.4f}°")
    print(f"  lat: {grid_lats.min():.4f} → {grid_lats.max():.4f}°")

    if not first_zarr:
        with xr.open_zarr(output_path) as ds_existing:
            existing_times = ds_existing.time.values

    for i in range(len(ds_lai.time)):
        time_val = ds_lai.time.isel(time=i).values.item()
        time_val_converted = pd.Timestamp(
            time_val.year, time_val.month, time_val.day
        ).to_datetime64()

        if not first_zarr:
            if time_val_converted in existing_times:
                print(f'Time {time_val_converted} exists. Skipping')
                continue

        lai_masked = _lai_qc_mask(
            ds_lai.isel(time=i).Lai_500m,
            ds_lai.isel(time=i).FparLai_QC,
            ds_lai.isel(time=i).FparExtra_QC,
        )
        result = resampler.get_average(lai_masked.values)[::-1]  # flip N→S to S→N to match ascending y_coords

        lai_da = xr.Dataset(
            {'LAI_aggregated': xr.DataArray(
                result[np.newaxis, ...],          # (1, n_y, n_x)
                dims   = ['time', 'lat', 'lon'],
                coords = {
                    'time': [time_val_converted],
                    'lat' : y_coords,             # projected y in metres
                    'lon' : x_coords,             # projected x in metres
                },
                attrs = {'units': 'LAI', 'resolution_m': resolution_m},
            )},
            attrs = {'crs': proj_crs, 'resolution_m': resolution_m},
        )

        if i == 0 and first_zarr:
            lai_da.to_zarr(
                output_path,
                mode     = 'w',
                encoding = {'LAI_aggregated': {'chunks': (1, n_y, n_x)}},
            )
        else:
            lai_da.to_zarr(output_path, append_dim='time')

        print(f'Finished time {time_val_converted}')
        del lai_masked, result

    # Write latitude / longitude once as static (lat, lon) variables —
    # they are time-invariant so no need to repeat per timestep.
    if not first_zarr:
        return

    geo = xr.Dataset(
        {
            'latitude': xr.DataArray(
                grid_lats, dims=['lat', 'lon'],
                coords={'lat': y_coords, 'lon': x_coords},
                attrs={'units': 'degrees_north'},
            ),
            'longitude': xr.DataArray(
                grid_lons, dims=['lat', 'lon'],
                coords={'lat': y_coords, 'lon': x_coords},
                attrs={'units': 'degrees_east'},
            ),
        }
    )
    geo.to_zarr(
        output_path, mode='a',
        encoding={
            'latitude':  {'chunks': (n_y, n_x)},
            'longitude': {'chunks': (n_y, n_x)},
        },
    )


if __name__ == "__main__":
    # lai = '/path/to/data-root/Data/LAI_TX/MCD15A3H.061_500m_aid0001_20231231_20251231.nc'
    # lai = '/path/to/data-root/Data/LAI_TX/MCD15A3H.061_500m_aid0001_20191231_20231231.nc'
    # lai = '/path/to/data-root/Data/LAI_TX/MCD15A3H.061_500m_aid0001_20141231_20191231.nc'
    lai = '/path/to/data-root/Data/LAI_TX/MCD15A3H.061_500m_aid0001_2002_2014.nc'

    ds_lai = xr.open_dataset(lai)

    # ── 1 km example ──────────────────────────────────────────────────────────
    aggregate_lai(
        ds_lai,
        output_path  = '/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km.zarr',
        resolution_m = 1000,
        first_zarr   = False #True,
    )

    # ── 10 km example (uncomment to run) ──────────────────────────────────────
    # aggregate_lai(
    #     ds_lai,
    #     output_path  = '/path/to/data-root/gfs_trials/lai_aggregated/lai_tx_10km.zarr',
    #     resolution_m = 10000,
    #     first_zarr   = True,
    # )
