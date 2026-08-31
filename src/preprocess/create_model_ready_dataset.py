"""Combine LAI and weather together for model input.
"""

import os
import shutil

import numpy as np
import xarray as xr


def upsample_lai(
    lai_zarr,
    geo_zarr=None,   # raw lai zarr that holds 2-D latitude/longitude variables;
                     # if None, latitude/longitude are not added to the output
):
    # Upsample LAI to daily by forward filling
    lai_ds = xr.open_zarr(lai_zarr)
    lai_subset = lai_ds[["LAI_norm", "LAI_flag", "lat_norm", "lon_norm"]]

    daily_time = xr.date_range(
         start=str(lai_ds.time[0].values)[:10],
        end=str(lai_ds.time[-1].values)[:10],
        freq="1D",
        use_cftime=False
    )

    lai_time_vars = lai_subset[["LAI_norm", "LAI_flag"]]
    lai_static    = lai_subset[["lat_norm", "lon_norm"]]

    lai_daily = lai_time_vars.reindex(
        time=daily_time,
        method="ffill"    # carry forward to daily timesteps
    )

    # fix the flag for the made-up daily timesteps
    obs_times = lai_ds.time.values.astype("datetime64[ns]")
    # build a boolean mask: True if this daily timestep is a real 4-day obs
    is_real_obs = np.isin(daily_time, obs_times)   # shape: (n_days,)
    is_real_obs_da = xr.DataArray(is_real_obs, dims=["time"], coords={"time": lai_daily.time})

    # anywhere that is NOT a real obs timestep → set flag to 2
    lai_daily["LAI_flag"] = lai_daily["LAI_flag"].where(is_real_obs_da, other=2)

    # Add static vars back (lat_norm, lon_norm)
    lai_daily = xr.merge([lai_daily, lai_static])

    # Add real geographic coordinates from the raw lai zarr (time-invariant)
    if geo_zarr is not None:
        geo_ds = xr.open_zarr(geo_zarr)
        for var in ("latitude", "longitude"):
            if var in geo_ds:
                lai_daily[var] = geo_ds[var].astype(np.float32)
            else:
                print(f"  Warning: '{var}' not found in {geo_zarr}, skipping.")

    return lai_daily


def prepare_dataset(
    weather_zarr,
    lai_zarr,
    output_zarr,
    weather_vars=None,
    temporal_upsample_lai=True,
    chunks=None,
    geo_zarr=None,   # raw lai zarr that holds 2-D latitude/longitude variables
):

    if temporal_upsample_lai:
        lai_daily = upsample_lai(lai_zarr, geo_zarr=geo_zarr)
    else:
        lai_daily = xr.open_zarr(lai_zarr)
    weather_ds = xr.open_zarr(weather_zarr)

    if weather_vars is None:
        weather_vars = ['tmin_norm', 'tmax_norm', 'prcp_norm', 'prcp_14d_norm', 'prcp_30d_norm', 'prcp_60d_norm']
        if 'weather_flag' in weather_ds:
            weather_vars.append('weather_flag')
            print("Weather flags: using combined weather_flag")
        else:
            weather_vars += ['temp_flag', 'prcp_flag']
            print("Weather flags: using separate temp_flag and prcp_flag")

    weather_subset = weather_ds[weather_vars]

    # Rename weather spatial dims to match LAI's (lat, lon) so that isel and
    # merge operate on a single shared coordinate system.
    rename_map = {d: t for d, t in [('y', 'lat'), ('x', 'lon')] if d in weather_subset.dims}
    if rename_map:
        weather_subset = weather_subset.rename(rename_map)
        print(f"Renamed weather dims: {rename_map}")

    print(f"LAI:      {lai_daily.time.values[0]} → {lai_daily.time.values[-1]}")
    print(f"Weather:  {weather_ds.time.values[0]} → {weather_ds.time.values[-1]}")

    ds = xr.merge([weather_subset, lai_daily], join="inner")

    if ds.sizes["time"] == 0:
        raise ValueError("No overlapping time period between LAI and Weather data")

    print(f"Overlap:  {ds.time.values[0]} → {ds.time.values[-1]}")
    print(f"Days:     {ds.sizes['time']}")

    # Cast dtypes: flag variables → uint8 (binary), all others → float32
    all_vars = ["LAI_norm", "LAI_flag"] + weather_vars
    for var in all_vars:
        if "_flag" in var:
            ds[var] = ds[var].astype(np.uint8)
        else:
            ds[var] = ds[var].astype(np.float32)

    # Static 2-D vars — not in the time-indexed variable list, handled separately
    geo_vars = [v for v in ("latitude", "longitude", "lat_norm", "lon_norm") if v in ds]

    # Rechunk all data variables to the requested layout.
    # Explicit encoding overrides zarr chunk metadata carried over from the source
    # stores (xr.open_zarr preserves it, which would otherwise cause chunk mismatches).
    if chunks is not None:
        for var in all_vars:
            chunk_dict = dict(zip(ds[var].dims, chunks))
            ds[var] = ds[var].chunk(chunk_dict)
        encoding = {var: {"chunks": list(chunks)} for var in all_vars}
        # lat/lon are (lat, lon) only — use spatial chunk sizes from the last two dims
        spatial_chunks = list(chunks[-2:])
        for var in geo_vars:
            chunk_dict = dict(zip(ds[var].dims, spatial_chunks))
            ds[var] = ds[var].chunk(chunk_dict)
        for var in geo_vars:
            encoding[var] = {"chunks": spatial_chunks}
        ds.to_zarr(output_zarr, mode='w', encoding=encoding)
    else:
        ds.to_zarr(output_zarr, mode='w')


def rechunk_static_vars(output_zarr, vars=("lat_norm", "lon_norm"), spatial_chunks=(100, 100)):
    """Rechunk 2-D static variables (no time dim) in an existing zarr store in-place."""
    ds = xr.open_zarr(output_zarr)
    missing = [v for v in vars if v not in ds]
    if missing:
        raise ValueError(f"Variables not found in {output_zarr}: {missing}")

    subset = xr.Dataset({v: ds[v] for v in vars}).load()  # load into memory before deleting from store

    import zarr as _zarr
    store = _zarr.open(str(output_zarr), mode="a")
    for v in vars:
        if v in store:
            del store[v]

    for v in vars:
        chunk_dict = dict(zip(subset[v].dims, spatial_chunks))
        subset[v] = subset[v].chunk(chunk_dict)

    encoding = {v: {"chunks": list(spatial_chunks)} for v in vars}
    subset.to_zarr(output_zarr, mode="a", encoding=encoding)
    print(f"Rechunked {list(vars)} → {output_zarr}")


def restore_latlon_norm(geo_zarr, output_zarr, spatial_chunks=(100, 100)):
    """Recompute lat_norm/lon_norm from geo_zarr coordinates and write into output_zarr.

    Use this when lat_norm/lon_norm in output_zarr are corrupted (e.g. NaN after
    a failed rechunk). Reads lat/lon coordinate arrays from geo_zarr, applies the
    same CONUS min-max normalisation used in latlon_norm.py, then rechunks and
    writes the result back into output_zarr.
    """
    ds = xr.open_zarr(geo_zarr)
    lat_2d = ds["latitude"].values.astype(np.float32)   # (lat, lon) 2-D array
    lon_2d = ds["longitude"].values.astype(np.float32)

    lat_min, lat_max = float(lat_2d.min()), float(lat_2d.max())
    lon_min, lon_max = float(lon_2d.min()), float(lon_2d.max())

    lat_norm_2d = (lat_2d - lat_min) / (lat_max - lat_min)
    lon_norm_2d = (lon_2d - lon_min) / (lon_max - lon_min)

    subset = xr.Dataset({
        "lat_norm": xr.DataArray(lat_norm_2d, dims=["lat", "lon"]),
        "lon_norm": xr.DataArray(lon_norm_2d, dims=["lat", "lon"]),
    })

    import zarr as _zarr
    store = _zarr.open(str(output_zarr), mode="a")
    for v in ("lat_norm", "lon_norm"):
        if v in store:
            del store[v]

    for v in ("lat_norm", "lon_norm"):
        subset[v] = subset[v].chunk({"lat": spatial_chunks[0], "lon": spatial_chunks[1]})

    encoding = {v: {"chunks": list(spatial_chunks)} for v in ("lat_norm", "lon_norm")}
    subset.to_zarr(output_zarr, mode="a", encoding=encoding)
    print(f"lat_norm range: [{lat_norm_2d.min():.4f}, {lat_norm_2d.max():.4f}]")
    print(f"lon_norm range: [{lon_norm_2d.min():.4f}, {lon_norm_2d.max():.4f}]")
    print(f"Restored lat_norm/lon_norm → {output_zarr}")


def add_geo_to_existing_zarr(geo_zarr, output_zarr, spatial_chunks=(100, 100)):
    """Append latitude/longitude from geo_zarr into an already-built output_zarr.

    Safe to run on an existing zarr — uses mode='a' so all other variables
    are left untouched.  If latitude/longitude already exist in output_zarr
    they will be overwritten.
    """
    geo_ds = xr.open_zarr(geo_zarr)
    missing = [v for v in ("latitude", "longitude") if v not in geo_ds]
    if missing:
        raise ValueError(f"Variables not found in geo_zarr: {missing}")

    geo = xr.Dataset({
        "latitude":  geo_ds["latitude"].astype(np.float32),
        "longitude": geo_ds["longitude"].astype(np.float32),
    })
    encoding = {v: {"chunks": list(spatial_chunks)} for v in ("latitude", "longitude")}
    geo.to_zarr(output_zarr, mode="a", encoding=encoding)
    print(f"Added latitude/longitude → {output_zarr}")


def add_lc_to_dataset_annual(lc_zarr, output_zarr, spatial_chunks=(100, 100)):
    """Store LC as annual slices instead of daily — index at batch time."""
    import zarr as _zarr

    lc_ds = xr.open_zarr(lc_zarr).sortby("time")
    if hasattr(lc_ds.indexes["time"], "to_datetimeindex"):
        lc_ds["time"] = lc_ds.indexes["time"].to_datetimeindex()
    out_ds = xr.open_zarr(output_zarr)

    lc_vars = sorted(v for v in lc_ds if v.startswith("lc_"))
    if not lc_vars:
        raise ValueError(f"No 'lc_*' variables found in {lc_zarr}")

    lat_dim = "lat" if "lat" in out_ds.dims else "y"
    lon_dim = "lon" if "lon" in out_ds.dims else "x"

    lc_times = lc_ds.time.values  # (n_ann,) — keep as annual
    n_ann = len(lc_times)
    n_lat = out_ds.sizes[lat_dim]
    n_lon = out_ds.sizes[lon_dim]

    print(f"LC variables : {lc_vars}")
    print(f"LC years     : {n_ann} ({str(lc_times[0])[:10]} → {str(lc_times[-1])[:10]})")

    # Load annual LC into memory (~41 MB)
    print("Loading annual LC data...")
    lc_annual = {v: lc_ds[v].values for v in lc_vars}  # each (n_ann, lat, lon)

    store = _zarr.open_group(output_zarr, mode='a')
    for v in lc_vars:
        var_path = os.path.join(output_zarr, v)
        if os.path.exists(var_path):
            shutil.rmtree(var_path)

        # Annual time dimension, no daily expansion
        arr = store.create_array(
            v,
            shape=(n_ann, n_lat, n_lon),
            chunks=(1, *spatial_chunks),  # 1 year per chunk is fine here
            dtype='uint8',
            fill_value=0,
            dimension_names=['year', lat_dim, lon_dim],
            overwrite=True,
        )
        arr.attrs.update(lc_ds[v].attrs)
        arr.attrs['lc_years'] = [str(t)[:10] for t in lc_times]

        for ai in range(n_ann):
            arr[ai] = lc_annual[v][ai].astype(np.uint8)
            print(f"  {v} [{ai+1}/{n_ann}] {str(lc_times[ai])[:10]} written")

    print(f"\nAdded annual {lc_vars} → {output_zarr}")
    return [str(t)[:10] for t in lc_times]  # return year list for Dataset use



if __name__ == "__main__":
    weather_zarr = '/path/to/data-root/Data/station_data_2026/TX_domain_interpolated/1km/weather_norm.zarr'
    lai_zarr     = '/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km_processed.zarr'
    geo_zarr     = '/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km.zarr'  # raw zarr with latitude/longitude
    lc_zarr      = '/path/to/data-root/Data/Land_cover/lc_tx_1km_PFT_processed_6groups.zarr'
    output_zarr  = '/path/to/data-root/Data/model_input/tx_1km_rechunked.zarr'

    weather_vars = ['tmin_norm', 'tmax_norm', 'prcp_norm', 'prcp_14d_norm', 'temp_flag', 'prcp_flag']

    # prepare_dataset(weather_zarr, lai_zarr, output_zarr, weather_vars=weather_vars,
    #                chunks=(30, 100, 100), geo_zarr=geo_zarr)

    # add_lc_to_dataset_annual(lc_zarr, output_zarr)

    # rechunk_static_vars(output_zarr, vars=("lat_norm", "lon_norm"), spatial_chunks=(100, 100))

    restore_latlon_norm(geo_zarr, output_zarr, spatial_chunks=(100, 100))



