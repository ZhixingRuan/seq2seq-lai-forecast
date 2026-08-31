"""LAI processing pipeline for the 1-km projected LAI zarr (lai_tx_1km.zarr).

Steps (all in-memory, single write):
  1. lai_nan_fill : create LAI_flag (1=valid, 0=NaN), fill NaN → 0
  2. lai_norm     : z-score LAI using training-period valid pixels only
  3. coord_norm   : min-max normalize projected y/x coordinates to [0, 1]
                    NOTE: normalizes projected northing/easting (metres/km), NOT
                    geographic lat/lon in degrees — see Step 3 comment for details.

Output variables:
  LAI_aggregated  : NaN-filled raw LAI
  LAI_flag        : 1 = originally valid, 0 = was NaN
  LAI_norm        : z-scored LAI (invalid pixels set to 0.0)
  y_norm          : min-max normalized northing  (0 = south edge, 1 = north edge)
  x_norm          : min-max normalized easting   (0 = west edge,  1 = east edge)
"""

import xarray as xr
import numpy as np


def _spatial_dims(ds):
    """Return (y_dim, x_dim) — supports y/x (projected) or lat/lon."""
    for y, x in [("y", "x"), ("lat", "lon")]:
        if y in ds.dims and x in ds.dims:
            return y, x
    raise ValueError(f"Cannot find spatial dims in {list(ds.dims)}")


def process_lai(
    input_zarr: str,
    output_zarr: str,
    train_end: str,
    test_start: str,
):
    """Full LAI processing pipeline: nan-fill → normalize → coord-normalize.

    Parameters
    ----------
    input_zarr  : raw LAI zarr (e.g. lai_tx_1km.zarr from lai_aggregate.py)
    output_zarr : path to write processed output
    train_end   : last date of training period (e.g. "2022-12-31")
    test_start  : first date of test period    (e.g. "2023-01-01")
    """
    ds = xr.open_zarr(input_zarr).sortby("time")
    y_dim, x_dim = _spatial_dims(ds)

    times = ds.time.values
    print(f"Input zarr    : {input_zarr}")
    print(f"Dims          : {dict(ds.dims)}")
    print(f"Time range    : {str(times[0])[:10]} → {str(times[-1])[:10]}  ({len(times)} steps)")
    print(f"Training end  : {train_end}  |  Test start: {test_start}")

    lai = ds["LAI_aggregated"]

    # ── Step 1: NaN fill + flag ───────────────────────────────────────────────
    print("\n[1/3] NaN fill + flag...")

    flag = xr.where(lai.isnull(), 0, 1).astype(np.int8)
    flag.name = "LAI_flag"
    flag.attrs = {
        "long_name"       : "LAI validity flag",
        "description"     : "1 = valid (original value present), 0 = invalid (NaN filled with 0)",
        "flag_values"     : [0, 1],
        "flag_meanings"   : "invalid valid",
    }

    lai_filled = lai.fillna(0.0)
    lai_filled.attrs = lai.attrs.copy()
    lai_filled.attrs["_FillValue_note"] = (
        "NaN values replaced with 0; see LAI_flag for original NaN locations"
    )

    n_invalid = int(flag.sum().compute())
    n_total   = flag.size
    print(f"  Valid pixels  : {n_invalid:,} / {n_total:,}  ({100*n_invalid/n_total:.2f}%)")

    # ── Step 2: LAI normalization (z-score on training valid pixels) ──────────
    print("\n[2/3] LAI normalization (z-score)...")

    lai_train       = lai_filled.sel(time=slice(None, train_end))
    flag_train      = flag.sel(time=slice(None, train_end))
    lai_train_valid = lai_train.where(flag_train == 1)

    print(f"  Training period: {str(lai_train.time.values[0])[:10]} → {str(lai_train.time.values[-1])[:10]}")
    print(f"  Computing stats over valid pixels...")
    lai_mean = float(lai_train_valid.mean().compute())
    lai_std  = float(lai_train_valid.std().compute())
    print(f"  LAI mean : {lai_mean:.4f}")
    print(f"  LAI std  : {lai_std:.4f}")

    lai_norm = (lai_filled - lai_mean) / lai_std
    lai_norm = lai_norm.where(flag == 1, other=0.0)   # invalid pixels → 0
    lai_norm.attrs = lai.attrs.copy()
    lai_norm.attrs.update({
        "long_name"       : "Normalized LAI (z-score)",
        "normalization"   : "z-score",
        "norm_mean"       : lai_mean,
        "norm_std"        : lai_std,
        "norm_computed_on": f"valid training pixels up to {train_end}",
        "nan_handling"    : "invalid pixels (LAI_flag=0) set to 0.0",
    })

    # Sanity check (deferred — printed after save)
    lai_test = lai_norm.sel(time=slice(test_start, None)).where(
        flag.sel(time=slice(test_start, None)) == 1)

    # ── Step 3: coordinate normalization (min-max → [0, 1]) ──────────────────
    # NOTE: this normalizes the *projected* y/x coordinates (units: metres / km),
    # not geographic lat/lon in degrees.  The data lives on a projected CRS
    # (e.g. EPSG:3083 Texas Centric Albers Equal Area), so y_dim/x_dim are
    # northing/easting, not latitude/longitude.
    # If you intended to feed geographic coordinates to the model, regenerate
    # the zarr from lat/lon values instead (see lai_aggregate.py).
    print("\n[3/3] Coordinate normalization (min-max)...")

    y_vals = ds[y_dim].values
    x_vals = ds[x_dim].values

    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    x_min, x_max = float(x_vals.min()), float(x_vals.max())

    y_norm_1d = (y_vals - y_min) / (y_max - y_min)
    x_norm_1d = (x_vals - x_min) / (x_max - x_min)

    # Broadcast to 2D (y, x)
    y_norm_2d, x_norm_2d = np.meshgrid(y_norm_1d, x_norm_1d, indexing="ij")

    y_norm_da = xr.DataArray(
        y_norm_2d, dims=[y_dim, x_dim],
        attrs={
            "long_name"    : f"Normalized {y_dim} coordinate",
            "normalization": "min-max",
            "norm_min"     : y_min,
            "norm_max"     : y_max,
            "units_original": ds[y_dim].attrs.get("units", "m"),
        },
    )
    x_norm_da = xr.DataArray(
        x_norm_2d, dims=[y_dim, x_dim],
        attrs={
            "long_name"    : f"Normalized {x_dim} coordinate",
            "normalization": "min-max",
            "norm_min"     : x_min,
            "norm_max"     : x_max,
            "units_original": ds[x_dim].attrs.get("units", "m"),
        },
    )

    print(f"  {y_dim}_norm range: [{y_norm_1d.min():.4f}, {y_norm_1d.max():.4f}]")
    print(f"  {x_dim}_norm range: [{x_norm_1d.min():.4f}, {x_norm_1d.max():.4f}]")

    # ── Build output dataset and write ────────────────────────────────────────
    print("\nBuilding output dataset...")

    chunk_t  = 1
    chunk_y  = len(ds[y_dim])
    chunk_x  = len(ds[x_dim])
    out_chunks = (chunk_t, chunk_y, chunk_x)

    ds_out = xr.Dataset(
        {
            "LAI_aggregated": lai_filled,
            "LAI_flag"      : flag,
            "LAI_norm"      : lai_norm,
            f"{y_dim}_norm" : y_norm_da,
            f"{x_dim}_norm" : x_norm_da,
        },
        coords=ds.coords,
        attrs=ds.attrs,
    )

    encoding = {
        "LAI_aggregated": {"chunks": out_chunks},
        "LAI_flag"      : {"chunks": out_chunks, "dtype": "int8"},
        "LAI_norm"      : {"chunks": out_chunks},
        f"{y_dim}_norm" : {"chunks": (chunk_y, chunk_x)},
        f"{x_dim}_norm" : {"chunks": (chunk_y, chunk_x)},
    }

    ds_out.to_zarr(output_zarr, mode="w", encoding=encoding)
    print(f"\nSaved → {output_zarr}")

    # ── Sanity check ─────────────────────────────────────────────────────────
    ds_check = xr.open_zarr(output_zarr).sortby("time")
    flag_c   = ds_check["LAI_flag"]

    train_valid = ds_check["LAI_norm"].sel(time=slice(None, train_end)).where(
        flag_c.sel(time=slice(None, train_end)) == 1)
    test_valid  = ds_check["LAI_norm"].sel(time=slice(test_start, None)).where(
        flag_c.sel(time=slice(test_start, None)) == 1)

    print("\n--- Sanity check ---")
    print(f"  Train LAI_norm mean (≈0.0) : {float(train_valid.mean().compute()):.4f}")
    print(f"  Train LAI_norm std  (≈1.0) : {float(train_valid.std().compute()):.4f}")
    print(f"  Test  LAI_norm mean        : {float(test_valid.mean().compute()):.4f}")
    n_invalid_out = int((flag_c == 0).sum().compute())
    print(f"  LAI_flag=0 (invalid) cells : {n_invalid_out:,}")


if __name__ == "__main__":
    INPUT_ZARR  = "/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km.zarr"
    OUTPUT_ZARR = "/path/to/data-root/Data/LAI_TX_1km/lai_tx_1km_processed_.zarr"
    TRAIN_END   = "2019-12-31"
    TEST_START  = "2020-01-01"

    process_lai(INPUT_ZARR, OUTPUT_ZARR, train_end=TRAIN_END, test_start=TEST_START)
