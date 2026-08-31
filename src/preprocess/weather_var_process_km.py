"""Normalize tmax, tmin, precip from projected (y/x) zarrs. Add rolling prcp sums.
z-score.

Inputs:
  tmax_tmin_zarr : multi-year zarr with tmax, tmin  (y, x dims in metres)
  prcp_zarr      : multi-year zarr with prcp         (y, x dims in metres)

Precip: log(1+x) first, then z-score.
tp_reconstructed = np.expm1(tp_norm * tp_std + tp_mean)
"""

import gc

import dask
import xarray as xr
import numpy as np

MAX_GAP_CELLS = 10    # max contiguous NaN cells to fill with interpolate_na; for rolling prcp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spatial_dims(ds):
    """Return (y_dim, x_dim) names — supports both lat/lon and y/x conventions."""
    for y, x in [("y", "x"), ("lat", "lon")]:
        if y in ds.dims and x in ds.dims:
            return y, x
    raise ValueError(f"Cannot find spatial dims in {list(ds.dims)}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interp_norm_flag(da, y_dim, x_dim, target_chunks, max_gap):
    """Spatially interpolate, return (norm_filled, flag) — each computed separately."""
    interp = (
        da.chunk({x_dim: -1})
          .interpolate_na(dim=x_dim, method="linear", max_gap=max_gap)
          .chunk({y_dim: -1})
          .interpolate_na(dim=y_dim, method="linear", max_gap=max_gap)
          .chunk(target_chunks)
    )
    flag   = (~interp.isnull()).astype(float).chunk(target_chunks)
    filled = interp.fillna(0.0)
    return filled, flag


def _make_attrs(orig_attrs, method, mean, std, train_end, note=None):
    attrs = orig_attrs.copy()
    attrs.update({
        "normalization"   : method,
        "norm_mean"       : mean,
        "norm_std"        : std,
        "norm_computed_on": f"training data up to {train_end}",
        "nan_handling"    : "interpolate_na(x+y); large gaps filled with 0.0 and flagged",
    })
    if note:
        attrs["norm_note"] = note
    return attrs


# ── Step 1: normalize ────────────────────────────────────────────────────────

def normalize_weather(
    tmax_tmin_zarr: str,
    prcp_zarr: str,
    output_zarr: str,
    train_end: str,
    test_start: str,
    max_gap: int = MAX_GAP_CELLS,
):
    """Normalize tmax, tmin, prcp and write to output_zarr.

    tmax, tmin : plain z-score
    prcp       : log1p then z-score

    Each variable is processed and written independently to minimize memory use.
    NaN handling:
      1. interpolate_na along x then y for small gaps (≤ max_gap cells)
      2. remaining NaNs flagged (temp_flag, prcp_flag) and filled with 0.0
    """
    ds_tt   = xr.open_zarr(tmax_tmin_zarr)[["tmax", "tmin"]].sortby("time")
    ds_prcp = xr.open_zarr(prcp_zarr)[["prcp"]].sortby("time")
    y_dim, x_dim = _spatial_dims(ds_tt)

    times = ds_tt.time.values
    print(f"  Time range : {str(times[0])[:10]} → {str(times[-1])[:10]}  ({len(times)} steps)")
    print(f"  Training   : up to {train_end}  |  Test from: {test_start}")

    target_chunks = {"time": 1, y_dim: len(ds_tt[y_dim]), x_dim: len(ds_tt[x_dim])}
    out_chunks    = [target_chunks[k] for k in ("time", y_dim, x_dim)]

    # ── tmax ──────────────────────────────────────────────────────────────────
    print("\n  [1/5] tmax_norm...")
    tmax = ds_tt["tmax"]
    tmax_mean = float(tmax.sel(time=slice(None, train_end)).mean().compute())
    tmax_std  = float(tmax.sel(time=slice(None, train_end)).std().compute())
    print(f"    stats: mean={tmax_mean:.4f}  std={tmax_std:.4f}")
    tmax_norm, _ = _interp_norm_flag((tmax - tmax_mean) / tmax_std, y_dim, x_dim, target_chunks, max_gap)
    tmax_norm.attrs = _make_attrs(tmax.attrs, "z-score", tmax_mean, tmax_std, train_end)
    xr.Dataset({"tmax_norm": tmax_norm}, coords=ds_tt.coords).to_zarr(
        output_zarr, mode="w", encoding={"tmax_norm": {"chunks": out_chunks}})
    del tmax, tmax_norm
    print(f"    saved → {output_zarr}")

    # ── tmin ──────────────────────────────────────────────────────────────────
    print("\n  [2/5] tmin_norm...")
    tmin = ds_tt["tmin"]
    tmin_mean = float(tmin.sel(time=slice(None, train_end)).mean().compute())
    tmin_std  = float(tmin.sel(time=slice(None, train_end)).std().compute())
    print(f"    stats: mean={tmin_mean:.4f}  std={tmin_std:.4f}")
    tmin_norm, _ = _interp_norm_flag((tmin - tmin_mean) / tmin_std, y_dim, x_dim, target_chunks, max_gap)
    tmin_norm.attrs = _make_attrs(tmin.attrs, "z-score", tmin_mean, tmin_std, train_end)
    xr.Dataset({"tmin_norm": tmin_norm}).to_zarr(
        output_zarr, mode="a", encoding={"tmin_norm": {"chunks": out_chunks}})
    del tmin, tmin_norm
    print(f"    saved")

    # ── temp_flag (tmax & tmin both valid after interp) ───────────────────────
    print("\n  [3/5] temp_flag...")
    tmax_interp = (ds_tt["tmax"].chunk({x_dim: -1})
                                .interpolate_na(dim=x_dim, method="linear", max_gap=max_gap)
                                .chunk({y_dim: -1})
                                .interpolate_na(dim=y_dim, method="linear", max_gap=max_gap))
    tmin_interp = (ds_tt["tmin"].chunk({x_dim: -1})
                                .interpolate_na(dim=x_dim, method="linear", max_gap=max_gap)
                                .chunk({y_dim: -1})
                                .interpolate_na(dim=y_dim, method="linear", max_gap=max_gap))
    temp_flag = ((~tmax_interp.isnull()) & (~tmin_interp.isnull())).astype(float).chunk(target_chunks)
    temp_flag.attrs = {"description": "1 = tmax and tmin both valid after spatial interpolation, 0 = gap"}
    xr.Dataset({"temp_flag": temp_flag}).to_zarr(
        output_zarr, mode="a", encoding={"temp_flag": {"chunks": out_chunks}})
    del tmax_interp, tmin_interp, temp_flag
    print(f"    saved")

    # ── prcp_norm ─────────────────────────────────────────────────────────────
    print("\n  [4/5] prcp_norm...")
    prcp     = ds_prcp["prcp"]
    prcp_log = np.log1p(prcp)
    prcp_mean = float(prcp_log.sel(time=slice(None, train_end)).mean().compute())
    prcp_std  = float(prcp_log.sel(time=slice(None, train_end)).std().compute())
    print(f"    stats: mean={prcp_mean:.4f}  std={prcp_std:.4f}  (log1p-transformed)")
    prcp_norm, _ = _interp_norm_flag((prcp_log - prcp_mean) / prcp_std, y_dim, x_dim, target_chunks, max_gap)
    prcp_norm.attrs = _make_attrs(prcp.attrs, "log1p then z-score", prcp_mean, prcp_std, train_end,
                                  note="apply log1p before inverting: x = expm1(norm * std + mean)")
    xr.Dataset({"prcp_norm": prcp_norm}).to_zarr(
        output_zarr, mode="a", encoding={"prcp_norm": {"chunks": out_chunks}})
    del prcp_log, prcp_norm
    print(f"    saved")

    # ── prcp_flag ─────────────────────────────────────────────────────────────
    print("\n  [5/5] prcp_flag...")
    prcp_interp = (prcp.chunk({x_dim: -1})
                       .interpolate_na(dim=x_dim, method="linear", max_gap=max_gap)
                       .chunk({y_dim: -1})
                       .interpolate_na(dim=y_dim, method="linear", max_gap=max_gap))
    prcp_flag = (~prcp_interp.isnull()).astype(float).chunk(target_chunks)
    prcp_flag.attrs = {"description": "1 = prcp valid after spatial interpolation, 0 = gap"}
    xr.Dataset({"prcp_flag": prcp_flag}).to_zarr(
        output_zarr, mode="a", encoding={"prcp_flag": {"chunks": out_chunks}})
    del prcp, prcp_interp, prcp_flag
    print(f"    saved")

    print(f"\nDone → {output_zarr}")

    # Sanity check
    ds_check = xr.open_zarr(output_zarr).sortby("time")
    print(f"\n--- Sanity check (train should be ~0.0 mean, ~1.0 std on valid pixels) ---")
    for var, flag_var in [("tmax_norm", "temp_flag"), ("tmin_norm", "temp_flag"), ("prcp_norm", "prcp_flag")]:
        flag_da  = ds_check[flag_var]
        train_da = ds_check[var].sel(time=slice(None, train_end)).where(
                       flag_da.sel(time=slice(None, train_end)) == 1)
        test_da  = ds_check[var].sel(time=slice(test_start, None)).where(
                       flag_da.sel(time=slice(test_start, None)) == 1)
        print(f"  {var}  (masked by {flag_var}):")
        print(f"    train mean={float(train_da.mean().compute()):.4f}  std={float(train_da.std().compute()):.4f}")
        print(f"    test  mean={float(test_da.mean().compute()):.4f}  (can differ)")

    temp_gap = float((ds_check["temp_flag"] == 0).mean().compute())
    prcp_gap = float((ds_check["prcp_flag"] == 0).mean().compute())
    print(f"\n  temp_flag=0 fraction: {temp_gap:.4f} ({temp_gap*100:.2f}%)")
    print(f"  prcp_flag=0 fraction: {prcp_gap:.4f} ({prcp_gap*100:.2f}%)")


# ── Step 2: rolling prcp sums ────────────────────────────────────────────────

def add_rolling_prcp(
    prcp_zarr: str,
    output_zarr: str,
    train_end: str,
    test_start: str,
    windows: list[int] = (14, 30, 60),
    max_gap: int = MAX_GAP_CELLS,
):
    """Append rolling prcp sums (z-scored) to output_zarr.

    Parameters
    ----------
    prcp_zarr : path to zarr with prcp  (from interpolate_IDW_km.py)
    windows   : list of window sizes in days, e.g. [14, 30, 60] or [30]
                Each window W produces prcp_{W}d_norm (z-scored).

    Pipeline: raw prcp → interpolate_na(x+y) → rolling sum → z-score (train stats) → fillna(0.0)
    Must be called after normalize_weather so that output_zarr already exists.
    """
    ds_in = xr.open_zarr(prcp_zarr)[["prcp"]].sortby("time")
    prcp  = ds_in["prcp"]
    y_dim, x_dim = _spatial_dims(ds_in)

    times = ds_in.time.values
    print(f"  Time range: {str(times[0])[:10]} → {str(times[-1])[:10]}  ({len(times)} steps)")
    print(f"  Training period: up to {train_end}  |  Test from: {test_start}")

    ds_out_base   = xr.open_zarr(output_zarr).sortby("time")
    target_chunks = {"time": 1, y_dim: len(ds_out_base[y_dim]), x_dim: len(ds_out_base[x_dim])}

    # Use a larger time chunk for rolling to avoid boundary artifacts,
    # but NOT time: -1 which would load all timesteps into memory at once.
    # Use spatial tiles instead of full spatial dims to limit peak memory:
    #   tile × tile × roll_time_chunk × 4 bytes  (e.g. 256×256×365×4 ≈ 96 MB per chunk)
    roll_time_chunk = max(365, max(windows) * 4)
    ny = len(ds_out_base[y_dim])
    nx = len(ds_out_base[x_dim])
    prcp_interp = (
        prcp.chunk({x_dim: -1})
        .interpolate_na(dim=x_dim, method="linear", max_gap=max_gap)
        .chunk({y_dim: -1})
        .interpolate_na(dim=y_dim, method="linear", max_gap=max_gap)
        .chunk({"time": roll_time_chunk, y_dim: ny, x_dim: nx})
    )

    out_chunks = [target_chunks[k] for k in ("time", y_dim, x_dim)]

    for W in windows:
        print(f"\n  prcp_{W}d_norm...")
        min_p = max(1, W // 2)
        raw   = prcp_interp.rolling(time=W, min_periods=min_p).sum()

        raw_train = raw.sel(time=slice(None, train_end))
        mean, std = dask.compute(raw_train.mean(), raw_train.std())  # single scheduler pass
        mean, std = float(mean), float(std)
        del raw_train
        print(f"    stats: mean={mean:.4f}  std={std:.4f}")

        norm_key     = f"prcp_{W}d_norm"
        norm_chunked = ((raw - mean) / std).fillna(0.0).chunk(target_chunks)
        norm_chunked.attrs = {
            "normalization"   : f"z-score of {W}-day rolling sum",
            "norm_mean"       : mean,
            "norm_std"        : std,
            "norm_computed_on": f"training data up to {train_end}",
            "rolling_input"   : "raw prcp (spatially interpolated)",
            "nan_handling"    : "filled with 0.0 after normalization",
        }
        xr.Dataset({norm_key: norm_chunked}).to_zarr(
            output_zarr, mode="a", encoding={norm_key: {"chunks": out_chunks}})
        del raw, norm_chunked
        gc.collect()
        print(f"    saved {norm_key}")

    print(f"\nDone → {output_zarr}")

    # Sanity check
    ds_check  = xr.open_zarr(output_zarr).sortby("time")
    norm_keys = [f"prcp_{W}d_norm" for W in windows]
    print("\n--- Sanity check (train should be ~0.0 mean, ~1.0 std) ---")
    for var in norm_keys:
        train_da = ds_check[var].sel(time=slice(None, train_end))
        test_da  = ds_check[var].sel(time=slice(test_start, None))
        print(f"  {var}:")
        print(f"    train mean={float(train_da.mean().compute()):.4f}  std={float(train_da.std().compute()):.4f}")
        print(f"    test  mean={float(test_da.mean().compute()):.4f}  (can differ)")


# ── Step 4: combine flags ─────────────────────────────────────────────────────

def check_flag_agreement(weather_zarr: str, disagree_threshold: float = 0.05):
    """Print how often temp_flag and prcp_flag agree / disagree.

    Case 1: temp=1, prcp=1  → all good
    Case 2: temp=0, prcp=0  → all missing
    Case 3: temp=1, prcp=0  → temp ok, precip missing
    Case 4: temp=0, prcp=1  → precip ok, temp missing

    If cases 3+4 < disagree_threshold: combines both into a single weather_flag
    (1 = both valid) and writes it back to the zarr.
    """
    ds = xr.open_zarr(weather_zarr)
    t = ds["temp_flag"].values.ravel()
    p = ds["prcp_flag"].values.ravel()

    total = len(t)
    c1 = int(((t == 1) & (p == 1)).sum())
    c2 = int(((t == 0) & (p == 0)).sum())
    c3 = int(((t == 1) & (p == 0)).sum())
    c4 = int(((t == 0) & (p == 1)).sum())

    disagree_frac = (c3 + c4) / total

    print(f"\n--- Flag agreement check ({total:,} total pixel-timesteps) ---")
    print(f"  Case 1 (temp=1, prcp=1): {c1:>10,}  ({100*c1/total:5.2f}%)")
    print(f"  Case 2 (temp=0, prcp=0): {c2:>10,}  ({100*c2/total:5.2f}%)")
    print(f"  Case 3 (temp=1, prcp=0): {c3:>10,}  ({100*c3/total:5.2f}%)")
    print(f"  Case 4 (temp=0, prcp=1): {c4:>10,}  ({100*c4/total:5.2f}%)")
    print(f"\n  Disagreement (cases 3+4): {disagree_frac*100:.2f}%")

    if disagree_frac < disagree_threshold:
        print("  → flags nearly identical; combining into a single weather_flag.")
        weather_flag = ((ds["temp_flag"] == 1) & (ds["prcp_flag"] == 1)).astype(float)
        weather_flag.attrs = {
            "description": "1 = tmax, tmin, and prcp all valid (temp_flag & prcp_flag combined)",
        }
        chunks = [c[0] for c in ds["temp_flag"].chunks]
        xr.Dataset({"weather_flag": weather_flag}).to_zarr(
            weather_zarr, mode="r+", encoding={"weather_flag": {"chunks": chunks}}
        )
        print(f"  → weather_flag written to {weather_zarr}")
        print("  → temp_flag and prcp_flag are kept but weather_flag should be used.")
    else:
        print("  → flags carry independent information; keep temp_flag and prcp_flag separate.")


if __name__ == "__main__":
    TMAX_TMIN_ZARR = "/path/to/data-root/Data/station_data_2026/TX_domain_interpolated/1km/tmax_tmin_1km_barnes20km.zarr"
    PRCP_ZARR      = "/path/to/data-root/Data/station_data_2026/TX_domain_interpolated/1km/prcp_1km_idw10km.zarr"
    OUTPUT_ZARR    = "/path/to/data-root/Data/station_data_2026/TX_domain_interpolated/1km/weather_norm.zarr"
    TRAIN_END   = "2021-12-31"
    TEST_START  = "2022-01-01"

    normalize_weather(TMAX_TMIN_ZARR, PRCP_ZARR, OUTPUT_ZARR, train_end=TRAIN_END, test_start=TEST_START)
    add_rolling_prcp(PRCP_ZARR, OUTPUT_ZARR, train_end=TRAIN_END, test_start=TEST_START, windows=[14])
    check_flag_agreement(OUTPUT_ZARR)
