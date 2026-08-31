"""
Preprocessing PFT land type for model input.
Group land cover classes into 5 groups:
- 1: trees. Evergreen Needleleaf Trees, Evergreen Broadleaf Trees, Deciduous Needleleaf Trees, Deciduous Broadleaf Trees
- 2: shurbs
- 3: grass
- 4: cropland. Cereal crop,broadleaf crop
- 5. Others
Use one hot encoding for the 5 groups, i.e. 5 additional channels in the input data.
"""

import numpy as np
import xarray as xr

# MODIS LC_Type5 (PFT) class values
#   0  = Water
#   1  = Evergreen Needleleaf Trees
#   2  = Evergreen Broadleaf Trees
#   3  = Deciduous Needleleaf Trees
#   4  = Deciduous Broadleaf Trees
#   5  = Shrubs
#   6  = Grass
#   7  = Cereal Croplands
#   8  = Broadleaf Croplands
#   9  = Urban and Built-up Lands
#   10 = Permanent Snow and Ice
#   11 = Barren
#   255 / NaN = Fill / Unclassified → Others
"""
# Classes assigned to each group (everything not listed → "others")
_GROUP_CLASSES = {
    "trees":  [1, 2, 3, 4],
    "shrubs": [5],
    "grass":  [6],
    "crops":  [7, 8],
}
GROUP_NAMES = ["trees", "shrubs", "grass", "crops", "others"]

_GROUP_ATTRS = {
    "trees":  {"long_name": "Trees (one-hot)",   "pft_classes": "1,2,3,4",
               "description": "Evergreen Needleleaf, Evergreen Broadleaf, "
                              "Deciduous Needleleaf, Deciduous Broadleaf Trees"},
    "shrubs": {"long_name": "Shrubs (one-hot)",  "pft_classes": "5",
               "description": "Shrubs"},
    "grass":  {"long_name": "Grass (one-hot)",   "pft_classes": "6",
               "description": "Grass"},
    "crops":  {"long_name": "Croplands (one-hot)", "pft_classes": "7,8",
               "description": "Cereal Croplands and Broadleaf Croplands"},
    "others": {"long_name": "Others (one-hot)",  "pft_classes": "0,9,10,11,255,NaN",
               "description": "Water, Urban, Snow/Ice, Barren, Fill, and unclassified"},
}
"""

_GROUP_CLASSES = {
    "ev_trees":        [1, 2],
    "dc_trees":        [3, 4],
    "shrub":           [5],
    "grass":           [6],
    "c_crop":          [7],
    "b_crop":  [8],
}
GROUP_NAMES = ["ev_trees", "dc_trees", "shrub", "grass", "c_crop", "b_crop"]

_GROUP_ATTRS = {
    "ev_trees": {
        "long_name":    "Evergreen Trees (one-hot)",
        "pft_classes":  "1,2",
        "description":  "Evergreen Needleleaf and Evergreen Broadleaf Trees",
    },
    "dc_trees": {
        "long_name":    "Deciduous Trees (one-hot)",
        "pft_classes":  "3,4",
        "description":  "Deciduous Needleleaf and Deciduous Broadleaf Trees",
    },
    "shrub": {
        "long_name":    "Shrub (one-hot)",
        "pft_classes":  "5",
        "description":  "Shrubs",
    },
    "grass": {
        "long_name":    "Grass (one-hot)",
        "pft_classes":  "6",
        "description":  "Grass",
    },
    "c_crop": {
        "long_name":    "Cereal Cropland (one-hot)",
        "pft_classes":  "7",
        "description":  "Cereal Croplands",
    },
    "b_crop": {
        "long_name":    "Broadleaf Cropland (one-hot)",
        "pft_classes":  "8",
        "description":  "Broadleaf Croplands",
    },
}


def process_lc(
    input_zarr: str,
    output_zarr: str,
    lc_var: str = "LC_aggregated",
):
    """Map aggregated PFT classes to one-hot encoded group channels.

    Parameters
    ----------
    input_zarr  : zarr produced by lc_aggregate.py  (contains LC_aggregated)
    output_zarr : path to write the processed output
    lc_var      : variable name in input_zarr holding the class values
    """
    ds = xr.open_zarr(input_zarr).sortby("time")
    lc = ds[lc_var]

    times  = ds.time.values
    n_time = len(times)
    print(f"Input zarr  : {input_zarr}")
    print(f"Dims        : {dict(ds.dims)}")
    print(f"Time range  : {str(times[0])[:10]} → {str(times[-1])[:10]}  ({n_time} steps)")

    lc_np = lc.values  # (time, lat, lon)

    # ── Pad 2025 from the last available year if missing ──────────────────────
    last_t    = times[-1]
    last_year = int(str(last_t)[:4])
    if last_year < 2025:
        # Preserve the cftime type (e.g. DatetimeJulian) or fall back to datetime64
        if hasattr(last_t, 'month'):
            new_time = type(last_t)(2025, last_t.month, last_t.day)
        else:
            last_date_str = str(last_t)[:10]
            new_time = np.datetime64("2025" + last_date_str[4:])
        lc_np  = np.concatenate([lc_np, lc_np[[-1]]], axis=0)  # copy last year
        times  = np.append(times, new_time)
        n_time += 1
        print(f"  2025 not in input; copied from {str(last_t)[:10]} → {str(new_time)[:10]}")
    else:
        print(f"  2025 already present in input data")

    # ── Build one-hot arrays ──────────────────────────────────────────────────
    print("\nBuilding one-hot encoded PFT groups...")

    one_hot = {}
    assigned = np.zeros(lc_np.shape, dtype=bool)  # tracks which pixels are claimed

    for group in GROUP_NAMES:
        mask = np.zeros(lc_np.shape, dtype=np.uint8)
        for cls in _GROUP_CLASSES[group]:
            mask |= (lc_np == cls).astype(np.uint8)
        one_hot[group] = mask
        assigned |= mask.astype(bool)
        pct = 100.0 * mask.mean()
        print(f"  {group:8s}: {pct:.2f}% of all pixels (averaged over time)")

    # Sanity check: every pixel is assigned to at most one group
    total = sum(one_hot[g].astype(int) for g in GROUP_NAMES)
    assert (total <= 1).all(), "One-hot sanity check failed: pixel assigned to >1 group"
    unassigned_pct = 100.0 * (total == 0).mean()
    print(f"  {'(others/mask)':15s}: {unassigned_pct:.2f}% of pixels unassigned (water/urban/barren/snow)")
    print("  One-hot sanity check passed (no pixel assigned to >1 group)")

    # ── Assemble output dataset ───────────────────────────────────────────────
    lat_dim = "lat" if "lat" in ds.dims else "y"
    lon_dim = "lon" if "lon" in ds.dims else "x"

    coords = {
        "time": times,
        lat_dim: ds[lat_dim],
        lon_dim: ds[lon_dim],
    }

    data_vars = {}
    for group in GROUP_NAMES:
        attrs = _GROUP_ATTRS[group].copy()
        attrs["source_var"] = lc_var
        data_vars[f"lc_{group}"] = xr.DataArray(
            one_hot[group],
            dims=["time", lat_dim, lon_dim],
            coords=coords,
            attrs=attrs,
        )

    ds_out = xr.Dataset(data_vars, attrs=ds.attrs)

    # ── Chunk and write ───────────────────────────────────────────────────────
    n_lat = ds.sizes[lat_dim]
    n_lon = ds.sizes[lon_dim]
    time_chunks = (1, n_lat, n_lon)

    encoding = {f"lc_{g}": {"chunks": time_chunks, "dtype": "uint8"} for g in GROUP_NAMES}

    ds_out.to_zarr(output_zarr, mode="w", encoding=encoding)
    print(f"\nSaved → {output_zarr}")
    print(f"Variables: {list(data_vars.keys())}")


if __name__ == "__main__":
    INPUT_ZARR  = "/path/to/data-root/Data/Land_cover/lc_tx_1km_PFT.zarr"
    OUTPUT_ZARR = "/path/to/data-root/Data/Land_cover/lc_tx_1km_PFT_processed_6groups.zarr"

    process_lc(INPUT_ZARR, OUTPUT_ZARR)
