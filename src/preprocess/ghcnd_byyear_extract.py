"""GHCND by-year file extractor.
Get stations within the domain using ghcnd_station_analysis.py.
Read GHCND by-year CSV files and extreact TMAX, TMIN, PRCP for stations within the domain.
"""

import os
import pandas as pd
import numpy as np

VARIABLES = {"TMAX", "TMIN", "PRCP"}

def load_station_ids(txt_path):
    with open(txt_path) as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"  Loaded {len(ids):,} station IDs from {txt_path}")
    return ids


TEMP_VARS = {"TMAX", "TMIN"}  # stored in tenths of °C


def process_year(filepath, station_ids, year, out_dir):
    print(f"  Reading {os.path.basename(filepath)}...")

    df = pd.read_csv(
        filepath,
        header=None,
        usecols=[0, 1, 2, 3],                          # only first 4 columns
        names=["station_id", "date", "element", "value"],
        dtype={"station_id": str, "date": str,
               "element": str, "value": str},
    )
    print(f"    Total rows: {len(df):,}")

    # ── Filter to domain stations and variables we want ──
    df = df[
        df["station_id"].isin(station_ids) &
        df["element"].isin(VARIABLES)
    ].copy()
    print(f"    Rows after filter: {len(df):,}")

    if df.empty:
        return {}

    # ── Parse ──
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"]  = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date"].notna() & df["value"].notna()].copy()

    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for var in VARIABLES:
        sub = df[df["element"] == var].sort_values(["date", "station_id"])
        if sub.empty:
            continue
        out = sub[["station_id", "date", "value"]].copy()
        out.columns = ["station_id", "date", var.lower()]
        if var in TEMP_VARS:
            out[var.lower()] = out[var.lower()] / 10.0  # tenths of °C → °C
        out = out.reset_index(drop=True)
        results[var] = out
        print(f"    {var}: {len(out):,} records, "
              f"{out['station_id'].nunique():,} stations")

        out_path = os.path.join(out_dir, f"{var}_{year}.txt")
        out.to_csv(out_path, index=False)
        print(f"    Saved {out_path}")

    return results
 

if __name__ == "__main__":
    STATION_IDS_FILE = "/path/to/data-root/Data/station_data_2026/analysis_plots/ghcnd_station_ids_2degree_buffer.txt"          # path to station IDs txt file
    GHCND_DIR        = "/path/to/data-root/Data/station_data_2026/by_year"          # directory containing GHCND by-year CSVs
    OUT_DIR          = "/path/to/data-root/Data/station_data_2026/TX_domain"          # output directory for results
    YEARS            = range(2002, 2026)   # e.g. range(2000, 2021)

    station_ids = load_station_ids(STATION_IDS_FILE)

    for year in YEARS:
        print(f"\nProcessing {year}...")
        filepath = os.path.join(GHCND_DIR, f"{year}.csv")
        if not os.path.exists(filepath):
            print(f"  File not found, skipping: {filepath}")
            continue
        process_year(filepath, station_ids, year, OUT_DIR)

    print("\nDone.")
