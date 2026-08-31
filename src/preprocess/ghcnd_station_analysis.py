"""GHCND station spacing analysis
Use the station inventory and compute nearest-neighbor distances to understand the spatial distribution of stations.
NOTE: This is a preliminary analysis since the station inventory contains all the stations.
But for each day the station which provides records might vary. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree


def parse_stations(text):
    """
    Parse ghcnd-stations.txt.
    Verified column positions from actual file:
      ACW00011604  17.1167  -61.7833   10.1    ST JOHNS COOLIDGE FLD
      0-10  : Station ID
      11-19 : Latitude
      20-29 : Longitude
      30-36 : Elevation
      38+   : Name
    """
    records = []
    for line in text.splitlines():
        if len(line) < 30:
            continue
        try:
            sid  = line[0:11].strip()
            lat  = float(line[11:20].strip())
            lon  = float(line[20:30].strip())
            elev = float(line[30:37].strip()) if line[30:37].strip() else np.nan
            name = line[38:].strip() if len(line) > 38 else ""
            records.append({
                "station_id": sid,
                "lat":  lat,
                "lon":  lon,
                "elev": elev,
                "name": name
            })
        except ValueError:
            continue
    return pd.DataFrame(records)


def compute_spacing(stations, k=1):
    """
    Compute k nearest-neighbor distances in km using equirectangular
    approximation — accurate for state-scale domain like Texas.
    """
    lat_ref = stations["lat"].mean()
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_ref))

    xy = np.column_stack([
        stations["lon"].values * km_per_deg_lon,
        stations["lat"].values * km_per_deg_lat
    ])

    tree = cKDTree(xy)
    # k+1 to skip self (distance=0)
    dists, _ = tree.query(xy, k=k + 1)
    return dists[:, 1:]


def suggest_kappa(d_bar_km, E=1.0):
    """
    R = 1.33 * d_bar  (radius of influence)
    kappa = R^2 / (4 * E)

    E = 1.0 → larger κ → broader influence → MORE smooth
    E = 0.2 → smaller κ → narrower influence → LESS smooth
    Koch et al. suggest E in range [0.2, 1.0].
    E = 1.0 less smooth, E = 0.2 more smooth.
    """
    R = 1.33 * d_bar_km
    return R ** 2 / (4 * E)



def print_summary(nn_km, label):
    d = nn_km
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  N stations     : {len(d):,}")
    print(f"  Mean spacing   : {np.mean(d):.1f} km")
    print(f"  Median spacing : {np.median(d):.1f} km")
    print(f"  Std dev        : {np.std(d):.1f} km")
    print(f"  10th pct       : {np.percentile(d, 10):.1f} km")
    print(f"  25th pct       : {np.percentile(d, 25):.1f} km")
    print(f"  75th pct       : {np.percentile(d, 75):.1f} km")
    print(f"  90th pct       : {np.percentile(d, 90):.1f} km")
    print(f"  Max spacing    : {np.max(d):.1f} km")
    d_bar = np.median(d)
    kappa = suggest_kappa(d_bar)
    print(f"\n  Barnes κ (Koch et al. 1983):")
    print(f"    d_bar (median) = {d_bar:.1f} km")
    print(f"    κ = (1.33×d_bar)² = {kappa:.0f} km²")
    print(f"{'='*55}")
    return d_bar, kappa


def plot_stations(stations_search, lat_min, lat_max, lon_min, lon_max,
                  buffer_deg, out_path=None):
    """
    Map of stations coloured by zone:
      - blue  : inside the target domain
      - orange: in the buffer ring only
    Includes state boundaries and lat/lon gridlines.
    """
    mask_target = (
        (stations_search["lat"] >= lat_min) &
        (stations_search["lat"] <= lat_max) &
        (stations_search["lon"] >= lon_min) &
        (stations_search["lon"] <= lon_max)
    )
    in_target = stations_search[mask_target]
    in_buffer = stations_search[~mask_target]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

    # map extent: full search domain with a small visual padding
    pad = 0.5
    ax.set_extent(
        [lon_min - buffer_deg - pad, lon_max + buffer_deg + pad,
         lat_min - buffer_deg - pad, lat_max + buffer_deg + pad],
        crs=proj
    )

    # state and country boundaries
    ax.add_feature(cfeature.STATES, linewidth=0.7, edgecolor="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="dimgray")

    # gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1.0,
                      color="gray", linestyle="--", alpha=1.0)
    gl.top_labels = False
    gl.right_labels = False

    # stations
    ax.scatter(in_buffer["lon"], in_buffer["lat"],
               s=6, color="tab:orange", alpha=0.6, transform=proj,
               label=f"Buffer ring ({len(in_buffer):,})")
    ax.scatter(in_target["lon"], in_target["lat"],
               s=6, color="tab:blue", alpha=0.8, transform=proj,
               label=f"Target domain ({len(in_target):,})")

    # target-domain bounding box
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min, lat_max - lat_min,
        linewidth=1.5, edgecolor="black", facecolor="none",
        linestyle="--", transform=proj, label="Target domain boundary"
    )
    ax.add_patch(rect)

    ax.set_title(f"GHCND stations — target domain + {buffer_deg}° buffer")
    ax.legend(loc="lower right", markerscale=3, fontsize=11)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"  Plot saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    station_data = "/path/to/data-root/Data/station_data_2026/ghcnd-stations.txt"
    output_dir = '/path/to/data-root/Data/station_data_2026/analysis_plots'
    # Bounding box (CONUS)
    # LAT_MIN, LAT_MAX =  24.0,  50.0
    # LON_MIN, LON_MAX = -125.0, -66.0

    # Target domain (Texas)
    LAT_MIN, LAT_MAX = 25.8, 37.2
    LON_MIN, LON_MAX = -108.25, -92.5

    # Buffer added around target domain
    BUFFER_DEG = 2 # 2.0

    stations = parse_stations(open(station_data).read())

    search_lat_min = LAT_MIN - BUFFER_DEG
    search_lat_max = LAT_MAX + BUFFER_DEG
    search_lon_min = LON_MIN - BUFFER_DEG
    search_lon_max = LON_MAX + BUFFER_DEG

    mask_search = (
        (stations["lat"] >= search_lat_min) &
        (stations["lat"] <= search_lat_max) &
        (stations["lon"] >= search_lon_min) &
        (stations["lon"] <= search_lon_max)
    )
    stations_search = stations[mask_search].reset_index(drop=True)
    print(f"\nTarget domain    : lat [{LAT_MIN}, {LAT_MAX}]  "
          f"lon [{LON_MIN}, {LON_MAX}]")
    print(f"Station domain   : target + {BUFFER_DEG}° buffer")
    print(f"Stations in search domain : {len(stations_search):,}")

    nn_search = compute_spacing(stations_search).flatten()

    d_bar, kappa = print_summary(
        nn_search,
        f"Target domain stations (spacing computed with {BUFFER_DEG}° buffer)"
    )

    plot_stations(
        stations_search,
        LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
        BUFFER_DEG,
        out_path=f"{output_dir}/ghcnd_station_map_2degree_buffer.png"
    )

    station_id_path = f"{output_dir}/ghcnd_station_ids_2degree_buffer.txt"
    with open(station_id_path, "w") as f:
        f.write("\n".join(stations_search["station_id"].tolist()))
    print(f"  Station IDs saved → {station_id_path}  ({len(stations_search):,} stations)")
