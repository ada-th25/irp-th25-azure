import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goes_local_fetch import upload_stack_as_geotiff  # format-agnostic, still used as-is
from mtg_fetch import fetch_mtg_local_stack, MTG_BANDS
from pc_covariates import fetch_covariates_window

import io
import gc
import json
import random
import tempfile
import threading
import warnings
import shutil
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from azure.storage.blob import ContainerClient

warnings.filterwarnings("ignore", message=".*transport does not support per-request timeout.*")

# ==========================================
# Azure Initialisation & Credentials
# ==========================================
conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_NAME = "wildfire-irp-data-th25"
VIIRS_CSV_BLOB_PATH = "viirs/viirs_snpp_fire_archive_med_europe_2024-09-24_to_present.csv"
OUTPUT_PREFIX = "processed_dataset_mtg"

_thread_local = threading.local()
blob_lock = threading.Lock()


def get_thread_container_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)
    return _thread_local.client


container_client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)

print("Caching existing blobs from Azure Container for resume verification...")
existing_blobs = set(b.name for b in container_client.list_blobs(name_starts_with=f"{OUTPUT_PREFIX}/"))
print(f"Found {len(existing_blobs)} pre-existing blobs in output prefix '{OUTPUT_PREFIX}/'.")


def azure_blob_exists(blob_name: str) -> bool:
    with blob_lock:
        if blob_name in existing_blobs:
            return True
    client = get_thread_container_client()
    return client.get_blob_client(blob_name).exists()


def add_existing_blob(blob_name: str):
    with blob_lock:
        existing_blobs.add(blob_name)


# ==========================================
# Constants
# ==========================================
# Mediterranean Europe + Sub-Saharan Africa -- MTG-I1 at 0 deg E doesn't give
# usable geometry over the Americas, so the old AMERICAS_LON_BOUNDS box is
# dropped rather than reused.
MTG_REGIONS = {
    "med_europe": [-10.0, 35.0, 30.0, 45.0],              # lon_min, lat_min, lon_max, lat_max
    "sub_saharan_africa": [10.0, -20.0, 40.0, 0.0],
}
VIIRS_POOL_BUFFER_DEG = 2.0

GOES_NATIVE_SCALE_M = 2000  # kept only as a manifest label for legacy comparison; unused for fetch now
MTG_HRFI_NATIVE_SCALE_M = 1000  # 1km IR bands; VIS/NIR HRFI are 0.5km, downsampled together
LANDSAT_GRID_SCALE = 30
VIIRS_GRID_SCALE = 375
VIIRS_TARGET_EXTENT_M = 7680
VIIRS_TARGET_PIXELS = int(VIIRS_TARGET_EXTENT_M / VIIRS_GRID_SCALE)
VIIRS_CONTEXT_RATIO = 1
VIIRS_CONTEXT_EXTENT_M = VIIRS_CONTEXT_RATIO * VIIRS_TARGET_EXTENT_M


def load_viirs_pool(start_date, end_date):
    """Local-disk cache of the raw CSV (avoids re-downloading from Azure on
    every run) + usecols/dtype at read_csv time (avoids ever loading the
    columns we don't use, which is cheaper than loading everything and
    dropping columns afterward).

    Cache filename is derived from VIIRS_CSV_BLOB_PATH so different source
    CSVs (Americas GOES run vs EU/Africa MTG run, etc.) never collide on
    the same /var/tmp cache file."""
    cache_key = VIIRS_CSV_BLOB_PATH.replace("/", "_")
    temp_csv_path = f"/var/tmp/viirs_archive_{cache_key}"
    blob_client = container_client.get_blob_client(VIIRS_CSV_BLOB_PATH)
    if not blob_client.exists():
        raise FileNotFoundError(f"VIIRS archive not found in Azure container at {VIIRS_CSV_BLOB_PATH}.")

    if not os.path.exists(temp_csv_path):
        print(f"Streaming VIIRS archive CSV from Azure to disk ({temp_csv_path}) ...")
        with open(temp_csv_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
    else:
        print(f"Using existing local cache at {temp_csv_path} ...")

    print("Loading VIIRS DataFrame into memory (only needed columns)...")
    use_cols = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence"]
    dtypes = {
        "latitude": "float32",
        "longitude": "float32",
        "frp": "float32",
        "acq_time": "int16",
    }
    df = pd.read_csv(temp_csv_path, usecols=use_cols, dtype=dtypes)
    df.columns = df.columns.str.strip()

    if 'acq_date' not in df.columns:
        raise KeyError("Missing required column 'acq_date' in source CSV.")

    df = df.dropna(subset=['acq_date', 'acq_time'])
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    acq_time_str = df['acq_time'].astype(int).apply(lambda t: f"{t:04d}")
    df['acq_datetime'] = pd.to_datetime(
        df['acq_date'].dt.strftime('%Y-%m-%d') + ' ' + acq_time_str,
        format='%Y-%m-%d %H%M'
    )

    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].astype(str).str.strip().str.lower()
        df = df[df['confidence'].isin(['n', 'h', 'nominal', 'high'])]

    df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
    print(f"VIIRS pool after temporal filtering ({start_date} to {end_date}): {len(df)} detections.")

    # Drop everything except what clustering/rasterizing actually needs --
    # confidence/acq_date/acq_time/acq_time_str are done their job now.
    df = df[['longitude', 'latitude', 'frp', 'acq_datetime']].copy()
    gc.collect()
    return df


def write_manifest(blob_name: str, data: dict):
    client = get_thread_container_client()
    b_client = client.get_blob_client(blob_name)
    b_client.upload_blob(json.dumps(data, indent=2), overwrite=True, content_type="application/json")
    add_existing_blob(blob_name)


def get_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def assign_viirs_cells(df, cell_extent_m=VIIRS_TARGET_EXTENT_M):
    cell_deg = cell_extent_m / 111320.0
    df['cell_lon'] = (df['longitude'] // cell_deg).astype(int)
    df['cell_lat'] = (df['latitude'] // cell_deg).astype(int)
    df['overpass_window'] = df['acq_datetime'].dt.floor('30min')
    return df, cell_deg


def cluster_id_for(overpass_window, cell_lon, cell_lat):
    return f"{overpass_window.strftime('%Y%m%dT%H%M')}_{cell_lon}_{cell_lat}"


def cluster_already_done(cluster_id):
    target_path = f"{OUTPUT_PREFIX}/viirs_pairs/targets/VIIRS_{cluster_id}.tif"
    mtg_path = f"{OUTPUT_PREFIX}/viirs_pairs/mtg_input/MTG_{cluster_id}.tif"
    manifest_path = f"{OUTPUT_PREFIX}/viirs_pairs/manifest/{cluster_id}.json"
    cov_path = f"{OUTPUT_PREFIX}/viirs_pairs/covariates/COV_{cluster_id}.tif"
    return (azure_blob_exists(target_path) and azure_blob_exists(mtg_path) and
            azure_blob_exists(manifest_path) and azure_blob_exists(cov_path))


def stream_pending_clusters_balanced(df, cell_deg, total_max_clusters, start_year, end_year, max_per_pass=2):
    """
    Streams clusters evenly distributed across years, months, and distinct overpasses.
    Caps clusters per single satellite pass to ensure spatial & temporal spread.
    """
    years = list(range(start_year, end_year + 1))
    clusters_per_year = total_max_clusters // len(years) if len(years) > 0 else total_max_clusters

    print(f"\n[BALANCED SAMPLING] Target per year ({start_year}-{end_year}): {clusters_per_year} clusters")

    df['year'] = df['acq_datetime'].dt.year
    df['month'] = df['acq_datetime'].dt.month

    # Year -> Month -> List of (overpass_window, cluster_items)
    hierarchy = defaultdict(lambda: defaultdict(list))

    for yr in years:
        df_yr = df[df['year'] == yr]
        if df_yr.empty:
            continue

        # Group by month and overpass window
        for (overpass_window, cell_lon, cell_lat), group in df_yr.groupby(['overpass_window', 'cell_lon', 'cell_lat'], sort=False):
            m = overpass_window.month
            hierarchy[yr][m].append((overpass_window, cell_lon, cell_lat, group))

    # Shuffle clusters within each month/pass to avoid geographic bias
    for yr in hierarchy:
        for m in hierarchy[yr]:
            random.shuffle(hierarchy[yr][m])

    total_yielded = 0
    total_skipped = 0

    # Round-robin loop across YEARS and MONTHS simultaneously
    active_years = [yr for yr in years if yr in hierarchy]
    year_counts = {yr: 0 for yr in active_years}

    while active_years and total_yielded < total_max_clusters:
        for yr in list(active_years):
            if year_counts[yr] >= clusters_per_year:
                active_years.remove(yr)
                continue

            months = sorted(hierarchy[yr].keys())
            if not months:
                active_years.remove(yr)
                continue

            # Pick 1 cluster from an active month in this year
            yielded_this_year_step = False
            for m in list(months):
                if not hierarchy[yr][m]:
                    del hierarchy[yr][m]
                    continue

                # Take a cluster
                overpass_window, cell_lon, cell_lat, group = hierarchy[yr][m].pop(0)
                cluster_id = cluster_id_for(overpass_window, cell_lon, cell_lat)

                if cluster_already_done(cluster_id):
                    total_skipped += 1
                    continue

                peak_time = group.loc[group['frp'].idxmax(), 'acq_datetime']
                time_spread_mins = (group['acq_datetime'] - peak_time).dt.total_seconds().abs() / 60.0
                cluster_group = group[time_spread_mins <= 15.0]

                center_lon = (cell_lon + 0.5) * cell_deg
                center_lat = (cell_lat + 0.5) * cell_deg
                utm_crs = get_utm_epsg(center_lon, center_lat)

                year_counts[yr] += 1
                total_yielded += 1
                yielded_this_year_step = True
                yield (cluster_id, center_lon, center_lat, peak_time, utm_crs, cluster_group)
                break  # Yielded 1 item for this year, move to next year in round-robin

    print(f"\n[BALANCED SAMPLING COMPLETE] Total Yielded: {total_yielded}, Total Skipped: {total_skipped}")


# ==========================================
# Worker Task
# ==========================================
def process_viirs_item(item):
    """Rasterizes the target AND uploads the full quadruplet, all inside the
    worker thread -- rasterization happens right before upload, not in a
    separate main-thread pre-pass."""
    (cluster_id, center_lon, center_lat, peak_time, utm_crs, cluster_group_df) = item

    mtg_prefix = f"{OUTPUT_PREFIX}/viirs_pairs/mtg_input/MTG_{cluster_id}"
    cov_prefix = f"{OUTPUT_PREFIX}/viirs_pairs/covariates/COV_{cluster_id}"
    target_path_tif = f"{OUTPUT_PREFIX}/viirs_pairs/targets/VIIRS_{cluster_id}.tif"
    manifest_path = f"{OUTPUT_PREFIX}/viirs_pairs/manifest/{cluster_id}.json"

    thread_scratch = tempfile.mkdtemp(prefix=f"mtg_task_{cluster_id}_", dir="/var/tmp")

    try:
        client = get_thread_container_client()
        target_dt = peak_time.to_pydatetime() if hasattr(peak_time, "to_pydatetime") else peak_time

        # --- 1. Target rasterization (in-thread, freed as soon as uploaded) ---
        if not azure_blob_exists(target_path_tif):
            half_extent = VIIRS_TARGET_EXTENT_M / 2.0
            transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            cx_utm, cy_utm = transformer_to_utm.transform(center_lon, center_lat)

            xmin, xmax = cx_utm - half_extent, cx_utm + half_extent
            ymin, ymax = cy_utm - half_extent, cy_utm + half_extent
            width = height = VIIRS_TARGET_PIXELS
            actual_scale_m = VIIRS_TARGET_EXTENT_M / VIIRS_TARGET_PIXELS

            local_array = np.zeros((height, width), dtype=np.float32)
            gdf = gpd.GeoDataFrame(
                cluster_group_df,
                geometry=gpd.points_from_xy(cluster_group_df.longitude, cluster_group_df.latitude),
                crs="EPSG:4326"
            ).to_crs(utm_crs)

            gdf = gdf.sort_values(by='frp', ascending=True)
            for _, row in gdf.iterrows():
                px, py = row.geometry.x, row.geometry.y
                col = int(np.floor((px - xmin) / actual_scale_m))
                row_idx = int(np.floor((ymax - py) / actual_scale_m))
                if 0 <= col < width and 0 <= row_idx < height:
                    local_array[row_idx, col] = row['frp']

            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
            mem_file = io.BytesIO()
            with rasterio.open(
                mem_file, 'w', driver='GTiff', height=height, width=width, count=1,
                dtype='float32', crs=utm_crs, transform=transform
            ) as dst:
                dst.write(local_array, 1)
            mem_file.seek(0)

            client.get_blob_client(target_path_tif).upload_blob(
                mem_file, overwrite=True, content_type='image/tiff'
            )
            add_existing_blob(target_path_tif)
            del local_array, gdf, mem_file

        # --- 2. MTG context ---
        if azure_blob_exists(f"{mtg_prefix}.tif"):
            mtg_status = "ok"
        else:
            stack, matched, reason = fetch_mtg_local_stack(
                center_lon=center_lon, center_lat=center_lat, target_dt=target_dt,
                extent_m=VIIRS_CONTEXT_EXTENT_M, utm_crs=utm_crs,
                local_cache_dir=thread_scratch,
            )
            if matched:
                to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
                cx, cy = to_utm.transform(center_lon, center_lat)
                half = VIIRS_CONTEXT_EXTENT_M / 2.0
                upload_stack_as_geotiff(
                    stack, cx - half, cy - half, cx + half, cy + half,
                    utm_crs, client, f"{mtg_prefix}.tif",
                    band_names=tuple(MTG_BANDS)
                )
                add_existing_blob(f"{mtg_prefix}.tif")
                mtg_status = "ok"
            else:
                mtg_status = reason
            del stack
        mtg_dispatched = (mtg_status == "ok")

        # --- 3. Covariates ---
        if not azure_blob_exists(f"{cov_prefix}.tif"):
            cov_stack, (cxmin, cymin, cxmax, cymax) = fetch_covariates_window(
                center_lon, center_lat, VIIRS_CONTEXT_EXTENT_M, utm_crs, scale_m=LANDSAT_GRID_SCALE
            )
            upload_stack_as_geotiff(
                cov_stack, cxmin, cymin, cxmax, cymax, utm_crs, client,
                f"{cov_prefix}.tif", band_names=("elevation", "landcover")
            )
            add_existing_blob(f"{cov_prefix}.tif")
            del cov_stack

        # --- 4. Manifest ---
        write_manifest(manifest_path, {
            "id": cluster_id, "branch": "viirs",
            "date": target_dt.strftime("%Y-%m-%d"),
            "peak_datetime": peak_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "center": [center_lon, center_lat], "crs": utm_crs,
            "target_pixels_nominal": VIIRS_TARGET_PIXELS,
            "target_extent_m": VIIRS_TARGET_EXTENT_M,
            "target_scale_m": VIIRS_TARGET_EXTENT_M / VIIRS_TARGET_PIXELS,
            "context_ratio": VIIRS_CONTEXT_RATIO,
            "context_extent_m": VIIRS_CONTEXT_EXTENT_M,
            "mtg_native_scale_m": MTG_HRFI_NATIVE_SCALE_M,
            "mtg_bands": MTG_BANDS,
            "covariate_scale_m": LANDSAT_GRID_SCALE,
            "n_detections": int(len(cluster_group_df)),
            "max_frp": float(cluster_group_df['frp'].max()),
            "mtg_dispatched": mtg_dispatched,
            "mtg_fetch_status": mtg_status,
            "covariate_source": "planetary-computer",
            "mtg_source": "eumetsat-mtg-fci-hrfi",
            "paths": {
                "target": target_path_tif,
                "mtg_input": f"{mtg_prefix}.tif",
                "covariates": f"{cov_prefix}.tif",
            }
        })

        return "ok"

    except Exception as e:
        return f"Fail cluster {cluster_id}: {e}"

    finally:
        if os.path.exists(thread_scratch):
            shutil.rmtree(thread_scratch)


def run_viirs_branch(df, cell_deg, start_year, end_year, max_clusters=10, dispatch_workers=4, batch_size=None):
    if batch_size is None:
        batch_size = dispatch_workers * 5

    print(f"\nStreaming VIIRS clusters evenly (target: {max_clusters} total, "
          f"batch_size={batch_size}, workers={dispatch_workers})...")

    total_done = 0
    total_failed = 0
    batch = []

    def _flush_batch(batch):
        nonlocal total_done, total_failed
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=dispatch_workers) as executor:
            futures = {executor.submit(process_viirs_item, item): item[0] for item in batch}
            for future in as_completed(futures):
                c_id = futures[future]
                try:
                    status = future.result()
                    if status == "ok":
                        total_done += 1
                        print(f"    [SUCCESS] ({total_done} done so far) {c_id}")
                    else:
                        total_failed += 1
                        print(f"    [!] {status}")
                except Exception as e:
                    total_failed += 1
                    print(f"    [!] Thread crash for {c_id}: {e}")
        gc.collect()

    for item in stream_pending_clusters_balanced(df, cell_deg, max_clusters, start_year, end_year):
        batch.append(item)
        if len(batch) >= batch_size:
            _flush_batch(batch)
            batch = []

    _flush_batch(batch)  # Remainder
    print(f"\nDone. {total_done} succeeded, {total_failed} failed this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="viirs")
    parser.add_argument("--start-year", type=int, default=2024,
                         help="First year (inclusive) of VIIRS detections to process this run. "
                              "2024 is the floor for usable MTG FCI coverage.")
    parser.add_argument("--end-year", type=int, default=2025,
                         help="Last year (inclusive) of VIIRS detections to process this run.")
    parser.add_argument("--max-clusters", type=int, default=20_000,
                         help="Max NEW (not-yet-complete) clusters to process this run.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.mode != "viirs":
        print(f"Unknown mode '{args.mode}'. Only 'viirs' is supported in this edition.")
        sys.exit(1)

    start_date = f"{args.start_year}-01-01"
    end_date = f"{args.end_year}-12-31"

    print(f"\n=== MODE: viirs | years {args.start_year}-{args.end_year} | "
          f"max_clusters={args.max_clusters} | workers={args.workers} ===")

    df_viirs = load_viirs_pool(start_date, end_date)

    # Union of the MTG-coverable regions, replacing the old single
    # AMERICAS_LON_BOUNDS box.
    region_mask = pd.Series(False, index=df_viirs.index)
    for name, (lon_min, lat_min, lon_max, lat_max) in MTG_REGIONS.items():
        region_mask |= (
            (df_viirs['longitude'] >= lon_min) & (df_viirs['longitude'] <= lon_max) &
            (df_viirs['latitude'] >= lat_min) & (df_viirs['latitude'] <= lat_max)
        )
    pool = df_viirs[region_mask].copy()
    del df_viirs, region_mask
    gc.collect()

    pool, cell_deg = assign_viirs_cells(pool)

    run_viirs_branch(
        pool, cell_deg,
        start_year=args.start_year,
        end_year=args.end_year,
        max_clusters=args.max_clusters,
        dispatch_workers=args.workers,
    )
