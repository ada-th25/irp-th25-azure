import io
import os
import re
import json
import zipfile
import tempfile
import gdown
import ee
import rasterio
import time
import threading
import warnings
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pyproj import Transformer
from google.cloud import storage
from google.auth import default
import numpy as np
import pandas as pd

# ==========================================
# INITIALISATION
# ==========================================
credentials, project = default()
storage_client = storage.Client(credentials=credentials)
ee.Initialize(credentials=credentials, project='wildfire-irp')

BUCKET_NAME = "wildfire-irp-data-th25"
bucket = storage_client.bucket(BUCKET_NAME)

print("Caching existing files from GCS...")
existing_blobs = set(b.name for b in bucket.list_blobs(prefix="processed_dataset/"))

OUTPUT_PREFIX = "processed_dataset"

# ==========================================
# REGION / TIME CONSTANTS
# ==========================================
AMERICAS_LON_BOUNDS = [-170.0, -30.0]

VIIRS_POOL_BUFFER_DEG = 2.0

# Same time period as the Pereira et al. Landsat-8 dataset (Aug-Sep 2020).
VIIRS_DATE_RANGE = ("2020-08-01", "2020-09-30")

# ==========================================
# RESOLUTION / EXTENT CONSTANTS
# ==========================================
GOES_TEMPORAL_GUARD_MIN = 15     # Max temporal gap allowed between target time and GOES scene (minutes)
GOES_NATIVE_SCALE_M = 2000        # Native GOES ABI grid spacing

LANDSAT_GRID_SCALE = 30           # 30m target resolution for Landsat branch
VIIRS_GRID_SCALE = 375            # 375m target resolution for VIIRS branch

LANDSAT_TARGET_PIXELS = 256
LANDSAT_TARGET_EXTENT_M = LANDSAT_TARGET_PIXELS * LANDSAT_GRID_SCALE  # 7680m - Pereira 256x256 @ 30m patches

VIIRS_TARGET_PIXELS = 32
VIIRS_TARGET_EXTENT_M = VIIRS_TARGET_PIXELS * VIIRS_GRID_SCALE  # 12000m

# Context ratios for GOES contextual windowing around the target patch. 1x means the context window is the same size as the target patch.
# This is a hyperparameter that can be tuned to balance the amount of contextual information vs. the size of the input tensor.
# This is because GOES has a much coarser resolution than Landsat or VIIRS, which means the GOES patch only covers a few pixels of the target patch.
# Increasing the context ratio allows the model to see more of the surrounding area, which can help it learn better representations of the fire dynamics.
LANDSAT_CONTEXT_RATIO = 1
VIIRS_CONTEXT_RATIO = 1

LANDSAT_CONTEXT_EXTENT_M = LANDSAT_CONTEXT_RATIO * LANDSAT_TARGET_EXTENT_M  # 30720m (~15x15 GOES px)
VIIRS_CONTEXT_EXTENT_M = VIIRS_CONTEXT_RATIO * VIIRS_TARGET_EXTENT_M        # 120000m (~60x60 GOES px)

MIN_ACTIVE_PIXELS = 2

# ==========================================
# COVARIATES
# ==========================================
DEM_ASSET = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().rename(['elevation'])
LC_ASSET = ee.ImageCollection("ESA/WorldCover/v100").select('Map').mosaic().rename(['landcover'])
COVARIATE_STACK = DEM_ASSET.addBands(LC_ASSET.toFloat())

# ==========================================
# NASA VIIRS ARCHIVE: LOAD + FILTER
# ==========================================
VIIRS_CSV_PATH = "raw-viirs/fire_archive_SV-C2_762484.csv"

if not os.path.exists(VIIRS_CSV_PATH):
    raise FileNotFoundError(f"Missing local NASA catalog asset at: {VIIRS_CSV_PATH}.")

print("Loading NASA VIIRS Archive CSV into memory...")
df_viirs = pd.read_csv(VIIRS_CSV_PATH)
df_viirs.columns = df_viirs.columns.str.strip()

if 'acq_date' not in df_viirs.columns:
    raise KeyError("Missing required column 'acq_date' in source file.")
df_viirs['acq_date'] = pd.to_datetime(df_viirs['acq_date'])
print(f"Loaded {len(df_viirs)} raw archive detections.")

if 'confidence' in df_viirs.columns:
    df_viirs['confidence'] = df_viirs['confidence'].astype(str).str.strip().str.lower()
    before = len(df_viirs)
    df_viirs = df_viirs[df_viirs['confidence'].isin(['n', 'h', 'nominal', 'high'])]
    print(f"Confidence filter (nominal/high only): kept {len(df_viirs)}/{before} detections.")
else:
    print("Warning: no 'confidence' column found in archive CSV - confidence filter skipped.")

df_viirs = df_viirs[
    (df_viirs['acq_date'] >= VIIRS_DATE_RANGE[0]) & (df_viirs['acq_date'] <= VIIRS_DATE_RANGE[1])
]
print(f"VIIRS branch pool after Aug-Sep-2020 filtering: {len(df_viirs)} detections.")

# ==========================================
# CORE UTILITIES
# ==========================================
def gcs_exists(path_prefix):
    return any(name.startswith(path_prefix) for name in existing_blobs)

# Thread-safe tracker arrays for bounding box processing
collected_patch_bboxes = []
bbox_lock = threading.Lock()

def compute_viirs_pool_bbox(patch_bboxes, buffer_deg=VIIRS_POOL_BUFFER_DEG):
    if not patch_bboxes:
        return None
    lon_mins, lat_mins, lon_maxs, lat_maxs = zip(*patch_bboxes)
    return [
        min(lon_mins) - buffer_deg,
        min(lat_mins) - buffer_deg,
        max(lon_maxs) + buffer_deg,
        max(lat_maxs) + buffer_deg,
    ]

def write_manifest(path, data):
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')

EE_DISPATCH_WORKERS = 6
QUEUE_CHECK_INTERVAL = 25
_dispatch_count = 0
_dispatch_count_lock = threading.Lock()

def wait_for_queue_slot():
    global _dispatch_count
    with _dispatch_count_lock:
        _dispatch_count += 1
        should_check = (_dispatch_count % QUEUE_CHECK_INTERVAL == 0)

    if not should_check:
        return

    while True:
        tasks = ee.batch.Task.list()
        active_tasks = len([t for t in tasks if t.status()['state'] in ['RUNNING', 'READY', 'PENDING']])
        if active_tasks < 2500:
            break
        print(f"    [WAIT] GEE Engine saturated ({active_tasks} tasks). Sleeping 45s...")
        time.sleep(45)

def build_roi(center_lon, center_lat, extent_m):
    return ee.Geometry.Point([center_lon, center_lat]).buffer(extent_m / 2).bounds()

def parse_zip_patch_metadata(file_bytes, filename):
    meta_match = re.search(r'LC08_[A-Z0-9]{4}_\d{6}_(\d{8})', filename)
    if not meta_match:
        return None
    date_str = meta_match.group(1)
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    file_bytes.seek(0)
    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            bounds = src.bounds
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon_min, lat_min = t.transform(bounds.left, bounds.bottom)
            lon_max, lat_max = t.transform(bounds.right, bounds.top)
            lon_cen, lat_cen = t.transform((bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2)
    return {
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "center": [lon_cen, lat_cen],
        "date": formatted_date,
        "id": filename.replace(".tif", "")
    }

def get_utm_epsg(lon, lat):
    """Return the EPSG code for the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

# ==========================================
# FEATURE EXTRACTORS (GOES / VIIRS / COV)
# ==========================================
def fetch_goes_fire_features(context_roi, lon_cen, target_date_ee):
    """Returns a GOES MCMIPC image stack masked completely server-side 
    if no scenes fall inside the temporal buffer window."""
    satellite = "NOAA/GOES/17/MCMIPC" if lon_cen < -100.0 else "NOAA/GOES/16/MCMIPC"
    
    col = (ee.ImageCollection(satellite)
           .filterBounds(context_roi)
           .filterDate(target_date_ee.advance(-12, 'hour'), target_date_ee.advance(12, 'hour'))
           .select(['CMI_C07', 'CMI_C14']))

    target_millis = target_date_ee.millis()
    
    col_with_time = col.map(lambda img: img.set(
        'time_diff', ee.Number(img.get('system:time_start')).subtract(target_millis).abs()
    ))
    
    closest = ee.Image(col_with_time.sort('time_diff').first())
    tfi = closest.select('CMI_C07').subtract(closest.select('CMI_C14')).rename('GOES_TFI').toInt16()
    final_stack = closest.select(['CMI_C07', 'CMI_C14']).addBands(tfi).clip(context_roi)

    # Convert temporal constraints to structural pixel masks
    max_diff_ms = GOES_TEMPORAL_GUARD_MIN * 60 * 1000
    time_condition = ee.Number(closest.get('time_diff')).lte(max_diff_ms)
    
    return final_stack.updateMask(time_condition)

# ==========================================
# BRANCH A: GOES <-> LANDSAT
# ==========================================
def process_landsat_sample(meta, fb, mask_source="intersection"):
    sample_id = meta['id']

    target_prefix = f"{OUTPUT_PREFIX}/landsat_pairs/targets/LANDSAT_{sample_id}"
    goes_prefix = f"{OUTPUT_PREFIX}/landsat_pairs/goes_context/GOES_{sample_id}"
    cov_prefix = f"{OUTPUT_PREFIX}/landsat_pairs/covariates/COV_{sample_id}"
    manifest_path = f"{OUTPUT_PREFIX}/landsat_pairs/manifest/{sample_id}.json"

    try:
        fb.seek(0)
        with rasterio.open(fb) as src:
            mask = src.read(1)
        n_active_pixels = int(np.sum(mask == 1))
        if n_active_pixels < MIN_ACTIVE_PIXELS:
            print(f"    [SKIP landsat] Only {n_active_pixels} active fire pixel(s) "
                  f"(< {MIN_ACTIVE_PIXELS}) for {sample_id}")
            return
    except Exception as e:
        print(f"    [!] Failed to verify Landsat mask contents for {sample_id}: {e}")
        return

    center_lon, center_lat = meta['center']
    utm_crs = get_utm_epsg(center_lon, center_lat)
    context_roi = build_roi(center_lon, center_lat, LANDSAT_CONTEXT_EXTENT_M)

    if not gcs_exists(target_prefix):
        blob = bucket.blob(f"{target_prefix}.tif")
        fb.seek(0)
        blob.upload_from_file(fb, content_type='image/tiff')
        print(f"    [SAVED] Landsat target: {sample_id} (n_active_pixels={n_active_pixels}, "
              f"mask_source={mask_source})")

    if not gcs_exists(goes_prefix):
        goes_img = fetch_goes_fire_features(context_roi, center_lon, ee.Date(meta['date']))
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=goes_img, description=f"goes_ls_{sample_id}",
            bucket=BUCKET_NAME, fileNamePrefix=goes_prefix, region=context_roi,
            scale=GOES_NATIVE_SCALE_M, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] GOES context export: {sample_id}")

    if not gcs_exists(cov_prefix):
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=COVARIATE_STACK.clip(context_roi), description=f"cov_ls_{sample_id}",
            bucket=BUCKET_NAME, fileNamePrefix=cov_prefix, region=context_roi,
            scale=LANDSAT_GRID_SCALE, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] Covariate export: {sample_id}")

    write_manifest(manifest_path, {
        "id": sample_id,
        "branch": "landsat",
        "mask_source": mask_source,
        "date": meta['date'],
        "center": [center_lon, center_lat],
        "crs": utm_crs,
        "n_active_pixels": n_active_pixels,
        "target_pixels": LANDSAT_TARGET_PIXELS,
        "target_extent_m": LANDSAT_TARGET_EXTENT_M,
        "target_scale_m": LANDSAT_GRID_SCALE,
        "context_ratio": LANDSAT_CONTEXT_RATIO,
        "context_extent_m": LANDSAT_CONTEXT_EXTENT_M,
        "goes_scale_m": GOES_NATIVE_SCALE_M,
        "covariate_scale_m": LANDSAT_GRID_SCALE,
        "paths": {
            "target": f"{target_prefix}.tif",
            "goes_context": f"{goes_prefix}.tif",
            "covariates": f"{cov_prefix}.tif",
        }
    })

# ==========================================
# BRANCH B: GOES <-> VIIRS
# ==========================================
def assign_viirs_cells(df, cell_extent_m=VIIRS_TARGET_EXTENT_M):
    df = df.copy()
    cell_deg = cell_extent_m / 111320.0
    df['cell_lon'] = (df['longitude'] // cell_deg).astype(int)
    df['cell_lat'] = (df['latitude'] // cell_deg).astype(int)
    df['date_str'] = df['acq_date'].dt.strftime('%Y%m%d')
    return df, cell_deg

def process_viirs_cluster(cluster_id, center_lon, center_lat, ee_viirs_date, group_df):
    target_prefix = f"processed_dataset/viirs_pairs/targets/VIIRS_{cluster_id}"
    goes_prefix = f"processed_dataset/viirs_pairs/goes_context/GOES_{cluster_id}"
    cov_prefix = f"processed_dataset/viirs_pairs/covariates/COV_{cluster_id}"
    manifest_path = f"processed_dataset/viirs_pairs/manifest/{cluster_id}.json"

    target_roi = build_roi(center_lon, center_lat, VIIRS_TARGET_EXTENT_M)
    context_roi = build_roi(center_lon, center_lat, VIIRS_CONTEXT_EXTENT_M)

    if not gcs_exists(target_prefix):
        features = [
            ee.Feature(ee.Geometry.Point([float(row['longitude']), float(row['latitude'])]), {'frp': float(row['frp'])})
            for _, row in group_df.iterrows()
        ]
        viirs_img = (ee.FeatureCollection(features)
                     .reduceToImage(properties=['frp'], reducer=ee.Reducer.max())
                     .unmask(0)
                     .rename('VIIRS_FRP')
                     .clip(target_roi))
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=viirs_img, description=f"viirs_frp_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=target_prefix, region=target_roi,
            scale=VIIRS_GRID_SCALE, crs='EPSG:4326', fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] VIIRS target export: {cluster_id}")

    if not gcs_exists(goes_prefix):
        goes_img = fetch_goes_fire_features(context_roi, center_lon, ee_viirs_date)
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=goes_img, description=f"goes_vx_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=goes_prefix, region=context_roi,
            scale=GOES_NATIVE_SCALE_M, crs='EPSG:4326', fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] GOES context export: {cluster_id}")

    if not gcs_exists(cov_prefix):
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=COVARIATE_STACK.clip(context_roi), description=f"cov_vx_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=cov_prefix, region=context_roi,
            scale=VIIRS_GRID_SCALE, crs='EPSG:4326', fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] Covariate export: {cluster_id}")

    write_manifest(manifest_path, {
        "id": cluster_id,
        "branch": "viirs",
        "date": group_df['acq_date'].iloc[0].strftime("%Y-%m-%d"),
        "center": [center_lon, center_lat],
        "target_pixels": VIIRS_TARGET_PIXELS,
        "target_extent_m": VIIRS_TARGET_EXTENT_M,
        "target_scale_m": VIIRS_GRID_SCALE,
        "context_ratio": VIIRS_CONTEXT_RATIO,
        "context_extent_m": VIIRS_CONTEXT_EXTENT_M,
        "goes_scale_m": GOES_NATIVE_SCALE_M,
        "covariate_scale_m": VIIRS_GRID_SCALE,
        "n_detections": int(len(group_df)),
        "max_frp": float(group_df['frp'].max()),
        "paths": {
            "target": f"{target_prefix}.tif",
            "goes_context": f"{goes_prefix}.tif",
            "covariates": f"{cov_prefix}.tif",
        }
    })

def run_viirs_branch(region_bbox, max_clusters=10, dispatch_workers=EE_DISPATCH_WORKERS):
    if region_bbox is None:
        print("\n[SKIP] No Landsat patches were processed this run - skipping VIIRS branch.")
        return

    print(f"\nProcessing decoupled GOES -> VIIRS branch (region={region_bbox})...")
    pool = df_viirs[
        (df_viirs['longitude'] >= region_bbox[0]) & (df_viirs['longitude'] <= region_bbox[2]) &
        (df_viirs['latitude'] >= region_bbox[1]) & (df_viirs['latitude'] <= region_bbox[3])
    ]
    print(f"VIIRS spatial pool for this region: {len(pool)} detections.")

    df, cell_deg = assign_viirs_cells(pool)
    groups = list(df.groupby(['date_str', 'cell_lon', 'cell_lat']))
    if len(groups) > max_clusters:
        print(f"[TEST MODE] {len(groups)} candidate clusters found; capping at {max_clusters}.")
        groups = groups[:max_clusters]

    def _build_and_process(key, group):
        date_str, cell_lon, cell_lat = key
        center_lon = (cell_lon + 0.5) * cell_deg
        center_lat = (cell_lat + 0.5) * cell_deg

        best_row = group.sort_values('frp', ascending=False).iloc[0]
        acq_time_str = f"{int(best_row['acq_time']):04d}"
        rep_datetime = datetime.combine(
            best_row['acq_date'].date(),
            datetime.strptime(acq_time_str, "%H%M").time()
        )
        ee_viirs_date = ee.Date(rep_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"))

        cluster_id = f"{date_str}_{cell_lon}_{cell_lat}"
        process_viirs_cluster(cluster_id, center_lon, center_lat, ee_viirs_date, group)
        return cluster_id

    print(f"Dispatching {len(groups)} VIIRS clusters with {dispatch_workers} workers...")
    with ThreadPoolExecutor(max_workers=dispatch_workers) as executor:
        futures = {executor.submit(_build_and_process, key, group): key for key, group in groups}
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"    [!] Skipping VIIRS cluster {key}: {e}")

# ==========================================
# MASTER LOOP: LANDSAT BRANCH
# ==========================================
def process_public_zip(file_id, file_name, max_sub_archives=15, max_tiffs_per_sub=2,
                       dispatch_workers=EE_DISPATCH_WORKERS):
    print(f"\nProcessing Master Archive: {file_name}...")
    tmp_path = os.path.join(tempfile.gettempdir(), file_name)

    if not os.path.exists(tmp_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_path, quiet=False, use_cookies=False)

    extract_dir = os.path.join(tempfile.gettempdir(), file_name.replace(".zip", "_extracted"))
    os.makedirs(extract_dir, exist_ok=True)
    work_items = []

    try:
        with zipfile.ZipFile(tmp_path, 'r') as parent:
            parent.extractall(extract_dir)

        nested_zips = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) 
                       if f.endswith('.zip') and ('masks' in f.lower())]

        sub_archive_count = 0
        for n_path in nested_zips:
            if sub_archive_count >= max_sub_archives:
                print(f"[TEST MODE] Reached maximum sub-archives ({max_sub_archives}). Stopping.")
                break

            print(f" -> Accessing Sub-Archive: {os.path.basename(n_path)}")
            sub_archive_count += 1

            child_extract_dir = n_path.replace(".zip", "_contents")
            with zipfile.ZipFile(n_path, 'r') as child:
                child.extractall(child_extract_dir)

            tiff_files = [f for f in os.listdir(child_extract_dir) 
                          if f.lower().endswith('.tif') and 'intersection' in f.lower()]

            attempts = 0
            for member_name in tiff_files:
                if attempts >= max_tiffs_per_sub:
                    break
                attempts += 1

                member_path = os.path.join(child_extract_dir, member_name)
                
                with open(member_path, 'rb') as zf:
                    fb = io.BytesIO(zf.read())
                    meta = parse_zip_patch_metadata(fb, member_name)
                    if not meta:
                        continue

                    lon_cen = meta["center"][0]
                    if not (AMERICAS_LON_BOUNDS[0] <= lon_cen <= AMERICAS_LON_BOUNDS[1]):
                        continue

                    with bbox_lock:
                        collected_patch_bboxes.append(meta["bbox"])
                    work_items.append((meta, fb))

            shutil.rmtree(child_extract_dir)

    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Dispatching {len(work_items)} Landsat samples with {dispatch_workers} workers...")
    with ThreadPoolExecutor(max_workers=dispatch_workers) as executor:
        futures = {executor.submit(process_landsat_sample, meta, fb): meta['id'] for meta, fb in work_items}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"    [!] Error processing {sample_id}: {e}")

# ==========================================
# EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    import sys

    urls = {
        "North_America1.zip": "1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M",
        "North_America2.zip": "1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi",
        "South_America.zip": "1hnBUmYkvFYohTf9hTnKiyp2MXVevglK7"
    }

    MAX_SUB_ARCHIVES = 10_000
    MAX_TIFFS_PER_SUB = 10_000
    MAX_VIIRS_CLUSTERS = 3000

    mode = sys.argv[1] if len(sys.argv) > 1 else "landsat"

    if mode == "landsat":
        print("\n=== MODE: landsat (Branch A only) ===")
        for name, fid in urls.items():
            try:
                process_public_zip(
                    fid,
                    name,
                    max_sub_archives=MAX_SUB_ARCHIVES,
                    max_tiffs_per_sub=MAX_TIFFS_PER_SUB
                )
            except Exception as e:
                print(f"Fatal Interrupt processing {name}: {e}")

        n_this_run = len(collected_patch_bboxes)
        n_total_manifests = sum(
            1 for name in existing_blobs
            if name.startswith("processed_dataset/landsat_pairs/manifest/") and name.endswith(".json")
        )
        print(f"\n[SUMMARY] Landsat patches processed this run: {n_this_run}")
        print(f"[SUMMARY] Total Landsat manifests in GCS (incl. prior runs): {n_total_manifests}")
        print(f"[SUMMARY] region_bbox for VIIRS branch: {compute_viirs_pool_bbox(collected_patch_bboxes)}")
        print("\nNext step: run with `python3 dataset_pipeline.py viirs` "
              "(adjust MAX_VIIRS_CLUSTERS first if desired).")

    elif mode == "viirs":
        print("\n=== MODE: viirs (Branch B only) ===")

        bboxes = []
        for name in existing_blobs:
            if name.startswith("processed_dataset/landsat_pairs/manifest/") and name.endswith(".json"):
                try:
                    data = json.loads(bucket.blob(name).download_as_text())
                    center = data.get("center")
                    extent_m = data.get("context_extent_m")
                    if center and extent_m:
                        half_deg = (extent_m / 2) / 111320.0  # approx degrees
                        lon_c, lat_c = center
                        bboxes.append([lon_c - half_deg, lat_c - half_deg, lon_c + half_deg, lat_c + half_deg])
                except Exception:
                    continue

        if not bboxes:
            print("No usable Landsat manifests found in GCS; "
                  "falling back to a wide Americas bbox.")
            region_bbox = [AMERICAS_LON_BOUNDS[0], -56.0, AMERICAS_LON_BOUNDS[1], 75.0]
        else:
            region_bbox = compute_viirs_pool_bbox(bboxes)
            print(f"Derived region_bbox from {len(bboxes)} existing Landsat manifests: {region_bbox}")

        try:
            run_viirs_branch(region_bbox=region_bbox, max_clusters=MAX_VIIRS_CLUSTERS)
        except Exception as e:
            print(f"Fatal Interrupt processing VIIRS branch: {e}")

    else:
        print(f"Unknown mode '{mode}'. Use 'landsat' or 'viirs'.")