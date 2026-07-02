import io
import os
import re
import json
import glob
import zipfile
import tempfile
import gdown
import ee
import rasterio
import time
import threading
import warnings
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pyproj import Transformer
from google.cloud import storage
from google.auth import default
import numpy as np
import pandas as pd

# Suppress the noisy httplib2 transport warning log
warnings.filterwarnings("ignore", message=".*transport does not support per-request timeout.*")

# ==========================================
# Initialisation & Credentials
# ==========================================
credentials, project = default()
storage_client = storage.Client(credentials=credentials)
ee.Initialize(credentials=credentials, project='wildfire-irp')

BUCKET_NAME = "wildfire-irp-data-th25"
bucket = storage_client.bucket(BUCKET_NAME)

# ==========================================
# Output Prefix
# ==========================================
OUTPUT_PREFIX = "processed_dataset_r1"

print("Caching existing files from GCS...")
existing_blobs = set(b.name for b in bucket.list_blobs(prefix=f"{OUTPUT_PREFIX}/"))

# ==========================================
# Region / Time Constants
# ==========================================
AMERICAS_LON_BOUNDS = [-170.0, -30.0]
VIIRS_POOL_BUFFER_DEG = 2.0
VIIRS_DATE_RANGE = ("2020-01-01", "2020-12-31")

# ==========================================
# Resolution / Extent Constants

# NOTE: Pereira et al. patches are 256x256 px @ 30m = 7680m per side
# ==========================================
GOES_TEMPORAL_GUARD_MIN = 60     # Max temporal gap allowed between target time and GOES scene (minutes)
GOES_NATIVE_SCALE_M = 2000        # Native GOES ABI grid spacing

LANDSAT_GRID_SCALE = 30           # 30m target resolution for Landsat branch
VIIRS_GRID_SCALE = 375            # 375m target resolution for VIIRS branch

LANDSAT_TARGET_PIXELS = 256
LANDSAT_TARGET_EXTENT_M = LANDSAT_TARGET_PIXELS * LANDSAT_GRID_SCALE  # 7680m - Pereira 256x256 @ 30m patches

VIIRS_TARGET_EXTENT_M = 7500 # 20 x 375m
VIIRS_TARGET_PIXELS = int(VIIRS_TARGET_EXTENT_M / VIIRS_GRID_SCALE) # 20

# Context-to-target ratio. This gives GOES and context data a zoomed-out window
# A ratio of 1 means context == target extent.
LANDSAT_CONTEXT_RATIO = 1
VIIRS_CONTEXT_RATIO = 1

LANDSAT_CONTEXT_EXTENT_M = LANDSAT_CONTEXT_RATIO * LANDSAT_TARGET_EXTENT_M 
VIIRS_CONTEXT_EXTENT_M = VIIRS_CONTEXT_RATIO * VIIRS_TARGET_EXTENT_M

print(f"[CONFIG] OUTPUT_PREFIX={OUTPUT_PREFIX}, "
      f"LANDSAT_CONTEXT_RATIO={LANDSAT_CONTEXT_RATIO}, VIIRS_CONTEXT_RATIO={VIIRS_CONTEXT_RATIO} "
      f"(context_extent == target_extent when ratio == 1)")

# Drop patches with fewer than 10 active pixels
MIN_ACTIVE_PIXELS = 10

# ==========================================
# Covariates
# ==========================================
DEM_ASSET = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().rename(['elevation'])
LC_ASSET = ee.ImageCollection("ESA/WorldCover/v100").select('Map').mosaic().rename(['landcover'])
COVARIATE_STACK = DEM_ASSET.unmask(0).addBands(LC_ASSET.unmask(0).toFloat())

# ==========================================
# NASA VIIRS Archive: Load + Filter
# ==========================================
VIIRS_CSV_PATH = "raw_viirs/viirs_snpp_fire_archive_americas_2020jan_dec.csv"
VIIRS_CSV_GCS_PATH = "raw_viirs/viirs_snpp_fire_archive_americas_2020jan_dec.csv"  # path inside the bucket

print(f"Downloading VIIRS archive CSV from GCS: gs://{BUCKET_NAME}/{VIIRS_CSV_GCS_PATH} ...")
viirs_blob = bucket.blob(VIIRS_CSV_GCS_PATH)
if not viirs_blob.exists():
    raise FileNotFoundError(
        f"VIIRS archive not found in GCS at gs://{BUCKET_NAME}/{VIIRS_CSV_GCS_PATH}. "
        f"Upload it once with: gsutil cp {VIIRS_CSV_PATH} gs://{BUCKET_NAME}/{VIIRS_CSV_GCS_PATH}"
    )
df_viirs = pd.read_csv(io.BytesIO(viirs_blob.download_as_bytes()))
df_viirs.columns = df_viirs.columns.str.strip()

if 'acq_date' not in df_viirs.columns:
    raise KeyError("Missing required column 'acq_date' in source file.")

before = len(df_viirs)
df_viirs = df_viirs.dropna(subset=['acq_date', 'acq_time'])
dropped = before - len(df_viirs)
if dropped:
    print(f"Dropped {dropped} malformed row(s) with missing acq_date/acq_time.")

df_viirs['acq_date'] = pd.to_datetime(df_viirs['acq_date'])
df_viirs['acq_time_str'] = df_viirs['acq_time'].astype(int).apply(lambda t: f"{t:04d}")
df_viirs['acq_datetime'] = pd.to_datetime(
    df_viirs['acq_date'].dt.strftime('%Y-%m-%d') + ' ' + df_viirs['acq_time_str'],
    format='%Y-%m-%d %H%M'
)
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
print(f"VIIRS branch pool after Jan-Dec-2020 filtering: {len(df_viirs)} detections.")

# ==========================================
# Core Utilities
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

EE_DISPATCH_WORKERS = 12
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
        active_tasks = len([t for t in tasks if t.status()['state'] in ['RUNNING', 'READY']])
        if active_tasks < 2500:
            break
        print(f"    [WAIT] GEE Engine saturated ({active_tasks} tasks). Sleeping 45s...")
        time.sleep(45)

def build_roi(center_lon, center_lat, extent_m, target_crs):
    """Builds a projection-aligned square bounding box."""
    # Define point in lat/lon, transform to target UTM, buffer in meters, and get UTM bounds
    pt = ee.Geometry.Point([center_lon, center_lat], 'EPSG:4326')
    return pt.transform(target_crs).buffer(extent_m / 2).bounds(maxError=1, proj=target_crs)

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
            lon_cen, lat_cen = t.transform((bounds.left + bounds.right) / 2,
                                            (bounds.bottom + bounds.top) / 2)

    # Look up precise acquisition time from GEE Landsat metadata
    # Scene ID encodes path/row: LC08_L1TP_PPPRRR_YYYYMMDD_...
    scene_match = re.search(r'LC08_[A-Z0-9]{4}_(\d{3})(\d{3})_(\d{8})', filename)
    acquisition_datetime = formatted_date  # fallback: date only
    if scene_match:
        try:
            path = int(scene_match.group(1))
            row  = int(scene_match.group(2))
            d    = scene_match.group(3)
            next_day = (datetime.strptime(d, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            ls_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
                      .filterDate(formatted_date, next_day)
                      .filter(ee.Filter.eq('WRS_PATH', path))
                      .filter(ee.Filter.eq('WRS_ROW',  row)))
            ls_time = ls_col.first().get('system:time_start').getInfo()
            if ls_time:
                acquisition_datetime = datetime.utcfromtimestamp(
                    ls_time / 1000
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass  # fall back to date-only if GEE lookup fails

    return {
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "center": [lon_cen, lat_cen],
        "date": formatted_date,
        "datetime": acquisition_datetime,
        "id": filename.replace(".tif", "")
    }

def get_utm_epsg(lon, lat):
    """Return the EPSG code for the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

# ==========================================
# Feature Extractors (GOES / VIIRS / COV)
# ==========================================
def fetch_goes_fire_features(context_roi, lon_cen, target_date_ee):
    context_roi_wgs84 = context_roi.transform('EPSG:4326', maxError=100)

    col = (ee.ImageCollection("NOAA/GOES/16/MCMIPF")
           .merge(ee.ImageCollection("NOAA/GOES/17/MCMIPF"))
           .filterBounds(context_roi_wgs84)
           .filterDate(target_date_ee.advance(-3, 'hour'), target_date_ee.advance(3, 'hour'))
           .select(['CMI_C07', 'CMI_C14']))

    target_millis = target_date_ee.millis()

    def scale_to_bt(img):
        time_diff = ee.Number(img.get('system:time_start')).subtract(target_millis).abs()
        c07_bt = (img.select('CMI_C07')
                     .multiply(ee.Number(img.get('CMI_C07_scale')))
                     .add(ee.Number(img.get('CMI_C07_offset'))))
        c14_bt = (img.select('CMI_C14')
                     .multiply(ee.Number(img.get('CMI_C14_scale')))
                     .add(ee.Number(img.get('CMI_C14_offset'))))
        return (c07_bt.rename('CMI_C07')
                      .addBands(c14_bt.rename('CMI_C14'))
                      .set('system:time_start', img.get('system:time_start'))
                      .set('time_diff', time_diff))

    col = col.map(scale_to_bt)

    dummy_img = (ee.Image.constant([0.0, 0.0])
                 .rename(['CMI_C07', 'CMI_C14'])
                 .set('time_diff', ee.Number(9999999999))
                 .set('system:time_start', ee.Number(0)))

    safe_col  = col.merge(ee.ImageCollection([dummy_img]))
    closest   = ee.Image(safe_col.sort('time_diff').first())

    tfi         = closest.select('CMI_C07').subtract(closest.select('CMI_C14')).rename('GOES_TFI')
    final_stack = closest.select(['CMI_C07', 'CMI_C14']).addBands(tfi).toFloat()

    max_diff_ms    = GOES_TEMPORAL_GUARD_MIN * 60 * 1000
    empty_fallback = (ee.Image.constant([0.0, 0.0, 0.0])
                      .rename(['CMI_C07', 'CMI_C14', 'GOES_TFI'])
                      .toFloat())

    time_condition = ee.Number(closest.get('time_diff')).lte(max_diff_ms)

    return ee.Image(ee.Algorithms.If(
        time_condition,
        final_stack,
        empty_fallback
    )).toFloat()

# ==========================================
# Branch A: GOES <-> Landsat
# ==========================================
def process_landsat_sample(meta, fb, mask_source="intersection"):
    sample_id = meta['id']

    target_prefix = f"{OUTPUT_PREFIX}/landsat_pairs/targets/LANDSAT_{sample_id}"
    goes_prefix = f"{OUTPUT_PREFIX}/landsat_pairs/goes_input/GOES_{sample_id}"
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
    context_roi = build_roi(center_lon, center_lat, LANDSAT_CONTEXT_EXTENT_M, utm_crs)

    if not gcs_exists(target_prefix):
        blob = bucket.blob(f"{target_prefix}.tif")
        fb.seek(0)
        blob.upload_from_file(fb, content_type='image/tiff')
        print(f"    [SAVED] Landsat target: {sample_id} (n_active_pixels={n_active_pixels}, "
              f"mask_source={mask_source})")

    goes_blob = bucket.blob(f"{goes_prefix}.tif")
    goes_dispatched = goes_blob.exists()
    if not goes_dispatched:
        goes_img = fetch_goes_fire_features(context_roi, center_lon, ee.Date(meta['datetime']))
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=goes_img, description=f"goes_ls_{sample_id}",
            bucket=BUCKET_NAME, fileNamePrefix=goes_prefix, region=context_roi,
            scale=GOES_NATIVE_SCALE_M, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] GOES context export: {sample_id}")
        goes_dispatched = True

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
        "datetime": meta['datetime'],
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
        "goes_dispatched": goes_dispatched,
        "paths": {
            "target": f"{target_prefix}.tif",
            "goes_input": f"{goes_prefix}.tif",
            "covariates": f"{cov_prefix}.tif",
        }
    })

# ==========================================
# Branch B: GOES <-> VIIRS
# ==========================================
def assign_viirs_cells(df, cell_extent_m=VIIRS_TARGET_EXTENT_M):
    df = df.copy()
    cell_deg = cell_extent_m / 111320.0
    df['cell_lon'] = (df['longitude'] // cell_deg).astype(int)
    df['cell_lat'] = (df['latitude'] // cell_deg).astype(int)
    VIIRS_TEMPORAL_WINDOW_MIN = 30
    df['overpass_window'] = df['acq_datetime'].dt.floor(f'{VIIRS_TEMPORAL_WINDOW_MIN}min')
    return df, cell_deg

def process_viirs_cluster(cluster_id, center_lon, center_lat, ee_viirs_date, group_df):
    target_prefix = f"{OUTPUT_PREFIX}/viirs_pairs/targets/VIIRS_{cluster_id}"
    goes_prefix = f"{OUTPUT_PREFIX}/viirs_pairs/goes_input/GOES_{cluster_id}"
    cov_prefix = f"{OUTPUT_PREFIX}/viirs_pairs/covariates/COV_{cluster_id}"
    manifest_path = f"{OUTPUT_PREFIX}/viirs_pairs/manifest/{cluster_id}.json"

    utm_crs = get_utm_epsg(center_lon, center_lat)
    target_roi = build_roi(center_lon, center_lat, VIIRS_TARGET_EXTENT_M, utm_crs)
    context_roi = build_roi(center_lon, center_lat, VIIRS_CONTEXT_EXTENT_M, utm_crs)

    if not gcs_exists(target_prefix):
        features = [
            ee.Feature(ee.Geometry.Point([float(row['longitude']), float(row['latitude'])]), {'frp': float(row['frp'])})
            for _, row in group_df.iterrows()
        ]
        viirs_img = (ee.FeatureCollection(features)
                     .reduceToImage(properties=['frp'], reducer=ee.Reducer.max())
                     .clip(target_roi)
                     .unmask(0)
                     .rename('VIIRS_FRP'))
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=viirs_img, description=f"viirs_frp_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=target_prefix, region=target_roi,
            scale=VIIRS_GRID_SCALE, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] VIIRS target export: {cluster_id}")

    goes_blob = bucket.blob(f"{goes_prefix}.tif")
    goes_dispatched = goes_blob.exists()
    if not goes_dispatched:
        goes_img = fetch_goes_fire_features(context_roi, center_lon, ee_viirs_date)
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=goes_img, description=f"goes_vx_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=goes_prefix, region=context_roi,
            scale=GOES_NATIVE_SCALE_M, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] GOES context export: {cluster_id}")
        goes_dispatched = True

    if not gcs_exists(cov_prefix):
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=COVARIATE_STACK.clip(context_roi), description=f"cov_vx_{cluster_id}",
            bucket=BUCKET_NAME, fileNamePrefix=cov_prefix, region=context_roi,
            scale=LANDSAT_GRID_SCALE, crs=utm_crs, fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] Covariate export: {cluster_id}")

    write_manifest(manifest_path, {
        "id": cluster_id,
        "branch": "viirs",
        "date": group_df['acq_date'].iloc[0].strftime("%Y-%m-%d"),
        "center": [center_lon, center_lat],
        "crs": utm_crs,
        "target_pixels_nominal": VIIRS_TARGET_PIXELS,  # 7500m/375m=20 exactly
        "target_extent_m": VIIRS_TARGET_EXTENT_M,
        "target_scale_m": VIIRS_GRID_SCALE,
        "context_ratio": VIIRS_CONTEXT_RATIO,
        "context_extent_m": VIIRS_CONTEXT_EXTENT_M,
        "goes_scale_m": GOES_NATIVE_SCALE_M,
        "covariate_scale_m": LANDSAT_GRID_SCALE,
        "n_detections": int(len(group_df)),
        "max_frp": float(group_df['frp'].max()),
        "goes_dispatched": goes_dispatched,
        "paths": {
            "target": f"{target_prefix}.tif",
            "goes_input": f"{goes_prefix}.tif",
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
    groups = list(df.groupby(['overpass_window', 'cell_lon', 'cell_lat']))
    if len(groups) > max_clusters:
        print(f"[TEST MODE] {len(groups)} candidate clusters found; capping at {max_clusters}.")
        groups = groups[:max_clusters]

    VIIRS_INTRA_CLUSTER_MAX_SPREAD_MIN = 15
    
    def _build_and_process(key, group):
        overpass_window, cell_lon, cell_lat = key
    
        # Compute geographic centre from cell indices
        center_lon = (cell_lon + 0.5) * cell_deg
        center_lat = (cell_lat + 0.5) * cell_deg
    
        # Drop detections more than 15 min from the peak-FRP detection's time
        peak_time = group.loc[group['frp'].idxmax(), 'acq_datetime']
        time_spread = (group['acq_datetime'] - peak_time).abs()
        group = group[time_spread <= pd.Timedelta(minutes=VIIRS_INTRA_CLUSTER_MAX_SPREAD_MIN)]
    
        # Representative datetime from peak-FRP detection
        ee_viirs_date = ee.Date(peak_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    
        cluster_id = f"{overpass_window.strftime('%Y%m%dT%H%M')}_{cell_lon}_{cell_lat}"
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
# Master Loop: Landsat Branch
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

            attempts = 0
            with zipfile.ZipFile(n_path, 'r') as child:
                for member_name in child.namelist():
                    if member_name.lower().endswith('.tif') and 'intersection' in member_name.lower():
                        if attempts >= max_tiffs_per_sub:
                            break
                        
                        # Stream the file directly from the zip bytes into rasterio memory
                        with child.open(member_name) as member_file:
                            file_bytes = io.BytesIO(member_file.read())
                            
                            with rasterio.MemoryFile(file_bytes) as memfile:
                                with memfile.open() as src:
                                    mask = src.read(1)
                                    n_active_pixels = int(np.sum(mask == 1))
                        
                        # Instantly drop sparse patches from the memory pipeline
                        if n_active_pixels < MIN_ACTIVE_PIXELS:
                            print(f"    [SKIP landsat] Only {n_active_pixels} active fire pixel(s) (< {MIN_ACTIVE_PIXELS}) for {member_name}")
                            continue

                        # Reuse the stream to parse metadata and append to queue
                        meta = parse_zip_patch_metadata(file_bytes, member_name)
                        if not meta:
                            continue

                        lon_cen = meta["center"][0]
                        if not (AMERICAS_LON_BOUNDS[0] <= lon_cen <= AMERICAS_LON_BOUNDS[1]):
                            continue

                        with bbox_lock:
                            collected_patch_bboxes.append(meta["bbox"])
                        
                        # Store file bytes to avoid reading from zip again inside the executor
                        work_items.append((meta, file_bytes))
                        attempts += 1

    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)

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
# Execution Loop
# ==========================================
def process_flat_zip(file_id, file_name, mask_source, max_tiffs=10_000,
                      dispatch_workers=EE_DISPATCH_WORKERS):
    print(f"\nProcessing Flat Archive: {file_name} (mask_source={mask_source})...")
    tmp_path = os.path.join(tempfile.gettempdir(), file_name)

    if not os.path.exists(tmp_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_path, quiet=False, use_cookies=False)

    extract_dir = os.path.join(tempfile.gettempdir(), file_name.replace(".zip", "_extracted"))
    os.makedirs(extract_dir, exist_ok=True)
    work_items = []

    try:
        with zipfile.ZipFile(tmp_path, 'r') as parent:
            parent.extractall(extract_dir)

        tiff_paths = glob.glob(os.path.join(extract_dir, "**", "*.tif"), recursive=True)
        print(f"Found {len(tiff_paths)} tif patches in {file_name}")

        attempts = 0
        for member_path in tiff_paths:
            if attempts >= max_tiffs:
                print(f"[TEST MODE] Reached maximum tiffs ({max_tiffs}). Stopping.")
                break
            attempts += 1

            member_name = os.path.basename(member_path)
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

    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Dispatching {len(work_items)} samples from {file_name} with {dispatch_workers} workers...")
    with ThreadPoolExecutor(max_workers=dispatch_workers) as executor:
        futures = {
            executor.submit(process_landsat_sample, meta, fb, mask_source): meta['id']
            for meta, fb in work_items
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"    [!] Error processing {sample_id}: {e}")

if __name__ == "__main__":

    urls = {
        "North_America1.zip": "1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M",
        "North_America2.zip": "1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi",
        "South_America.zip": "1hnBUmYkvFYohTf9hTnKiyp2MXVevglK7"
    }

    # Manually-annotated validation patches (Pereira et al.): 13 large
    # Landsat-8 scenes from Sept 2020.
    # Same 256x256 @ 30m / binary {0,1} convention as the continental
    # archives, but a single flat zip rather than nested per-scene
    # sub-archives - handled by process_flat_zip().
    MANUAL_ANNOTATIONS_FILE_ID = "1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf"
    MANUAL_ANNOTATIONS_FILE_NAME = "manual_annotations_patches.zip"

    MAX_SUB_ARCHIVES = 10_000     # effectively unbounded - each zip has <1000 sub-archives
    MAX_TIFFS_PER_SUB = 10_000    # effectively unbounded - typically 1 intersection tiff per sub-archive
    MAX_VIIRS_CLUSTERS = 5000

    mode = sys.argv[1] if len(sys.argv) > 1 else "test_all"

    if mode == "test_all":
        print("\n=== MODE: test_all (Sequential validation of Landsat + VIIRS Branches) ===")
        
        # RUN LANDSAT BRANCH
        print("\n--- Step 1: Running Landsat/GOES Branch ---")
        try:
            process_flat_zip(
                MANUAL_ANNOTATIONS_FILE_ID,
                MANUAL_ANNOTATIONS_FILE_NAME,
                mask_source="manual_v1",
                max_tiffs=MAX_TIFFS_PER_SUB
            )
        except Exception as e:
            print(f"Fatal Interrupt processing {MANUAL_ANNOTATIONS_FILE_NAME}: {e}")
    
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

        # COMPUTE SPATIAL BOUNDING BOX
        n_this_run = len(collected_patch_bboxes)
        print(f"\n[SUMMARY] Landsat patches processed this test run: {n_this_run}")
        
        if n_this_run == 0:
            print("\n[WARNING] No Landsat patches collected this run. Pulling historic manifests from GCS...")
            bboxes = []
            for name in existing_blobs:
                if name.startswith(f"{OUTPUT_PREFIX}/landsat_pairs/manifest/") and name.endswith(".json"):
                    try:
                        data = json.loads(bucket.blob(name).download_as_text())
                        center = data.get("center")
                        extent_m = data.get("context_extent_m")
                        if center and extent_m:
                            half_deg = (extent_m / 2) / 111320.0
                            lon_c, lat_c = center
                            bboxes.append([lon_c - half_deg, lat_c - half_deg, lon_c + half_deg, lat_c + half_deg])
                    except Exception:
                        continue
            region_bbox = compute_viirs_pool_bbox(bboxes) if bboxes else [AMERICAS_LON_BOUNDS[0], -56.0, AMERICAS_LON_BOUNDS[1], 75.0]
        else:
            region_bbox = compute_viirs_pool_bbox(collected_patch_bboxes)
            
        print(f"Target region_bbox calculated for testing VIIRS branch: {region_bbox}")

        # RUN VIIRS BRANCH
        print("\n--- Step 2: Running VIIRS/GOES Branch ---")
        try:
            run_viirs_branch(region_bbox=region_bbox, max_clusters=MAX_VIIRS_CLUSTERS)
        except Exception as e:
            print(f"Fatal Interrupt processing VIIRS branch: {e}")
            
        print("\n=== [TEST COMPLETE] Component export commands dispatched to Earth Engine. ===")

    elif mode == "landsat":
        print("\n=== MODE: landsat (Branch A only) ===")
        
        try:
            process_flat_zip(
                MANUAL_ANNOTATIONS_FILE_ID,
                MANUAL_ANNOTATIONS_FILE_NAME,
                mask_source="manual_v1",
                max_tiffs=MAX_TIFFS_PER_SUB
            )
        except Exception as e:
            print(f"Fatal Interrupt processing {MANUAL_ANNOTATIONS_FILE_NAME}: {e}")
    
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
            if name.startswith(f"{OUTPUT_PREFIX}/landsat_pairs/manifest/") and name.endswith(".json")
        )
        print(f"\n[SUMMARY] Landsat patches processed this run: {n_this_run}")
        print(f"[SUMMARY] Total Landsat manifests in GCS (incl. prior runs): {n_total_manifests}")
        print(f"[SUMMARY] region_bbox for VIIRS branch: {compute_viirs_pool_bbox(collected_patch_bboxes)}")

    elif mode == "viirs":
        print("\n=== MODE: viirs (Branch B only) ===")

        bboxes = []
        for name in existing_blobs:
            if name.startswith(f"{OUTPUT_PREFIX}/landsat_pairs/manifest/") and name.endswith(".json"):
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