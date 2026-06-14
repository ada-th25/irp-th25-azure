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

LANDSAT_CONTEXT_RATIO = 4
VIIRS_CONTEXT_RATIO = 10

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

# Geographic footprint of the Landsat patches actually processed this run -
# used to spatially scope the decoupled VIIRS branch (see VIIRS_POOL_BUFFER_DEG).
collected_patch_bboxes = []

def compute_viirs_pool_bbox(patch_bboxes, buffer_deg=VIIRS_POOL_BUFFER_DEG):
    """Union bounding box of all processed Landsat patches, expanded by
    `buffer_deg` in each direction. Returns None if no patches were collected."""
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
    """Small JSON sidecar so target/GOES-context/covariate files for a given
    sample id can be matched up unambiguously at training time."""
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')

def wait_for_queue_slot():
    while True:
        tasks = ee.batch.Task.list()
        active_tasks = len([t for t in tasks if t.status()['state'] in ['RUNNING', 'READY', 'PENDING']])
        if active_tasks < 2500:
            break
        print(f"    [WAIT] GEE Engine saturated ({active_tasks} tasks). Sleeping 45s...")
        time.sleep(45)

def build_roi(center_lon, center_lat, extent_m):
    """Square ROI of side `extent_m`, centred on (center_lon, center_lat)."""
    return ee.Geometry.Point([center_lon, center_lat]).buffer(extent_m / 2).bounds()

def parse_zip_patch_metadata(file_bytes, filename):
    meta_match = re.search(r'LC08_[A-Z0-9]{4}_\d{6}_(\d{8})', filename)
    if not meta_match:
        return None
    date_str = meta_match.group(1)
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    file_bytes.seek(0)  # Safety reset before reading
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

# ==========================================
# FEATURE EXTRACTORS (GOES / VIIRS / COV)
# ==========================================
def fetch_goes_fire_features(roi, lon_cen, target_date_ee, target_scale):
    """
    Fetches GOES channels and computes the thermal difference matrix.
    Applies Contextual Windowing (60km buffer) to extract regional shapes.
    """
    # Buffer out 30,000 meters in every direction from center
    roi_center = roi.centroid()
    context_roi = roi_center.buffer(30000).bounds()

    satellite = "NOAA/GOES/17/MCMIPC" if lon_cen < -100.0 else "NOAA/GOES/16/MCMIPC"
    col = (ee.ImageCollection(satellite)
           .filterBounds(context_roi)
           .filterDate(target_date_ee.advance(-12, 'hour'), target_date_ee.advance(12, 'hour'))
           .select(['CMI_C07', 'CMI_C14']))
 
    if col.size().getInfo() == 0:
        return None, None
 
    target_millis = target_date_ee.millis()
    closest = ee.Image(col.map(lambda img: img.set('time_diff', ee.Number(img.get('system:time_start')).subtract(target_millis).abs())).sort('time_diff').first())
    
    if ee.Number(closest.get('time_diff')).getInfo() > GOES_TEMPORAL_GUARD_MIN * 60 * 1000:
        return None, None
 
    # TFI calculation on the expanded region
    tfi = closest.select('CMI_C07').subtract(closest.select('CMI_C14')).rename('GOES_TFI')
    return closest.addBands(tfi).clip(context_roi), context_roi

# ==========================================
# BRANCH A: GOES <-> LANDSAT
# ==========================================
def process_landsat_sample(meta, fb):
    sample_id = meta['id']

    target_prefix = f"processed_dataset/landsat_pairs/targets/LANDSAT_{sample_id}"
    goes_prefix = f"processed_dataset/landsat_pairs/goes_context/GOES_{sample_id}"
    cov_prefix = f"processed_dataset/landsat_pairs/covariates/COV_{sample_id}"
    manifest_path = f"processed_dataset/landsat_pairs/manifest/{sample_id}.json"

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
    context_roi = build_roi(center_lon, center_lat, LANDSAT_CONTEXT_EXTENT_M)

    if not gcs_exists(target_prefix):
        blob = bucket.blob(f"{target_prefix}.tif")
        fb.seek(0)
        blob.upload_from_file(fb, content_type='image/tiff')
        print(f"    [SAVED] Landsat target: {sample_id} (n_active_pixels={n_active_pixels})")

    if not gcs_exists(goes_prefix):
        goes_img = fetch_goes_fire_features(context_roi, center_lon, ee.Date(meta['date']))
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=goes_img, description=f"goes_ls_{sample_id}",
            bucket=BUCKET_NAME, fileNamePrefix=goes_prefix, region=context_roi,
            scale=GOES_NATIVE_SCALE_M, crs='EPSG:4326', fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] GOES context export: {sample_id}")

    if not gcs_exists(cov_prefix):
        wait_for_queue_slot()
        ee.batch.Export.image.toCloudStorage(
            image=COVARIATE_STACK.clip(context_roi), description=f"cov_ls_{sample_id}",
            bucket=BUCKET_NAME, fileNamePrefix=cov_prefix, region=context_roi,
            scale=LANDSAT_GRID_SCALE, crs='EPSG:4326', fileFormat='GeoTIFF'
        ).start()
        print(f"    [DISPATCHED] Covariate export: {sample_id}")

    write_manifest(manifest_path, {
        "id": sample_id,
        "branch": "landsat",
        "date": meta['date'],
        "center": [center_lon, center_lat],
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
# EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    urls = {"North_America1.zip": "1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M"}