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

LANDSAT_TARGET_EXTENT_M = 128 * LANDSAT_GRID_SCALE    # 3840m - fixed by Pereira 128x128 at 30m patches

VIIRS_TARGET_PIXELS = 32
VIIRS_TARGET_EXTENT_M = VIIRS_TARGET_PIXELS * VIIRS_GRID_SCALE  # 12000m

CONTEXT_RATIO = 10  # context extent = CONTEXT_RATIO x target extent (centre-aligned, fixed)
LANDSAT_CONTEXT_EXTENT_M = CONTEXT_RATIO * LANDSAT_TARGET_EXTENT_M  # 38400m  (~19x19 GOES px)
VIIRS_CONTEXT_EXTENT_M = CONTEXT_RATIO * VIIRS_TARGET_EXTENT_M      # 120000m (~60x60 GOES px)

# ==========================================
# COVARIATES
# ==========================================
DEM_ASSET = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().rename(['elevation'])
LC_ASSET = ee.ImageCollection("ESA/WorldCover/v100").select('Map').mosaic().rename(['landcover'])
COVARIATE_STACK = DEM_ASSET.addBands(LC_ASSET.toFloat())

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
# EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    urls = {"North_America1.zip": "1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M"}