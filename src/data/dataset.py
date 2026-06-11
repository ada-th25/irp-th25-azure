import io, os, re, zipfile, tempfile, gdown, ee, rasterio, time
from datetime import datetime, timedelta
from pyproj import Transformer
from google.cloud import storage
from google.auth import default

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
 
AMERICAS_LON_BOUNDS = [-170.0, -30.0]
 
GOES_TEMPORAL_GUARD_MIN = 15        # max |GOES time - target time| allowed, minutes
VIIRS_SEARCH_WINDOW_DAYS = 3        # +/- days around Landsat date to look for VIIRS detections
VIIRS_PASS_GROUP_HOURS = 6          # group detections within this window of the chosen overpass
 
COVARIATE_SCALE = 30                # DEM+LC export resolution (matches Landsat target grid)
LANDSAT_PAIR_SCALE = 30             # GOES export resolution for the Landsat-pair input
VIIRS_RASTER_SCALE = 375            # VIIRS target + GOES export resolution for the VIIRS-pair
 
# Pre-load global covariate assets once to minimise API overhead inside the loops
DEM_ASSET = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().rename(['elevation'])
LC_ASSET = ee.Image("ESA/WorldCover/v100").select('Map').rename(['landcover'])
COVARIATE_STACK = DEM_ASSET.addBands(LC_ASSET)

# ==========================================
# GCS HELPERS
# ==========================================
def gcs_exists(path_prefix):
    """True if any object exists under this prefix (export-in-progress or completed)."""
    return any(name.startswith(path_prefix) for name in existing_blobs)
 
def wait_for_queue_slot():
    """Monitors GEE task engine to ensure queue limits are not violated."""
    while True:
        tasks = ee.batch.Task.list()
        active_tasks = len([t for t in tasks if t.status()['state'] in ['RUNNING', 'READY', 'PENDING']])
        if active_tasks < 2500:
            break
        print(f"    [WAIT] GEE Engine queue saturated ({active_tasks} active tasks). Sleeping 45s...")
        time.sleep(45)

# ==========================================
# EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    urls = {"North_America1.zip": "1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M"}