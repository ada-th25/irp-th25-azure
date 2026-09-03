"""
Fetch static covariates (DEM + landcover) for dynamic Bootleg Fire patches.

Dynamically iterates through the active fire centers produced by 
run_bootleg_case_study.py, deduplicates physical tile locations, 
and exports ONE 256x256 @ 30m covariate patch per unique spatial location.
"""

import time
import ee
import pandas as pd
from google.cloud import storage
from google.auth import default

# ==========================================
# Configuration (Matches run_bootleg_case_study.py)
# ==========================================
COVARIATE_EXTENT_M = 7680
COVARIATE_SCALE_M = 30  # Yields 256x256 pixels

BUCKET_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "case_studies/bootleg_2021_dynamic"
QA_MANIFEST_PATH = f"{OUTPUT_PREFIX}/goes_hourly_qa.csv"


def get_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def build_roi(center_lon, center_lat, extent_m, target_crs):
    pt = ee.Geometry.Point([center_lon, center_lat], "EPSG:4326")
    return pt.transform(target_crs).buffer(extent_m / 2).bounds(maxError=1, proj=target_crs)


def main():
    credentials, project = default()
    ee.Initialize(credentials=credentials, project="wildfire-irp")
    
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)

    # 1. Read QA Manifest to retrieve dynamic center coordinates
    blob = bucket.blob(QA_MANIFEST_PATH)
    if not blob.exists():
        raise FileNotFoundError(
            f"QA manifest not found at gs://{BUCKET_NAME}/{QA_MANIFEST_PATH}. "
            "Please run `run_bootleg_case_study.py` first."
        )
    
    qa_df = pd.read_csv(f"gs://{BUCKET_NAME}/{QA_MANIFEST_PATH}")
    valid_frames = qa_df[qa_df["status"] == "ok"].copy()

    if valid_frames.empty:
        print("[COVARIATES] No valid frames found in QA manifest.")
        return

    # 2. Spatial Deduplication: Round centers to 3 decimal places (~100m)
    # This prevents exporting duplicate 30m tiles for identical/near-identical centers.
    valid_frames["spatial_key"] = (
        valid_frames["center_lon"].round(3).astype(str) + "_" + 
        valid_frames["center_lat"].round(3).astype(str)
    )
    
    unique_tiles = valid_frames.drop_duplicates(subset=["spatial_key"])
    print(f"[COVARIATES] Found {len(valid_frames)} total hourly frames across {len(unique_tiles)} unique physical tile locations.")

    # 3. Base Earth Engine Image (Copernicus DEM + ESA WorldCover)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename(["elevation"])
    lc = ee.ImageCollection("ESA/WorldCover/v100").select("Map").mosaic().rename(["landcover"])
    covariate_stack = dem.unmask(0).addBands(lc.unmask(0).toFloat())

    tasks = []

    # 4. Iterate over unique spatial locations
    for _, row in unique_tiles.iterrows():
        lon = float(row["center_lon"])
        lat = float(row["center_lat"])
        
        # Spatial key used in filename
        loc_str = f"lon{lon:.3f}_lat{lat:.3f}"
        cov_prefix = f"{OUTPUT_PREFIX}/covariates/COV_{loc_str}"
        file_blob = f"{cov_prefix}.tif"

        # CACHE CHECK: Avoid re-exporting if the physical tile already exists on GCS
        if bucket.blob(file_blob).exists():
            print(f"    [SKIP / CACHED] {loc_str} already exists in GCS.")
            continue

        utm_crs = get_utm_epsg(lon, lat)
        context_roi = build_roi(lon, lat, COVARIATE_EXTENT_M, utm_crs)

        task = ee.batch.Export.image.toCloudStorage(
            image=covariate_stack.clip(context_roi),
            description=f"cov_{loc_str}",
            bucket=BUCKET_NAME,
            fileNamePrefix=cov_prefix,
            region=context_roi,
            scale=COVARIATE_SCALE_M,
            crs=utm_crs,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append((task, loc_str))
        print(f"    [STARTED] Exporting covariate patch for center ({lon:.3f}, {lat:.3f})")

    # 5. Monitor all pending tasks
    if not tasks:
        print("[COVARIATES] All required covariate patches are already cached on GCS!")
        return

    print(f"[COVARIATES] Waiting for {len(tasks)} Earth Engine export tasks to complete...")
    for task, loc_str in tasks:
        while task.active():
            time.sleep(5)
        print(f"    [COMPLETED] {loc_str}: {task.status()['state']}")

    print("[COVARIATES] All static covariate tiles successfully saved to GCS.")


if __name__ == "__main__":
    main()