import io
import os
import s3fs
import requests
import numpy as np
import pandas as pd
import re
import tempfile
from pyresample import geometry, kd_tree
from satpy import Scene
from google.cloud import storage

# ==========================================
# CONFIGURATION & INFRASTRUCTURE SETUP
# ==========================================
BUCKET_NAME = "wildfire-irp-data-th25"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

fs = s3fs.S3FileSystem(anon=True)

ROI = [38.7, 38.9, -122.9, -122.7] 
BBOX = f"{ROI[2]},{ROI[0]},{ROI[3]},{ROI[1]}"  
MAP_KEY = "694f59d089b8dab51c7e4c1094067706"

def upload_matrix_to_gcs(array, destination_blob_name):
    """Streams a processed NumPy array directly to GCS without writing to local disk."""
    blob = bucket.blob(destination_blob_name)
    file_stream = io.BytesIO()
    np.save(file_stream, array)
    file_stream.seek(0)
    blob.upload_from_file(file_stream, content_type='application/octet-stream')
    print(f"Uploaded: gs://{BUCKET_NAME}/{destination_blob_name}")

# ==========================================
# PHASE 1: VIIRS VECTOR PROCESSING WITH TIMEZONE CORRECTION
# ==========================================
def fetch_viirs_ground_truth(start_date, number_of_days=5):
    """
    Fetches VIIRS S-NPP data and ensures timestamps are explicitly parsed as UTC.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{BBOX}/{number_of_days}/{start_date}"
    
    print(f"Requesting target fire points from NASA API for starting date: {start_date}...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data from NASA API: {response.status_code}")
        return None
        
    df = pd.read_csv(io.StringIO(response.text))
    
    if df.empty or 'latitude' not in df.columns:
        print("No active fire points found for this window/ROI.")
        return None

    # Filter spatially
    spatial_mask = (df['latitude'] >= ROI[0]) & (df['latitude'] <= ROI[1]) & \
                   (df['longitude'] >= ROI[2]) & (df['longitude'] <= ROI[3])
    filtered_df = df[spatial_mask].copy()
    
    if filtered_df.empty:
        print("No fire points matched the strict spatial boundaries.")
        return None

    # Combine date and time
    filtered_df['datetime_raw'] = pd.to_datetime(filtered_df['acq_date'] + ' ' + 
                                                 filtered_df['acq_time'].astype(str).str.zfill(4), 
                                                 format='%Y-%m-%d %H%M')
    
    # NASA FIRMS API returns UTC timestamps for global queries. 
    # If the response contains a local time variation or lacks an explicit zone,
    # localize it to UTC to ensure compatibility with AWS S3 structures.
    filtered_df['datetime'] = filtered_df['datetime_raw'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
    
    return filtered_df

# ==========================================
# PHASE 2: GOES-17 STREAMING & PROCESSING
# ==========================================
def fetch_and_process_goes(target_datetime):
    target_utc = target_datetime.tz_convert('UTC')
    year = target_utc.strftime('%Y')
    day_of_year = target_utc.strftime('%j')
    hour = target_utc.strftime('%H')
    
    s3_dir = f"noaa-goes17/ABI-L1b-RadC/{year}/{day_of_year}/{hour}/"
    print(f"  → Querying S3 path: {s3_dir}")
    
    try:
        files = fs.ls(s3_dir)
        print(f"  → Found {len(files)} files in directory")
    except Exception as e:
        print(f"  → S3 ls FAILED: {e}")
        return None

    ch07_files = [f for f in files if re.search(r'C07_G17', f)]
    print(f"  → {len(ch07_files)} C07 channel files found")
    
    if not ch07_files:
        return None

    best_file = None
    min_diff = pd.Timedelta(minutes=5.0)
    
    for f in ch07_files:
        try:
            time_str = f.split('_s')[1][:13]
            file_dt = pd.to_datetime(time_str, format='%Y%j%H%M%S').tz_localize('UTC')
            diff = abs(file_dt - target_utc)
            print(f"    file: {os.path.basename(f)}, dt={file_dt}, diff={diff}")  # temp
            if diff < min_diff:
                min_diff = diff
                best_file = f
        except (IndexError, ValueError) as e:
            print(f"    Parse error on {f}: {e}")
            continue

    if not best_file:
        print(f"No matching GOES-17 file within temporal threshold for {target_utc}")
        return None

    print(f"Found matching GOES file: {os.path.basename(best_file)} (Offset: {min_diff})")

    # Download to a temp file — h5netcdf/netCDF4 can't stream from S3 directly
    original_filename = os.path.basename(best_file)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, original_filename)

    try:
        print(f"  → Downloading to temp file...")
        fs.get(best_file, tmp_path)

        scn = Scene(reader="abi_l1b", filenames=[tmp_path])
        scn.load(['C07'])
        cropped_scn = scn.crop(ll_bbox=(ROI[2], ROI[0], ROI[3], ROI[1]))
        goes_matrix = cropped_scn['C07'].values
        return goes_matrix

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

# ==========================================
# VIIRS Rasterization
# ==========================================
def rasterize_viirs_to_goes_grid(viirs_df, goes_scene, target_datetime):
    """Rasterize VIIRS point FRP detections onto the GOES pixel grid."""
    
    # Get the GOES area definition (geographic grid) from the cropped scene
    area_def = goes_scene['C07'].attrs['area']
    
    # Filter VIIRS points to this timestamp (±6 min window to catch overpass)
    window = pd.Timedelta(minutes=6)
    mask = abs(viirs_df['datetime'] - target_datetime) <= window
    pts = viirs_df[mask]
    
    if pts.empty:
        return np.zeros((area_def.height, area_def.width), dtype=np.float32)
    
    # Build a swath definition from VIIRS point locations
    swath_def = geometry.SwathDefinition(
        lons=pts['longitude'].values,
        lats=pts['latitude'].values
    )
    
    # Resample FRP values onto the GOES grid using nearest-neighbour
    frp_grid = kd_tree.resample_nearest(
        swath_def,
        pts['frp'].values.astype(np.float32),
        area_def,
        radius_of_influence=2000,  # 2 km — matches GOES resolution
        fill_value=0.0
    )
    
    return frp_grid

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    START_DATE = "2019-10-24" 
    
    viirs_targets = fetch_viirs_ground_truth(start_date=START_DATE, number_of_days=5)
    
    if viirs_targets is not None:
        unique_timestamps = viirs_targets['datetime'].unique()
        print(f"Found {len(unique_timestamps)} unique fire timestamps to process.")

        for ts in unique_timestamps:
            ts_dt = pd.Timestamp(ts)
            print(f"\nProcessing timestamp (UTC): {ts_dt}")
            
            goes_data = fetch_and_process_goes(ts_dt)
            if goes_data is None:
                continue
                
            viirs_rasterized = np.zeros_like(goes_data) 
            
            ts_str = ts_dt.strftime('%Y%m%d_%H%M%S')
            upload_matrix_to_gcs(goes_data, f"processed_dataset/inputs/GOES_{ts_str}.npy")
            upload_matrix_to_gcs(viirs_rasterized, f"processed_dataset/ground_truth/VIIRS_{ts_str}.npy")
    else:
        print("Pipeline execution stopped: No source data generated.")