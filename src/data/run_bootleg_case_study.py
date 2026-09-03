"""
Bootleg Fire 2021 Case Study Driver (Dynamic Tracking + Smooth Centroid Carry-Forward).
"""

import io
import json
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from pyproj import Transformer
from google.cloud import storage
from google.auth import default

from goes_local_fetch import fetch_goes_local_stack, upload_stack_as_geotiff, GDAL_IO_LOCK

# ==========================================
# Bootleg Fire Configuration
# ==========================================
# 72-Hour Focused Progression Window
EVENT_START_UTC = pd.Timestamp("2021-07-06 12:00:00", tz="UTC")
EVENT_END_UTC   = pd.Timestamp("2021-07-18 12:00:00", tz="UTC")

GOES_EXTENT_M  = 7680     # Yields 4x4 @ 1920m
VIIRS_EXTENT_M = 7680     # Yields 20x20 @ 384m
VIIRS_SCALE_M  = 384.0
VIIRS_PIXELS   = 20

GOES_SATELLITES = ("G16", "G17")

# FIX 1: Isolated path prefix to prevent cache-collisions with old static runs
OUTPUT_PREFIX = "case_studies/bootleg_2021_dynamic"
BUCKET_NAME = "wildfire-irp-data-th25"

FALLBACK_LON = -121.421
FALLBACK_LAT = 42.616

credentials, project = default()
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket(BUCKET_NAME)


def get_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def load_viirs_bootleg(viirs_csv_path):
    if viirs_csv_path.startswith("gs://"):
        blob_path = viirs_csv_path.replace(f"gs://{BUCKET_NAME}/", "")
        df = pd.read_csv(io.BytesIO(bucket.blob(blob_path).download_as_bytes()))
    else:
        df = pd.read_csv(viirs_csv_path)
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["acq_date", "acq_time"])
    df["acq_time_str"] = df["acq_time"].astype(int).apply(lambda t: f"{t:04d}")
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time_str"],
        format="%Y-%m-%d %H%M", utc=True
    )
    return df


def get_active_fire_center(ts, viirs_df, acr273_gdf, last_center=None):
    """
    Calculates active thermal centroid near ACR273 boundary.
    FIX 2: Uses last_center carry-forward to avoid jumping back to ignition point when VIIRS has no pass.
    """
    t_start = ts - pd.Timedelta(minutes=60)
    t_end   = ts + pd.Timedelta(minutes=60)
    
    sub_df = viirs_df[(viirs_df["acq_datetime"] >= t_start) & (viirs_df["acq_datetime"] <= t_end)]
    
    if sub_df.empty:
        # Carry forward previous hour's active center if available
        return last_center if last_center is not None else (FALLBACK_LON, FALLBACK_LAT)

    gdf = gpd.GeoDataFrame(
        sub_df, geometry=gpd.points_from_xy(sub_df.longitude, sub_df.latitude), crs="EPSG:4326"
    )
    
    boundary_line = acr273_gdf.geometry.boundary.union_all()
    near_boundary = gdf[gdf.geometry.distance(boundary_line) < 0.03]
    
    if not near_boundary.empty:
        return float(near_boundary.geometry.x.mean()), float(near_boundary.geometry.y.mean())
    
    return float(gdf.geometry.x.mean()), float(gdf.geometry.y.mean())


# ==========================================
# Step 1: Hourly GOES Loop
# ==========================================
def run_goes_hourly(viirs_csv_path, acr273_path):
    acr273_gdf = gpd.read_file(acr273_path).to_crs("EPSG:4326")
    viirs_df = load_viirs_bootleg(viirs_csv_path)

    hours = pd.date_range(EVENT_START_UTC, EVENT_END_UTC, freq="1h", tz="UTC")
    print(f"[GOES] Processing timeline: {len(hours)} hours ({EVENT_START_UTC} -> {EVENT_END_UTC})")

    qa_rows = []
    current_center = (FALLBACK_LON, FALLBACK_LAT)

    for ts in hours:
        sample_id = ts.strftime("%Y%m%dT%H%M")
        goes_prefix = f"{OUTPUT_PREFIX}/goes_hourly/GOES_{sample_id}"
        blob = bucket.blob(f"{goes_prefix}.tif")

        # Update dynamic center with carry-forward memory
        current_center = get_active_fire_center(ts, viirs_df, acr273_gdf, last_center=current_center)
        center_lon, center_lat = current_center

        if blob.exists():
            # FIX 3: Record center_lon/center_lat even on cached hits
            qa_rows.append({
                "timestamp": ts.isoformat(), 
                "status": "cached",
                "center_lon": center_lon, 
                "center_lat": center_lat
            })
            continue

        utm_crs = get_utm_epsg(center_lon, center_lat)

        with GDAL_IO_LOCK:
            to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            cx, cy = to_utm.transform(center_lon, center_lat)
        half = GOES_EXTENT_M / 2.0
        xmin, xmax, ymin, ymax = cx - half, cx + half, cy - half, cy + half

        stack, matched = fetch_goes_local_stack(
            center_lon, center_lat, ts.to_pydatetime().replace(tzinfo=None), 
            GOES_EXTENT_M, utm_crs, satellites=GOES_SATELLITES
        )

        if not matched:
            qa_rows.append({
                "timestamp": ts.isoformat(), 
                "status": "no_scan",
                "center_lon": center_lon, 
                "center_lat": center_lat
            })
            continue

        if np.all(stack == 0):
            qa_rows.append({
                "timestamp": ts.isoformat(), 
                "status": "all_zero",
                "center_lon": center_lon, 
                "center_lat": center_lat
            })
            continue

        upload_stack_as_geotiff(stack, xmin, ymin, xmax, ymax, utm_crs, bucket, f"{goes_prefix}.tif")
        zero_frac = float(np.mean(stack == 0.0))
        print(f"    [SAVED] {sample_id} at center ({center_lon:.3f}, {center_lat:.3f})")
        
        qa_rows.append({
            "timestamp": ts.isoformat(), 
            "status": "ok", 
            "zero_fraction": zero_frac,
            "center_lon": center_lon, 
            "center_lat": center_lat
        })

    qa_df = pd.DataFrame(qa_rows)
    qa_path = f"{OUTPUT_PREFIX}/goes_hourly_qa.csv"
    bucket.blob(qa_path).upload_from_string(qa_df.to_csv(index=False), content_type="text/csv")
    print(f"[GOES] Done. QA written to gs://{BUCKET_NAME}/{qa_path}")
    return qa_df


# ==========================================
# Step 2: VIIRS Overpasses
# ==========================================
def rasterize_viirs_overpass(group_df, overpass_ts, center_lon, center_lat):
    utm_crs = get_utm_epsg(center_lon, center_lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = VIIRS_EXTENT_M / 2.0
    xmin, xmax, ymin, ymax = cx - half, cx + half, cy - half, cy + half

    arr = np.zeros((VIIRS_PIXELS, VIIRS_PIXELS), dtype=np.float32)
    gdf = gpd.GeoDataFrame(
        group_df, geometry=gpd.points_from_xy(group_df.longitude, group_df.latitude), crs="EPSG:4326"
    ).to_crs(utm_crs)
    gdf = gdf.sort_values(by="frp", ascending=True)

    for _, row in gdf.iterrows():
        px, py = row.geometry.x, row.geometry.y
        col = int(np.floor((px - xmin) / VIIRS_SCALE_M))
        r = int(np.floor((ymax - py) / VIIRS_SCALE_M))
        if 0 <= col < VIIRS_PIXELS and 0 <= r < VIIRS_PIXELS:
            arr[r, col] = row["frp"]

    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, VIIRS_PIXELS, VIIRS_PIXELS)
    mem_file = io.BytesIO()
    with GDAL_IO_LOCK, rasterio.open(
        mem_file, "w", driver="GTiff", height=VIIRS_PIXELS, width=VIIRS_PIXELS,
        count=1, dtype="float32", crs=utm_crs, transform=transform,
    ) as dst:
        dst.write(arr, 1)
    mem_file.seek(0)

    sample_id = overpass_ts.strftime("%Y%m%dT%H%M")
    target_path = f"{OUTPUT_PREFIX}/viirs_overpasses/VIIRS_{sample_id}.tif"
    bucket.blob(target_path).upload_from_file(mem_file, content_type="image/tiff")
    return target_path, int(len(group_df)), float(group_df["frp"].max())


def run_viirs_overpasses(df_viirs, acr273_path, window_min=30):
    acr273_gdf = gpd.read_file(acr273_path).to_crs("EPSG:4326")
    df_viirs = df_viirs[
        (df_viirs["acq_datetime"] >= EVENT_START_UTC) & 
        (df_viirs["acq_datetime"] <= EVENT_END_UTC)
    ].copy()
    
    df_viirs["overpass_window"] = df_viirs["acq_datetime"].dt.floor(f"{window_min}min")

    manifest_rows = []
    current_center = (FALLBACK_LON, FALLBACK_LAT)

    for overpass_ts, group in df_viirs.groupby("overpass_window"):
        current_center = get_active_fire_center(overpass_ts, df_viirs, acr273_gdf, last_center=current_center)
        center_lon, center_lat = current_center
        
        target_path, n_det, max_frp = rasterize_viirs_overpass(group, overpass_ts, center_lon, center_lat)
        nearest_hour = overpass_ts.round("1h")
        
        manifest_rows.append({
            "overpass_timestamp": overpass_ts.isoformat(),
            "nearest_goes_hour": nearest_hour.isoformat(),
            "n_detections": n_det,
            "max_frp": max_frp,
            "center_lon": center_lon,
            "center_lat": center_lat,
            "viirs_target_path": target_path,
            "goes_path": f"{OUTPUT_PREFIX}/goes_hourly/GOES_{nearest_hour.strftime('%Y%m%dT%H%M')}.tif",
        })
        print(f"    [SAVED] VIIRS overpass {overpass_ts} (n={n_det}) -> pairs with GOES {nearest_hour}")

    manifest_path = f"{OUTPUT_PREFIX}/viirs_goes_pairs_manifest.json"
    bucket.blob(manifest_path).upload_from_string(json.dumps(manifest_rows, indent=2), content_type="application/json")
    print(f"[VIIRS] Complete. Manifest -> gs://{BUCKET_NAME}/{manifest_path}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    viirs_gcs_uri = f"gs://{BUCKET_NAME}/case_studies/bootleg_2021/raw_viirs/bootleg_viirs_archive.csv"
    acr273_geojson = "acr273_boundary.geojson"

    if mode in ("all", "goes"):
        run_goes_hourly(viirs_gcs_uri, acr273_geojson)

    if mode in ("all", "viirs"):
        df_viirs = load_viirs_bootleg(viirs_gcs_uri)
        run_viirs_overpasses(df_viirs, acr273_geojson)