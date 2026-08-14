"""GOES-16/17 GCS Data Fetching and Spatial Reprojection Pipeline.

This module provides thread-safe utilities to search, download, cache, and reproject
multichannel full-disk imagery from the GOES-R series (GOES-16/17) satellite.
It converts raw ABI-L2-MCMIPF NetCDF files into spatial target tensors (CMI_C07, CMI_C14,
and Thermal Fire Index) reprojected to local coordinate reference systems (e.g. UTM).
"""

import os
import re
import threading
import datetime
import tempfile

import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
from google.cloud import storage

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
GOES_TEMPORAL_GUARD_MIN = 60
GOES_SCAN_SEARCH_HOURS = 3
GOES_CACHE_DIR = os.path.join(tempfile.gettempdir(), "goes_scan_cache")
GOES_CACHE_MAX_BYTES = 20 * 1024 ** 3  # 20 GB cap; evict oldest-accessed scans beyond this
os.makedirs(GOES_CACHE_DIR, exist_ok=True)

_storage_client = storage.Client()

# Thread-safe cache: maps (satellite, scan_key) -> local file path.
# Prevents two threads racing to download the same full-disk scan.
_scan_cache = {}
_scan_cache_lock = threading.Lock()
GDAL_IO_LOCK = threading.Lock()
_netcdf_io_lock = GDAL_IO_LOCK
_in_use_paths = {}

# One lock per scan_key, so concurrent threads wanting the SAME scan
# serialise on download+rename, while different scans still proceed in parallel
_download_locks = {}
_download_locks_guard = threading.Lock()


def _get_download_lock(scan_key):
    with _download_locks_guard:
        if scan_key not in _download_locks:
            _download_locks[scan_key] = threading.Lock()
        return _download_locks[scan_key]


def _mark_in_use(path):
    with _scan_cache_lock:
        _in_use_paths[path] = _in_use_paths.get(path, 0) + 1
 
 
def _unmark_in_use(path):
    with _scan_cache_lock:
        remaining = _in_use_paths.get(path, 0) - 1
        if remaining <= 0:
            _in_use_paths.pop(path, None)
        else:
            _in_use_paths[path] = remaining

def _evict_lru_if_needed(incoming_path, cache_dir=GOES_CACHE_DIR):
    """Keep active cache directory under GOES_CACHE_MAX_BYTES by deleting oldest-accessed
    .nc files first. Runs under _scan_cache_lock, so it never races a download.
    Skips the file currently being written (incoming_path) and any in-flight
    .part files."""
    try:
        entries = []
        total = 0
        # MODIFIED: Scan the resolved active directory
        for fname in os.listdir(cache_dir):
            if not fname.endswith(".nc"):
                continue
            fpath = os.path.join(cache_dir, fname)
            if fpath == incoming_path:
                continue
            if _in_use_paths.get(fpath, 0) > 0:
                continue  # currently being read by another thread, don't evict
            try:
                st = os.stat(fpath)
            except FileNotFoundError:
                continue
            total += st.st_size
            entries.append((st.st_atime, st.st_size, fpath))
 
        if total <= GOES_CACHE_MAX_BYTES:
            return
 
        # Oldest-accessed first
        entries.sort(key=lambda e: e[0])
        for atime, size, fpath in entries:
            if total <= GOES_CACHE_MAX_BYTES:
                break
            try:
                os.remove(fpath)
                total -= size
                # Drop it from the in-memory cache map too so callers re-download
                # rather than returning a path to a now-deleted file
                stale_keys = [k for k, v in _scan_cache.items() if v == fpath]
                for k in stale_keys:
                    _scan_cache.pop(k, None)
                    with _download_locks_guard:
                        _download_locks.pop(k, None)
            except FileNotFoundError:
                continue
    except Exception as e:
        print(f"  [WARN] GOES cache eviction failed: {e}")


def _list_candidate_blobs(target_utc_dt, satellite="G16"):
    """List candidate MCMIPF blobs across a window of hours around the target time."""
    goes_bucket = _storage_client.bucket(f"gcp-public-data-goes-{satellite[1:]}")
    start_search = target_utc_dt - datetime.timedelta(hours=1)
    candidates = []

    for h_offset in range(GOES_SCAN_SEARCH_HOURS):
        current_check_dt = start_search + datetime.timedelta(hours=h_offset)
        year = current_check_dt.strftime("%Y")
        doy = current_check_dt.strftime("%j")
        hour = current_check_dt.strftime("%H")
        hour_prefix = f"ABI-L2-MCMIPF/{year}/{doy}/{hour}/"

        for blob in goes_bucket.list_blobs(prefix=hour_prefix):
            match = re.search(r"_s(\d+)_", blob.name)
            if not match:
                continue
            try:
                scan_dt = datetime.datetime.strptime(match.group(1)[:13], "%Y%j%H%M%S")
            except ValueError:
                continue
            candidates.append((scan_dt, blob))

    return candidates


def find_closest_goes_blob(target_utc_dt, satellite="G16"):
    """Return (scan_dt, blob, time_diff) for the closest scan, or (None, None, None)."""
    candidates = _list_candidate_blobs(target_utc_dt, satellite=satellite)
    if not candidates:
        return None, None, None

    scan_dt, blob = min(candidates, key=lambda c: abs(c[0] - target_utc_dt))
    time_diff = abs(scan_dt - target_utc_dt)
    return scan_dt, blob, time_diff


def _reproject_bands(nc_path, xmin, ymin, xmax, ymax, width, height, dst_crs,
                     target_shape=None, target_transform=None):
    with _netcdf_io_lock, xr.open_dataset(nc_path, mask_and_scale=True) as ds:
        # Sort coordinates to match standard grid indexing
        ds = ds.sortby("y").sortby("x")

        proj_var = ds.goes_imager_projection
        goes_crs = CRS.from_cf(proj_var.attrs)
        sat_height = proj_var.attrs.get("perspective_point_height")
        if sat_height is None:
            raise ValueError("perspective_point_height missing from NetCDF metadata.")

        # Scaled GOES fixed-grid coordinates to meters
        ds = ds.assign_coords(x=ds.x * sat_height, y=ds.y * sat_height)

        c07 = ds["CMI_C07"].rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(goes_crs)
        c14 = ds["CMI_C14"].rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs(goes_crs)

        # Handle target framing parameters with fallbacks
        final_shape = target_shape if target_shape is not None else (height, width)
        final_transform = target_transform if target_transform is not None else from_bounds(xmin, ymin, xmax, ymax, width, height)

        c07_utm = c07.rio.reproject(
            dst_crs=dst_crs, 
            shape=final_shape,
            transform=final_transform,
            resampling=rasterio.enums.Resampling.nearest
        )
        c14_utm = c14.rio.reproject(
            dst_crs=dst_crs, 
            shape=final_shape,
            transform=final_transform,
            resampling=rasterio.enums.Resampling.nearest
        )

        stack = np.stack(
            [c07_utm.values, c14_utm.values, c07_utm.values - c14_utm.values],
            axis=0,
        ).astype(np.float32)
        
        stack[~np.isfinite(stack)] = 0.0
        return stack


def fetch_goes_local_stack(center_lon, center_lat, target_dt, extent_m, utm_crs,
                            satellites=("G16",), target_shape=None, target_transform=None,
                            local_cache_dir=None):
    with GDAL_IO_LOCK:
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        cx, cy = to_utm.transform(center_lon, center_lat)
    half = extent_m / 2.0
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half
 
    # Use target_shape dimensions if explicitly provided
    if target_shape is not None:
        height, width = target_shape
    else:
        scale_m = 2000
        width = height = max(1, int(round(extent_m / scale_m)))
 
    best = None  # (time_diff, scan_dt, blob, satellite)
    for sat in satellites:
        scan_dt, blob, time_diff = find_closest_goes_blob(target_dt, satellite=sat)
        if blob is None:
            continue
        if best is None or time_diff < best[0]:
            best = (time_diff, scan_dt, blob, sat)
 
    if best is None or best[0] > datetime.timedelta(minutes=GOES_TEMPORAL_GUARD_MIN):
        return np.zeros((3, height, width), dtype=np.float32), False
 
    time_diff, scan_dt, blob, sat = best
    
    # Pass local_cache_dir down to cache downloader
    nc_path = _download_scan_cached(blob, scan_dt, sat, local_cache_dir=local_cache_dir)
 
    try:
        # Pass along the structural constraints
        stack = _reproject_bands(
            nc_path, xmin, ymin, xmax, ymax, width, height, utm_crs,
            target_shape=target_shape, target_transform=target_transform
        )
    finally:
        _unmark_in_use(nc_path)
    return stack, True


def upload_stack_as_geotiff(stack, xmin, ymin, xmax, ymax, crs, bucket, dest_blob_path,
                             band_names=("CMI_C07", "CMI_C14", "GOES_TFI")):
    """Write a (bands, H, W) stack directly to GCS as a GeoTIFF - no GEE export task."""
    import io as _io

    bands, height, width = stack.shape
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    mem_file = _io.BytesIO()
    with GDAL_IO_LOCK, rasterio.open(
        mem_file, "w", driver="GTiff",
        height=height, width=width, count=bands,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(stack[i], i + 1)
        if band_names and len(band_names) == bands:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)

    mem_file.seek(0)
    bucket.blob(dest_blob_path).upload_from_file(mem_file, content_type="image/tiff")


def fetch_and_upload_goes_local(center_lon, center_lat, target_dt, extent_m, utm_crs,
                                 bucket, dest_blob_path, satellites=("G16",)):
    """
    Convenience wrapper: fetch the local GOES stack and upload it to GCS in one call.
    Returns True if a real (non-fallback) scan was used, False if it was dropped.
    """
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    with GDAL_IO_LOCK:
        cx, cy = to_utm.transform(center_lon, center_lat)
    half = extent_m / 2.0
    xmin, xmax, ymin, ymax = cx - half, cx + half, cy - half, cy + half

    stack, matched = fetch_goes_local_stack(
        center_lon, center_lat, target_dt, extent_m, utm_crs, satellites=satellites
    )
    
    # Only upload if a valid scan was actually matched. 
    # If matched is False, discard the all-zero matrix.
    if matched:
        upload_stack_as_geotiff(stack, xmin, ymin, xmax, ymax, utm_crs, bucket, dest_blob_path)
    else:
        print(f"  [DROP] No valid GOES scan within temporal guard for {dest_blob_path}. Skipping upload.")

    return matched


def _download_scan_cached(blob, scan_dt, satellite, local_cache_dir=None):
    """Returns a local path with its in-use refcount already incremented.
    Callers MUST call _unmark_in_use(path) when done reading it."""
    scan_key = (satellite, scan_dt.strftime("%Y%m%dT%H%M%S"))
    
    cache_dir = local_cache_dir if local_cache_dir is not None else GOES_CACHE_DIR
 
    with _scan_cache_lock:
        cached_path = _scan_cache.get(scan_key)
        if cached_path and os.path.exists(cached_path) and _is_valid_nc(cached_path):
            os.utime(cached_path, None)  # mark as recently used for LRU eviction
            _in_use_paths[cached_path] = _in_use_paths.get(cached_path, 0) + 1
            return cached_path
        
        local_path = os.path.join(cache_dir, f"{satellite}_{scan_key[1]}.nc")
        _scan_cache[scan_key] = local_path
        _evict_lru_if_needed(local_path, cache_dir=cache_dir)
        _in_use_paths[local_path] = _in_use_paths.get(local_path, 0) + 1

    download_lock = _get_download_lock(scan_key)
    with download_lock:
        if not os.path.exists(local_path) or not _is_valid_nc(local_path):
            tmp_path = local_path + f".{os.getpid()}.{threading.get_ident()}.part"
            blob.download_to_filename(tmp_path)
            os.replace(tmp_path, local_path)
 
    return local_path


def _is_valid_nc(path):
    try:
        with _netcdf_io_lock, xr.open_dataset(path) as ds:
            _ = ds.goes_imager_projection
        return True
    except Exception:
        return False