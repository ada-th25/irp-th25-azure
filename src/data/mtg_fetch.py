import os
import shutil
import time
import threading
import datetime
import tempfile
import io as _io

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import from_bounds
from pyproj import Transformer
import zipfile

import eumdac

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MTG_TEMPORAL_GUARD_MIN = 15          # FCI repeat cycle is 10 min; allow one missed cycle
MTG_HRFI_PRODUCT_TYPE = "MTIFCI1CRRADHRFI"  # confirmed via `eumdac describe -c EO:EUM:DAT:0665`
MTG_TAILOR_PRODUCT = "FCIL1HRFI"
MTG_HRFI_COLLECTION = "EO:EUM:DAT:0665"        # HRFI Collection

MTG_BANDS_1KM = [
    "ir_38_hr_effective_radiance", 
    "ir_105_hr_effective_radiance"
]
MTG_BANDS_HALFKM = [
    "vis_06_hr_effective_radiance", 
    "nir_22_hr_effective_radiance"
]
MTG_BANDS = MTG_BANDS_HALFKM + MTG_BANDS_1KM
MTG_MAX_PRODUCT_CANDIDATES = 3  # retry against next-closest product on chain failure
MTG_CACHE_DIR = os.path.join(tempfile.gettempdir(), "mtg_scan_cache")
MTG_CUSTOMISATION_TIMEOUT_S = 600
MTG_CUSTOMISATION_POLL_S = 5
os.makedirs(MTG_CACHE_DIR, exist_ok=True)

_token_lock = threading.Lock()
_token = None
_datastore = None
_datatailor = None

GDAL_IO_LOCK = threading.Lock()
_netcdf_io_lock = GDAL_IO_LOCK

DATA_TAILOR_SEMAPHORE = threading.Semaphore(2)

def _get_clients():
    """Lazily create a single shared token/datastore/datatailor per process."""
    global _token, _datastore, _datatailor
    with _token_lock:
        token_expired = True
        if _token is not None:
            exp = _token.expiration
            now = datetime.datetime.now(datetime.timezone.utc) if exp.tzinfo is not None else datetime.datetime.utcnow()
            token_expired = exp < now

        if token_expired:
            key = os.environ["EUMETSAT_CONSUMER_KEY"]
            secret = os.environ["EUMETSAT_CONSUMER_SECRET"]
            _token = eumdac.AccessToken((key, secret))
            _datastore = eumdac.DataStore(_token)
            _datatailor = eumdac.DataTailor(_token)
        return _datastore, _datatailor


def find_closest_mtg_product(target_utc_dt, collection_id=MTG_HRFI_COLLECTION):
    """Search the Data Store for the FCI repeat cycle closest to target_utc_dt."""
    datastore, _ = _get_clients()
    collection = datastore.get_collection(collection_id)

    window_start = target_utc_dt - datetime.timedelta(hours=1)
    window_end = target_utc_dt + datetime.timedelta(hours=1)

    products = list(collection.search(dtstart=window_start, dtend=window_end, type=MTG_HRFI_PRODUCT_TYPE))
    if not products:
        return None, None, None

    def _sensing_start(p):
        return p.sensing_start

    best = min(products, key=lambda p: abs(_sensing_start(p) - target_utc_dt))
    time_diff = abs(_sensing_start(best) - target_utc_dt)
    return best, _sensing_start(best), time_diff


def _bbox_from_utm_window(xmin, ymin, xmax, ymax, utm_crs, pad_deg=0.3):
    """Convert a UTM window to a lat/lon bbox for the Data Tailor ROI filter."""
    to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        lon, lat = to_ll.transform(x, y)
        lons.append(lon)
        lats.append(lat)
    return (min(lons) - pad_deg, min(lats) - pad_deg, max(lons) + pad_deg, max(lats) + pad_deg)

def _run_single_chain(datatailor, product, bands, bbox, local_cache_dir, suffix):
    lon_min, lat_min, lon_max, lat_max = bbox
    chain = eumdac.tailor_models.Chain(
        product=MTG_TAILOR_PRODUCT,
        format="geotiff",
        projection="native",
        filter={"bands": bands},
        roi={"NSWE": [lat_max, lat_min, lon_min, lon_max]},
    )

    customisation = datatailor.new_customisation(product, chain=chain)

    try:
        start_time = time.time()
        while True:
            # Hard limit: kill job if it takes longer than MTG_CUSTOMISATION_TIMEOUT_S
            if time.time() - start_time > MTG_CUSTOMISATION_TIMEOUT_S:
                try:
                    customisation.kill()
                except Exception:
                    pass
                raise TimeoutError(f"Data Tailor customisation timed out after {MTG_CUSTOMISATION_TIMEOUT_S}s for {getattr(product, '_id', str(product))}")

            # Safely query status without allowing network socket reads to freeze the thread
            try:
                status = customisation.status
            except Exception:
                time.sleep(MTG_CUSTOMISATION_POLL_S)
                continue

            if status == "DONE":
                break
            elif status not in ("QUEUED", "RUNNING"):
                raise RuntimeError(f"Data Tailor customisation entered unexpected state '{status}' for {getattr(product, '_id', str(product))}")

            time.sleep(MTG_CUSTOMISATION_POLL_S)

        tif_outputs = [f for f in customisation.outputs if f.endswith((".tif", ".tiff"))]
        if not tif_outputs:
            raise RuntimeError(f"Data Tailor customisation produced no GeoTIFF outputs for {getattr(product, '_id', str(product))}")

        os.makedirs(local_cache_dir, exist_ok=True)
        local_path = os.path.join(local_cache_dir, f"mtg_{customisation._id}_{suffix}.tif")
        
        with customisation.stream_output(tif_outputs[0]) as stream, open(local_path, "wb") as f:
            shutil.copyfileobj(stream, f)

        return local_path

    finally:
        try:
            customisation.delete()
        except Exception:
            pass
# def _run_single_chain(datatailor, product, bands, bbox, local_cache_dir, suffix):
#     lon_min, lat_min, lon_max, lat_max = bbox
#     chain = eumdac.tailor_models.Chain(
#         product=MTG_TAILOR_PRODUCT,
#         format="geotiff",
#         projection="native",  # Explicitly specify 'native' to prevent EPCT GIS NetCDF/HDF crashes
#         filter={"bands": bands},
#         roi={
#             "NSWE": [lat_max, lat_min, lon_min, lon_max]
#         },
#     )

#     customisation = datatailor.new_customisation(product, chain=chain)

#     try:
#         start = time.time()
#         while customisation.status in ("QUEUED", "RUNNING"):
#             if time.time() - start > MTG_CUSTOMISATION_TIMEOUT_S:
#                 try:
#                     customisation.kill()
#                 except Exception:
#                     pass
#                 raise TimeoutError(f"Data Tailor customisation timed out for product {getattr(product, '_id', str(product))}")
#             time.sleep(MTG_CUSTOMISATION_POLL_S)

#         if customisation.status != "DONE":
#             raise RuntimeError(f"Data Tailor customisation failed ({customisation.status}) for {getattr(product, '_id', str(product))}: "
#                                f"{getattr(customisation, 'logfile', '')}")

#         tif_outputs = [f for f in customisation.outputs if f.endswith((".tif", ".tiff"))]
#         if not tif_outputs:
#             raise RuntimeError(f"Data Tailor customisation produced no GeoTIFF outputs for {getattr(product, '_id', str(product))}")

#         os.makedirs(local_cache_dir, exist_ok=True)
#         local_path = os.path.join(local_cache_dir, f"mtg_{customisation._id}_{suffix}.tif")
#         with customisation.stream_output(tif_outputs[0]) as stream, open(local_path, "wb") as f:
#             shutil.copyfileobj(stream, f)

#         return local_path
#     finally:
#         try:
#             customisation.delete()
#         except Exception:
#             pass


def _run_data_tailor(product, bbox, local_cache_dir):
    """Submits two separate Data Tailor jobs (one for 0.5km bands, one for 1km bands)."""
    _, datatailor = _get_clients()

    with DATA_TAILOR_SEMAPHORE:
        tif_05km = _run_single_chain(
            datatailor, product, MTG_BANDS_HALFKM, bbox, local_cache_dir, "05km"
        )
        tif_1km = _run_single_chain(
            datatailor, product, MTG_BANDS_1KM, bbox, local_cache_dir, "1km"
        )

    return tif_05km, tif_1km


def _reproject_mtg_tif(tif_path_05km, tif_path_1km, xmin, ymin, xmax, ymax, 
                       dst_crs, scale_m=1000.0):
    """Reprojects MTG bands onto an independent coarse grid (default 1000m / 1km)."""
    width = max(1, int(round((xmax - xmin) / scale_m)))
    height = max(1, int(round((ymax - ymin) / scale_m)))
    
    coarse_shape = (height, width)
    coarse_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    with GDAL_IO_LOCK:
        da_05 = rioxarray.open_rasterio(tif_path_05km, masked=True)
        da_10 = rioxarray.open_rasterio(tif_path_1km, masked=True)

        da_05_utm = da_05.rio.reproject(
            dst_crs=dst_crs,
            shape=coarse_shape,
            transform=coarse_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        
        da_10_utm = da_10.rio.reproject(
            dst_crs=dst_crs,
            shape=coarse_shape,
            transform=coarse_transform,
            resampling=rasterio.enums.Resampling.nearest,
        )

        stack_05 = da_05_utm.values.astype(np.float32)
        stack_10 = da_10_utm.values.astype(np.float32)

        stack = np.concatenate([stack_05, stack_10], axis=0)
        stack[~np.isfinite(stack)] = 0.0
        return stack


def find_mtg_product_candidates(target_utc_dt, collection_id=MTG_HRFI_COLLECTION, limit=3):
    """Search the Data Store for candidate FCI repeat cycles near target_utc_dt."""
    datastore, _ = _get_clients()
    collection = datastore.get_collection(collection_id)

    window_start = target_utc_dt - datetime.timedelta(minutes=MTG_TEMPORAL_GUARD_MIN)
    window_end = target_utc_dt + datetime.timedelta(minutes=MTG_TEMPORAL_GUARD_MIN)

    products = list(collection.search(dtstart=window_start, dtend=window_end, type=MTG_HRFI_PRODUCT_TYPE))
    if not products:
        return []

    def _sensing_start(p):
        return p.sensing_start

    sorted_candidates = sorted(products, key=lambda p: abs(_sensing_start(p) - target_utc_dt))
    return sorted_candidates[:limit]


def fetch_mtg_local_stack(center_lon, center_lat, target_dt, extent_m=8000.0, utm_crs="EPSG:32631",
                           local_cache_dir=None):
    """Fetches MTG data and resamples it to a coarse 1km grid (8x8 pixels for 8000m extent)."""
    with GDAL_IO_LOCK:
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        cx, cy = to_utm.transform(center_lon, center_lat)
        
    half = extent_m / 2.0
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    candidates = find_mtg_product_candidates(target_dt, limit=MTG_MAX_PRODUCT_CANDIDATES)

    if not candidates:
        coarse_dim = max(1, int(round(extent_m / 1000.0)))
        return np.zeros((len(MTG_BANDS), coarse_dim, coarse_dim), dtype=np.float32), False, "no_product_in_search_window"

    bbox = _bbox_from_utm_window(xmin, ymin, xmax, ymax, utm_crs)
    cache_dir = local_cache_dir if local_cache_dir is not None else MTG_CACHE_DIR

    last_error = None
    for product in candidates:
        try:
            tif_05km, tif_1km = _run_data_tailor(product, bbox, cache_dir)
            
            stack = _reproject_mtg_tif(
                tif_05km, tif_1km, xmin, ymin, xmax, ymax, 
                dst_crs=utm_crs, scale_m=1000.0
            )
            return stack, True, "ok"
        except Exception as e:
            last_error = e
            continue

    coarse_dim = max(1, int(round(extent_m / 1000.0)))
    return np.zeros((len(MTG_BANDS), coarse_dim, coarse_dim), dtype=np.float32), False, f"error: {last_error}"
