"""
DEM + landcover covariate fetching via Microsoft Planetary Computer.

Replaces the GEE COVARIATE_STACK + ee.batch.Export.image.toCloudStorage path.
Confirmed asset keys (2026-09): cop-dem-glo-30 -> "data", esa-worldcover -> "map".
Both collections are hosted as Azure-native COGs, so reads from an Azure VM
are same-cloud.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer
import planetary_computer as pc
from pystac_client import Client as STACClient

_catalog = STACClient.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

DEM_COLLECTION = "cop-dem-glo-30"
DEM_ASSET_KEY = "data"

LC_COLLECTION = "esa-worldcover"
LC_ASSET_KEY = "map"


def _bbox_wgs84_from_utm_window(center_lon, center_lat, buffer_deg=0.05):
    # Cheap WGS84 bbox for the STAC spatial search - doesn't need to be exact,
    # just needs to comfortably contain the UTM window we'll warp into.
    return [
        center_lon - buffer_deg, center_lat - buffer_deg,
        center_lon + buffer_deg, center_lat + buffer_deg,
    ]


def _read_and_warp(collection, asset_key, bbox_wgs84, dst_crs, dst_transform,
                    dst_shape, resampling):
    search = _catalog.search(collections=[collection], bbox=bbox_wgs84, limit=20)
    items = list(search.items())
    if not items:
        return np.zeros(dst_shape, dtype=np.float32)

    dst = np.zeros(dst_shape, dtype=np.float32)
    for item in items:
        href = item.assets[asset_key].href
        with rasterio.open(href) as src:
            band = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=band,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=0.0
            )
            mask = (band != 0) & (dst == 0)
            dst[mask] = band[mask]

    return dst


def fetch_covariates_window(center_lon, center_lat, extent_m, utm_crs, scale_m=30):
    """
    Returns (stack, bounds) where stack is a (2, H, W) float32 array
    [elevation, landcover] on a grid of the given UTM extent/scale, and
    bounds is (xmin, ymin, xmax, ymax) in utm_crs for upload.
    """
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = extent_m / 2.0
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    width = height = max(1, int(round(extent_m / scale_m)))
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    bbox_wgs84 = _bbox_wgs84_from_utm_window(center_lon, center_lat)

    elevation = _read_and_warp(
        DEM_COLLECTION, DEM_ASSET_KEY, bbox_wgs84, utm_crs, dst_transform,
        (height, width), Resampling.bilinear
    )
    landcover = _read_and_warp(
        LC_COLLECTION, LC_ASSET_KEY, bbox_wgs84, utm_crs, dst_transform,
        (height, width), Resampling.nearest  # categorical - never interpolate
    )

    stack = np.stack([elevation, landcover], axis=0).astype(np.float32)
    return stack, (xmin, ymin, xmax, ymax)
