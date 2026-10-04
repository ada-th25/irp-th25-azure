import numpy as np
import rasterio
from azure.core.exceptions import ResourceNotFoundError

def compute_validity(container_client, manifest_entry: dict) -> dict:
    """
    Single source of truth for sample validity. Mirrors the checks
    currently in WildfireDataset._filter_complete exactly, so that
    split assignment and actual training data never diverge.
    """
    paths = manifest_entry.get("paths", {})
    target_path = paths.get("target", "")
    goes_path = paths.get("goes_input", "")
    cov_path = paths.get("covariates", "")

    try:
        goes_raw = container_client.get_blob_client(goes_path).download_blob().readall()
    except ResourceNotFoundError:
        return {"is_valid": False, "invalid_reason": "missing_goes"}
    except Exception:
        return {"is_valid": False, "invalid_reason": "goes_read_error"}

    try:
        target_raw = container_client.get_blob_client(target_path).download_blob().readall()
    except ResourceNotFoundError:
        return {"is_valid": False, "invalid_reason": "missing_target_or_cov"}
    except Exception:
        return {"is_valid": False, "invalid_reason": "target_read_error"}

    if not container_client.get_blob_client(cov_path).exists():
        return {"is_valid": False, "invalid_reason": "missing_target_or_cov"}

    try:
        with rasterio.MemoryFile(goes_raw) as mf, mf.open() as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = 0.0
        data[~np.isfinite(data)] = 0.0
        if np.all(data == 0):
            return {"is_valid": False, "invalid_reason": "goes_all_zero"}
    except Exception:
        return {"is_valid": False, "invalid_reason": "goes_read_error"}

    try:
        with rasterio.MemoryFile(target_raw) as mf, mf.open() as src:
            target_data = src.read().astype(np.float32)
        if np.all(target_data == 0):
            return {"is_valid": False, "invalid_reason": "target_all_zero"}
    except Exception:
        return {"is_valid": False, "invalid_reason": "target_read_error"}

    return {"is_valid": True, "invalid_reason": None}
