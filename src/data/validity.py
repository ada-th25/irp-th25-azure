import numpy as np
import rasterio

def compute_validity(bucket, manifest_entry: dict) -> dict:
    """
    Single source of truth for sample validity. Mirrors the checks
    currently in WildfireDataset._filter_complete exactly, so that
    split assignment and actual training data never diverge.
    """
    paths = manifest_entry.get("paths", {})
    target_path = paths.get("target", "")
    goes_path = paths.get("goes_input", "")
    cov_path = paths.get("covariates", "")

    if not (bucket.blob(target_path).exists() and bucket.blob(cov_path).exists()):
        return {"is_valid": False, "invalid_reason": "missing_target_or_cov"}

    goes_blob = bucket.blob(goes_path)
    if not goes_blob.exists():
        return {"is_valid": False, "invalid_reason": "missing_goes"}

    try:
        raw = goes_blob.download_as_bytes()
        with rasterio.MemoryFile(raw) as mf:
            with mf.open() as src:
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
        raw = bucket.blob(target_path).download_as_bytes()
        with rasterio.MemoryFile(raw) as mf:
            with mf.open() as src:
                target_data = src.read().astype(np.float32)
        if np.all(target_data == 0):
            return {"is_valid": False, "invalid_reason": "target_all_zero"}
    except Exception:
        return {"is_valid": False, "invalid_reason": "target_read_error"}

    return {"is_valid": True, "invalid_reason": None}