import json
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F
from google.cloud import storage

# ------------------------------------------------------------------
# Data Split
# ------------------------------------------------------------------
def load_split_ids(bucket_name: str, branch: str,
                   split: str, output_prefix: str = "processed_dataset") -> set[str]:
    """
    Read a split text file from GCS and return the set of sample IDs.
 
    Args:
        bucket_name: GCS bucket (e.g. 'wildfire-irp-data-th25')
        branch: 'landsat' or 'viirs'
        split: 'train', 'val', or 'test'
        output_prefix: top-level GCS prefix (default 'processed_dataset')
 
    Returns:
        Set of sample ID strings.
 
    Raises:
        FileNotFoundError if the split file does not exist, run
        split_data.py first.
    """
    path   = f"{output_prefix}/splits/{branch}/{split}.txt"
    bucket = storage.Client().bucket(bucket_name)
    blob   = bucket.blob(path)
 
    if not blob.exists():
        raise FileNotFoundError(
            f"Split file not found: gs://{bucket_name}/{path}\n"
            f"Run split_data.py first to generate splits."
        )
 
    content = blob.download_as_text().strip()
    ids     = set(line.strip() for line in content.splitlines() if line.strip())
    print(f"[load_split_ids] Loaded {len(ids)} IDs for {branch}/{split}")
    return ids

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class WildfireDataset(Dataset):
    """
    Loads (target, goes, covariates) triplets for either the
    landsat or viirs branch from GCS manifests.

    Spatial layout after __getitem__:
        Landsat branch:
            goes       -> (3, 15, 15)   bilinear snap from ~15-16px GEE output
            covariates -> (2, 256, 256)  centre-cropped from 1024px context export
            target     -> (1, 256, 256)  untouched Pereira patch

        VIIRS branch:
            goes       -> (3, 15, 15)   bilinear snap from 15px GEE output
            covariates -> (2, 80, 80)   full context-extent covariate (30,000m @ 375m)
            target     -> (1, 20, 20)   nearest snap from GEE output (7500m @ 375m)

    Completeness filtering:
        Samples are silently dropped at init time if any of the three
        files are missing from GCS, or if the GOES raster is all-masked
        (all zeros across all bands, indicates a failed/empty GEE export).
        The number of dropped samples is reported via self.n_dropped.
    """

    def __init__(self, bucket_name, output_prefix, branch="landsat", split_ids=None):
        self.bucket = storage.Client().bucket(bucket_name)
        self.branch = branch
        self.output_prefix = output_prefix

        if branch == "landsat":
            self.target_size = (256, 256)
            self.cov_size = (256, 256)
            self.goes_size = (15, 15)
        else:  # viirs
            self.target_size = (20, 20)
            self.cov_size = (20, 20)
            self.goes_size = (15, 15)

        manifest_prefix = f"{output_prefix}/{branch}_pairs/manifest/"
        raw_samples = []
        for blob in self.bucket.list_blobs(prefix=manifest_prefix):
            if blob.name.endswith(".json"):
                try:
                    raw_samples.append(json.loads(blob.download_as_text()))
                except Exception:
                    pass

        if split_ids is not None:
            before = len(raw_samples)
            raw_samples = [m for m in raw_samples if m.get("id") in split_ids]
            print(f"[WildfireDataset/{branch}] Split filter: "
                  f"{len(raw_samples)}/{before} samples in provided split.")

        self.samples, self.n_dropped = self._filter_complete(raw_samples)
        
        if self.n_dropped > 0:
            print(
                f"[WildfireDataset/{branch}] Dropped {self.n_dropped}/{len(raw_samples)} "
                f"incomplete or all-masked samples. "
                f"{len(self.samples)} usable pairs remaining."
            )
        else:
            print(f"[WildfireDataset/{branch}] {len(self.samples)} usable pairs.")

    # ------------------------------------------------------------------
    # Completeness filtering
    # ------------------------------------------------------------------

    def _blob_exists(self, gcs_path):
        return self.bucket.blob(gcs_path).exists()

    def _goes_is_valid(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        if not blob.exists():
            return False
        try:
            raw = blob.download_as_bytes()
            with rasterio.MemoryFile(raw) as mf:
                with mf.open() as src:
                    data   = src.read().astype(np.float32)
                    nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = 0.0
            data[~np.isfinite(data)] = 0.0
    
            # All-zero -> failed GEE export
            if np.all(data == 0):
                return False
    
        except Exception:
            return False
        return True

    def _filter_complete(self, raw_samples):
        valid = []
        dropped = 0
        for m in raw_samples:
            paths = m.get("paths", {})
            target_path = paths.get("target", "")
            goes_path   = paths.get("goes_input", "")
            cov_path    = paths.get("covariates", "")

            # All three paths must exist in GCS
            if not all([
                self._blob_exists(target_path),
                self._blob_exists(cov_path),
            ]):
                dropped += 1
                continue

            # GOES must exist and not be all-masked
            if not self._goes_is_valid(goes_path):
                dropped += 1
                continue

            # Target must have at least 1 non-zero pixel
            try:
                raw = self.bucket.blob(target_path).download_as_bytes()
                with rasterio.MemoryFile(raw) as mf:
                    with mf.open() as src:
                        data = src.read().astype(np.float32)
                if np.all(data == 0):
                    dropped += 1
                    continue
            except Exception:
                dropped += 1
                continue

            valid.append(m)
        return valid, dropped

    # ------------------------------------------------------------------
    # Raster I/O
    # ------------------------------------------------------------------

    def _read_gcs_raster(self, gcs_path, fill_nodata=0.0):
        blob = self.bucket.blob(gcs_path)
        raw  = blob.download_as_bytes()
        with rasterio.MemoryFile(raw) as memfile:
            with memfile.open() as src:
                data   = src.read().astype(np.float32)
                nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = fill_nodata
        # Always replace NaN regardless of nodata metadata
        data[~np.isfinite(data)] = fill_nodata
        return torch.from_numpy(data)

    def _centre_crop(self, tensor, size):
        _, h, w = tensor.shape
        th, tw  = size
        if h < th or w < tw:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(max(h, th), max(w, tw)),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            _, h, w = tensor.shape
        top  = (h - th) // 2
        left = (w - tw) // 2
        return tensor[:, top:top+th, left:left+tw]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        manifest = self.samples[idx]
        paths    = manifest["paths"]
    
        target = self._read_gcs_raster(paths["target"])
        goes   = self._read_gcs_raster(paths["goes_input"])
        cov    = self._read_gcs_raster(paths["covariates"])
    
        # Snap GOES to canonical low-res size (handles ±1px GEE snapping).
        goes = F.interpolate(
            goes.unsqueeze(0),
            size=self.goes_size,        # (15, 15)
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                    # (3, 15, 15)
    
        # Upsample GOES separately for concatenation with covariates.
        # goes stays (3, 15, 15); goes_up is (3, 256, 256) or (3, 20, 20).
        goes_up = F.interpolate(
            goes.unsqueeze(0),
            size=self.target_size,      # (256, 256) or (20, 20)
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    
        # Snap covariates to target size (handles ±1-2px GEE grid snapping).
        if cov.shape[-2:] != self.cov_size:
            cov = F.interpolate(
                cov.unsqueeze(0),
                size=self.cov_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)                # (2, 256, 256) or (2, 20, 20)
    
        # Snap target to canonical size (GEE snapping can produce ±1px).
        if target.shape[-2:] != self.target_size:
            target = F.interpolate(
                target.unsqueeze(0),
                size=self.target_size,
                mode="nearest",
            ).squeeze(0)                # (1, 256, 256) or (1, 20, 20)
    
        # Concatenate upsampled GOES with covariates into the model input tensor.
        inp = torch.cat([goes_up, cov], dim=0)   # (5, 256, 256) or (5, 20, 20)
    
        return {
            "input":      inp,       # (5, H, W) model input
            "target":     target,    # (1, H, W) fire mask
            "goes":       goes,      # (3, 15, 15) raw GOES, for diagnostics
            "covariates": cov,       # (2, H, W) DEM + land cover
            "sample_id":  manifest["id"],
            "date":       manifest["date"],
        }