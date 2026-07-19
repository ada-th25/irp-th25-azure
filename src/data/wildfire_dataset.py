import os
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
                   split: str, output_prefix: str = "processed_dataset",
                   splits_dir: str = "splits") -> set[str]:
    """
    Read a split text file from GCS and return the set of sample IDs.
 
    Args:
        bucket_name: GCS bucket (e.g. 'wildfire-irp-data-th25')
        branch: 'landsat' or 'viirs'
        split: 'train', 'val', or 'test'
        output_prefix: top-level GCS prefix (default 'processed_dataset')
        splits_dir: which splits folder to read from, e.g. 'splits'
                    (full dataset) or 'splits_dev500' (subsampled).
                    Must match the --suffix used in split_data.py.
 
    Returns:
        Set of sample ID strings.
 
    Raises:
        FileNotFoundError if the split file does not exist, run
        split_data.py first.
    """
    path = f"{output_prefix}/{splits_dir}/{branch}/{split}.txt"
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(path)
 
    if not blob.exists():
        raise FileNotFoundError(
            f"Split file not found: gs://{bucket_name}/{path}\n"
            f"Run split_data.py first to generate splits."
        )
 
    content = blob.download_as_text().strip()
    ids = set(line.strip() for line in content.splitlines() if line.strip())
    print(f"[load_split_ids] Loaded {len(ids)} IDs for {branch}/{split}")
    return ids

# ------------------------------------------------------------------
# Default sizes per branch, based on measured native resolutions:
#
#   landsat: target 256x256, covariates 256x256
#            goes native 4x4
#   viirs:   target  20x20 - NOT resized, model input
#            (goes_up + covariates) stays at FULL native
#            256x256 (30m) resolution. The model's 256x256 output
#            is downsampled to the native 20x20 target grid before
#            the loss/metrics compare them
#            Stage 1 predicts at VIIRS resolution, using
#            full-resolution inputs to get there.
# ------------------------------------------------------------------
DEFAULT_SIZES = {
    "landsat": {"target_size": (256, 256), "cov_size": (256, 256), "goes_size": (4, 4)},
    "viirs":   {"target_size": (20, 20),   "cov_size": (256, 256), "goes_size": (4, 4)},
}

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class WildfireDataset(Dataset):
    """
    Loads (target, goes, covariates) triplets for either the
    landsat or viirs branch from GCS manifests.

     Spatial layout after __getitem__ (defaults, see DEFAULT_SIZES):
        Landsat branch:
            goes       -> (3, 4, 4)     native GOES context (context_ratio=1)
            goes_up    -> (3, 256, 256) upsampled for concatenation
            covariates -> (2, 256, 256) native (30m)
            target     -> (1, 256, 256) untouched fire patch
 
        VIIRS branch (stage 1 -- full-res input, native-res output):
            goes       -> (3, 4, 4)     native GOES context
            goes_up    -> (3, 256, 256) upsampled for concatenation
            covariates -> (2, 256, 256) native (30m), UNCHANGED -- model
                                        input keeps full spatial detail
            target     -> (1, 20, 20)   native VIIRS grid, no resize needed
                                        (binarized from raw FRP)

    Completeness filtering:
        Samples are silently dropped at init time if any of the three
        files are missing from GCS, or if the GOES raster is all-masked
        (all zeros across all bands, indicates a failed/empty GEE export).
        The number of dropped samples is reported via self.n_dropped.
    """

    def __init__(self, bucket_name, output_prefix, branch="landsat", split_ids=None,
                 target_size=None, cov_size=None, goes_size=None, 
                 dilate_target=False, heatmap_target=False):

        # Ensure mutual exclusion as requested
        if dilate_target and heatmap_target:
            raise ValueError("dilate_target and heatmap_target cannot both be True. "
                             "They are mutually exclusive target representation strategies.")
        
        self.bucket_name = bucket_name
        self.branch = branch
        self.output_prefix = output_prefix
        self.dilate_target = dilate_target
        self.heatmap_target = heatmap_target
        self._bucket = None       # created per-process, see self.bucket property
        self._client_pid = None   # tracks which process created self._bucket
 
        defaults = DEFAULT_SIZES.get(branch, DEFAULT_SIZES["landsat"])
        self.target_size = tuple(target_size) if target_size is not None else defaults["target_size"]
        self.cov_size = tuple(cov_size) if cov_size is not None else defaults["cov_size"]
        self.goes_size = tuple(goes_size) if goes_size is not None else defaults["goes_size"]
 
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
 
        print(f"[WildfireDataset/{branch}] sizes: target={self.target_size}, "
              f"cov={self.cov_size}, goes={self.goes_size}")

    # ------------------------------------------------------------------
    # Fork-safe lazy GCS client
    # ------------------------------------------------------------------
    @property
    def bucket(self):
        """
        Lazily create a storage.Client()/bucket in the CURRENT process.
 
        google-cloud-storage's client holds open HTTP connections
        (urllib3 pools) that are not fork-safe: if a client created in
        the main process is inherited by DataLoader worker processes via
        fork(), concurrent workers can corrupt/interleave reads on the
        duplicated sockets, surfacing as intermittent
        google.cloud.storage.exceptions.DataCorruption errors.
 
        Checking os.getpid() ensures each worker process builds its own
        fresh client on first access, rather than reusing one inherited
        from the parent at fork time.
        """
        current_pid = os.getpid()
        if self._bucket is None or self._client_pid != current_pid:
            self._bucket = storage.Client().bucket(self.bucket_name)
            self._client_pid = current_pid
        return self._bucket

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
                    data = src.read().astype(np.float32)
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
        valid = [m for m in raw_samples if m.get("is_valid") is True]
        dropped = len(raw_samples) - len(valid)
        return valid, dropped

    # ------------------------------------------------------------------
    # Raster I/O
    # ------------------------------------------------------------------

    def _read_gcs_raster(self, gcs_path, fill_nodata=0.0):
        blob = self.bucket.blob(gcs_path)
        raw  = blob.download_as_bytes()
        with rasterio.MemoryFile(raw) as memfile:
            with memfile.open() as src:
                data = src.read().astype(np.float32)
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
    # Gaussian Heatmap Generator
    # ------------------------------------------------------------------

    def make_gaussian_heatmap_target(self, binary_target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Converts binary keypoints into continuous Gaussian target heatmaps.
        For every pixel with value 1.0, a Gaussian distribution is rendered around it.
        Underlapping Gaussians retain the maximum value at each pixel.
        """
        _, h, w = binary_target.shape
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y_grid = y_grid.to(binary_target.device).float()
        x_grid = x_grid.to(binary_target.device).float()

        # Locate all active/non-zero peak coordinates (handles both {0, 1} and raw continuous inputs)
        pos_indices = torch.nonzero(binary_target[0] > 0.0, as_tuple=False)
        if len(pos_indices) == 0:
            return torch.zeros_like(binary_target)

        # Draw Gaussians around each point, taking the element-wise maximum
        heatmap = torch.zeros((h, w), device=binary_target.device)
        for py, px in pos_indices:
            # Squared Euclidean distance
            dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
            channel_heatmap = torch.exp(-dist_sq / (2 * (sigma ** 2)))
            heatmap = torch.max(heatmap, channel_heatmap)

        return heatmap.unsqueeze(0)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        manifest = self.samples[idx]
        paths = manifest["paths"]
    
        target = self._read_gcs_raster(paths["target"])
        goes = self._read_gcs_raster(paths["goes_input"])
        cov = self._read_gcs_raster(paths["covariates"])

        # Per-channel standardization for GOES (C07, C14, TFI), computed
        # from compute_goes_stats.py over the training split
        goes_mean = torch.tensor([305.98227561473846, 296.6176749324799, 9.364600682258606]).view(-1, 1, 1)
        goes_std  = torch.tensor([12.870277949475222, 9.583713562045117, 7.1649856621255585]).view(-1, 1, 1)
        goes = (goes - goes_mean) / goes_std

        cov[0] = cov[0] / 1000.0          # DEM in km instead of meters
        cov[1] = cov[1] / 100.0           # land cover code roughly to [0,1]
    
        # Snap GOES to canonical low-res size (handles ±1px GEE snapping).
        goes = F.interpolate(
            goes.unsqueeze(0),
            size=self.goes_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
 
        # Upsample GOES separately for concatenation with covariates/target
        goes_up = F.interpolate(
            goes.unsqueeze(0),
            size=self.cov_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
 
        # Snap covariates to cov_size
        if cov.shape[-2:] != self.cov_size:
            cov = F.interpolate(
                cov.unsqueeze(0),
                size=self.cov_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
 
        # Snap target to canonical size
        if target.shape[-2:] != self.target_size:
            target = F.interpolate(
                target.unsqueeze(0),
                size=self.target_size,
                mode="nearest",
            ).squeeze(0)
 
        if self.branch == "viirs":
            target = (target > 0).float()
            
            # Apply dilation OR heatmap depending on config
            if self.dilate_target:
                target = F.max_pool2d(
                    target.unsqueeze(0), kernel_size=3, stride=1, padding=1
                ).squeeze(0)
                target = (target > 0).float()
            elif self.heatmap_target:
                # Convert binary targets into standard continuous Gaussian targets
                # Default sigma = 1.0 fits small grids like 20x20 nicely
                target = self.make_gaussian_heatmap_target(target, sigma=1.0)
 
        # Concatenate upsampled GOES with covariates into the model input
        inp = torch.cat([goes_up, cov], dim=0)
 
        return {
            "input":      inp,       # (goes_ch + cov_ch, H, W) model input
            "target":     target,    # (1, H, W) fire mask, H,W = target_size
            "goes":       goes,      # (3, goes_H, goes_W) raw GOES, for diagnostics
            "covariates": cov,       # (2, H, W) DEM + land cover, H,W = cov_size
            "sample_id":  manifest["id"],
            "date":       manifest["date"],
        }