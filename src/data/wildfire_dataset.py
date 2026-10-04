"""
Wildfire Dataset and Storage Utilities.

Provides Azure Blob Storage dataset loading interfaces for multi-modal
wildfire datasets containing GOES satellite, DEM, land cover, and high-resolution
ground-truth fire rasters. Handles fork-safe lazy connection pooling, dynamic
raster spatial resampling, target dilation/heatmap transforms, and split ID resolution.
"""

import os
import json
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F
from azure.storage.blob import ContainerClient, BlobClient

# ------------------------------------------------------------------
# Data Split
# ------------------------------------------------------------------
def load_split_ids(container_name: str, branch: str,
                   split: str, output_prefix: str = "processed_dataset",
                   splits_dir: str = "splits") -> set[str]:
    """
    Downloads split ID tracking manifest from Azure Storage.

    Args:
        container_name: Azure Storage container identifier.
        branch: Target dataset branch ('landsat' or 'viirs').
        split: Target dataset partition ('train', 'val', or 'test').
        output_prefix: Root directory path in the target container.
        splits_dir: Specific directory containing split text files.

    Returns:
        Set[str]: Unique sample identifier strings contained within the split file.

    Raises:
        FileNotFoundError: If target split file does not exist in Azure Storage.
    """
    path = f"{output_prefix}/{splits_dir}/{branch}/{split}.txt"
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    blob_client = BlobClient.from_connection_string(conn_str, container_name, blob_name=path)

    if not blob_client.exists():
        raise FileNotFoundError(
            f"Split file not found in container '{container_name}': {path}\n"
            f"Run split_data.py first to generate splits."
        )

    content = blob_client.download_blob().readall().decode("utf-8").strip()
    ids = set(line.strip() for line in content.splitlines() if line.strip())
    print(f"[load_split_ids] Loaded {len(ids)} IDs for {branch}/{split}")
    return ids

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
    landsat or viirs branch from Azure Blob Storage manifests.

     Spatial layout after __getitem__ (defaults, see DEFAULT_SIZES):
        Landsat branch:
            goes       -> (3, 4, 4)     native GOES context (context_ratio=1)
            goes_up    -> (3, 256, 256) upsampled for concatenation
            covariates -> (11, 256, 256) native (30m)
            target     -> (1, 256, 256) untouched fire patch

        VIIRS branch (stage 1 -- full-res input, native-res output):
            goes       -> (3, 4, 4)     native GOES context
            goes_up    -> (3, 256, 256) upsampled for concatenation
            covariates -> (11, 256, 256) native (30m), model
                                        input keeps full spatial detail
            target     -> (1, 20, 20)   native VIIRS grid, no resize needed
                                        (binarized from raw FRP)

    Completeness filtering:
        Samples are silently dropped at init time if any of the three
        files are missing from Azure Blob Storage, or if the GOES raster
        is all-masked (all zeros across all bands, indicates a failed/empty
        GOES fetch).
    """

    def __init__(self, container_name, output_prefix, branch="landsat", split_ids=None,
                 target_size=None, cov_size=None, goes_size=None,
                 binarize_target=None, dilate_target=False, heatmap_target=False, dilation_kernel_size=None,
                 goes_mean=None, goes_std=None, dem_mean=None, dem_std=None, heatmap_sigma=1.0,
                 manifest_workers=32, drop_dem=False, drop_landcover=False,
                 use_stage1_pred=False, stage1_pred_key="stage1_pred", augment=False):

        # Ensure mutual exclusion as requested
        if dilate_target and heatmap_target:
            raise ValueError("dilate_target and heatmap_target cannot both be True. "
                             "They are mutually exclusive target representation strategies.")

        self.container_name = container_name
        self._container_client = None   # created per-process, see self.container property
        self._client_pid = None         # tracks which process created self._container_client
        self.branch = branch
        self.output_prefix = output_prefix
        self.dilate_target = dilate_target
        self.heatmap_target = heatmap_target
        self.heatmap_sigma = heatmap_sigma
        self.use_stage1_pred = use_stage1_pred
        self.stage1_pred_key = stage1_pred_key   # key in manifest["paths"] pointing to the stage-1 raster
        self.augment = augment
        self.drop_dem = drop_dem
        self.drop_landcover = drop_landcover

        if binarize_target is None:
            binarize_target = (branch == "viirs")
        self.binarize_target = binarize_target

        if dilation_kernel_size is None:
            self.dilation_kernel_size = 5 if branch == "landsat" else 3
        else:
            self.dilation_kernel_size = dilation_kernel_size

        # GOES per-channel standardisation stats (computed via compute_goes_stats.py
        # over the training split)
        default_goes_mean = [305.98227561473846, 296.6176749324799, 9.364600682258606]
        default_goes_std  = [12.870277949475222, 9.583713562045117, 7.1649856621255585]
        self.goes_mean = torch.tensor(goes_mean if goes_mean is not None else default_goes_mean).view(-1, 1, 1)
        self.goes_std  = torch.tensor(goes_std if goes_std is not None else default_goes_std).view(-1, 1, 1)

        # DEM standardisation
        default_dem_mean = [0.3390881556710402]
        default_dem_std  = [0.5048759609237181]
        self.dem_mean = torch.tensor(dem_mean if dem_mean is not None else default_dem_mean).view(-1, 1, 1)
        self.dem_std  = torch.tensor(dem_std if dem_std is not None else default_dem_std).view(-1, 1, 1)

        defaults = DEFAULT_SIZES.get(branch, DEFAULT_SIZES["landsat"])
        self.target_size = tuple(target_size) if target_size is not None else defaults["target_size"]
        self.cov_size = tuple(cov_size) if cov_size is not None else defaults["cov_size"]
        self.goes_size = tuple(goes_size) if goes_size is not None else defaults["goes_size"]

        manifest_prefix = f"{output_prefix}/{branch}_pairs/manifest/"
        raw_samples = self._load_manifests(manifest_prefix, split_ids, manifest_workers)

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
    # Manifest loading
    # ------------------------------------------------------------------
    def _download_manifest_json(self, blob_name):
        """Downloads and parses Azure Blob JSON manifest safely."""
        try:
            blob_client = self.container.get_blob_client(blob_name)
            content = blob_client.download_blob().readall()
            return json.loads(content.decode("utf-8"))
        except Exception:
            return None

    def _load_manifests(self, manifest_prefix, split_ids, manifest_workers):
        """Fetches JSON manifest files in parallel using ThreadPoolExecutor worker pools."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        candidate_blobs = []
        used_direct_path = False

        if split_ids is not None:
            probe_names = [f"{manifest_prefix}{sid}.json" for sid in split_ids]

            def _check_exists(b_name):
                return b_name, self.container.get_blob_client(b_name).exists()

            with ThreadPoolExecutor(max_workers=manifest_workers) as ex:
                results = list(ex.map(_check_exists, probe_names))

            found = [b_name for b_name, exists in results if exists]
            missing_count = len(probe_names) - len(found)

            if missing_count == 0:
                candidate_blobs = found
                used_direct_path = True
            elif len(found) > 0:
                print(f"[WildfireDataset/{self.branch}] {missing_count} split IDs not found "
                      f"via direct manifest path; falling back to full listing.")
            else:
                print(f"[WildfireDataset/{self.branch}] Direct manifest path found nothing; "
                      f"falling back to full listing.")

        if not used_direct_path:
            candidate_blobs = [
                blob.name for blob in self.container.list_blobs(name_starts_with=manifest_prefix)
                if blob.name.endswith(".json")
            ]

        raw_samples = []
        with ThreadPoolExecutor(max_workers=manifest_workers) as ex:
            futures = [ex.submit(self._download_manifest_json, b_name) for b_name in candidate_blobs]
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    raw_samples.append(result)

        return raw_samples

    # ------------------------------------------------------------------
    # Fork-safe lazy Azure client
    # ------------------------------------------------------------------
    @property
    def container(self):
        """
        Lazily initializes Azure ContainerClient bound to current PID.

        Prevents socket-sharing issues during PyTorch DataLoader worker processes.
        """
        current_pid = os.getpid()
        if self._container_client is None or self._client_pid != current_pid:
            conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
            self._container_client = ContainerClient.from_connection_string(
                conn_str, container_name=self.container_name
            )
            self._client_pid = current_pid
        return self._container_client

    # ------------------------------------------------------------------
    # Completeness filtering
    # ------------------------------------------------------------------

    def _blob_exists(self, blob_path):
        return self.container.get_blob_client(blob_path).exists()

    def _goes_is_valid(self, blob_path):
        blob_client = self.container.get_blob_client(blob_path)
        if not blob_client.exists():
            return False
        try:
            raw = blob_client.download_blob().readall()
            with rasterio.MemoryFile(raw) as mf:
                with mf.open() as src:
                    data = src.read().astype(np.float32)
                    nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = 0.0
            data[~np.isfinite(data)] = 0.0

            if np.all(data == 0):
                return False

        except Exception:
            return False
        return True

    def _filter_complete(self, raw_samples):
        """Filters manifest entries based on `is_valid` flag and stage 1 key existence."""
        valid = [m for m in raw_samples if m.get("is_valid") is True]
        if self.use_stage1_pred:
            before = len(valid)
            valid = [m for m in valid if self.stage1_pred_key in m.get("paths", {})]
            dropped_stage1 = before - len(valid)
            if dropped_stage1 > 0:
                print(f"[WildfireDataset/{self.branch}] Dropped {dropped_stage1} samples "
                      f"missing '{self.stage1_pred_key}' path.")
        dropped = len(raw_samples) - len(valid)
        return valid, dropped

    # ------------------------------------------------------------------
    # Raster I/O
    # ------------------------------------------------------------------

    def _read_azure_raster(self, blob_path, fill_nodata=0.0):
        """
        Reads binary raster data from Azure Blob Storage into a PyTorch FloatTensor.

        Replaces NoData values and non-finite NaN/Inf values with fill value.

        Args:
            blob_path: Azure blob path within the container.
            fill_nodata: Target numerical replacement value for non-finite entries.

        Returns:
            torch.Tensor: Spatial raster tensor representation.
        """
        blob_client = self.container.get_blob_client(blob_path)
        raw = blob_client.download_blob().readall()
        with rasterio.MemoryFile(raw) as memfile:
            with memfile.open() as src:
                data = src.read().astype(np.float32)
                nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = fill_nodata
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
        Converts binary active fire coordinates into a 2D continuous Gaussian heatmap.

        Args:
            binary_target: Single-channel spatial target tensor (1, H, W).
            sigma: Gaussian distribution standard deviation / kernel radius spread.

        Returns:
            torch.Tensor: Continuous target probability heatmap with shape (1, H, W).
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
    # Onehot Landcover
    # ------------------------------------------------------------------

    LANDCOVER_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]

    def _onehot_landcover(self, lc_code):
        """Encodes raw land cover class code raster into multi-channel one-hot tensor."""
        # lc_code: (1, H, W) raw class codes
        stacks = [(lc_code[0] == c).float() for c in self.LANDCOVER_CLASSES]
        return torch.stack(stacks, dim=0)  # (N, H, W)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Returns total valid samples loaded into dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches, preprocesses, resamples, and constructs model training batch dict.

        Args:
            idx: Integer dataset sample index.

        Returns:
            Dict[str, Union[torch.Tensor, str]]: Batch dictionary containing model inputs
            ("input"), target masks ("target"), GOES base tensor ("goes"),
            covariates ("covariates"), sample string ID ("sample_id"), and date string ("date").
        """
        manifest = self.samples[idx]
        paths = manifest["paths"]

        target = self._read_azure_raster(paths["target"])
        goes = self._read_azure_raster(paths["goes_input"])
        cov = self._read_azure_raster(paths["covariates"])

        goes = (goes - self.goes_mean) / self.goes_std

        dem = cov[0:1] / 1000.0                    # km
        dem = (dem - self.dem_mean) / self.dem_std  # standardise
        lc_onehot = self._onehot_landcover(cov[1:2])
        cov = torch.cat([dem, lc_onehot], dim=0)   # (1+N, H, W)

        if self.drop_dem:
            cov[0:1] = 0.0
        if self.drop_landcover:
            cov[1:11] = 0.0

        # Snap GOES to canonical low-res size (handles +/-1px snapping).
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

        if self.binarize_target:
            target = (target > 0).float()
            if self.dilate_target:
                padding = self.dilation_kernel_size // 2
                target = F.max_pool2d(
                    target.unsqueeze(0),
                    kernel_size=self.dilation_kernel_size,
                    stride=1,
                    padding=padding,
                ).squeeze(0)
                target = (target > 0).float()
            elif self.heatmap_target:
                target = self.make_gaussian_heatmap_target(
                    target, sigma=self.heatmap_sigma
                )

        input_channels = [goes_up]

        if self.use_stage1_pred:
            stage1_pred = self._read_azure_raster(paths[self.stage1_pred_key])
            if stage1_pred.shape[-2:] != self.cov_size:
                stage1_pred = F.interpolate(
                    stage1_pred.unsqueeze(0), size=self.cov_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            input_channels.append(stage1_pred)

        input_channels.append(cov)
        inp = torch.cat(input_channels, dim=0)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                inp = torch.flip(inp, dims=[-1])       # horizontal flip
                target = torch.flip(target, dims=[-1])
            if torch.rand(1).item() < 0.5:
                inp = torch.flip(inp, dims=[-2])       # vertical flip
                target = torch.flip(target, dims=[-2])

        return {
            "input":      inp,       # (goes_ch + cov_ch, H, W) model input
            "target":     target,    # (1, H, W) fire mask, H,W = target_size
            "goes":       goes,      # (3, goes_H, goes_W) raw GOES
            "covariates": cov,       # (C, H, W) DEM + land cover, H,W = cov_size
            "sample_id":  manifest["id"],
            "date":       manifest["date"],
        }
