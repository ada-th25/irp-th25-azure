import json
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F
from google.cloud import storage


class WildfireDataset(Dataset):
    """
    Loads (target, goes, covariates) triplets for either the
    landsat or viirs branch from GCS manifests.

    Spatial layout after __getitem__:
        Landsat branch:
            goes       -> (3, 15, 15)    bilinear snap from ~15-16px GEE output
            covariates -> (2, 256, 256)  centre-cropped from 1024px context export
            target     -> (1, 256, 256)  Pereira et al. Landsat patch

        VIIRS branch:
            goes       -> (3, 15, 15)   bilinear snap from 15px GEE output
            covariates -> (2, 80, 80)   full context-extent covariate (30,000m @ 375m)
            target     -> (1, 20, 20)   nearest snap from GEE output (7500m @ 375m)

    Completeness filtering:
        Samples are dropped at init time if any of the three
        files are missing from GCS, or if the GOES raster is all-masked
        (all zeros across all bands — indicates a failed/empty GEE export).
        The number of dropped samples is reported via self.n_dropped.
    """

    def __init__(self, bucket_name, output_prefix, branch="landsat"):
        self.bucket = storage.Client().bucket(bucket_name)
        self.branch = branch

        if branch == "landsat":
            self.target_size = (256, 256)
            self.goes_size   = (15, 15)
            self.cov_size    = (256, 256)
            self.cov_mode    = "crop"
        else:  # viirs
            self.target_size = (20, 20)
            self.goes_size   = (15, 15)
            self.cov_size    = (80, 80)
            self.cov_mode    = "resize"

        manifest_prefix = f"{output_prefix}/{branch}_pairs/manifest/"
        raw_samples = []
        for blob in self.bucket.list_blobs(prefix=manifest_prefix):
            if blob.name.endswith(".json"):
                data = json.loads(blob.download_as_text())
                raw_samples.append(data)

        self.samples, self.n_dropped = self._filter_complete(raw_samples)
        if self.n_dropped > 0:
            print(
                f"[WildfireDataset/{branch}] Dropped {self.n_dropped}/{len(raw_samples)} "
                f"incomplete or all-masked samples. "
                f"{len(self.samples)} usable pairs remaining."
            )