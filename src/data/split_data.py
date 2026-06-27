import argparse
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from google.cloud import storage
from google.auth import default

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BUCKET_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO is implicit: 1 - TRAIN - VAL
N_STRATA = 4      # number of fire-size quantile bins for stratification
RANDOM_SEED = 42  # fixed seed for reproducibility across all experiments


def load_manifests(bucket, branch: str) -> list[dict]:
    """Download and parse all manifest JSONs for a given branch."""
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"  Loading {len(blobs)} manifests for branch '{branch}'...")
    manifests = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            try:
                manifests.append(json.loads(blob.download_as_text()))
            except Exception as e:
                print(f"  [WARN] Could not parse {blob.name}: {e}")
    return manifests


def stratified_split(
    sample_ids: list[str],
    pixel_counts: list[int],
    train_ratio: float,
    val_ratio: float,
    n_strata: int,
    seed: int,
) -> tuple[list, list, list]:
    """
    Stratified train/val/test split by fire size using
    sklearn.StratifiedShuffleSplit.

    Steps:
    1. Bin continuous pixel counts into n_strata quantile bins
    2. First split: train+val vs test, stratified by bin
    3. Second split: train vs val from the train+val portion,
       stratified by bin
    """
    sample_ids = np.array(sample_ids)
    counts = np.array(pixel_counts)

    # Assign each sample to a fire-size quantile bin
    edges = np.percentile(counts, np.linspace(0, 100, n_strata + 1))
    bins = np.clip(np.digitize(counts, edges[1:-1]), 0, n_strata - 1)

    print(f"  Bin distribution: "
          + ", ".join(f"bin{i}={np.sum(bins==i)}" for i in range(n_strata)))

    # Split into (train+val) vs test
    test_ratio = 1.0 - train_ratio - val_ratio
    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_val_idx, test_idx = next(sss_outer.split(sample_ids, bins))

    # Split train+val into train vs val
    # val_ratio as a fraction of the train+val portion
    # If we want val = 15% of total, train+val = 85% of total
    # so val fraction of train+val = 0.15 / 0.85
    val_of_trainval = val_ratio / (train_ratio + val_ratio)
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=val_of_trainval, random_state=seed
    )
    train_idx, val_idx = next(
        sss_inner.split(sample_ids[train_val_idx], bins[train_val_idx])
    )
    # Remap local indices back to global positions
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_ids = sample_ids[train_idx].tolist()
    val_ids = sample_ids[val_idx].tolist()
    test_ids = sample_ids[test_idx].tolist()

    print(f"  Split: train={len(train_ids)} / "
          f"val={len(val_ids)} / test={len(test_ids)}")

    return train_ids, val_ids, test_ids