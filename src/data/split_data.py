"""
Dataset Stratification and Split Generation Pipeline (Azure Native Edition).

Generates reproducible train/validation/test splits stratified by fire size
(pixel counts or hotspot detections) for Landsat and VIIRS branches. Reads
sample manifest metadata directly from Azure Blob Storage and writes
split ID lists as plain-text blobs.

Pre-requisite:
    Must run `src/utils/enrich_manifests.py` before generating splits to ensure
    `is_valid` metadata flags are populated across all manifest JSON files.

Usage example:
    python src/data/split_data.py \
        --branch viirs \
        --max-samples 500 \
        --suffix 500_samples_test \
        --overwrite
"""

import os
import argparse
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from azure.storage.blob import ContainerClient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CONTAINER_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_gcp"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO is implicit: 1 - TRAIN - VAL
N_STRATA = 4      # number of fire-size quantile bins for stratification
RANDOM_SEED = 42  # fixed seed for reproducibility across all experiments


def load_manifests(container_client, branch: str) -> list[dict]:
    """
    Downloads and parses all manifest JSON records for a specified dataset branch.

    Args:
        container_client: Active azure.storage.blob.ContainerClient.
        branch: Target dataset branch ('landsat' or 'viirs').

    Returns:
        List[Dict[str, Any]]: List of parsed sample manifest dictionaries.
    """
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    blob_names = [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
    print(f"  Loading {len(blob_names)} manifests for branch '{branch}'...")
    manifests = []
    for name in blob_names:
        if name.endswith(".json"):
            try:
                content = container_client.get_blob_client(name).download_blob().readall()
                manifests.append(json.loads(content))
            except Exception as e:
                print(f"  [WARN] Could not parse {name}: {e}")
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
    Executes a two-stage stratified train/val/test split based on fire pixel counts.

    Divides continuous fire sizes into quantile bins and uses `StratifiedShuffleSplit`
    to guarantee identical size distributions across all generated partitions.

    Args:
        sample_ids: Identifiers for the dataset samples.
        pixel_counts: Target fire pixel or detection counts per sample.
        train_ratio: Proportion of dataset allocated to training set.
        val_ratio: Proportion of dataset allocated to validation set.
        n_strata: Number of percentile bins to aggregate fire sizes.
        seed: Random state seed for reproducible splitting.

    Returns:
        Tuple[List[str], List[str], List[str]]: Lists containing sample IDs for
        (train_ids, val_ids, test_ids).
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


def write_split(container_client, branch: str, split_name: str, ids: list[str],
                 suffix: str = "") -> None:
    """
    Serialises a split ID list to a plain-text blob in Azure Blob Storage.

    Args:
        container_client: Active azure.storage.blob.ContainerClient.
        branch: Dataset branch name ('landsat' or 'viirs').
        split_name: Partition name ('train', 'val', or 'test').
        ids: Collection of sample ID strings to write.
        suffix: Optional directory suffix for experimental splits.
    """
    splits_dir = f"splits_{suffix}" if suffix else "splits"
    path = f"{OUTPUT_PREFIX}/{splits_dir}/{branch}/{split_name}.txt"
    content = "\n".join(sorted(ids))
    container_client.get_blob_client(path).upload_blob(
        content, overwrite=True, content_type="text/plain"
    )
    print(f"  Written {len(ids):4d} IDs -> {CONTAINER_NAME}/{path}")


def split_exists(container_client, branch: str, suffix: str = "") -> bool:
    """
    Checks if a training split file already exists in Azure for a given branch.

    Args:
        container_client: Active azure.storage.blob.ContainerClient.
        branch: Dataset branch name ('landsat' or 'viirs').
        suffix: Optional directory suffix.

    Returns:
        bool: True if split exists, False otherwise.
    """
    splits_dir = f"splits_{suffix}" if suffix else "splits"
    return container_client.get_blob_client(
        f"{OUTPUT_PREFIX}/{splits_dir}/{branch}/train.txt"
    ).exists()


def run_branch(container_client, branch: str, overwrite: bool,
                max_samples: int = None, suffix: str = "") -> None:
    """
    Filters valid records, calculates splits, and exports split metadata for a branch.

    Args:
        container_client: Active azure.storage.blob.ContainerClient.
        branch: Target dataset branch ('landsat' or 'viirs').
        overwrite: Forces regeneration if destination split files exist.
        max_samples: Subsamples the target dataset before split execution.
        suffix: Directory suffix name for saving experiment splits.
    """
    print(f"\n{'='*60}")
    print(f"Branch: {branch}" + (f" (suffix={suffix})" if suffix else ""))
    print(f"{'='*60}")

    if split_exists(container_client, branch, suffix) and not overwrite:
        print(f"  Splits already exist for '{branch}'. "
              f"Use --overwrite to regenerate.")
        return

    manifests = load_manifests(container_client, branch)
    if not manifests:
        print(f"  [SKIP] No manifests found for branch '{branch}'.")
        return

    n_before = len(manifests)
    manifests = [m for m in manifests if m.get("is_valid") is True]
    n_dropped = n_before - len(manifests)
    if n_dropped:
        print(f"  Filtered {n_dropped}/{n_before} invalid samples "
            f"(missing/corrupted rasters) before splitting.")
    if not manifests:
        print(f"  [SKIP] No valid manifests remain for branch '{branch}'.")
        return

    # Extract IDs and fire pixel counts from manifests
    # Both landsat (n_active_pixels) and viirs (n_detections) branches
    # store a meaningful activity count, use whichever is present.
    sample_ids = []
    pixel_counts = []
    for m in manifests:
        sid = m.get("id")
        count = m.get("n_active_pixels") or m.get("n_detections") or 1
        if sid:
            sample_ids.append(sid)
            pixel_counts.append(int(count))

    if max_samples is not None:
        rng = np.random.default_rng(RANDOM_SEED)
        pool_idx = rng.permutation(len(sample_ids))[:max_samples]
        sample_ids = [sample_ids[i] for i in pool_idx]
        pixel_counts = [pixel_counts[i] for i in pool_idx]
        print(f"  [SMOKE] Subsampled pool to {len(sample_ids)} samples "
              f"before splitting (80/10/10 ratio preserved).")

    print(f"  Total samples: {len(sample_ids)}")
    print(f"  Fire size — min: {min(pixel_counts)}, "
          f"median: {int(np.median(pixel_counts))}, "
          f"max: {max(pixel_counts)}")
    print(f"  Stratifying into {N_STRATA} fire-size bins...")

    train_ids, val_ids, test_ids = stratified_split(
        sample_ids, pixel_counts,
        TRAIN_RATIO, VAL_RATIO, N_STRATA, RANDOM_SEED
    )

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"\n  Final split: train={len(train_ids)} / "
          f"val={len(val_ids)} / test={len(test_ids)} "
          f"(total={total})")

    write_split(container_client, branch, "train", train_ids, suffix)
    write_split(container_client, branch, "val",   val_ids, suffix)
    write_split(container_client, branch, "test",  test_ids, suffix)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser(description="Create stratified data splits.")
    parser.add_argument("--branch",    choices=["landsat", "viirs", "both"],
                        default="both")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate splits even if they already exist.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap each split to N samples (for smoke testing/fine tuning)")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output split dir")
    args = parser.parse_args()

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)

    branches = (["landsat", "viirs"] if args.branch == "both"
                else [args.branch])

    for branch in branches:
        run_branch(container_client, branch, args.overwrite,
                   max_samples=args.max_samples, suffix=args.suffix)

    print("\nSplits complete.")


if __name__ == "__main__":
    main()
