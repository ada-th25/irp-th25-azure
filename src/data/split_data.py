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


def write_split(bucket, branch: str, split_name: str, ids: list[str]) -> None:
    path = f"{OUTPUT_PREFIX}/splits/{branch}/{split_name}.txt"
    content = "\n".join(sorted(ids))
    bucket.blob(path).upload_from_string(content, content_type="text/plain")
    print(f"  Written {len(ids):4d} IDs -> gs://{bucket.name}/{path}")


def split_exists(bucket, branch: str) -> bool:
    return bucket.blob(
        f"{OUTPUT_PREFIX}/splits/{branch}/train.txt"
    ).exists()


def run_branch(bucket, branch: str, overwrite: bool) -> None:
    print(f"\n{'='*60}")
    print(f"Branch: {branch}")
    print(f"{'='*60}")

    if split_exists(bucket, branch) and not overwrite:
        print(f"  Splits already exist for '{branch}'. "
              f"Use --overwrite to regenerate.")
        return

    manifests = load_manifests(bucket, branch)
    if not manifests:
        print(f"  [SKIP] No manifests found for branch '{branch}'.")
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

    write_split(bucket, branch, "train", train_ids)
    write_split(bucket, branch, "val",   val_ids)
    write_split(bucket, branch, "test",  test_ids)
    print(f"  Done.")

def main():
    parser = argparse.ArgumentParser(description="Create stratified data splits.")
    parser.add_argument("--branch",    choices=["landsat", "viirs", "both"],
                        default="both")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate splits even if they already exist.")
    args = parser.parse_args()

    credentials, _ = default()
    bucket = storage.Client(credentials=credentials).bucket(BUCKET_NAME)

    branches = (["landsat", "viirs"] if args.branch == "both"
                else [args.branch])

    for branch in branches:
        run_branch(bucket, branch, args.overwrite)

    print("\nSplits complete.")


if __name__ == "__main__":
    main()