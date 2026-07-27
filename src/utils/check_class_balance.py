"""
Compute the fraction of 'fire' pixels (target > 0) across target
rasters, using the same manifest + split-loading logic as training.

Usage:
    python check_class_balance.py --bucket wildfire-irp-data-th25 \
        --prefix processed_dataset_gcp --branch viirs \
        --splits-dir splits_500_samples_test --split train
"""

import argparse
import json
import numpy as np
import rasterio
from google.cloud import storage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--branch", default="viirs")
    p.add_argument("--splits-dir", default=None,
                    help="If given, restrict to this split's IDs (e.g. splits_500_samples_test).")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--max-samples", type=int, default=None,
                    help="Cap number of samples inspected (for speed).")
    return p.parse_args()


def main():
    args = parse_args()
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    split_ids = None
    if args.splits_dir:
        path = f"{args.prefix}/{args.splits_dir}/{args.branch}/{args.split}.txt"
        blob = bucket.blob(path)
        content = blob.download_as_text().strip()
        split_ids = set(line.strip() for line in content.splitlines() if line.strip())
        print(f"Restricting to {len(split_ids)} IDs from {path}")

    manifest_prefix = f"{args.prefix}/{args.branch}_pairs/manifest/"
    manifests = []
    for blob in bucket.list_blobs(prefix=manifest_prefix):
        if not blob.name.endswith(".json"):
            continue
        try:
            m = json.loads(blob.download_as_text())
        except Exception:
            continue
        if m.get("is_valid") is not True:
            continue
        if split_ids is not None and m.get("id") not in split_ids:
            continue
        manifests.append(m)

    if args.max_samples:
        manifests = manifests[:args.max_samples]

    print(f"Inspecting {len(manifests)} target rasters...\n")

    fire_fractions = []
    empty_count = 0

    for i, m in enumerate(manifests):
        target_path = m["paths"]["target"]
        try:
            raw = bucket.blob(target_path).download_as_bytes()
            with rasterio.MemoryFile(raw) as mf:
                with mf.open() as src:
                    data = src.read().astype(np.float32)
                    nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = 0.0
            data[~np.isfinite(data)] = 0.0
        except Exception as e:
            print(f"  [ERROR] {m.get('id')}: {e}")
            continue

        binary = (data > 0)
        frac = binary.mean()
        fire_fractions.append(frac)
        if binary.sum() == 0:
            empty_count += 1

        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(manifests)} processed")

    fire_fractions = np.array(fire_fractions)
    print("\n" + "=" * 50)
    print(f"Samples inspected      : {len(fire_fractions)}")
    print(f"Samples with ZERO fire : {empty_count} ({100*empty_count/len(fire_fractions):.1f}%)")
    print(f"Mean fire pixel frac   : {fire_fractions.mean():.6f}")
    print(f"Median fire pixel frac : {np.median(fire_fractions):.6f}")
    print(f"Max fire pixel frac    : {fire_fractions.max():.6f}")
    if fire_fractions.mean() > 0:
        print(f"Implied pos:neg ratio  : 1:{(1/fire_fractions.mean() - 1):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()