"""
Usage: python -m src.utils.enrich_manifests --branch viirs
"""

import argparse
import json
from google.cloud import storage
from google.auth import default
from src.data.validity import compute_validity
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_gcp"

def process_blob(bucket, blob, dry_run):
    if not blob.name.endswith(".json"):
        return None
    try:
        manifest = json.loads(blob.download_as_text())
    except Exception as e:
        return ("warn", blob.name, str(e))
    result = compute_validity(bucket, manifest)
    manifest.update(result)
    if not dry_run:
        blob.upload_from_string(json.dumps(manifest, indent=2), content_type="application/json")
    return ("ok", result)

def enrich_branch(bucket, branch, dry_run, max_workers=32):
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".json")]
    print(f"Enriching {len(blobs)} manifests for branch '{branch}'...")
    n_valid, n_invalid, reasons = 0, 0, {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_blob, bucket, b, dry_run) for b in blobs]
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            if r[0] == "warn":
                print(f"  [WARN] Could not parse {r[1]}: {r[2]}")
                continue
            result = r[1]
            if result["is_valid"]:
                n_valid += 1
            else:
                n_invalid += 1
                reasons[result["invalid_reason"]] = reasons.get(result["invalid_reason"], 0) + 1
    print(f"  Branch '{branch}': valid={n_valid}, invalid={n_invalid}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["landsat", "viirs", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute and report validity without writing back to GCS.")
    args = parser.parse_args()

    credentials, _ = default()
    bucket = storage.Client(credentials=credentials).bucket(BUCKET_NAME)

    branches = ["landsat", "viirs"] if args.branch == "both" else [args.branch]
    for branch in branches:
        enrich_branch(bucket, branch, args.dry_run)

if __name__ == "__main__":
    main()