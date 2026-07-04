import argparse
import json
from google.cloud import storage
from google.auth import default
from src.data.validity import compute_validity

BUCKET_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_r1"

def enrich_branch(bucket, branch: str, dry_run: bool):
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Enriching {len(blobs)} manifests for branch '{branch}'...")

    n_valid, n_invalid, reasons = 0, 0, {}
    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue
        try:
            manifest = json.loads(blob.download_as_text())
        except Exception as e:
            print(f"  [WARN] Could not parse {blob.name}: {e}")
            continue

        result = compute_validity(bucket, manifest)
        manifest.update(result)

        if result["is_valid"]:
            n_valid += 1
        else:
            n_invalid += 1
            reasons[result["invalid_reason"]] = reasons.get(result["invalid_reason"], 0) + 1

        if not dry_run:
            blob.upload_from_string(json.dumps(manifest, indent=2),
                                     content_type="application/json")

    print(f"  Branch '{branch}': valid={n_valid}, invalid={n_invalid}")
    if reasons:
        print(f"  Invalid breakdown: {reasons}")
    if dry_run:
        print("  [DRY RUN] No manifests were modified.")

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