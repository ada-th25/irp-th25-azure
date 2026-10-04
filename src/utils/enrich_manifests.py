"""
Usage: python -m src.utils.enrich_manifests --branch viirs
"""

import argparse
import json
import os
from azure.storage.blob import ContainerClient
from src.data.validity import compute_validity
from concurrent.futures import ThreadPoolExecutor, as_completed

CONTAINER_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_gcp"

def process_blob(container_client, blob_name, dry_run):
    if not blob_name.endswith(".json"):
        return None
    try:
        content = container_client.get_blob_client(blob_name).download_blob().readall()
        manifest = json.loads(content)
    except Exception as e:
        return ("warn", blob_name, str(e))
    result = compute_validity(container_client, manifest)
    manifest.update(result)
    if not dry_run:
        container_client.get_blob_client(blob_name).upload_blob(
            json.dumps(manifest, indent=2), overwrite=True, content_type="application/json"
        )
    return ("ok", result)

def enrich_branch(container_client, branch, dry_run, max_workers=32):
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    blob_names = [b.name for b in container_client.list_blobs(name_starts_with=prefix) if b.name.endswith(".json")]
    print(f"Enriching {len(blob_names)} manifests for branch '{branch}'...")
    n_valid, n_invalid, reasons = 0, 0, {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_blob, container_client, b, dry_run) for b in blob_names]
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
    if reasons:
        print("  Invalid reason breakdown:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:6d}  {reason}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["landsat", "viirs", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute and report validity without writing back to Azure.")
    args = parser.parse_args()

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)

    branches = ["landsat", "viirs"] if args.branch == "both" else [args.branch]
    for branch in branches:
        enrich_branch(container_client, branch, args.dry_run)

if __name__ == "__main__":
    main()
