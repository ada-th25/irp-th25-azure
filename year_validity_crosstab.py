"""
Cross-tab validity (and invalid_reason) by year -- diagnoses whether specific
years' clusters are disproportionately landing in the invalid bucket, which
the plain --count breakdown doesn't show (it only tabulates years for
ALREADY-valid samples).

Usage: python year_validity_crosstab.py
"""

import json
import os
from collections import defaultdict

from azure.storage.blob import ContainerClient

CONTAINER_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_gcp"
BRANCH = "viirs"


def main():
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    client = ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)

    prefix = f"{OUTPUT_PREFIX}/{BRANCH}_pairs/manifest/"
    names = [b.name for b in client.list_blobs(name_starts_with=prefix) if b.name.endswith(".json")]
    print(f"Downloading {len(names)} manifests...")

    # year -> {"valid": n, "missing_goes": n, "missing_target_or_cov": n, ...}
    by_year = defaultdict(lambda: defaultdict(int))

    for i, name in enumerate(names):
        try:
            content = client.get_blob_client(name).download_blob().readall()
            m = json.loads(content)
        except Exception:
            continue

        date = m.get("date", "")
        year = date[:4] if len(date) >= 4 else "unknown"

        if m.get("is_valid") is True:
            by_year[year]["valid"] += 1
        else:
            reason = m.get("invalid_reason", "unflagged")
            by_year[year][reason] += 1

        if (i + 1) % 10000 == 0:
            print(f"  ...{i+1}/{len(names)}")

    all_reasons = sorted({r for counts in by_year.values() for r in counts})
    header = ["year"] + all_reasons
    print("\n" + "\t".join(header))
    for year in sorted(by_year):
        row = [year] + [str(by_year[year].get(r, 0)) for r in all_reasons]
        print("\t".join(row))


if __name__ == "__main__":
    main()
