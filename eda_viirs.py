"""
EDA / visualization / valid-sample-count utility for the VIIRS branch.

Usage:
    python eda_viirs.py --count                 # just count valid/invalid
    python eda_viirs.py --eda                    # summary stats + histograms
    python eda_viirs.py --visualize 6             # plot 6 random sample triplets
    python eda_viirs.py --count --eda --visualize 6   # do all three

Read-only against Azure Blob Storage -- safe to run alongside enrich_manifests.py.
"""

import argparse
import io
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless VM -- write PNGs, don't try to open a window
import matplotlib.pyplot as plt
import rasterio
from azure.storage.blob import ContainerClient

CONTAINER_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset_gcp"
OUT_DIR = "eda_output"


def get_client():
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    return ContainerClient.from_connection_string(conn_str, CONTAINER_NAME)


def load_all_manifests(client, branch="viirs"):
    prefix = f"{OUTPUT_PREFIX}/{branch}_pairs/manifest/"
    names = [b.name for b in client.list_blobs(name_starts_with=prefix) if b.name.endswith(".json")]
    print(f"Found {len(names)} manifest blobs for branch '{branch}'. Downloading...")
    manifests = []
    for i, name in enumerate(names):
        try:
            content = client.get_blob_client(name).download_blob().readall()
            manifests.append(json.loads(content))
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        if (i + 1) % 5000 == 0:
            print(f"  ...{i+1}/{len(names)}")
    return manifests


def read_raster(client, blob_path):
    raw = client.get_blob_client(blob_path).download_blob().readall()
    with rasterio.MemoryFile(raw) as mf, mf.open() as src:
        return src.read().astype(np.float32)


# ------------------------------------------------------------------
# 1. Counting
# ------------------------------------------------------------------
def do_count(manifests):
    total = len(manifests)
    valid = [m for m in manifests if m.get("is_valid") is True]
    invalid = [m for m in manifests if m.get("is_valid") is False]
    unflagged = total - len(valid) - len(invalid)

    print(f"\n{'='*60}\nCOUNT\n{'='*60}")
    print(f"Total manifests:     {total}")
    print(f"Valid:               {len(valid)}")
    print(f"Invalid:             {len(invalid)}")
    print(f"Unflagged (no is_valid field yet): {unflagged}")

    if invalid:
        reasons = Counter(m.get("invalid_reason", "unknown") for m in invalid)
        print("\nInvalid reason breakdown:")
        for reason, count in reasons.most_common():
            print(f"  {count:6d}  {reason}")

    # Coverage by year, from the cluster_id's date prefix (YYYYMMDD)
    by_year = Counter()
    for m in valid:
        date = m.get("date", "")
        if len(date) >= 4:
            by_year[date[:4]] += 1
    if by_year:
        print("\nValid samples by year:")
        for year in sorted(by_year):
            print(f"  {year}: {by_year[year]}")


# ------------------------------------------------------------------
# 2. EDA plots
# ------------------------------------------------------------------
def do_eda(manifests):
    os.makedirs(OUT_DIR, exist_ok=True)
    valid = [m for m in manifests if m.get("is_valid") is True]
    if not valid:
        print("No valid samples to run EDA on.")
        return

    n_detections = [m.get("n_detections", 0) for m in valid]
    max_frp = [m.get("max_frp", 0) for m in valid]
    goes_status = Counter(m.get("goes_fetch_status", "unknown") for m in valid)
    lons = [m["center"][0] for m in valid if "center" in m]
    lats = [m["center"][1] for m in valid if "center" in m]
    by_year = Counter(m.get("date", "")[:4] for m in valid if m.get("date"))

    print(f"\n{'='*60}\nEDA SUMMARY\n{'='*60}")
    print(f"n_detections  — min={min(n_detections)}, median={int(np.median(n_detections))}, "
          f"max={max(n_detections)}, mean={np.mean(n_detections):.1f}")
    print(f"max_frp       — min={min(max_frp):.1f}, median={np.median(max_frp):.1f}, "
          f"max={max(max_frp):.1f}")
    print(f"GOES fetch status breakdown: {dict(goes_status)}")
    print(f"  -> {goes_status.get('no_scan', 0)}/{len(valid)} samples have an ALL-ZERO GOES "
          f"raster (no scan found within the temporal guard) -- these are structurally valid "
          f"but carry no real GOES signal. Worth knowing before training.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(n_detections, bins=50)
    axes[0, 0].set_title("VIIRS detections per cluster")
    axes[0, 0].set_yscale("log")

    axes[0, 1].hist(max_frp, bins=50)
    axes[0, 1].set_title("Max FRP per cluster")
    axes[0, 1].set_yscale("log")

    years_sorted = sorted(by_year)
    axes[0, 2].bar(years_sorted, [by_year[y] for y in years_sorted])
    axes[0, 2].set_title("Valid samples by year")
    axes[0, 2].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(list(goes_status.keys()), list(goes_status.values()))
    axes[1, 0].set_title("GOES fetch status")

    axes[1, 1].scatter(lons, lats, s=1, alpha=0.3)
    axes[1, 1].set_title("Spatial coverage (cluster centers)")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")

    axes[1, 2].axis("off")  # spare panel

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "eda_summary.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved EDA summary plot -> {out_path}")


# ------------------------------------------------------------------
# 3. Visualize sample triplets
# ------------------------------------------------------------------
def do_visualize(client, manifests, n):
    os.makedirs(OUT_DIR, exist_ok=True)
    valid = [m for m in manifests if m.get("is_valid") is True]
    if not valid:
        print("No valid samples to visualize.")
        return

    sample = random.sample(valid, min(n, len(valid)))

    for i, m in enumerate(sample):
        sid = m["id"]
        paths = m["paths"]
        try:
            target = read_raster(client, paths["target"])       # (1, H, W)
            goes = read_raster(client, paths["goes_input"])      # (3, H, W) -- C07, C14, TFI
            cov = read_raster(client, paths["covariates"])       # (2, H, W) -- DEM, landcover
        except Exception as e:
            print(f"  [SKIP] {sid}: could not read rasters ({e})")
            continue

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f"{sid}  |  n_detections={m.get('n_detections')}  "
                     f"max_frp={m.get('max_frp'):.1f}  goes_status={m.get('goes_fetch_status')}")

        im0 = axes[0].imshow(target[0], cmap="hot")
        axes[0].set_title("VIIRS target (FRP)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(goes[0], cmap="inferno")
        axes[1].set_title("GOES C07 (BT)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(goes[2], cmap="RdBu_r")
        axes[2].set_title("GOES TFI (C07-C14)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        im3 = axes[3].imshow(cov[0], cmap="terrain")
        axes[3].set_title("DEM (elevation)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(cov[1], cmap="tab20")
        axes[4].set_title("Landcover class")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{i:02d}_{sid}.png")
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  Saved -> {out_path}")

    print(f"\n{len(sample)} sample plots written to {OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="viirs", choices=["landsat", "viirs"])
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--eda", action="store_true")
    parser.add_argument("--visualize", type=int, default=0, metavar="N")
    args = parser.parse_args()

    if not (args.count or args.eda or args.visualize):
        print("Nothing to do -- pass --count, --eda, and/or --visualize N.")
        return

    client = get_client()
    manifests = load_all_manifests(client, args.branch)

    if args.count:
        do_count(manifests)
    if args.eda:
        do_eda(manifests)
    if args.visualize:
        do_visualize(client, manifests, args.visualize)


if __name__ == "__main__":
    main()
