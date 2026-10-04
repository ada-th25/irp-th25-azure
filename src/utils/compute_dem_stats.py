"""
Usage: python -m src.utils.compute_dem_stats --branch viirs --suffix 500_samples_test
"""
import argparse
import torch

from src.data.wildfire_dataset import WildfireDataset, load_split_ids

CONTAINER_NAME = "wildfire-irp-data-th25"
PREFIX = "processed_dataset_gcp"


def main():
    parser = argparse.ArgumentParser(description="Compute DEM normalization stats.")
    parser.add_argument("--branch", choices=["landsat", "viirs"], default="viirs",
                         help="Which branch's train split to compute stats over.")
    parser.add_argument("--suffix", default="",
                         help="Splits dir suffix, e.g. 500_samples_test "
                              "(matches --suffix passed to split_data.py). "
                              "Leave empty for the full 'splits' dir.")
    args = parser.parse_args()

    splits_dir = f"splits_{args.suffix}" if args.suffix else "splits"
    print(f"Loading split IDs from: {CONTAINER_NAME}/{PREFIX}/{splits_dir}/{args.branch}/train.txt")
    train_ids = load_split_ids(CONTAINER_NAME, args.branch, "train",
                                output_prefix=PREFIX, splits_dir=splits_dir)

    # Initialise dataset with the loaded IDs
    ds = WildfireDataset(CONTAINER_NAME, PREFIX, args.branch, split_ids=train_ids)

    # Accumulate sum and sum-of-squares over the DEM channel (cov[0]), in KM
    n_pixels = 0
    ch_sum = torch.zeros(1, dtype=torch.float64)
    ch_sqsum = torch.zeros(1, dtype=torch.float64)

    for i in range(len(ds)):
        cov = ds._read_azure_raster(ds.samples[i]["paths"]["covariates"]).double()  # (2, h, w) raw
        dem_km = cov[0:1] / 1000.0  # (1, h, w)
        ch_sum += dem_km.sum(dim=(1, 2))
        ch_sqsum += (dem_km ** 2).sum(dim=(1, 2))
        n_pixels += dem_km.shape[1] * dem_km.shape[2]
        if i % 500 == 0:
            print(f"{i}/{len(ds)}")

    mean = ch_sum / n_pixels
    var = ch_sqsum / n_pixels - mean ** 2
    std = var.sqrt()

    print("DEM mean (km):", mean.tolist())
    print("DEM std  (km):", std.tolist())


if __name__ == "__main__":
    main()
