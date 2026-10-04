"""
Usage: python -m src.utils.compute_goes_stats --branch viirs --suffix 500_samples_test
"""
import argparse
import torch

from src.data.wildfire_dataset import WildfireDataset, load_split_ids

CONTAINER_NAME = "wildfire-irp-data-th25"
PREFIX = "processed_dataset_gcp"


def main():
    parser = argparse.ArgumentParser(description="Compute per-channel GOES normalization stats.")
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

    # Accumulate per-channel sum and sum-of-squares over the raw GOES tensor
    n_pixels = 0
    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sqsum = torch.zeros(3, dtype=torch.float64)

    for i in range(len(ds)):
        goes = ds._read_azure_raster(ds.samples[i]["paths"]["goes_input"]).double()  # (3, h, w) raw, pre-normalization
        ch_sum += goes.sum(dim=(1, 2))
        ch_sqsum += (goes ** 2).sum(dim=(1, 2))
        n_pixels += goes.shape[1] * goes.shape[2]
        if i % 500 == 0:
            print(f"{i}/{len(ds)}")

    mean = ch_sum / n_pixels
    var = ch_sqsum / n_pixels - mean ** 2
    std = var.sqrt()

    print("GOES channel mean:", mean.tolist())
    print("GOES channel std :", std.tolist())


if __name__ == "__main__":
    main()
