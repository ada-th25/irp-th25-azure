"""
Usage: python -m src.utils.compute_goes_stats
"""

import torch
import fsspec
from src.data.wildfire_dataset import WildfireDataset

bucket_name = "wildfire-irp-data-th25"
prefix = "processed_dataset_gcp"
branch = "viirs"

# Read GCS split file directly, bypassing load_split_ids path constraints
split_path = f"gs://{bucket_name}/{prefix}/splits_500_samples_test/{branch}/train.txt"
print(f"Loading split IDs from: {split_path}")

with fsspec.open(split_path, "r") as f:
    train_ids = [line.strip() for line in f if line.strip()]

# Initialise dataset with the loaded IDs
ds = WildfireDataset(bucket_name, prefix, branch, split_ids=train_ids)

# Accumulate per-channel sum and sum-of-squares over the raw GOES tensor
n_pixels = 0
ch_sum = torch.zeros(3, dtype=torch.float64)
ch_sqsum = torch.zeros(3, dtype=torch.float64)

for i in range(len(ds)):
    goes = ds._read_gcs_raster(ds.samples[i]["paths"]["goes_input"]).double()  # (3, h, w) raw, pre-normalization
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