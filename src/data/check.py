import os
import io
import json
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from azure.storage.blob import ContainerClient

conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container_name = "wildfire-irp-data-th25"
client = ContainerClient.from_connection_string(conn_str, container_name)

# Pick a cluster_id to inspect
CLUSTER_ID = "20241019T1030_372_611"  # replace with any completed cluster ID

paths = {
    "Target (VIIRS)": f"processed_dataset_mtg/viirs_pairs/targets/VIIRS_{CLUSTER_ID}.tif",
    "MTG Context": f"processed_dataset_mtg/viirs_pairs/mtg_input/MTG_{CLUSTER_ID}.tif",
    "Covariates": f"processed_dataset_mtg/viirs_pairs/covariates/COV_{CLUSTER_ID}.tif",
}

manifest_path = f"processed_dataset_mtg/viirs_pairs/manifest/{CLUSTER_ID}.json"

# 1. Print Manifest Metadata
print("=== MANIFEST METADATA ===")
manifest_blob = client.get_blob_client(manifest_path)
if manifest_blob.exists():
    data = json.loads(manifest_blob.download_blob().readall())
    print(json.dumps(data, indent=2))
else:
    print(f"Manifest missing for {CLUSTER_ID}")

# 2. Plot Rasters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (label, blob_path) in zip(axes, paths.items()):
    b_client = client.get_blob_client(blob_path)
    if not b_client.exists():
        ax.set_title(f"{label}\n(NOT FOUND)")
        continue
    
    stream = io.BytesIO(b_client.download_blob().readall())
    with rasterio.open(stream) as src:
        arr = src.read(1)
        ax.set_title(f"{label}\nShape: {arr.shape}\nCRS: {src.crs.to_epsg()}")
        im = ax.imshow(arr, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.savefig(f"/var/tmp/check_{CLUSTER_ID}.png")
print(f"\nSaved visualization plot to /var/tmp/check_{CLUSTER_ID}.png")
