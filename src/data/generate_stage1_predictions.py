"""
Runs a frozen stage-1 (GOES→VIIRS) checkpoint over every landsat-branch
sample and writes the resulting VIIRS-resolution prediction as a GCS
raster, then patches each sample's manifest JSON to point at it.

Usage:
    python src/data/generate_stage1_predictions.py \
        --config experiments/e2_haunet_cascaded_stage1_test.YAML \
        --stage1-ckpt checkpoints/e2_haunet_stage1/best.pt \
        --splits train val test
"""

import argparse
import io
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
import torch.nn.functional as F
import yaml
from google.cloud import storage

sys.path.append(str(Path(__file__).parent.parent))
from wildfire_dataset import WildfireDataset, load_split_ids
import importlib

MODEL_REGISTRY = {
    "haunet": "models.haunet",
    "autoencoder": "models.autoencoder",
}


def load_stage1_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model_name = cfg["model"]["name"]
    make_model = importlib.import_module(MODEL_REGISTRY[model_name]).make_model
    model = make_model(
        goes_channels=cfg["data"]["goes_channels"],
        cov_channels=cfg["data"]["cov_channels"],
        prior_fire_fraction=cfg["model"].get("prior_fire_fraction", 0.0085),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"  Loaded stage-1 model ({model_name}) from {ckpt_path}, "
          f"epoch {ckpt['epoch']}, best_{ckpt.get('best_metric','?')}={ckpt.get('best_score','?')}")
    return model, cfg, ckpt["epoch"], ckpt.get("best_metric", "?"), ckpt.get("best_scores", {})


def write_raster_to_gcs(bucket, gcs_path, array, ref_profile):
    """array: (1, H, W) float32. ref_profile: rasterio profile to copy geo metadata from."""
    profile = ref_profile.copy()
    profile.update(
        dtype="float32",
        count=array.shape[0],
        height=array.shape[1],
        width=array.shape[2],
        nodata=None,
    )
    buf = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(array.astype(np.float32))
        buf.write(memfile.read())
    buf.seek(0)
    bucket.blob(gcs_path).upload_from_file(buf, content_type="image/tiff")


def get_ref_profile(bucket, gcs_path):
    """Read geo profile off an existing raster (e.g. the VIIRS target) to stamp onto the prediction."""
    raw = bucket.blob(gcs_path).download_as_bytes()
    with MemoryFile(raw) as memfile:
        with memfile.open() as src:
            return src.profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Stage-1 experiment YAML (GOES->VIIRS), for dataset params.")
    parser.add_argument("--stage1-ckpt", required=True,
                        help="Path to trained stage-1 checkpoint (.pt), local or downloaded already.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to generate predictions for (need all of them for stage 2).")
    parser.add_argument("--output-key", default="stage1_pred",
                        help="Manifest 'paths' key to write, and GCS subfolder name.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--manifest-workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate rasters even if one already exists at the target GCS path. "
                             "Without this flag, existing rasters are skipped.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bucket_name = cfg["data"]["bucket_name"]
    output_prefix = cfg["data"]["output_prefix"]
    splits_dir = cfg["data"].get("splits_dir", "splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, stage1_cfg, stage1_epoch, stage1_best_metric, stage1_best_scores = load_stage1_model(args.stage1_ckpt, device)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    stage1_pred_prefix = f"{output_prefix}/landsat_pairs/{args.output_key}"

    for split in args.splits:
        print(f"\n{'='*60}\nSplit: {split}\n{'='*60}")
    
        split_ids = load_split_ids(bucket_name, "landsat", split, output_prefix, splits_dir)
    
        ds = WildfireDataset(
            bucket_name, output_prefix, branch="landsat", split_ids=split_ids,
            goes_mean=stage1_cfg["data"].get("goes_mean"),
            goes_std=stage1_cfg["data"].get("goes_std"),
            manifest_workers=args.manifest_workers,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=stage1_cfg["data"].get("num_workers", 4),
        )
        sample_lookup = {m["id"]: m for m in ds.samples}
        n_written = 0
    
        with torch.no_grad():
            for batch in loader:
                inp = batch["input"].to(device)
                sample_ids = batch["sample_id"]

                logits = model(inp)                    # -> (B, 1, 20, 20)
                probs = torch.sigmoid(logits).cpu().numpy()

                for i, sid in enumerate(sample_ids):
                    gcs_path = f"{stage1_pred_prefix}/{sid}.tif"
                    if not args.overwrite and bucket.blob(gcs_path).exists():
                        continue
                    pred = probs[i]
                    ref_path = sample_lookup[sid]["paths"]["target"]
                    ref_profile = get_ref_profile(bucket, ref_path)
                    write_raster_to_gcs(bucket, gcs_path, pred, ref_profile)
                    n_written += 1

        print(f"  Wrote {n_written} stage-1 prediction rasters to "
              f"gs://{bucket_name}/{stage1_pred_prefix}/")

        # --------------------------------------------------------------
        # Patch landsat-branch manifests to reference the new raster
        # --------------------------------------------------------------
        landsat_manifest_prefix = f"{output_prefix}/landsat_pairs/manifest/"
        patched, missing = 0, 0

        def patch_one(sid):
            blob = bucket.blob(f"{landsat_manifest_prefix}{sid}.json")
            if not blob.exists():
                return "missing"
            manifest = json.loads(blob.download_as_text())
            if sid in sample_lookup:
                manifest["paths"][args.output_key] = f"{stage1_pred_prefix}/{sid}.tif"
                manifest.setdefault("stage1_meta", {})[args.output_key] = {
                    "checkpoint": str(args.stage1_ckpt),
                    "epoch": stage1_epoch,
                    "best_metric": stage1_best_metric,
                }
            elif args.output_key in manifest["paths"]:
                del manifest["paths"][args.output_key]
                manifest.get("stage1_meta", {}).pop(args.output_key, None)
            else:
                return "skipped"
            blob.upload_from_string(json.dumps(manifest), content_type="application/json")
            return "ok"

        with ThreadPoolExecutor(max_workers=args.manifest_workers) as ex:
            futures = {ex.submit(patch_one, sid): sid for sid in split_ids}
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    patched += 1
                elif result == "missing":
                    missing += 1

        skipped_incomplete = len(split_ids) - len(sample_lookup)
        print(f"  Patched {patched} landsat manifests with '{args.output_key}' path "
              f"({missing} landsat manifests not found; "
              f"{skipped_incomplete} split IDs had no stage1 prediction generated — "
              f"incomplete/invalid at landsat level — and were skipped or cleaned).")

    print("\nDone. Can now train stage 2 with data.use_stage1_pred: true "
          f"and manifest key '{args.output_key}'.")


if __name__ == "__main__":
    main()