"""
Bootleg Fire case-study plot: GOES input vs Stage1 (GOES->VIIRS) prediction
vs VIIRS ground truth (where a real overpass exists), with the ACR273
project boundary overlaid on every panel, clipped to the patch extent.

Deliberately does NOT go through WildfireDataset -- that class assumes the
training pipeline's manifest/split-ID contract, which this ad hoc case-study
data doesn't have. Instead this script loads GOES/covariate/VIIRS GeoTIFFs
directly from the case-study GCS prefix and reuses only the model-building
and checkpoint-loading logic from plot_predictions.py, since those need to
match training exactly (especially goes_mean/goes_std normalization).

Usage:
    python -m src.utils.plot_bootleg_case_study \
        --config experiments/e2_haunet_cascaded_stage1.YAML \
        --checkpoint gs://.../best_val_auprc.pt \
        --timestamp 20210710T2100 \
        --boundary src/data/acr273_boundary.geojson \
        --out bootleg_20210710T2100.png
"""

import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import torch
import torch.nn.functional as F
import yaml
import pandas as pd
from google.cloud import storage
from google.auth import default

# Ensure project root is in sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports from src
from src.utils.plot_predictions import build_model, resolve_checkpoint_path
from postprocess import postprocess_prediction

BUCKET_NAME = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "case_studies/bootleg_2021_dynamic"

# Use google-auth default credentials
credentials, project = default()
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket(BUCKET_NAME)


def _read_tif(blob_path):
    """Utility to read a GeoTIFF from GCS into memory and extract data/metadata."""
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    data = blob.download_as_bytes()
    with rasterio.MemoryFile(data) as memfile:
        with memfile.open() as src:
            arr = src.read()
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
    return arr, bounds, crs, transform


def _clip_boundary_to_patch(boundary_gdf, bounds, target_crs):
    from shapely.geometry import box
    if boundary_gdf is None or boundary_gdf.empty:
        return boundary_gdf
    if boundary_gdf.crs != target_crs:
        boundary_gdf = boundary_gdf.to_crs(target_crs)

    # Convert filled polygon(s) to their boundary LINE before clipping --
    # otherwise clipping a polygon that fully contains the patch just
    # returns the patch's own bounding box, not the actual project edge.
    line_gdf = boundary_gdf.copy()
    line_gdf["geometry"] = line_gdf.geometry.boundary

    patch_box = box(*bounds)
    clipped_gdf = line_gdf.clip(patch_box)
    return clipped_gdf


def _boundary_to_pixel_coords(clipped_gdf, transform):
    """Converts spatial LineString/MultiLineString geometries into image pixel coordinates."""
    if clipped_gdf is None or clipped_gdf.empty:
        return []

    lines = []
    inv_transform = ~transform  # Maps (x, y) spatial -> (col, row) matrix

    for geom in clipped_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        
        # Handle LineString, MultiLineString, or geometry collections
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for g in geoms:
            if hasattr(g, "coords"):
                coords = list(g.coords)
                pixel_coords = [inv_transform * (x, y) for x, y in coords]
                lines.append(pixel_coords)

    return lines


def _burned_area_ha(prob_or_mask_arr, pixel_area_m2, threshold=0.5):
    """Calculates hectares within a patch exceeding a threshold."""
    n_px = int(np.sum(prob_or_mask_arr >= threshold))
    return (n_px * pixel_area_m2) / 10_000.0  # m^2 -> ha


def downsample_to_viirs_grid(pred_256: np.ndarray, target_size=(20, 20)) -> np.ndarray:
    """
    Downsamples 256x256 probability maps to a 20x20 VIIRS-equivalent spatial grid
    using area-averaging to aggregate fire probability density accurately.
    """
    pred_t = torch.from_numpy(pred_256).float().unsqueeze(0).unsqueeze(0)
    pred_down = F.adaptive_avg_pool2d(pred_t, target_size)
    return pred_down[0, 0].numpy()


def load_case_study_sample(timestamp_str):
    """Loads GOES, covariates, and VIIRS (if available) for one case-study sample."""
    goes_path = f"{OUTPUT_PREFIX}/goes_hourly/GOES_{timestamp_str}.tif"
    goes_result = _read_tif(goes_path)
    if goes_result is None:
        raise FileNotFoundError(f"No GOES frame at gs://{BUCKET_NAME}/{goes_path}")
    goes_arr, goes_bounds, goes_crs, goes_transform = goes_result

    try:
        qa_df = pd.read_csv(f"gs://{BUCKET_NAME}/{OUTPUT_PREFIX}/goes_hourly_qa.csv")
    except Exception as e:
        raise FileNotFoundError(f"Could not read QA manifest from GCS: {e}")
        
    match = qa_df[qa_df["timestamp"].astype(str).str.replace("-", "").str.replace(":", "").str.contains(timestamp_str)]
    
    if match.empty or "center_lon" not in qa_df.columns:
        raise ValueError(f"Could not find center coordinates for {timestamp_str} in QA manifest.")
        
    lon, lat = float(match.iloc[0]["center_lon"]), float(match.iloc[0]["center_lat"])
    loc_str = f"lon{lon:.3f}_lat{lat:.3f}"
    cov_path = f"{OUTPUT_PREFIX}/covariates/COV_{loc_str}.tif"
    cov_result = _read_tif(cov_path)
    if cov_result is None:
        raise FileNotFoundError(f"No covariate tile at gs://{BUCKET_NAME}/{cov_path}")
    cov_arr, cov_bounds, cov_crs, cov_transform = cov_result

    viirs_result = _read_tif(f"{OUTPUT_PREFIX}/viirs_overpasses/VIIRS_{timestamp_str}.tif")

    viirs_bounds = viirs_result[1] if viirs_result else None

    # Print alignment check using local variables before return
    print(f"[ALIGNMENT CHECK] Timestamp: {timestamp_str}")
    print(f"  GOES Patch Bounds : {goes_bounds}")
    print(f"  VIIRS Patch Bounds: {viirs_bounds}")
    print(f"  Center Coordinates: Lon {lon:.3f}, Lat {lat:.3f}")

    if viirs_bounds is not None and goes_bounds != viirs_bounds:
        print("  ⚠️ ALIGNMENT MISMATCH DETECTED: GOES and VIIRS rasters were centered differently!")
    elif viirs_bounds is not None:
        print("  SUCCESS: GOES and VIIRS rasters share the exact same bounding box.")

    return {
        "goes": goes_arr, "goes_bounds": goes_bounds, "goes_crs": goes_crs, "goes_transform": goes_transform,
        "cov": cov_arr, "cov_bounds": cov_bounds, "cov_crs": cov_crs, "cov_transform": cov_transform,
        "viirs": viirs_result[0] if viirs_result else None,
        "viirs_bounds": viirs_bounds,
        "viirs_crs": viirs_result[2] if viirs_result else None,
        "viirs_transform": viirs_result[3] if viirs_result else None,
        "center_lon": lon, "center_lat": lat,
    }


LANDCOVER_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]

def _preprocess_covariates(cov_raw, dem_mean=0.3390881556710402, dem_std=0.5048759609237181):
    """Transforms raw 2-channel covariates into 11 processed channels."""
    cov_t = torch.from_numpy(cov_raw).float()

    dem = cov_t[0:1] / 1000.0
    dem = (dem - dem_mean) / (dem_std + 1e-6)

    lc_code = cov_t[1:2]
    lc_stacks = [(lc_code[0] == c).float() for c in LANDCOVER_CLASSES]
    lc_onehot = torch.stack(lc_stacks, dim=0)

    return torch.cat([dem, lc_onehot], dim=0)


def run_inference(model, goes_raw, cov_raw, goes_mean, goes_std, device, target_size=(256, 256)):
    """Executes model forward pass matching HAUNet shape expectations."""
    num_goes_ch = min(len(goes_raw), len(goes_mean))
    goes_raw = goes_raw[:num_goes_ch]
    
    goes_mean = torch.tensor(goes_mean[:num_goes_ch]).view(1, -1, 1, 1).to(device)
    goes_std  = torch.tensor(goes_std[:num_goes_ch]).view(1, -1, 1, 1).to(device)

    goes_t = torch.from_numpy(goes_raw).float().to(device).unsqueeze(0)
    goes_norm = (goes_t - goes_mean) / (goes_std + 1e-6)

    goes_up = F.interpolate(goes_norm, size=target_size, mode="bilinear", align_corners=False)

    cov_processed = _preprocess_covariates(cov_raw).to(device).unsqueeze(0)
    if cov_processed.shape[-2:] != target_size:
        cov_processed = F.interpolate(cov_processed, size=target_size, mode="bilinear", align_corners=False)

    needed_ch = model.intro.in_channels if hasattr(model, 'intro') else 14
    current_ch = goes_up.shape[1] + cov_processed.shape[1]

    if current_ch < needed_ch:
        stage1_dummy = torch.zeros((1, needed_ch - current_ch, target_size[0], target_size[1]), device=device)
        inp = torch.cat([goes_up, stage1_dummy, cov_processed], dim=1)
    else:
        inp = torch.cat([goes_up, cov_processed], dim=1)

    model.eval()
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)

    return probs[0, 0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--timestamp", required=True, help="e.g. 20210710T2000")
    p.add_argument("--boundary", required=True, help="Path to ACR273 boundary geojson")
    p.add_argument("--out", default="bootleg_case_study.png")
    p.add_argument("--threshold", type=float, default=0.5, help="Absolute confidence threshold")
    p.add_argument("--mode", choices=["continuous", "binary"], default="continuous")
    args = p.parse_args()

    if not args.timestamp.endswith("00"):
        print(f"[WARN] Input timestamp {args.timestamp} is sub-hourly. Adjusting to top-of-hour...")
        args.timestamp = args.timestamp[:-2] + "00"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket_name = cfg["data"]["bucket_name"]
    gcs_ckpt_prefix = cfg["checkpointing"]["gcs_checkpoint_prefix"]
    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint, bucket_name, gcs_ckpt_prefix)
    ckpt = torch.load(resolved_checkpoint, map_location=device)
    model_cfg = ckpt.get("config", cfg)

    print("[BOOTLEG] Building model and loading checkpoint...")
    model = build_model(model_cfg, args.config).to(device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[BOOTLEG] Loading data for timestamp {args.timestamp} from GCS...")
    try:
        sample = load_case_study_sample(args.timestamp)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Load project boundary once
    boundary_gdf = gpd.read_file(args.boundary) if os.path.exists(args.boundary) else None

    # --- Boundary Diagnostics ---
    if boundary_gdf is not None:
        print(f"\n[BOUNDARY DIAGNOSTIC]")
        print(f"  Boundary File: {args.boundary}")
        print(f"  Boundary Original CRS: {boundary_gdf.crs}")
        print(f"  Boundary Total Bounds: {boundary_gdf.total_bounds}")
        print(f"  Sample GOES Bounds: {sample['goes_bounds']}")
        print(f"  Sample GOES CRS: {sample['goes_crs']}")

        clipped_test = _clip_boundary_to_patch(boundary_gdf, sample["goes_bounds"], sample["goes_crs"])
        print(f"  Clipped Geometries Count: {len(clipped_test)}")
        if len(clipped_test) == 0:
            print("  ⚠️ WARNING: Boundary does NOT intersect this patch area!\n")
        else:
            print("  SUCCESS: Boundary intersects patch. Vectors will draw.\n")

    goes_mean = cfg["data"].get("goes_mean")
    goes_std = cfg["data"].get("goes_std")

    print(f"[BOOTLEG] Running model inference (Stage 1)...")
    pred_256 = run_inference(model, sample["goes"], sample["cov"], goes_mean, goes_std, device)

    # 1. Downsample predicted probabilities to 20x20 VIIRS grid
    pred_20_prob = downsample_to_viirs_grid(pred_256, target_size=(20, 20))

    if args.mode == "binary":
        pred_display = postprocess_prediction(
            torch.from_numpy(pred_20_prob).unsqueeze(0).unsqueeze(0),
            method="fixed",
            threshold=args.threshold,
            min_component_size=2,
        )[0, 0].numpy()
    else:
        pred_display = pred_20_prob

    # --- Setup Figure Canvas ---
    have_viirs = sample["viirs"] is not None
    n_panels = 3 if have_viirs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # --- Panel A: High-Contrast GOES TFI ---
    # tfi_raw = sample["goes"][0] - sample["goes"][1]
    # goes_thresh = np.percentile(tfi_raw, 90.0)
    # tfi_masked = np.where(tfi_raw >= goes_thresh, tfi_raw, np.nan)

    # cmap_goes = plt.colormaps["YlOrRd"].copy()
    # cmap_goes.set_bad("white")

    # im0 = axes[0].imshow(tfi_masked, cmap=cmap_goes, interpolation="nearest")
    # axes[0].set_title(f"GOES TFI (Active Thermal Front)\n@ {args.timestamp} (4x4)")
    # plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r"$\Delta T$ (K)")

    # if boundary_gdf is not None:
    #     boundary_gdf_goes = _clip_boundary_to_patch(boundary_gdf, sample["goes_bounds"], sample["goes_crs"])
    #     for line in _boundary_to_pixel_coords(boundary_gdf_goes, sample["goes_transform"]):
    #         xs, ys = zip(*line)
    #         axes[0].plot(xs, ys, color="cyan", linewidth=2.0, linestyle="--")
    # axes[0].axis("off")
    
    # --- Panel A: Raw GOES Thermal Image (No 90th Percentile Thresholding) ---
    # Assuming Channel 0 is shortwave IR / TFI feature:
    tfi_raw = sample["goes"][0]

    im0 = axes[0].imshow(tfi_raw, cmap="YlOrRd", interpolation="nearest")
    axes[0].set_title(f"GOES Input Band\n@ {args.timestamp}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Brightness Temp / Signal")

    if boundary_gdf is not None:
        boundary_gdf_goes = _clip_boundary_to_patch(boundary_gdf, sample["goes_bounds"], sample["goes_crs"])
        for line in _boundary_to_pixel_coords(boundary_gdf_goes, sample["goes_transform"]):
            xs, ys = zip(*line)
            axes[0].plot(xs, ys, color="cyan", linewidth=2.0, linestyle="--")
    axes[0].axis("off")

    # --- Panel B: Stage 1 Probability Map (20x20 Grid Space) ---
    pred_thresh = args.threshold
    pred_display = np.where(pred_20_prob >= pred_thresh, pred_20_prob, np.nan)

    cmap_pred = plt.colormaps["inferno"].copy()
    cmap_pred.set_bad("white")

    im1 = axes[1].imshow(pred_display, cmap=cmap_pred, vmin=pred_thresh, vmax=1.0, interpolation="nearest")
    axes[1].set_title(f"Stage-1 Fire Probability\n(Filtered: $P \\geq {pred_thresh}$)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="$P(\\text{Active Fire})$")

    # Match transform to the downsampled grid resolution (VIIRS space if present, else GOES space)
    grid_crs = sample["viirs_crs"] if have_viirs else sample["goes_crs"]
    grid_bounds = sample["viirs_bounds"] if have_viirs else sample["goes_bounds"]
    grid_transform = sample["viirs_transform"] if have_viirs else sample["goes_transform"]

    if boundary_gdf is not None:
        boundary_gdf_pred = _clip_boundary_to_patch(boundary_gdf, grid_bounds, grid_crs)
        for line in _boundary_to_pixel_coords(boundary_gdf_pred, grid_transform):
            xs, ys = zip(*line)
            axes[1].plot(xs, ys, color="cyan", linewidth=2.0, linestyle="--")
    axes[1].axis("off")

    pred_mask_binary = (pred_20_prob >= pred_thresh).astype(float)
    pred_area_ha = _burned_area_ha(pred_mask_binary, pixel_area_m2=(384.0 * 12.8) ** 2 / 400.0, threshold=pred_thresh)
    axes[1].text(0.02, -0.08, f"High-Confidence Area ($P \\geq {pred_thresh}$): {pred_area_ha:.0f} ha",
                transform=axes[1].transAxes, fontsize=10, fontweight="bold")

    # --- Panel C: VIIRS Ground Truth ---
    if have_viirs:
        viirs_raw = sample["viirs"][0]
        viirs_display = np.where(viirs_raw <= 0, np.nan, viirs_raw)

        cmap_viirs = plt.colormaps["Reds"].copy()
        cmap_viirs.set_bad("whitesmoke")

        im2 = axes[2].imshow(viirs_display, cmap=cmap_viirs, interpolation="nearest")
        axes[2].set_title(f"VIIRS Ground Truth\n(Overlaid with $P={pred_thresh}$ Contour)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Active FRP")

        axes[2].contour(pred_20_prob, levels=[pred_thresh], colors="blue", linewidths=2.0)

        if boundary_gdf is not None:
            boundary_gdf_viirs = _clip_boundary_to_patch(boundary_gdf, sample["viirs_bounds"], sample["viirs_crs"])
            for line in _boundary_to_pixel_coords(boundary_gdf_viirs, sample["viirs_transform"]):
                xs, ys = zip(*line)
                axes[2].plot(xs, ys, color="cyan", linewidth=2.0, linestyle="--")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[BOOTLEG] Success! Saved visual comparison to -> {args.out}")


if __name__ == "__main__":
    main()