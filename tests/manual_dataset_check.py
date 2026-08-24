"""
manual_dataset_check.py
------------------------
Manual diagnostic script for validating the WildfireDataset pipeline
against real GCS data. NOT a pytest test file (requires live GCS access
and billing). Run manually when GCS access is available:
    python3 tests/manual_dataset_check.py
"""

import sys
import traceback
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data"))
from wildfire_dataset import WildfireDataset

BUCKET_NAME   = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset"

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors   = []
warnings = []

def check(condition, label, detail="", warn=False):
    tag = WARN if warn else (PASS if condition else FAIL)
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        if warn:
            warnings.append(label)
        else:
            errors.append(label)
    return condition


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset construction
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Dataset construction ===")

try:
    ds_ls = WildfireDataset(BUCKET_NAME, OUTPUT_PREFIX, branch="landsat")
    check(len(ds_ls) > 0, "Landsat dataset non-empty", f"{len(ds_ls)} samples")
except Exception as e:
    check(False, "Landsat dataset construction", str(e))
    ds_ls = None

try:
    ds_vr = WildfireDataset(BUCKET_NAME, OUTPUT_PREFIX, branch="viirs")
    check(len(ds_vr) > 0, "VIIRS dataset non-empty", f"{len(ds_vr)} samples")
except Exception as e:
    check(False, "VIIRS dataset construction", str(e))
    ds_vr = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tensor shapes
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Tensor shapes ===")

EXPECTED = {
    "landsat": {
        "goes":       (3, 15, 15),
        "target":     (1, 256, 256),
        "covariates": (2, 256, 256),
    },
    "viirs": {
        "goes":       (3, 15, 15),
        "target":     (1, 20, 20),
        "covariates": (2, 80, 80),
    },
}

for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        print(f"  Skipping {branch} — no samples")
        continue
    try:
        sample = ds[0]
        for key, expected_shape in EXPECTED[branch].items():
            tensor = sample[key]
            shape  = tuple(tensor.shape)
            check(
                shape == expected_shape,
                f"{branch}/{key} shape",
                f"got {shape}, expected {expected_shape}"
            )
    except Exception as e:
        check(False, f"{branch} __getitem__", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dtype and value range checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Dtype and value ranges ===")

for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        continue
    try:
        sample = ds[0]

        # All tensors should be float32
        for key in ["goes", "target", "covariates"]:
            check(
                sample[key].dtype == torch.float32,
                f"{branch}/{key} dtype float32",
                str(sample[key].dtype)
            )

        # GOES: raw int16 brightness temps scaled to float — expect values > 1000
        goes = sample["goes"]
        check(
            goes.max().item() > 200,
            f"{branch}/goes has non-trivial values (expect ~290-310K)",
            f"max={goes.max().item():.1f}"
        )
        check(
            not torch.isnan(goes).any(),
            f"{branch}/goes no NaNs"
        )
        check(
            not torch.isinf(goes).any(),
            f"{branch}/goes no Infs"
        )

        # Target: fire mask (Landsat 0/1) or FRP (VIIRS ≥ 0)
        target = sample["target"]
        check(
            target.min().item() >= 0,
            f"{branch}/target values ≥ 0",
            f"min={target.min().item():.4f}"
        )
        check(
            not torch.isnan(target).any(),
            f"{branch}/target no NaNs"
        )

        # Covariates: DEM can be negative (below sea level), land cover 0–100
        cov = sample["covariates"]
        check(
            not torch.isnan(cov).any(),
            f"{branch}/covariates no NaNs"
        )
        lc = cov[1]
        check(
            lc.max().item() <= 110,
            f"{branch}/land cover values in expected range",
            f"max={lc.max().item():.1f}",
            warn=True
        )

    except Exception as e:
        check(False, f"{branch} value checks", traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 4. DataLoader batch test
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. DataLoader batch test ===")

for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        continue
    try:
        loader = DataLoader(ds, batch_size=min(2, len(ds)), num_workers=0)
        batch  = next(iter(loader))

        exp = EXPECTED[branch]
        for key, expected_shape in exp.items():
            batch_shape = tuple(batch[key].shape)
            expected_batch = (min(2, len(ds)),) + expected_shape
            check(
                batch_shape == expected_batch,
                f"{branch} batch/{key} shape",
                f"got {batch_shape}, expected {expected_batch}"
            )

        check(
            isinstance(batch["sample_id"][0], str),
            f"{branch} batch/sample_id is string"
        )
        check(
            isinstance(batch["date"][0], str),
            f"{branch} batch/date is string"
        )

    except Exception as e:
        check(False, f"{branch} DataLoader batch", traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 5. Target sparsity distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. Target sparsity distribution ===")

for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        continue
    sparsities = []
    for i in range(len(ds)):
        target = ds[i]["target"].numpy()
        zero_frac = (target == 0).sum() / target.size
        sparsities.append(zero_frac)

    sparsities = np.array(sparsities)
    all_empty  = (sparsities == 1.0).sum()
    very_sparse = (sparsities > 0.95).sum()

    print(f"  {branch}: {len(ds)} samples")
    print(f"    mean sparsity   : {sparsities.mean()*100:.1f}%")
    print(f"    median sparsity : {np.median(sparsities)*100:.1f}%")
    print(f"    >95% empty      : {very_sparse}/{len(ds)}")
    print(f"    100% empty      : {all_empty}/{len(ds)}")

    check(
        all_empty < len(ds),
        f"{branch} not all targets 100% empty",
        f"{all_empty} fully empty samples"
    )
    check(
        all_empty / len(ds) < 0.5,
        f"{branch} <50% of targets are 100% empty",
        f"{all_empty/len(ds)*100:.0f}% fully empty",
        warn=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. GOES temporal alignment spot-check
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. GOES non-uniform (not all same value) ===")

for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        continue
    uniform_count = 0
    for i in range(len(ds)):
        goes = ds[i]["goes"].numpy()
        if goes.std() < 1.0:
            uniform_count += 1
    check(
        uniform_count == 0,
        f"{branch} all GOES images have spatial variation",
        f"{uniform_count} uniform/flat images found",
        warn=uniform_count > 0
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. No all-zero inputs or targets in any pair
# ══════════════════════════════════════════════════════════════════════════════
print("=== 7. No all-zero components in any pair ===")
 
for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    zero_goes, zero_target, zero_cov = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        if s["goes"].abs().max().item() == 0:
            zero_goes.append(s["sample_id"])
        if s["target"].abs().max().item() == 0:
            zero_target.append(s["sample_id"])
        if s["covariates"].abs().max().item() == 0:
            zero_cov.append(s["sample_id"])
 
    check(len(zero_goes) == 0,
          f"{branch} no all-zero GOES",
          f"{len(zero_goes)} all-zero: {zero_goes[:3]}")
    check(len(zero_target) == 0,
          f"{branch} no all-zero target",
          f"{len(zero_target)} all-zero: {zero_target[:3]}")
    check(len(zero_cov) == 0,
          f"{branch} no all-zero covariates",
          f"{len(zero_cov)} all-zero: {zero_cov[:3]}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8. Ground truth has sufficient active fire signal
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. Ground truth active fire signal ===")
 
# Landsat: binary mask, require at least 10 active pixels (same as pipeline filter)
# VIIRS:   FRP values, require at least 1 non-zero pixel
MIN_LANDSAT_PIXELS = 10
MIN_VIIRS_PIXELS   = 1
 
for branch, ds, min_px in [
    ("landsat", ds_ls, MIN_LANDSAT_PIXELS),
    ("viirs",   ds_vr, MIN_VIIRS_PIXELS),
]:
    low_signal = []
    pixel_counts = []
    for i in range(len(ds)):
        target = ds[i]["target"].numpy()
        n_active = (target > 0).sum()
        pixel_counts.append(int(n_active))
        if n_active < min_px:
            low_signal.append((ds[i]["sample_id"], int(n_active)))
 
    if pixel_counts:
        print(f"  {branch}: active pixel counts — "
              f"min={min(pixel_counts)} max={max(pixel_counts)} "
              f"mean={np.mean(pixel_counts):.1f}")
    else:
        print(f"  {branch}: no samples to report pixel counts for")
 
    check(
        len(low_signal) == 0,
        f"{branch} all targets have ≥{min_px} active fire pixels",
        f"{len(low_signal)} below threshold: {low_signal[:3]}",
        warn=True   # warn not error — sparse fires are valid, just flagged
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 9. GOES TFI hotspot overlaps with target fire region
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. GOES TFI / target spatial overlap ===")
 
# Strategy:
#   - Upsample GOES TFI (band 2, C07-C14) to target spatial size
#   - Identify top-k% hottest GOES pixels as "fire candidate" region
#   - Check that at least one target fire pixel falls within that region
#   - Since GOES covers 4x the extent, we only look at the central quarter
#
# This is a loose sanity check — misalignment is expected at GOES scale,
# but gross failures (GOES hotspot in opposite corner from fire) are flagged.
 
GOES_TFI_BAND    = 2   # band index in the (3,15,15) tensor: C07, C14, TFI
TOP_PCT          = 25  # top 25% of GOES TFI pixels = "warm region"
MIN_OVERLAP_PX   = 1   # at least 1 fire pixel must fall in the warm region
 
def upsample_goes_to_target(goes_tfi, target_shape):
    """Upsample GOES TFI (H,W) to target shape using nearest-neighbour."""
    h_ratio = target_shape[0] / goes_tfi.shape[0]
    w_ratio = target_shape[1] / goes_tfi.shape[1]
    return zoom(goes_tfi, (h_ratio, w_ratio), order=0)
 
for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    no_overlap = []
    skipped    = 0
 
    for i in range(len(ds)):
        s      = ds[i]
        target = s["target"].numpy()[0]       # (H, W)
        tfi    = s["goes"].numpy()[GOES_TFI_BAND]  # (15, 15)
 
        n_fire = (target > 0).sum()
        if n_fire == 0:
            skipped += 1
            continue
 
        # For Landsat: GOES covers 4x extent, so the target occupies the
        # central 1/4 of the GOES footprint spatially.
        # We crop GOES to the central quarter before upsampling.
        h, w = tfi.shape
        if branch == "landsat":
            # Central quarter = middle 50% in each dimension
            qh, qw = h // 4, w // 4
            tfi_crop = tfi[qh:h-qh, qw:w-qw]
        else:
            # VIIRS: GOES and target cover the same extent (ratio=1)
            tfi_crop = tfi
 
        tfi_up   = upsample_goes_to_target(tfi_crop, target.shape)
        threshold = np.percentile(tfi_up, 100 - TOP_PCT)
        warm_mask = tfi_up >= threshold
 
        overlap = int((warm_mask & (target > 0)).sum())
        if overlap < MIN_OVERLAP_PX:
            no_overlap.append((s["sample_id"], int(n_fire), overlap))
 
    total_checked = len(ds) - skipped
    check(
        len(no_overlap) == 0,
        f"{branch} GOES warm region overlaps target fire pixels "
        f"({total_checked} checked, {skipped} skipped — no fire pixels)",
        f"{len(no_overlap)} non-overlapping: {[x[0][:40] for x in no_overlap[:2]]}",
        warn=True  # warn not error — loose geometric check
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 10. GOES brightness temperature plausibility
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 10. GOES BT plausibility (physical range) ===")
 
# After scale factor (x0.01): C07 and C14 should be in ~200–350 K range
# Without scale factor (raw int16): values ~20000–35000
# We detect which case we're in and flag accordingly.
 
for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    if ds is None or len(ds) == 0:
        print(f"  Skipping {branch} — no samples")
        continue
    s     = ds[0]
    goes  = s["goes"].numpy()
    c07   = goes[0]
    c14   = goes[1]
 
    c07_mean = c07.mean()
    c14_mean = c14.mean()
 
    # Physical BT range in Kelvin: ~200K (cold clouds) to ~350K (hot surface)
    # If values are in range 200-400: scale factor applied correctly
    # If values are >1000: raw int16 values, scale factor not applied
    # If values are 0: empty/masked
 
    bt_plausible = 200 <= c07_mean <= 400 and 200 <= c14_mean <= 400
    raw_int16    = c07_mean > 1000
 
    check(
        bt_plausible,
        f"{branch}/goes BT in physical range (200–400 K)",
        f"C07 mean={c07_mean:.1f}, C14 mean={c14_mean:.1f}"
        + (" — looks like raw int16, scale factor (×0.01) not applied!" if raw_int16 else ""),
        warn=not bt_plausible
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 11. Covariate spatial diversity
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 11. Covariate spatial diversity ===")
 
for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    flat_dem, single_lc = [], []
 
    for i in range(len(ds)):
        cov = ds[i]["covariates"].numpy()
        dem = cov[0]
        lc  = cov[1]
 
        # DEM should have some elevation variation (std > 5m)
        if dem.std() < 5.0:
            flat_dem.append(ds[i]["sample_id"])
 
        # Land cover should have at least 2 distinct classes
        unique_classes = len(np.unique(np.round(lc[lc > 0])))
        if unique_classes < 2:
            single_lc.append(ds[i]["sample_id"])
 
    check(
        len(flat_dem) == 0,
        f"{branch} DEM has spatial variation (std > 5m)",
        f"{len(flat_dem)} flat DEMs",
        warn=True
    )
    check(
        len(single_lc) == 0,
        f"{branch} land cover has ≥2 classes per patch",
        f"{len(single_lc)} single-class patches",
        warn=True
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 12. Per-sample completeness summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 12. Per-sample completeness summary ===")
 
for branch, ds in [("landsat", ds_ls), ("viirs", ds_vr)]:
    print(f"\n  {branch} ({len(ds)} samples):")
    print(f"  {'ID':<55} {'GOES':>6} {'target':>8} {'cov':>6} {'fire_px':>8}")
    print(f"  {'-'*55} {'-'*6} {'-'*8} {'-'*6} {'-'*8}")
 
    all_complete = True
    for i in range(len(ds)):
        s      = ds[i]
        sid    = s["sample_id"][:52]
        goes   = s["goes"].numpy()
        target = s["target"].numpy()
        cov    = s["covariates"].numpy()
 
        goes_ok   = goes.max() > 0
        target_ok = target.max() > 0
        cov_ok    = cov.max() > 0
        fire_px   = int((target > 0).sum())
 
        g_str = "✓" if goes_ok   else "✗ EMPTY"
        t_str = "✓" if target_ok else "✗ EMPTY"
        c_str = "✓" if cov_ok    else "✗ EMPTY"
 
        print(f"  {sid:<55} {g_str:>6} {t_str:>8} {c_str:>6} {fire_px:>8}")
 
        if not (goes_ok and target_ok and cov_ok):
            all_complete = False
 
    check(all_complete, f"{branch} all pairs complete (no empty components)")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print(f"ERRORS   : {len(errors)}")
print(f"WARNINGS : {len(warnings)}")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
if warnings:
    print("\nWarnings:")
    for w in warnings:
        print(f"  ⚠ {w}")
if not errors:
    print("\n✓ Dataset is ready for training.")
else:
    print("\n✗ Fix errors before running full pipeline.")