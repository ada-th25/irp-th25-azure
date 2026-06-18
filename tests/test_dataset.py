"""
test_dataset.py
---------------
Validates the WildfireDataset and pipeline outputs before full-scale run.
Run from the repo root:
    python3 src/data/test_dataset.py
"""

import sys
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src/data")
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
            goes.max().item() > 100,
            f"{branch}/goes has non-trivial values",
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