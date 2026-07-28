# Data Pipeline

Run these steps **in order** to go from raw sources to a normalised, split, training-ready dataset. All scripts read/write GCS bucket `wildfire-irp-data-th25` under prefix `processed_dataset_gcp`.

## 1. Generate the dataset

```bash
python src/data/generate_dataset.py <landsat|viirs|test_all>
```

Builds the Landsat and/or VIIRS branches: fetches GOES context imagery, covariates (DEM/land cover), and target rasters, and writes a manifest JSON per sample to GCS. Resumable, already-complete pairs (target + GOES + covariates + manifest all present) are skipped on rerun.

## 2. Enrich manifests with validity flags

```bash
python -m src.utils.enrich_manifests --branch <landsat|viirs|both> [--dry-run]
```

Reads every manifest, runs `compute_validity` (`src/data/validity.py`) against the actual rasters, and writes `is_valid` / `invalid_reason` back into each manifest. **Must run before step 3**, `split_data.py` filters on `is_valid` and will silently produce empty splits without this step having run first.

## 3. Create train/val/test splits

```bash
python src/data/split_data.py --branch <landsat|viirs|both> \
    [--max-samples N] [--suffix <name>] [--overwrite]
```

Filters to valid manifests, then does a stratified 80/10/10 split by fire size (4 quantile bins), writing `train.txt` / `val.txt` / `test.txt` to `splits[_<suffix>]/<branch>/` in GCS. Use `--suffix` for smoke-test subsets (e.g. `--max-samples 500 --suffix 500_samples_test`).

## 4. Compute GOES normalisation stats

```bash
python -m src.utils.compute_goes_stats --branch <landsat|viirs> [--suffix <name>]
```

Loads `train.txt` for the VIIRS branch via `WildfireDataset` and computes per-channel GOES mean/std for normalisation at training time. The values can be used to define the GOES stats in the YAML files in `experiments/`.

## Supporting modules (not run directly)

- `src/data/validity.py`: validity-check logic used by `enrich_manifests.py`
- `src/data/wildfire_dataset.py`: `WildfireDataset`, used by training and by `compute_goes_stats.py`
- `src/data/goes_local_fetch.py`: GOES-16/17 local bucket fetch logic used by `generate_dataset.py`