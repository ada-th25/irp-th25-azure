# Wildfire Super-Resolution Pipeline

Deep learning pipeline for wildfire super-resolution downscaling (GOES → VIIRS → Landsat), developed for an MSc IRP with Artio.

## Structure
```text
├── src
│   ├── data/                                # Data scripts
│   │   ├── README.md                        # Data pipeline instructions
│   │   ├── generate_dataset.py.             # Main data pipeline
│   │   ├── generate_stage1_predictions.py.  # Generates & saves Stage-1 predictions for two-stage pipelines
│   │   ├── goes_local_fetch.py              # GOES-16/17 S3/GCS fetching from GCS bucket
│   │   ├── split_data.py                    # Data split generator
│   │   ├── validity.py                      # Quality control & raster validation checks
│   │   └── wildfire_dataset.py              # Multi-modal PyTorch Dataset implementation
│   ├── models/                              # Models
│   │   ├── autoencoder.py                   # Baseline autoencoder
│   │   └── haunet.py                        # HAUNet
│   └── utils/                               # Utilities
│       ├── check_class_balance.py           # Class distribution analysis
│       ├── compute_dem_stats.py             # Calculates DEM normalisation statistics
│       ├── compute_goes_stats.py            # Calculates GOES normalisation statistics
│       ├── enrich_manifests.py              # Appends validity flags to GCS JSON manifests
│       ├── plot_loss.py                     # Training loss curve visualisation
│       ├── plot_predictions.py              # Predictions visualisation script
│       └── postprocess.py                   # Raster post-processing and spatial ops
├── evaluate.py                              # Evaluation script
├── losses.py                                # Loss functions
├── metrics.py                               # Metrics functions
└── train.py                                 # Training script
```

## Pipeline order

To go from raw data sources to a trained and evaluated model, execute the following steps in sequence.

> **Note**: All data processing scripts interact directly with the Google Cloud Storage bucket:
> `gs://wildfire-irp-data-th25/processed_dataset_gcp/`

### 1. Build Dataset
`generate_dataset.py`:  build the dataset (GOES/VIIRS/Landsat/Covariates)

```bash
python src/data/generate_dataset.py <landsat|viirs|test_all>
```

### 2. Enrich Manifests with Validity Flags
`src/utils/enrich_manifests.py`:  validity filtering (writes is_valid/invalid_reason)

```bash
python -m src.utils.enrich_manifests --branch <landsat|viirs|both>
```

### 3. Data Splitting
`split_data.py`:  stratified 80/10/10 train/val/test split by fire size,
   filtered on is_valid

```bash
python src/data/split_data.py --branch <landsat|viirs|both> \
    [--max-samples N] [--suffix <name>] [--overwrite]
```

### 4. Compute Normalisation Statistics
`src/utils/compute_goes_stats.py` and `src/utils/compute_dem_stats`: per-channel GOES and DEM mean/std over train split to enter in experiments YAML file.

```bash
python -m src.utils.compute_goes_stats --branch <landsat|viirs> [--suffix <name>]
python -m src.utils.compute_dem_stats --branch <landsat|viirs> [--suffix <name>]
```

### 5. Model Training
`train.py`: Train model specified by a YAML configuration file.

```bash
python src/train.py --config experiments/<experiment_config>.YAML
```

### 6. Evaluation
`evaluate.py`: Evaluate model performance against the test split with metrics from metric script.

```bash
python src/evaluate.py --config experiments/<experiment_config>.YAML --checkpoint <path_to_model.pt>
```

## Key constants & spatial configurations

| Parameter | VIIRS Branch | Landsat Branch | GOES | DEM | Landcover|
| --- | --- | --- | --- | -- | -- |
| **Grid Scale (Resolution)** | 375m | 30m | 2000m | 30m | 10m |
| **Target Grid Size** | 20 × 20 pixels | 256 × 256 pixels | 4 x 4 pixels | 256 x 256 pixels | 256 x 256 pixels |
| **Target Extent** | 7,680m | 7,680m | 7,680m | 7,680m | 7,680m |
| **Input Shape** | `(1, 20, 20)` | `(1, 256, 256)` | `(3, 4, 4)` | `(1, 256, 256)` | `(10, 256, 256)` |

## Branches

- **viirs**: (for GOES -> VIIRS) target is binarized (target > 0), then either dilated (3x3 max-pool) or undilated
- **landsat** (for GOES -> Landsat or VIIRS -> Landsat): target returned as raw values, already binary, then either dilated (7x7 max-pool) or undilated.

## Configurations & Experiments

Experiments are managed using modular YAML files located under [`experiments/`](../experiments/). Each configuration file controls data pipelines, dataset transforms, model architecture variations, training hyper-parameters, loss functions, and evaluation/post-processing metrics.