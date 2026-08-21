# 🔥 Wildfire Super-Resolution: Direct and Cascaded Spatial Downscaling

This repository contains the source code, data pipelines, and documentation for an Independent Research Project (IRP) focused on the high-resolution spatial downscaling of wildfire data down to 30m resolution.

The core objective is to evaluate and compare two main downscaling pathways: **Cascaded Downscaling** (1km $\rightarrow$ 375m $\rightarrow$ 30m) and **Direct Downscaling** (1km $\rightarrow$ 30m) to analyse performance with Hybrid Attention U-Net (HAUNet) and autoencoder baseline.

> **⚠️ Academic Integrity Note:**
> This project is entirely Tsuyu Huang's (irp-th25) individual work. 
>In the repository's commit history, the username appears in two different formats:
>1. **Clickable Profile Name**: Commits made from personal setup (VSCode) where Tsuyu's GitHub account is linked.
>2. **Unclickable Text Name**: Commits pushed directly from a cloud-hosted Google Cloud Platform Instance / JupyterLab environment.
>
>Because the cloud environment used a default system email address instead of registered GitHub email, GitHub displays the author as unclickable plain text. Both sets of commits were authored exclusively by Tsuyu Huang across these two environments. No external collaborators were used. For any AI usage reference, check [references.md](references.md).

---

## 📊 Main Results & Summary

Best-performing configuration per architecture/pipeline group (by AUPRC), evaluated on Otsu-thresholded post-processed predictions at final Landsat (30m) resolution. See [`deliverables/tables/evaluation_table.tex`](deliverables/tables/evaluation_table.tex) for full results, and [`deliverables/tables/evaluation_summary_table.tex`](deliverables/tables/evaluation_summary_table.tex) for bootstrap confidence intervals.

| Architecture | Pipeline | AUPRC | IoU | Boundary F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Autoencoder** | Direct | 0.0120 | 0.0064 | 0.0050 |
| **Autoencoder** | Cascaded | 0.0106 | 0.0056 | 0.0040 |
| **HAUNet** | Direct | 0.0188 | 0.0044 | 0.0041 |
| **HAUNet** | Cascaded | 0.0142 | 0.0072 | 0.0071 |

*(Autoencoder Cascaded and HAUNet Direct use dilated-target configurations, consistent with the best-performing setup identified per architecture; see the report's Methods section for full ablation details.)*

* **Best Model Architecture:** No single architecture or pipeline is uniformly best. In the **direct** pipeline, HAUNet leads on AUPRC while the Autoencoder achieves better fixed-threshold spatial overlap (IoU, Precision). In the **cascaded** pipeline, HAUNet leads on every reported metric. Paired bootstrap tests (1,000 resamples, matched by patch) support consistent AUPRC advantage for HAUNet: it exceeds the Autoencoder in 84.2% of resamples in the direct pipeline (mean diff +0.0069, 95% CI [-0.0057, 0.0222]) and 87.1% in the cascaded pipeline (mean diff +0.0036, 95% CI [-0.0028, 0.0110]). Neither reaches conventional statistical significance given the limited test-set size (121 patches), but the consistency across both pipelines suggests architectural advantage. Model choice should therefore depend on whether ranking quality (AUPRC) or fixed-threshold spatial accuracy matters more.

* **Primary Conclusion:** All trained configurations clearly outperform naive All-Zero and Prevalence-Rate baselines, confirming the models learn genuine fire-related signal. The best AUPRC is roughly 30 times the theoretical random baseline. However, absolute performance remains modest across the board (AUPRC and IoU generally < 0.02), reflecting the difficulty of the extreme 2km to 30m resolution gap, severe class imbalance in the fire target, and small training dataset (1,202 total Landsat patches). Cascading through an intermediate VIIRS-resolution stage improves overlap-based metrics for HAUNet at some cost to AUPRC, with a smaller and less consistent effect for the Autoencoder. Area inflation analysis (predicted vs. true fire area) shows current models are better suited to coarse regional risk flagging than precise active fire prediction. These results should be read as a proof-of-concept demonstrating a working pipeline rather than a deployment-ready wildfire detection system.

## Repository Structure

```text
├── deliverables/               # Project plan and report
│   ├── project_plan.pdf        # Initial project scope and timeline
│   └── final_report.pdf        # Completed IRP final report
├── experiments/                # Model training configurations and hyperparameter settings (YAML)
├── requirements.txt
├── logbook/                    # Project tracking and meeting records
├── logs/                       # Summary of experiment evaluation results
├── notebooks/                  # Project notes and exploratory ipynb notebooks
├── src/                        # Main project codebase and source modules
│   ├── data/                   # Data generation, preprocessing, loading and pipeline scripts
│   ├── evaluate.py             # Evaluation script
│   ├── losses.py               # Loss script
│   ├── metrics.py              # Metrics script
│   ├── train.py                # Train script
│   ├── utils/                  # Utilities
│   └── models/                 # Model architectures
│       ├── autoencoder.py      # Autoencoder
│       └── haunet.py           # Hybrid Attention U-Net (HAUNet)
├── tests/                      # Test scripts
└── README.md
```

## Quickstart

### Environment Setup

```bash
# Clone the repository
git clone [https://github.com/ese-ada-lovelace-2025/irp-th25.git](https://github.com/ese-ada-lovelace-2025/irp-th25.git)
cd irp-th25

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Pipeline & Reproducibility

> **Note:** Dataset generation and preprocessing require access to the project's Google Cloud Storage bucket (`gs://wildfire-irp-data-th25/processed_dataset_gcp/`). Without GCS credentials, steps 1–4 below cannot be run. However, trained model checkpoints and evaluation logs (see [`logs/`](logs/)) are included in this repository, so evaluation and results reproduction (steps 5–6) can be run independently of raw data access.

Full instructions for reproducing the dataset from raw sources (GOES/VIIRS/Landsat/covariates), including manifest validation, train/val/test splitting, and normalisation statistics, are documented in [`src/README.md`](src/README.md). This covers the required pipeline order:

1. Build dataset (`generate_dataset.py`)
2. Enrich manifests with validity flags (`enrich_manifests.py`)
3. Split data (`split_data.py`)
4. Compute normalisation statistics (`compute_goes_stats.py`, `compute_dem_stats.py`)
5. Train model (`train.py`)
6. Evaluate (`evaluate.py`)

The Quickstart below assumes data has already been generated per the steps above; see `src/README.md` for full command details and required GCS bucket access.

## Running Experiments

See [`experiments/`](experiments/) to see full list of experiments, the configurations and hyperparameters for this project.

```bash
# Example:

# Train Direct HAUNet Model (1km -> 30m)
python src/train.py --config experiments/e1_haunet_direct.YAML 

# Train Cascaded HAUNet Model Stage 1 (1km -> 375m)
python src/train.py --config experiments/e2_haunet_cascaded_stage1.YAML

# Train Cascaded HAUNet Model Stage 2 (375m -> 30m)
python src/train.py --config experiments/e2_haunet_cascaded_stage2.YAML

# Evaluate Model
python src/evaluate.py \
        --config experiments/e1_haunet_direct.YAML \
        --checkpoint checkpoints/e1_haunet_direct/best.pt \
        --split test
```