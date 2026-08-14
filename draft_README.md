# 🔥 Wildfire Super-Resolution: Direct and Cascaded Spatial Downscaling

This repository contains the source code, data pipelines, and documentation for an Independent Research Project (IRP) focused on the high-resolution spatial downscaling of wildfire data down to 30m resolution.

The core objective is to evaluate and compare two main downscaling pathways: **Cascaded Downscaling** (1km $\rightarrow$ 375m $\rightarrow$ 30m) and **Direct Downscaling** (1km $\rightarrow$ 30m) to analyse spatial accuracy trade-offs with Hybrid Attention U-Net (HAUNet).

---

## 📊 Main Results & Summary

| Architecture | Pipeline | AUPRC | IOU | Boundary F1 | Pixel F1|
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Autoencoder** | Direct | *TBD* | *TBD* | *TBD* | *TBD* |
| **Autoencoder** | Cascaded | *TBD* | *TBD* | *TBD* | *TBD* |
| **HAUNet** | Direct | *TBD* | *TBD* | *TBD* | *TBD* |
| **HAUNet** | Cascaded | *TBD* | *TBD* | *TBD* | *TBD* |

* **Best Model Architecture:**
* **Primary Conclusion:**

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