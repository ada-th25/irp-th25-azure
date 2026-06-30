# Wildfire Super-Resolution: Direct and Cascaded Spatial Downscaling

This repository contains the source code, data pipelines, and documentation for an Independent Research Project (IRP) focused on the high-resolution spatial downscaling of wildfire data down to 30m resolution.

The core objective is to evaluate and compare two main downscaling pathways: **Cascaded Downscaling** (1km $\rightarrow$ 375m $\rightarrow$ 30m) and **Direct Downscaling** (1km $\rightarrow$ 30m) to analyse spatial accuracy trade-offs using state-of-the-art remote sensing spatial downscaling architectures.

---

## Repository Structure

```text
├── deliverables/               # Project plan and report
│   ├── project_plan.pdf        # Initial project scope and timeline
│   └── final_report.pdf        # Completed IRP final report
├── environment.yml             # Conda environment definition with required dependencies
├── logbook/                    # Project tracking and meeting records
├── notebooks/                  # Project notes and exploratory ipynb notebooks
├── src/                        # Main project codebase and source modules
│   ├── data/                   # Data generation, preprocessing, loading and pipeline scripts
│   └── models/                 # Model architectures
│       └── haunet              # HAUNet
├── tests/                      # Test scripts
└── README.md                   # Project overview