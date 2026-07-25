"""
Plot train/val loss curves from a training run's CSV log.

Handles ragged rows produced by CSVLogger when a run is resumed and a new
logger instance ends up appending rows with a different number of fields
than the file's original header (e.g. non-validation epochs interleaved
with validation epochs after a resume).

Usage:
    python plot_loss.py --csv logs_smoke/<exp_name>.csv
    python plot_loss.py --csv logs_smoke/<exp_name>.csv --out loss_curve.png
"""

import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Fixed key order used by log_row in train.py -- always epoch, train_loss, lr,
# then (on validation epochs only) the 8 val metrics in this exact order.
FULL_COLUMNS = [
    "epoch", "train_loss", "lr",
    "val_loss", "val_iou", "val_iou_t03", "val_ipsnr",
    "val_boundary_f1", "val_precision", "val_recall", "val_pixel_f1",
    "val_auprc", "val_centroid_dist_m",
]


def load_ragged_csv(path: str) -> pd.DataFrame:
    records = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip any header line(s) -- detect by non-numeric first field.
            try:
                float(row[0])
            except ValueError:
                continue
            n = len(row)
            if n <= len(FULL_COLUMNS):
                cols = FULL_COLUMNS[:n]
            else:
                print(f"[WARN] Row with {n} fields, more than expected {len(FULL_COLUMNS)}. "
                      f"Truncating: {row}")
                cols = FULL_COLUMNS
                row = row[:len(FULL_COLUMNS)]
            record = dict(zip(cols, row))
            records.append(record)

    df = pd.DataFrame(records)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
    return df


def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from training CSV log.")
    parser.add_argument("--csv", required=True, help="Path to the CSV log written by CSVLogger.")
    parser.add_argument("--out", default=None, help="Path to save the plot (e.g. loss_curve.png). "
                                                      "If omitted, just displays the plot.")
    args = parser.parse_args()

    df = load_ragged_csv(args.csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", color="tab:blue")

    if "val_loss" in df.columns:
        val_df = df.dropna(subset=["val_loss"])
        if not val_df.empty:
            ax.plot(val_df["epoch"], val_df["val_loss"], label="Val loss", color="tab:orange", marker="o")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()