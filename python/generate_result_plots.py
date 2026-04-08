"""
Generate two result graphs from saved training/conversion outputs.

Outputs:
- results/plot_training_curves.png
- results/plot_model_quality_size.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TRAIN_HISTORY = RESULTS_DIR / "training_history.csv"
TRAIN_METRICS = RESULTS_DIR / "training_metrics.json"
CONV_REPORT = RESULTS_DIR / "conversion_report.json"


def plot_training_curves():
    df = pd.read_csv(TRAIN_HISTORY)
    epochs = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, df["accuracy"], label="train_accuracy", linewidth=2)
    axes[0].plot(epochs, df["val_accuracy"], label="val_accuracy", linewidth=2)
    axes[0].set_title("Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, df["loss"], label="train_loss", linewidth=2)
    axes[1].plot(epochs, df["val_loss"], label="val_loss", linewidth=2)
    axes[1].set_title("Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = RESULTS_DIR / "plot_training_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_quality_and_size():
    with open(TRAIN_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(CONV_REPORT, "r", encoding="utf-8") as f:
        conv = json.load(f)

    class_report = metrics["classification_report"]
    labels = ["ON", "OFF", "DELAY"]
    f1_scores = [class_report[label]["f1-score"] for label in labels]

    model_size = conv["size_bytes"]
    target_size = conv["target_max_bytes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = axes[0].bar(labels, f1_scores, color=["#2E86AB", "#2A9D8F", "#F4A261"])
    axes[0].set_title("Per-Class F1 Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("F1 Score")
    axes[0].grid(axis="y", alpha=0.3)
    for b, score in zip(bars, f1_scores):
        axes[0].text(b.get_x() + b.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center")

    axes[1].bar(["Model", "Target"], [model_size, target_size], color=["#264653", "#E76F51"])
    axes[1].set_title("Model Size vs 2KB Target")
    axes[1].set_ylabel("Bytes")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].text(0, model_size + 35, f"{model_size} B", ha="center")
    axes[1].text(1, target_size + 35, f"{target_size} B", ha="center")

    fig.tight_layout()
    out = RESULTS_DIR / "plot_model_quality_size.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    missing = [p for p in [TRAIN_HISTORY, TRAIN_METRICS, CONV_REPORT] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required input files: {missing_str}")

    out1 = plot_training_curves()
    out2 = plot_quality_and_size()
    print(f"Generated: {out1}")
    print(f"Generated: {out2}")


if __name__ == "__main__":
    main()
