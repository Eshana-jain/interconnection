from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_model_comparison(metrics: dict, output_dir: Path) -> None:
    rows = []
    for task, task_metrics in metrics.items():
        if isinstance(task_metrics, dict):
            for model, values in task_metrics.items():
                if isinstance(values, dict):
                    row = {"task": task, "model": model}
                    row.update(values)
                    rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison.csv", index=False)


def make_eda_plots(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    data["withdrawn"].map({0: "Approved/Active", 1: "Withdrawn"}).value_counts(normalize=True).sort_index().plot(kind="bar", ax=ax)
    ax.set_ylabel("Share of records")
    ax.set_title("Withdrawal Class Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "withdrawal_class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    data.groupby("filing_year").size().plot(kind="bar", ax=ax)
    ax.set_ylabel("Project count")
    ax.set_title("Queue Entries by Filing Year")
    fig.tight_layout()
    fig.savefig(output_dir / "filing_year_volume.png", dpi=180)
    plt.close(fig)

    cost = data.dropna(subset=["cost_bucket"])
    if not cost.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        cost["cost_bucket"].value_counts().reindex(["0-1M", "1-5M", "5-20M", "20M+"]).plot(kind="bar", ax=ax)
        ax.set_ylabel("Labeled records")
        ax.set_title("Network Upgrade Cost Buckets")
        fig.tight_layout()
        fig.savefig(output_dir / "cost_bucket_distribution.png", dpi=180)
        plt.close(fig)


def make_result_plots(test: pd.DataFrame, timeline_predictions: np.ndarray | None, metrics: dict, output_dir: Path) -> None:
    if timeline_predictions is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        actual = test["timeline_months"].to_numpy(dtype=float)
        ax.scatter(actual, timeline_predictions, alpha=0.55, s=18)
        lims = [min(actual.min(), timeline_predictions.min()), max(actual.max(), timeline_predictions.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel("Actual timeline (months)")
        ax.set_ylabel("Predicted timeline (months)")
        ax.set_title("Timeline Prediction")
        fig.tight_layout()
        fig.savefig(output_dir / "timeline_predicted_vs_actual.png", dpi=180)
        plt.close(fig)

    cost_metrics = metrics.get("cost_bucket", {})
    if "cost_mlp_unconstrained" in cost_metrics and "cost_mlp_pinn_monotonic" in cost_metrics:
        labels = ["Unconstrained MLP", "PINN monotonic MLP"]
        f1 = [
            cost_metrics["cost_mlp_unconstrained"]["weighted_f1"],
            cost_metrics["cost_mlp_pinn_monotonic"]["weighted_f1"],
        ]
        violations = [
            cost_metrics["cost_mlp_unconstrained"]["implausibility_rate"],
            cost_metrics["cost_mlp_pinn_monotonic"]["implausibility_rate"],
        ]
        x = np.arange(len(labels))
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.bar(x - 0.18, f1, width=0.36, label="Weighted F1")
        ax1.set_ylabel("Weighted F1")
        ax1.set_ylim(0, 1)
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, violations, width=0.36, color="tab:red", label="Violation rate")
        ax2.set_ylabel("Monotonic violation rate")
        ax2.set_ylim(0, 1)
        ax1.set_xticks(x, labels, rotation=10)
        ax1.set_title("PINN Ablation: Accuracy vs Physical Plausibility")
        fig.tight_layout()
        fig.savefig(output_dir / "pinn_ablation_cost.png", dpi=180)
        plt.close(fig)


def print_summary(metrics: dict) -> None:
    print(json.dumps(metrics, indent=2))
