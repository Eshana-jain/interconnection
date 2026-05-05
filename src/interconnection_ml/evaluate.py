from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from .data import COST_BUCKETS, ProjectColumns
from .features import feature_frame


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


def make_advanced_analysis_plots(
    data: pd.DataFrame,
    test: pd.DataFrame,
    withdrawal_results: dict,
    timeline_results: dict,
    cost_results: dict,
    output_dir: Path,
) -> None:
    make_queue_story_dashboard(data, output_dir)
    make_withdrawal_diagnostics(test, withdrawal_results, output_dir)
    make_withdrawal_risk_surface(test, withdrawal_results, output_dir)
    make_feature_importance_plot(withdrawal_results, output_dir)
    make_timeline_residual_plot(test, timeline_results, output_dir)
    make_cost_confusion_plot(test, cost_results, output_dir)
    write_analysis_brief(data, withdrawal_results, timeline_results, cost_results, output_dir)


def make_queue_story_dashboard(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    yearly = data.groupby("filing_year").agg(projects=("project_id", "count"), withdrawal_rate=("withdrawn", "mean"))
    axes[0, 0].bar(yearly.index, yearly["projects"], color="#4C78A8")
    axes[0, 0].set_title("Queue Volume Accelerates Over Time")
    axes[0, 0].set_xlabel("Filing year")
    axes[0, 0].set_ylabel("Project count")

    axes[0, 1].plot(yearly.index, yearly["withdrawal_rate"], marker="o", color="#E45756")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Withdrawal Risk by Filing Year")
    axes[0, 1].set_xlabel("Filing year")
    axes[0, 1].set_ylabel("Withdrawal rate")

    fuel_counts = data["fuel_type"].value_counts(normalize=True).head(8).sort_values()
    axes[1, 0].barh(fuel_counts.index, fuel_counts.values, color="#72B7B2")
    axes[1, 0].set_title("Resource Mix")
    axes[1, 0].set_xlabel("Share of queue records")

    coverage = data.assign(has_cost=data["cost_bucket"].notna()).groupby("filing_year")["has_cost"].mean()
    axes[1, 1].plot(coverage.index, coverage.values, marker="s", color="#F58518")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Cost Label Coverage by Year")
    axes[1, 1].set_xlabel("Filing year")
    axes[1, 1].set_ylabel("Share with cost label")

    fig.suptitle("Interconnection Queue Story Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "queue_story_dashboard.png", dpi=200)
    plt.close(fig)


def make_withdrawal_diagnostics(test: pd.DataFrame, withdrawal_results: dict, output_dir: Path) -> None:
    if not withdrawal_results:
        return
    y_true = test["withdrawn"].astype(int).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for name, result in withdrawal_results.items():
        if result.scores is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, result.scores)
        precision, recall, _ = precision_recall_curve(y_true, result.scores)
        axes[0, 0].plot(fpr, tpr, label=f"{_pretty_name(name)} AUC={auc(fpr, tpr):.2f}")
        axes[0, 1].plot(recall, precision, label=_pretty_name(name))
        frac_pos, mean_pred = calibration_curve(y_true, result.scores, n_bins=8, strategy="quantile")
        axes[1, 0].plot(mean_pred, frac_pos, marker="o", label=_pretty_name(name))

    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].legend()

    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1, 0].set_title("Probability Calibration")
    axes[1, 0].set_xlabel("Mean predicted withdrawal probability")
    axes[1, 0].set_ylabel("Observed withdrawal frequency")
    axes[1, 0].legend()

    best_name = _best_model_name(withdrawal_results, "auc_roc")
    cm = confusion_matrix(y_true, withdrawal_results[best_name].predictions)
    _plot_confusion_matrix(axes[1, 1], cm, ["Approved/Active", "Withdrawn"], f"Best Model Confusion Matrix\n{_pretty_name(best_name)}")

    fig.tight_layout()
    fig.savefig(output_dir / "withdrawal_model_diagnostics.png", dpi=200)
    plt.close(fig)


def make_withdrawal_risk_surface(test: pd.DataFrame, withdrawal_results: dict, output_dir: Path) -> None:
    if "logistic_regression" not in withdrawal_results:
        return
    model = withdrawal_results["logistic_regression"].model
    mw_grid = np.geomspace(max(test["mw_capacity"].min(), 2), test["mw_capacity"].quantile(0.98), 45)
    congestion_grid = np.linspace(0, test["substation_congestion_24mo"].quantile(0.98), 40)
    base = _representative_row(test)
    rows = []
    for congestion in congestion_grid:
        for mw in mw_grid:
            row = base.copy()
            row["mw_capacity"] = mw
            row["log_mw_capacity"] = np.log1p(mw)
            row["substation_congestion_24mo"] = congestion
            rows.append(row)
    grid = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        risk = model.predict_proba(feature_frame(grid))[:, 1].reshape(len(congestion_grid), len(mw_grid))

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.contourf(mw_grid, congestion_grid, risk, levels=18, cmap="magma")
    contour = ax.contour(mw_grid, congestion_grid, risk, levels=[0.25, 0.5, 0.75, 0.9], colors="white", linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Requested capacity (MW, log scale)")
    ax.set_ylabel("Prior 24-month substation congestion")
    ax.set_title("Withdrawal Risk Surface\nHolding other features at a representative project")
    fig.colorbar(mesh, ax=ax, label="Predicted withdrawal probability")
    fig.tight_layout()
    fig.savefig(output_dir / "withdrawal_risk_surface.png", dpi=220)
    plt.close(fig)


def make_feature_importance_plot(withdrawal_results: dict, output_dir: Path) -> None:
    if "logistic_regression" not in withdrawal_results:
        return
    pipe = withdrawal_results["logistic_regression"].model
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[-14:]
    labels = [_clean_feature_name(names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(coefs[order] >= 0, "#E45756", "#4C78A8")
    ax.barh(labels, coefs[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Interpretable Withdrawal Drivers\nLogistic regression coefficients")
    ax.set_xlabel("Coefficient: higher means greater withdrawal risk")
    fig.tight_layout()
    fig.savefig(output_dir / "withdrawal_feature_importance.png", dpi=220)
    plt.close(fig)


def make_timeline_residual_plot(test: pd.DataFrame, timeline_results: dict, output_dir: Path) -> None:
    if not timeline_results:
        return
    best_name = _best_model_name(timeline_results, "r2")
    pred = timeline_results[best_name].predictions
    residual = pred - test["timeline_months"].to_numpy(dtype=float)
    plot_data = test[["filing_year", "iso_region", "substation_congestion_24mo"]].copy()
    plot_data["residual_months"] = residual
    yearly = plot_data.groupby("filing_year")["residual_months"].agg(["mean", "median"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].plot(yearly.index, yearly["mean"], marker="o", label="Mean residual")
    axes[0].plot(yearly.index, yearly["median"], marker="s", label="Median residual")
    axes[0].set_title(f"Timeline Residual Drift\n{_pretty_name(best_name)}")
    axes[0].set_xlabel("Filing year")
    axes[0].set_ylabel("Predicted minus actual months")
    axes[0].legend()

    axes[1].scatter(plot_data["substation_congestion_24mo"], residual, alpha=0.35, s=18, color="#54A24B")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Timeline Errors vs Congestion")
    axes[1].set_xlabel("Prior 24-month substation congestion")
    axes[1].set_ylabel("Predicted minus actual months")

    fig.tight_layout()
    fig.savefig(output_dir / "timeline_residual_analysis.png", dpi=200)
    plt.close(fig)


def make_cost_confusion_plot(test: pd.DataFrame, cost_results: dict, output_dir: Path) -> None:
    if not cost_results:
        return
    test_cost = test.dropna(subset=["cost_bucket"]).reset_index(drop=True)
    if test_cost.empty:
        return
    mapping = {bucket: i for i, bucket in enumerate(COST_BUCKETS)}
    y_true = test_cost["cost_bucket"].map(mapping).astype(int).to_numpy()
    best_name = _best_model_name(cost_results, "weighted_f1")
    y_pred = cost_results[best_name].predictions
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(COST_BUCKETS)))

    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_confusion_matrix(ax, cm, COST_BUCKETS, f"Cost Bucket Confusion Matrix\n{_pretty_name(best_name)}")
    fig.tight_layout()
    fig.savefig(output_dir / "cost_bucket_confusion_matrix.png", dpi=220)
    plt.close(fig)


def write_analysis_brief(data: pd.DataFrame, withdrawal_results: dict, timeline_results: dict, cost_results: dict, output_dir: Path) -> None:
    best_withdrawal = _best_model_name(withdrawal_results, "auc_roc") if withdrawal_results else "n/a"
    best_timeline = _best_model_name(timeline_results, "r2") if timeline_results else "n/a"
    best_cost = _best_model_name(cost_results, "weighted_f1") if cost_results else "n/a"
    text = f"""# Generated Analysis Brief

This brief is generated by the pipeline so the narrative stays tied to the latest run.

- Dataset size: {len(data):,} projects.
- Cost-label coverage: {data['cost_bucket'].notna().mean():.1%}.
- Withdrawal rate: {data['withdrawn'].mean():.1%}.
- Best withdrawal model by AUC: {_pretty_name(best_withdrawal)}.
- Best timeline model by R2: {_pretty_name(best_timeline)}.
- Best cost model by weighted F1: {_pretty_name(best_cost)}.

Recommended high-impact figures for slides:

- `queue_story_dashboard.png`
- `withdrawal_risk_surface.png`
- `withdrawal_model_diagnostics.png`
- `withdrawal_feature_importance.png`
- `timeline_residual_analysis.png`
- `cost_bucket_confusion_matrix.png`
- `pinn_ablation_cost.png`
"""
    (output_dir / "analysis_brief.md").write_text(text, encoding="utf-8")


def _best_model_name(results: dict, metric: str) -> str:
    return max(results, key=lambda name: results[name].metrics.get(metric, -np.inf))


def _representative_row(data: pd.DataFrame) -> dict:
    columns = ProjectColumns()
    row = {}
    for column in columns.numeric:
        row[column] = float(data[column].median())
    for column in columns.categorical:
        row[column] = str(data[column].mode(dropna=True).iloc[0])
    row["mw_capacity"] = float(np.expm1(row["log_mw_capacity"]))
    return row


def _plot_confusion_matrix(ax, cm: np.ndarray, labels: list[str], title: str) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized share")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)


def _pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


def _clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


def print_summary(metrics: dict) -> None:
    print(json.dumps(metrics, indent=2))
