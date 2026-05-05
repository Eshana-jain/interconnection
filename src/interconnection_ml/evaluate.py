from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    make_pinn_ablation_deep_dive(test, cost_results, output_dir)
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


def make_pinn_ablation_deep_dive(test: pd.DataFrame, cost_results: dict, output_dir: Path) -> None:
    required = ["cost_mlp_unconstrained", "cost_mlp_pinn_monotonic"]
    if not all(name in cost_results for name in required):
        return

    test_cost = test.dropna(subset=["cost_bucket"]).reset_index(drop=True)
    if test_cost.empty:
        return

    labels = ["Unconstrained\nMLP", "PINN Monotonic\nMLP"]
    model_names = ["cost_mlp_unconstrained", "cost_mlp_pinn_monotonic"]
    metrics = [cost_results[name].metrics for name in model_names]
    y_true = test_cost["cost_bucket"].map({bucket: i for i, bucket in enumerate(COST_BUCKETS)}).astype(int).to_numpy()
    shifts = [_expected_cost_shift(cost_results[name].model, test_cost) for name in model_names]

    fig = plt.figure(figsize=(14, 9), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_shift = fig.add_subplot(grid[0, 1])
    ax_cm_unconstrained = fig.add_subplot(grid[1, 0])
    ax_cm_pinn = fig.add_subplot(grid[1, 1])

    x = np.arange(len(labels))
    width = 0.32
    weighted_f1 = [m["weighted_f1"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]
    violations = [m["implausibility_rate"] for m in metrics]
    ax_metrics.bar(x - width / 2, weighted_f1, width, label="Weighted F1", color="#4C78A8")
    ax_metrics.bar(x + width / 2, accuracy, width, label="Accuracy", color="#72B7B2")
    for i, value in enumerate(weighted_f1):
        ax_metrics.text(i - width / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    for i, value in enumerate(accuracy):
        ax_metrics.text(i + width / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    ax_metrics.set_xticks(x, labels)
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_title("Predictive Performance")
    ax_metrics.set_ylabel("Held-out score")
    ax_metrics.legend(frameon=False, loc="lower right")

    parts = ax_shift.violinplot(shifts, positions=x, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#F58518", "#B83280"]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#1f2937")
    ax_shift.axhline(0, color="black", linewidth=1, linestyle="--")
    ax_shift.set_xticks(x, labels)
    ax_shift.set_title("Monotonic Stress Test")
    ax_shift.set_ylabel("E[cost bucket | higher MW] - E[cost bucket | base]")
    ax_shift.text(
        0.5,
        0.03,
        "Values below zero are physically implausible",
        transform=ax_shift.transAxes,
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    ax_shift.text(0, np.percentile(shifts[0], 88), f"viol.={violations[0]:.3f}", ha="center", fontsize=10, color="#7c2d12")
    ax_shift.text(1, np.percentile(shifts[1], 88), f"viol.={violations[1]:.3f}", ha="center", fontsize=10, color="#831843")

    for ax, name, title in [
        (ax_cm_unconstrained, "cost_mlp_unconstrained", "Unconstrained MLP Errors"),
        (ax_cm_pinn, "cost_mlp_pinn_monotonic", "PINN Monotonic MLP Errors"),
    ]:
        cm = confusion_matrix(y_true, cost_results[name].predictions, labels=np.arange(len(COST_BUCKETS)))
        _plot_confusion_matrix(ax, cm, COST_BUCKETS, title, colorbar=False)

    fig.suptitle("PINN Ablation Deep Dive: Constrained vs Unconstrained Cost Model", fontsize=17, fontweight="bold", color="#17324d")
    fig.text(
        0.5,
        0.925,
        "Same architecture, same data; the constrained version adds a monotonic residual evaluated on higher-MW collocation copies.",
        ha="center",
        fontsize=11.5,
        color="#475569",
    )
    fig.savefig(output_dir / "pinn_ablation_deep_dive.png", dpi=220)
    plt.close(fig)


def _expected_cost_shift(model_bundle: dict, raw: pd.DataFrame) -> np.ndarray:
    preprocessor = model_bundle["preprocessor"]
    network = model_bundle["network"]
    high = raw.copy()
    high["mw_capacity"] = high["mw_capacity"] * 1.35 + 5.0
    high["log_mw_capacity"] = np.log1p(high["mw_capacity"])
    x_low = torch.tensor(preprocessor.transform(feature_frame(raw)).astype(np.float32), dtype=torch.float32)
    x_high = torch.tensor(preprocessor.transform(feature_frame(high)).astype(np.float32), dtype=torch.float32)
    expected = torch.arange(len(COST_BUCKETS), dtype=torch.float32)
    network.eval()
    with torch.no_grad():
        low_score = torch.softmax(network(x_low), dim=1) @ expected
        high_score = torch.softmax(network(x_high), dim=1) @ expected
    return (high_score - low_score).cpu().numpy()


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
- `pinn_ablation_deep_dive.png`
"""
    (output_dir / "analysis_brief.md").write_text(text, encoding="utf-8")


def make_gpr_uncertainty_plot(gpr_results, output_dir: Path) -> None:
    if not getattr(gpr_results, "metrics", None):
        return
    y_true = gpr_results.y_true_log
    mean = gpr_results.mean_log
    lower = gpr_results.lower_log
    upper = gpr_results.upper_log
    std = gpr_results.std_log
    covered = (y_true >= lower) & (y_true <= upper)

    order = np.argsort(mean)
    display_n = min(160, len(order))
    display_idx = order[np.linspace(0, len(order) - 1, display_n).astype(int)]
    x = np.arange(display_n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})
    fig.patch.set_facecolor("#f8fafc")

    axes[0].fill_between(x, lower[display_idx], upper[display_idx], color="#9ecae9", alpha=0.42, label="90% prediction interval")
    axes[0].plot(x, mean[display_idx], color="#1f5f99", linewidth=2, label="GPR mean")
    axes[0].scatter(x, y_true[display_idx], c=np.where(covered[display_idx], "#2f855a", "#c53030"), s=16, alpha=0.85, label="Actual log cost")
    axes[0].set_title("Sorted Cost Predictions with 90% Intervals")
    axes[0].set_xlabel("Held-out cost-labeled projects, sorted by predicted cost")
    axes[0].set_ylabel("log(1 + upgrade cost in $M)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].scatter(mean, y_true, c=std, cmap="viridis", s=24, alpha=0.72)
    lims = [min(mean.min(), y_true.min()), max(mean.max(), y_true.max())]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_title("Predicted vs Actual Cost")
    axes[1].set_xlabel("Predicted log cost")
    axes[1].set_ylabel("Actual log cost")
    cbar = fig.colorbar(axes[1].collections[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Predictive std.")

    bins = pd.qcut(std, q=min(6, len(np.unique(std))), duplicates="drop")
    calib = pd.DataFrame({"std": std, "covered": covered, "bin": bins}).groupby("bin", observed=False).agg(mean_std=("std", "mean"), coverage=("covered", "mean"))
    axes[2].bar(np.arange(len(calib)), calib["coverage"], color="#72B7B2")
    axes[2].axhline(0.90, color="#c53030", linestyle="--", linewidth=1.5, label="Target 90%")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(np.arange(len(calib)), [f"{v:.2f}" for v in calib["mean_std"]], rotation=35)
    axes[2].set_title("Coverage by Uncertainty Level")
    axes[2].set_xlabel("Mean predictive std. in bin")
    axes[2].set_ylabel("Empirical interval coverage")
    axes[2].legend(frameon=False)

    metrics = gpr_results.metrics
    fig.suptitle(
        f"Gaussian Process Cost Uncertainty: PICP={metrics['picp_90']:.3f}, R2={metrics['r2_log_cost']:.3f}, MAE={metrics['mae_log_cost']:.3f}",
        fontsize=15,
        fontweight="bold",
        color="#17324d",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_dir / "gpr_cost_uncertainty_diagnostics.png", dpi=220)
    plt.close(fig)


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


def _plot_confusion_matrix(ax, cm: np.ndarray, labels: list[str], title: str, colorbar: bool = True) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    if colorbar:
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
