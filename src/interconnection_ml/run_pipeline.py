from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from .data import ProjectColumns, load_or_generate
from .evaluate import make_eda_plots, make_result_plots, print_summary, write_metrics, write_model_comparison
from .features import split_dataset
from .models import train_cost_bucket_models, train_gpr_cost_model, train_timeline_models, train_withdrawal_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interconnection queue ML project pipeline.")
    parser.add_argument("--data", default=None, help="Optional cleaned public queue CSV.")
    parser.add_argument("--synthetic-rows", type=int, default=12000, help="Rows to generate when --data is omitted.")
    parser.add_argument("--output", default="outputs", help="Directory for metrics and plots.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_generate(args.data, synthetic_rows=args.synthetic_rows, seed=args.seed)
    _validate_finite_numeric_data(data)
    splits = split_dataset(data, seed=args.seed)

    make_eda_plots(data, output_dir)

    with warnings.catch_warnings():
        # macOS Accelerate can emit noisy BLAS RuntimeWarnings even after finite
        # input validation; fail on data issues above, keep model output readable.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        withdrawal = train_withdrawal_models(splits.train, splits.test, seed=args.seed)
        timeline = train_timeline_models(splits.train, splits.test, seed=args.seed)
        cost_bucket = train_cost_bucket_models(splits.train, splits.test, seed=args.seed)
        gpr = train_gpr_cost_model(splits.train, splits.test, seed=args.seed)

    metrics = {
        "dataset": {
            "rows": int(len(data)),
            "train_rows": int(len(splits.train)),
            "validation_rows": int(len(splits.validation)),
            "test_rows": int(len(splits.test)),
            "cost_labeled_rows": int(data["cost_bucket"].notna().sum()),
            "withdrawal_rate": float(data["withdrawn"].mean()),
        },
        "withdrawal": {name: result.metrics for name, result in withdrawal.items()},
        "timeline": {name: result.metrics for name, result in timeline.items()},
        "cost_bucket": {name: result.metrics for name, result in cost_bucket.items()},
        "gpr_uncertainty": gpr,
    }

    timeline_pred = None
    if "mlp_regressor" in timeline:
        timeline_pred = timeline["mlp_regressor"].predictions
    elif "ridge_regression" in timeline:
        timeline_pred = timeline["ridge_regression"].predictions

    make_result_plots(splits.test, timeline_pred, metrics, output_dir)
    write_metrics(metrics, output_dir)
    write_model_comparison(metrics, output_dir)
    print_summary(metrics)


def _validate_finite_numeric_data(data) -> None:
    required_numeric = list(ProjectColumns().numeric) + ["timeline_months", "withdrawn"]
    numeric = data[required_numeric]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Input data contains non-finite numeric values after preprocessing.")


if __name__ == "__main__":
    main()
