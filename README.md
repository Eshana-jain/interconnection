# ML Interconnection Queue Intelligence

This repository implements the CEE 498 MLC project proposed in `CEE498 Project Proposal Eshana Jain.pdf`: a machine-learning pipeline for early-stage interconnection queue risk assessment.

The project reproduces the central training idea from Raissi, Perdikaris, and Karniadakis (2019), *Physics-Informed Neural Networks*, by combining ordinary supervised loss with a domain-informed residual. In this civil infrastructure setting, the residual enforces the proposed monotonicity constraint that network-upgrade cost should not decrease as requested capacity (`MW`) increases, all else equal.

## What It Builds

- Withdrawal prediction with logistic regression and neural-network classifiers.
- Study timeline prediction with linear/Ridge and neural-network regressors.
- Network upgrade cost bucket prediction with an unconstrained MLP and a physics-informed monotonic MLP.
- Gaussian process regression on the cost subset to report calibrated 90% prediction intervals.
- Reproducible figures and metrics for the final report and presentation.
- Draft final-report and presentation scaffolding in `docs/`.

## Quick Start

The pipeline runs out of the box with a synthetic queue dataset that matches the fields described in the proposal/progress update. The default row count is set to the proposal-scale working set. If a cleaned public queue CSV or Excel file is available, pass it with `--data`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m interconnection_ml.run_pipeline --output outputs
```

If dependencies are already installed globally, the final command is enough:

```bash
PYTHONPATH=src python3 -m interconnection_ml.run_pipeline --output outputs
```

## Optional Real Data

Prepare a CSV or Excel file with these columns, using the same names where possible:

- `project_id`
- `iso_region`
- `fuel_type`
- `mw_capacity`
- `filing_year`
- `voltage_kv`
- `queue_position`
- `substation_id`
- `withdrawn`
- `timeline_months`
- `network_upgrade_cost_musd` or `cost_bucket`
- Optional context columns: `county`, `deliverability_status`, `interconnection_service`, `distance_to_transmission_miles`, and `renewable_penetration_pct`

Then run:

```bash
PYTHONPATH=src python3 -m interconnection_ml.run_pipeline --data path/to/clean_queue.csv --output outputs
```

The loader imputes missing queue-position and voltage fields, engineers congestion/cohort features, and scopes cost modeling to rows with cost labels, matching the progress report.

## Outputs

After a successful run, `outputs/` contains:

- `metrics.json`: all evaluation metrics.
- `model_comparison.csv`: summary table for report figures.
- `*.png`: EDA plots, model diagnostics, risk surfaces, feature importance, timeline residuals, cost confusion matrices, and PINN ablation plots.
- `analysis_brief.md`: generated narrative notes for slides/report writing.

## Project Structure

```text
src/interconnection_ml/
  data.py          Dataset loading, synthetic fallback, feature engineering
  features.py      Splits and preprocessing
  models.py        Baselines, GPR, MLPs, monotonic cost model
  evaluate.py      Metrics and plots
  run_pipeline.py  End-to-end command-line pipeline
docs/
  project_context.md
  final_report.md
  presentation_outline.md
```

## LLM Use Statement

Cursor/GPT was used to help with syntax completion as well as git commands during code writing as well as model training. The project logic, modeling scope, and scientific framing follow the provided course documents and proposal, and were largely ideated and written by Eshana Jain.
