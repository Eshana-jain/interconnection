# ML Interconnection Queue Intelligence

This repository implements the CEE 498 MLC project proposed in `CEE498 Project Proposal Eshana Jain.pdf`: a machine-learning pipeline for early-stage interconnection queue risk assessment.

The project reproduces the central training idea from Raissi, Perdikaris, and Karniadakis (2019), *Physics-Informed Neural Networks*, by combining ordinary supervised loss with a domain-informed residual. In this civil infrastructure setting, the residual enforces the proposed monotonicity constraint that network-upgrade cost should not decrease as requested capacity (`MW`) increases, all else equal.

## What It Builds

- Withdrawal prediction with logistic regression and neural-network classifiers.
- Queue timeline prediction with Ridge and neural-network regressors.
- Reproducible figures, model metrics, and comparison tables.
- Optional cost-model code is included, but the public LBNL workbook committed here does not include project-level network-upgrade cost labels, so cost models are skipped in the default real-data run.

## Data Source

The default training data is real public interconnection queue data, not generated sample data:

- Raw workbook: `data/lbnl_ix_queue_data_file_thru2024_v2.xlsx`
- Cleaned model input: `data/lbnl_queue_cleaned.csv`
- Source: Lawrence Berkeley National Laboratory / Interconnection.fyi, *Queued Up: 2025 Edition*
- Raw workbook URL: <https://eta-publications.lbl.gov/sites/default/files/2025-08/lbnl_ix_queue_data_file_thru2024_v2.xlsx>

The raw workbook contains 36,441 project-level queue records through 2024. The cleaned CSV contains 31,039 rows after dropping rows without the fields needed for model training.

## Quick Start

The pipeline runs on `data/lbnl_queue_cleaned.csv` by default.

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

To rebuild the cleaned CSV from the raw LBNL workbook:

```bash
python3 scripts/prepare_lbnl_data.py
```

## Outputs

After a successful run, `outputs/` contains:

- `metrics.json`: all evaluation metrics.
- `model_comparison.csv`: summary table for report figures.
- `*.png`: EDA plots, withdrawal diagnostics, risk surfaces, feature importance, and timeline residual plots.

## Project Structure

```text
data/
  lbnl_ix_queue_data_file_thru2024_v2.xlsx
  lbnl_queue_cleaned.csv
  README.md
scripts/
  prepare_lbnl_data.py
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
