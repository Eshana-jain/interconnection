# Data

This folder contains the real public queue dataset used by the default ML run.

- `lbnl_ix_queue_data_file_thru2024_v2.xlsx` is the original Lawrence Berkeley National Laboratory / Interconnection.fyi workbook for *Queued Up: 2025 Edition*, downloaded from LBNL:
  <https://eta-publications.lbl.gov/sites/default/files/2025-08/lbnl_ix_queue_data_file_thru2024_v2.xlsx>
- `lbnl_queue_cleaned.csv` is the cleaned project-level CSV used by `src/interconnection_ml/run_pipeline.py`.

The cleaned CSV keeps project status, region, fuel type, capacity, filing year, location fields, service/status information, and engineered timeline labels. Network-upgrade cost fields are left blank because the public LBNL workbook does not include project-level upgrade-cost labels.

Regenerate the cleaned CSV from the raw workbook:

```bash
python3 scripts/prepare_lbnl_data.py
```

Run the ML pipeline on this real dataset:

```bash
PYTHONPATH=src python3 -m interconnection_ml.run_pipeline --output outputs
```
