"""Prepare the public LBNL interconnection queue workbook for the ML pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILE = "lbnl_ix_queue_data_file_thru2024_v2.xlsx"
CLEAN_FILE = "lbnl_queue_cleaned.csv"


def excel_serial_to_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="D", origin="1899-12-30", errors="coerce")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    raw_path = data_dir / RAW_FILE
    clean_path = data_dir / CLEAN_FILE

    raw = pd.read_excel(raw_path, sheet_name="03. Complete Queue Data", skiprows=1)
    data = pd.DataFrame()
    data["project_id"] = raw["q_id"].fillna(raw.index.to_series().map(lambda i: f"LBNL_{i:06d}")).astype(str)
    data["status"] = raw["q_status"].astype(str).str.lower()
    data["withdrawn"] = (data["status"] == "withdrawn").astype(int)
    data["iso_region"] = raw["region"].fillna(raw["entity"]).fillna("Unknown")
    data["fuel_type"] = raw["type_clean"].fillna(raw["type1"]).fillna("Other")
    data["mw_capacity"] = raw[["mw1", "mw2", "mw3"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    data["filing_year"] = pd.to_numeric(raw["q_year"], errors="coerce")
    data["queue_position"] = raw.groupby(["entity", "q_year"]).cumcount() + 1
    data["substation_id"] = raw["poi_name"].fillna("UNKNOWN")
    data["county"] = raw["county"].fillna("Unknown")
    data["state"] = raw["state"].fillna("Unknown")
    data["deliverability_status"] = raw["IA_status_clean"].fillna("Unknown")
    data["interconnection_service"] = raw["service"].fillna("Unknown")
    data["project_type"] = raw["project_type"].fillna("Unknown")
    data["utility"] = raw["utility"].fillna("Unknown")
    data["entity"] = raw["entity"].fillna("Unknown")

    q_date = excel_serial_to_datetime(raw["q_date"])
    on_date = excel_serial_to_datetime(raw["on_date"])
    wd_date = excel_serial_to_datetime(raw["wd_date"])
    ia_date = excel_serial_to_datetime(raw["ia_date"])
    prop_date = excel_serial_to_datetime(raw["prop_date"])

    event_date = wd_date.where(data["status"] == "withdrawn")
    event_date = event_date.fillna(on_date.where(data["status"] == "operational"))
    event_date = event_date.fillna(ia_date)
    event_date = event_date.fillna(prop_date)
    timeline_months = (event_date - q_date).dt.days / 30.4375
    data["timeline_months"] = timeline_months
    data["timeline_source"] = np.select(
        [
            wd_date.notna() & (data["status"] == "withdrawn"),
            on_date.notna() & (data["status"] == "operational"),
            ia_date.notna(),
            prop_date.notna(),
        ],
        ["withdrawal_date", "online_date", "interconnection_agreement_date", "proposed_online_date"],
        default="missing",
    )

    data["request_date"] = q_date.dt.date
    data["event_date"] = event_date.dt.date
    data["network_upgrade_cost_musd"] = np.nan
    data["cost_bucket"] = np.nan

    data = data.dropna(subset=["mw_capacity", "filing_year", "timeline_months", "withdrawn"])
    data = data[data["timeline_months"] >= 0].reset_index(drop=True)
    data.to_csv(clean_path, index=False)
    print(f"Wrote {len(data):,} cleaned rows to {clean_path}")


if __name__ == "__main__":
    main()
