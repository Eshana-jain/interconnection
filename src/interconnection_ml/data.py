from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ISO_REGIONS = ["MISO", "PJM", "CAISO", "ERCOT", "SPP", "NYISO", "ISO-NE"]
FUEL_TYPES = ["Solar", "Battery", "Wind", "Hybrid", "Gas", "Storage", "Nuclear", "Other"]
COST_BUCKETS = ["0-1M", "1-5M", "5-20M", "20M+"]
COUNTIES = [
    "Maricopa",
    "Kern",
    "Riverside",
    "Cook",
    "Harris",
    "King",
    "Dane",
    "Wayne",
    "Lancaster",
    "Erie",
    "Story",
    "Noble",
]
DELIVERABILITY_STATUSES = ["Full Capacity", "Energy Only", "Partial Deliverability"]
INTERCONNECTION_SERVICES = ["NRIS", "ERIS", "Capacity Resource", "Network Resource"]


@dataclass(frozen=True)
class ProjectColumns:
    categorical: tuple[str, ...] = (
        "iso_region",
        "fuel_type",
        "county",
        "deliverability_status",
        "interconnection_service",
    )
    numeric: tuple[str, ...] = (
        "log_mw_capacity",
        "filing_year",
        "voltage_kv",
        "queue_position",
        "substation_congestion_24mo",
        "queue_cohort_size",
        "county_queue_density",
        "distance_to_transmission_miles",
        "renewable_penetration_pct",
        "years_since_ferc2023",
    )


def load_or_generate(path: str | None, synthetic_rows: int, seed: int) -> pd.DataFrame:
    """Load a cleaned queue CSV or generate a proposal-aligned synthetic dataset."""
    if path:
        data_path = Path(path)
        if data_path.suffix.lower() in {".xlsx", ".xls"}:
            data = pd.read_excel(data_path)
        else:
            data = pd.read_csv(data_path)
        return prepare_dataset(data)
    return generate_synthetic_queue_data(n=synthetic_rows, seed=seed)


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    rename_map = {
        "queue_status": "status",
        "capacity_mw": "mw_capacity",
        "mw": "mw_capacity",
        "region": "iso_region",
        "resource_type": "fuel_type",
        "poi": "substation_id",
        "substation": "substation_id",
        "county_name": "county",
        "deliverability": "deliverability_status",
        "service_type": "interconnection_service",
        "distance_to_transmission": "distance_to_transmission_miles",
        "renewable_penetration": "renewable_penetration_pct",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    if "project_id" not in data:
        data["project_id"] = [f"project_{i:06d}" for i in range(len(data))]
    if "withdrawn" not in data:
        if "status" not in data:
            raise ValueError("Real data must include either 'withdrawn' or 'status'.")
        data["withdrawn"] = data["status"].astype(str).str.lower().str.contains("withdraw").astype(int)
    if "cost_bucket" not in data and "network_upgrade_cost_musd" in data:
        data["cost_bucket"] = pd.cut(
            data["network_upgrade_cost_musd"],
            bins=[-np.inf, 1, 5, 20, np.inf],
            labels=COST_BUCKETS,
        ).astype(str)

    required = ["iso_region", "fuel_type", "mw_capacity", "filing_year", "timeline_months"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    defaults = {
        "voltage_kv": 138.0,
        "queue_position": np.nan,
        "substation_id": "UNKNOWN",
        "county": "Unknown",
        "deliverability_status": "Unknown",
        "interconnection_service": "Unknown",
        "distance_to_transmission_miles": 0.0,
        "renewable_penetration_pct": 0.0,
    }
    for column, value in defaults.items():
        if column not in data:
            data[column] = value

    data["mw_capacity"] = pd.to_numeric(data["mw_capacity"], errors="coerce")
    data["filing_year"] = pd.to_numeric(data["filing_year"], errors="coerce").astype("Int64")
    data["timeline_months"] = pd.to_numeric(data["timeline_months"], errors="coerce")
    data["voltage_kv"] = pd.to_numeric(data["voltage_kv"], errors="coerce")
    data["queue_position"] = pd.to_numeric(data["queue_position"], errors="coerce")
    data["distance_to_transmission_miles"] = pd.to_numeric(data["distance_to_transmission_miles"], errors="coerce")
    data["renewable_penetration_pct"] = pd.to_numeric(data["renewable_penetration_pct"], errors="coerce")
    data = data.dropna(subset=["mw_capacity", "filing_year", "timeline_months", "withdrawn"])

    data["iso_region"] = data["iso_region"].fillna("Unknown").astype(str)
    data["fuel_type"] = data["fuel_type"].fillna("Other").astype(str)
    data["substation_id"] = data["substation_id"].fillna("UNKNOWN").astype(str)
    data["county"] = data["county"].fillna("Unknown").astype(str)
    data["deliverability_status"] = data["deliverability_status"].fillna("Unknown").astype(str)
    data["interconnection_service"] = data["interconnection_service"].fillna("Unknown").astype(str)

    numeric_fallbacks = {
        "voltage_kv": 138.0,
        "queue_position": 0.0,
        "distance_to_transmission_miles": 0.0,
        "renewable_penetration_pct": 0.0,
    }
    for column in numeric_fallbacks:
        data[column] = data.groupby("iso_region")[column].transform(lambda s: s.fillna(s.median()))
        data[column] = data[column].fillna(data[column].median())
        data[column] = data[column].fillna(numeric_fallbacks[column])

    if "substation_congestion_24mo" not in data:
        data = add_congestion_proxy(data)
    if "queue_cohort_size" not in data:
        data["queue_cohort_size"] = data.groupby(["iso_region", "filing_year"])["project_id"].transform("count")
    if "county_queue_density" not in data:
        data["county_queue_density"] = data.groupby(["county", "filing_year"])["project_id"].transform("count")

    data["log_mw_capacity"] = np.log1p(data["mw_capacity"].clip(lower=0))
    data["years_since_ferc2023"] = (data["filing_year"].astype(int) - 2023).clip(lower=0)
    data["withdrawn"] = data["withdrawn"].astype(int)
    return data.reset_index(drop=True)


def add_congestion_proxy(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["substation_id", "filing_year"]).copy()
    congestion = []
    for _, group in ordered.groupby("substation_id", sort=False):
        years = group["filing_year"].astype(int).to_numpy()
        counts = [int(((years < year) & (years >= year - 2)).sum()) for year in years]
        congestion.extend(counts)
    ordered["substation_congestion_24mo"] = congestion
    return ordered.sort_index()


def generate_synthetic_queue_data(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic public-queue-shaped dataset for reproducible demos."""
    rng = np.random.default_rng(seed)
    iso = rng.choice(ISO_REGIONS, size=n, p=[0.23, 0.21, 0.14, 0.13, 0.12, 0.09, 0.08])
    fuel = rng.choice(FUEL_TYPES, size=n, p=[0.45, 0.20, 0.16, 0.09, 0.04, 0.03, 0.01, 0.02])
    county = rng.choice(COUNTIES, size=n)
    deliverability = rng.choice(DELIVERABILITY_STATUSES, size=n, p=[0.38, 0.47, 0.15])
    service = rng.choice(INTERCONNECTION_SERVICES, size=n, p=[0.32, 0.48, 0.12, 0.08])
    filing_year = rng.choice(np.arange(2006, 2025), size=n, p=_year_probs())
    mw = np.exp(rng.normal(4.3, 1.0, size=n)).clip(2, 1200)
    voltage = rng.choice([69, 115, 138, 161, 230, 345, 500, 765], size=n, p=[0.08, 0.12, 0.20, 0.15, 0.22, 0.16, 0.06, 0.01])
    queue_position = rng.gamma(shape=2.0, scale=25.0, size=n).astype(int) + 1
    substation_id = np.array([f"{region}_{rng.integers(1, 180):03d}" for region in iso])
    distance_to_transmission = rng.gamma(shape=2.2, scale=4.0, size=n).clip(0.1, 60.0)
    renewable_penetration = np.clip(rng.normal(42, 18, size=n) + (filing_year - 2015) * 1.3, 5, 95)

    data = pd.DataFrame(
        {
            "project_id": [f"Q{year}{i:05d}" for i, year in enumerate(filing_year)],
            "iso_region": iso,
            "fuel_type": fuel,
            "mw_capacity": mw,
            "filing_year": filing_year,
            "voltage_kv": voltage,
            "queue_position": queue_position,
            "substation_id": substation_id,
            "county": county,
            "deliverability_status": deliverability,
            "interconnection_service": service,
            "distance_to_transmission_miles": distance_to_transmission,
            "renewable_penetration_pct": renewable_penetration,
        }
    )
    data = add_congestion_proxy(data)
    data["queue_cohort_size"] = data.groupby(["iso_region", "filing_year"])["project_id"].transform("count")
    data["county_queue_density"] = data.groupby(["county", "filing_year"])["project_id"].transform("count")

    iso_effect = data["iso_region"].map({"MISO": 0.45, "PJM": 0.35, "CAISO": 0.15, "ERCOT": -0.10, "SPP": 0.25, "NYISO": 0.05, "ISO-NE": 0.00}).to_numpy()
    fuel_effect = data["fuel_type"].map({"Solar": 0.25, "Battery": 0.20, "Wind": 0.05, "Hybrid": 0.15, "Gas": -0.20, "Storage": 0.10, "Nuclear": -0.35, "Other": 0.00}).to_numpy()
    deliverability_effect = data["deliverability_status"].map({"Full Capacity": 0.25, "Energy Only": -0.10, "Partial Deliverability": 0.10}).to_numpy()
    service_effect = data["interconnection_service"].map({"NRIS": 0.18, "ERIS": -0.08, "Capacity Resource": 0.10, "Network Resource": 0.16}).to_numpy()
    post_2018 = (data["filing_year"].to_numpy() >= 2018).astype(float)
    log_mw = np.log1p(data["mw_capacity"].to_numpy())
    congestion = data["substation_congestion_24mo"].to_numpy()
    cohort = data["queue_cohort_size"].to_numpy()
    county_density = data["county_queue_density"].to_numpy()
    distance = data["distance_to_transmission_miles"].to_numpy()
    renewables = data["renewable_penetration_pct"].to_numpy()

    withdrawal_logit = (
        -7.0
        + 0.72 * log_mw
        + 0.44 * congestion
        + 0.020 * cohort
        + 0.018 * county_density
        + 0.020 * distance
        + 0.012 * renewables
        + 0.95 * post_2018
        + 1.35 * iso_effect
        + 1.00 * fuel_effect
        + deliverability_effect
        + service_effect
    )
    withdrawal_prob = 1 / (1 + np.exp(-withdrawal_logit))
    data["withdrawn"] = rng.binomial(1, withdrawal_prob)

    log_cost = (
        -1.35
        + 0.74 * log_mw
        + 0.21 * congestion
        + 0.012 * county_density
        + 0.030 * distance
        + 0.0022 * data["voltage_kv"].to_numpy()
        + 0.006 * renewables
        + iso_effect
        + deliverability_effect
        + rng.normal(0, 0.50, size=n)
    )
    data["network_upgrade_cost_musd"] = np.exp(log_cost).clip(0.05, 250)
    data["cost_bucket"] = pd.cut(
        data["network_upgrade_cost_musd"],
        bins=[-np.inf, 1, 5, 20, np.inf],
        labels=COST_BUCKETS,
    ).astype(str)

    timeline = (
        5
        + 4.4 * log_mw
        + 1.7 * congestion
        + 0.045 * county_density
        + 0.060 * data["queue_position"].to_numpy()
        + 0.065 * distance
        + 0.020 * renewables
        + 2.6 * post_2018
        + 1.1 * iso_effect
        + deliverability_effect
        + rng.normal(0, 4.8, size=n)
    )
    data["timeline_months"] = timeline.clip(2, 96)

    # Match the progress report: cost labels are available for fewer than 40% of records.
    labeled_cost = rng.random(n) < 0.38
    data.loc[~labeled_cost, ["network_upgrade_cost_musd", "cost_bucket"]] = np.nan
    return prepare_dataset(data)


def _year_probs() -> np.ndarray:
    years = np.arange(2006, 2025)
    weights = np.linspace(0.5, 3.0, len(years))
    weights[years >= 2018] *= 1.7
    return weights / weights.sum()
