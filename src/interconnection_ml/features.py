from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ProjectColumns


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_dataset(data: pd.DataFrame, seed: int) -> SplitData:
    """Create the proposal's 80/10/10 split, stratified by region/year where feasible."""
    strata = data["iso_region"].astype(str) + "_" + (data["filing_year"].astype(int) // 3 * 3).astype(str)
    if strata.value_counts().min() < 2:
        strata = data["withdrawn"].astype(str)

    train, temp = train_test_split(data, test_size=0.20, random_state=seed, stratify=strata)
    temp_strata = temp["iso_region"].astype(str) + "_" + (temp["filing_year"].astype(int) // 3 * 3).astype(str)
    stratify_temp = temp_strata if temp_strata.value_counts().min() >= 2 else temp["withdrawn"].astype(str)
    validation, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=stratify_temp)
    return SplitData(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True))


def make_preprocessor(columns: ProjectColumns | None = None) -> ColumnTransformer:
    columns = columns or ProjectColumns()
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(columns.numeric)),
            ("cat", encoder, list(columns.categorical)),
        ],
        remainder="drop",
    )


def feature_frame(data: pd.DataFrame, columns: ProjectColumns | None = None) -> pd.DataFrame:
    columns = columns or ProjectColumns()
    return data[list(columns.numeric + columns.categorical)].copy()
