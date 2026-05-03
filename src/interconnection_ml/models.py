from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from .data import COST_BUCKETS
from .features import feature_frame, make_preprocessor


@dataclass
class SupervisedResults:
    model: object
    metrics: dict[str, float]
    predictions: np.ndarray


class TorchCostMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_withdrawal_models(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, SupervisedResults]:
    x_train = feature_frame(train)
    y_train = train["withdrawn"].astype(int)
    x_test = feature_frame(test)
    y_test = test["withdrawn"].astype(int)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(48, 24),
            alpha=1e-3,
            early_stopping=True,
            learning_rate_init=3e-4,
            max_iter=220,
            random_state=seed,
        ),
    }
    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = SupervisedResults(
            model=pipe,
            predictions=pred,
            metrics={
                "auc_roc": float(roc_auc_score(y_test, proba)),
                "f1_withdrawn": float(f1_score(y_test, pred)),
                "accuracy": float(accuracy_score(y_test, pred)),
            },
        )
    return results


def train_timeline_models(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, SupervisedResults]:
    x_train = feature_frame(train)
    y_train = train["timeline_months"].astype(float)
    x_test = feature_frame(test)
    y_test = test["timeline_months"].astype(float)

    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "mlp_regressor": TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(48, 24),
                alpha=1e-3,
                early_stopping=True,
                learning_rate_init=3e-4,
                max_iter=260,
                random_state=seed,
            ),
            transformer=StandardScaler(),
        ),
    }
    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        results[name] = SupervisedResults(
            model=pipe,
            predictions=pred,
            metrics={
                "mae_months": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
            },
        )
    return results


def train_cost_bucket_models(train: pd.DataFrame, test: pd.DataFrame, seed: int, epochs: int = 90) -> dict[str, SupervisedResults]:
    train_cost = train.dropna(subset=["cost_bucket"]).reset_index(drop=True)
    test_cost = test.dropna(subset=["cost_bucket"]).reset_index(drop=True)
    if len(train_cost) < 50 or len(test_cost) < 10:
        return {}

    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(feature_frame(train_cost)).astype(np.float32)
    x_high_train = preprocessor.transform(feature_frame(_make_high_mw_frame(train_cost))).astype(np.float32)
    x_test = preprocessor.transform(feature_frame(test_cost)).astype(np.float32)
    y_train = _encode_cost_bucket(train_cost["cost_bucket"])
    y_test = _encode_cost_bucket(test_cost["cost_bucket"])

    unconstrained = _train_cost_mlp(
        x_train=x_train,
        x_high_train=x_high_train,
        y_train=y_train,
        monotonic_weight=0.0,
        seed=seed,
        epochs=epochs,
    )
    constrained = _train_cost_mlp(
        x_train=x_train,
        x_high_train=x_high_train,
        y_train=y_train,
        monotonic_weight=0.55,
        seed=seed,
        epochs=epochs,
    )

    results = {}
    for name, model in {"cost_mlp_unconstrained": unconstrained, "cost_mlp_pinn_monotonic": constrained}.items():
        pred = _predict_cost_bucket(model, x_test)
        results[name] = SupervisedResults(
            model={"preprocessor": preprocessor, "network": model},
            predictions=pred,
            metrics={
                "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
                "accuracy": float(accuracy_score(y_test, pred)),
                "implausibility_rate": float(monotonic_violation_rate(model, preprocessor, test_cost)),
            },
        )
    return results


def train_gpr_cost_model(train: pd.DataFrame, test: pd.DataFrame, seed: int, max_train: int = 650) -> dict[str, float]:
    train_cost = train.dropna(subset=["network_upgrade_cost_musd"]).reset_index(drop=True)
    test_cost = test.dropna(subset=["network_upgrade_cost_musd"]).reset_index(drop=True)
    if len(train_cost) < 50 or len(test_cost) < 10:
        return {}

    if len(train_cost) > max_train:
        train_cost = train_cost.sample(max_train, random_state=seed).reset_index(drop=True)

    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(feature_frame(train_cost))
    x_test = preprocessor.transform(feature_frame(test_cost))
    y_train = np.log1p(train_cost["network_upgrade_cost_musd"].to_numpy(dtype=float))
    y_test = np.log1p(test_cost["network_upgrade_cost_musd"].to_numpy(dtype=float))

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.15)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, random_state=seed, n_restarts_optimizer=0)
    gpr.fit(x_train, y_train)
    mean, std = gpr.predict(x_test, return_std=True)
    lower = mean - 1.645 * std
    upper = mean + 1.645 * std
    picp = np.mean((y_test >= lower) & (y_test <= upper))
    return {
        "mae_log_cost": float(mean_absolute_error(y_test, mean)),
        "r2_log_cost": float(r2_score(y_test, mean)),
        "picp_90": float(picp),
        "mean_interval_width_log_cost": float(np.mean(upper - lower)),
    }


def _train_cost_mlp(
    x_train: np.ndarray,
    x_high_train: np.ndarray,
    y_train: np.ndarray,
    monotonic_weight: float,
    seed: int,
    epochs: int,
) -> TorchCostMLP:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TorchCostMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_high_tensor = torch.tensor(x_high_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    batch_size = min(128, len(x_train))

    for _ in range(epochs):
        permutation = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_tensor[idx]
            xhb = x_high_tensor[idx]
            yb = y_tensor[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if monotonic_weight > 0:
                loss = loss + monotonic_weight * _monotonic_penalty(model, xb, xhb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _monotonic_penalty(model: TorchCostMLP, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
    expected = torch.arange(len(COST_BUCKETS), dtype=torch.float32)
    low_score = torch.softmax(model(x_low), dim=1) @ expected
    high_score = torch.softmax(model(x_high), dim=1) @ expected
    return torch.relu(low_score - high_score).mean()


def monotonic_violation_rate(model: TorchCostMLP, preprocessor, raw: pd.DataFrame) -> float:
    if raw.empty:
        return float("nan")
    high = _make_high_mw_frame(raw)
    with torch.no_grad():
        x_low = torch.tensor(preprocessor.transform(feature_frame(raw)).astype(np.float32), dtype=torch.float32)
        x_high = torch.tensor(preprocessor.transform(feature_frame(high)).astype(np.float32), dtype=torch.float32)
        expected = torch.arange(len(COST_BUCKETS), dtype=torch.float32)
        low_score = torch.softmax(model(x_low), dim=1) @ expected
        high_score = torch.softmax(model(x_high), dim=1) @ expected
    return float((low_score > high_score + 1e-4).float().mean().item())


def _make_high_mw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    high = raw.copy()
    high["mw_capacity"] = high["mw_capacity"] * 1.35 + 5.0
    high["log_mw_capacity"] = np.log1p(high["mw_capacity"])
    return high


def _encode_cost_bucket(series: pd.Series) -> np.ndarray:
    mapping = {bucket: i for i, bucket in enumerate(COST_BUCKETS)}
    return series.map(mapping).astype(int).to_numpy()


def _predict_cost_bucket(model: TorchCostMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.argmax(logits, dim=1).cpu().numpy()
