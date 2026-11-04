"""
src/benchmarks.py — Multi-Model Benchmarking & Walk-Forward Time-Series Cross-Validation.

Compares forecasting models on historical retail data:
1. LightGBM Quantile Regressors (Q10, Q50, Q90)
2. XGBoost Quantile Regressors (Q10, Q50, Q90)
3. Seasonal Naive Baseline (Lag-7 persistence)
4. Rolling Window Moving Average Baseline

Evaluates out-of-sample across walk-forward rolling-origin folds on:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
- Pinball Loss (Q10, Q50, Q90)
- Empirical 80% Prediction Interval Coverage
- Training Duration (seconds)
- Inference Latency (ms per 1,000 predictions)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Metric Helpers ─────────────────────────────────────────────────────────────

def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y)."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Standard asymmetric pinball loss for quantile regression."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def calculate_coverage(y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """Fraction of true values falling within [q_lower, q_upper]."""
    if len(y_true) == 0:
        return 0.0
    inside = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(inside))


# ── Model Wrapper Definitions ──────────────────────────────────────────────────

class BaselineSeasonalNaive:
    """
    Seasonal Naive baseline: predicts sales as the observed value from
    `season_length` days ago (default 7 days).
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.history_: Optional[pd.DataFrame] = None
        self.median_fallback_: float = 0.0

    def fit(self, df_train: pd.DataFrame, target_col: str = "sales", date_col: str = "date"):
        self.target_col = target_col
        self.date_col = date_col
        self.history_ = df_train[[date_col, "store_nbr", "family", target_col]].copy()
        self.median_fallback_ = float(df_train[target_col].median() if len(df_train) > 0 else 0.0)
        return self

    def predict_quantiles(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (q10, q50, q90). For naive baseline, q50 is lag-7; q10 and q90 add historical spread."""
        if self.history_ is None:
            n = len(df_test)
            return np.zeros(n), np.zeros(n), np.zeros(n)

        # Merge test dates with date - 7 from history
        test_sub = df_test[[self.date_col, "store_nbr", "family"]].copy()
        test_sub["lag_date"] = test_sub[self.date_col] - pd.Timedelta(days=self.season_length)

        merged = pd.merge(
            test_sub,
            self.history_,
            left_on=["lag_date", "store_nbr", "family"],
            right_on=[self.date_col, "store_nbr", "family"],
            how="left",
            suffixes=("", "_hist")
        )

        q50 = merged[self.target_col].fillna(self.median_fallback_).to_numpy()
        q50 = np.maximum(0.0, q50)

        # Naive interval: ±25% margin
        q10 = np.maximum(0.0, q50 * 0.75)
        q90 = q50 * 1.25
        return q10, q50, q90


class BaselineRollingMean:
    """
    Rolling Mean baseline: predicts sales as the trailing rolling average
    over a window of `window_size` days.
    """
    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        self.means_: Dict[Tuple[int, str], float] = {}
        self.global_mean_: float = 0.0

    def fit(self, df_train: pd.DataFrame, target_col: str = "sales", date_col: str = "date"):
        self.global_mean_ = float(df_train[target_col].mean() if len(df_train) > 0 else 0.0)
        # Compute mean over the last window_size days for each store, family
        max_date = df_train[date_col].max()
        recent_df = df_train[df_train[date_col] > (max_date - pd.Timedelta(days=self.window_size))]

        grouped = recent_df.groupby(["store_nbr", "family"])[target_col].mean()
        self.means_ = grouped.to_dict()
        return self

    def predict_quantiles(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q50_vals = []
        for _, row in df_test.iterrows():
            key = (row["store_nbr"], row["family"])
            val = self.means_.get(key, self.global_mean_)
            q50_vals.append(max(0.0, float(val)))
        
        q50 = np.array(q50_vals)
        q10 = np.maximum(0.0, q50 * 0.70)
        q90 = q50 * 1.30
        return q10, q50, q90


class QuantileLGBMEnsemble:
    """LightGBM quantile regression ensemble for Q10, Q50, and Q90."""
    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.05, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models: Dict[float, LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        for alpha in [0.10, 0.50, 0.90]:
            model = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=2 ** self.max_depth - 1,
                min_child_samples=20,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X, y)
            self.models[alpha] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q10 = np.maximum(0.0, self.models[0.10].predict(X))
        q50 = np.maximum(0.0, self.models[0.50].predict(X))
        q90 = np.maximum(0.0, self.models[0.90].predict(X))
        # Monotonicity check
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q90, q50)
        return q10, q50, q90


class QuantileXGBEnsemble:
    """XGBoost quantile regression ensemble for Q10, Q50, and Q90."""
    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.05, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models: Dict[float, Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is not installed.")
        for alpha in [0.10, 0.50, 0.90]:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            # Ensure categorical types are handled or converted
            X_clean = X.copy()
            for col in X_clean.select_dtypes(include=["category", "object"]).columns:
                X_clean[col] = X_clean[col].astype("category")
            model.fit(X_clean, y)
            self.models[alpha] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=["category", "object"]).columns:
            X_clean[col] = X_clean[col].astype("category")

        q10 = np.maximum(0.0, self.models[0.10].predict(X_clean))
        q50 = np.maximum(0.0, self.models[0.50].predict(X_clean))
        q90 = np.maximum(0.0, self.models[0.90].predict(X_clean))
        # Monotonicity check
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q90, q50)
        return q10, q50, q90


# ── Benchmark Evaluation Data Structures ───────────────────────────────────────

@dataclass
class ModelBenchmarkResult:
    """Summary metrics for a benchmarked model."""
    model_name: str
    mae: float
    rmse: float
    wape: float
    pinball_q10: float
    pinball_q50: float
    pinball_q90: float
    coverage_80: float
    mean_interval_width: float
    training_time_sec: float
    inference_latency_ms: float
    n_samples: int


@dataclass
class BenchmarkComparisonReport:
    """Consolidated benchmark suite comparison report."""
    results: List[ModelBenchmarkResult]
    summary_df: pd.DataFrame
    n_folds: int
    test_horizon_days: int


# ── Walk-Forward Cross Validation ─────────────────────────────────────────────

def run_walk_forward_benchmark(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "sales",
    date_col: str = "date",
    n_splits: int = 3,
    test_horizon_days: int = 14,
    include_xgboost: bool = True,
) -> BenchmarkComparisonReport:
    """
    Performs expanding-window walk-forward time-series cross validation.
    
    In each split:
      - Train window expands chronologically.
      - Test window is the subsequent `test_horizon_days`.
      - Out-of-sample predictions are gathered and scored.
    """
    dates = pd.to_datetime(feature_df[date_col].unique())
    dates = np.sort(dates)
    total_dates = len(dates)

    required_dates = n_splits * test_horizon_days + 30
    if total_dates < required_dates:
        raise ValueError(
            f"Dataset has {total_dates} distinct dates, but at least {required_dates} "
            f"are needed for {n_splits} folds of {test_horizon_days} days."
        )

    # Initialize model registries
    model_types = [
        ("LightGBM Quantile", "lgbm"),
        ("Seasonal Naive (Lag-7)", "snaive"),
        ("Rolling Mean (14-Day)", "rmean"),
    ]
    if include_xgboost and XGB_AVAILABLE:
        model_types.insert(1, ("XGBoost Quantile", "xgb"))

    # Tracking storage
    tracking: Dict[str, Dict[str, Any]] = {
        name: {
            "y_trues": [],
            "q10s": [],
            "q50s": [],
            "q90s": [],
            "train_times": [],
            "infer_times": [],
        }
        for name, _ in model_types
    }

    # Generate fold date boundaries
    fold_cutoffs = []
    for i in range(n_splits, 0, -1):
        test_end_idx = total_dates - (i - 1) * test_horizon_days
        test_start_idx = test_end_idx - test_horizon_days
        fold_cutoffs.append((dates[test_start_idx], dates[test_end_idx - 1]))

    logger.info("Executing Walk-Forward CV across %d folds (test horizon = %d days)", n_splits, test_horizon_days)

    for fold_idx, (test_start, test_end) in enumerate(fold_cutoffs, 1):
        logger.info(
            "Fold %d/%d: Test from %s to %s",
            fold_idx, n_splits,
            pd.Timestamp(test_start).date(),
            pd.Timestamp(test_end).date(),
        )

        train_mask = pd.to_datetime(feature_df[date_col]) < test_start
        test_mask = (pd.to_datetime(feature_df[date_col]) >= test_start) & (pd.to_datetime(feature_df[date_col]) <= test_end)

        df_train = feature_df[train_mask].copy()
        df_test = feature_df[test_mask].copy()

        if len(df_test) == 0:
            continue

        X_train = df_train[feature_cols]
        y_train = df_train[target_col].to_numpy()
        X_test = df_test[feature_cols]
        y_test = df_test[target_col].to_numpy()

        for name, m_key in model_types:
            t0 = time.perf_counter()
            if m_key == "lgbm":
                model = QuantileLGBMEnsemble(n_estimators=50, max_depth=5)
                model.fit(X_train, y_train)
                train_sec = time.perf_counter() - t0

                t_infer = time.perf_counter()
                q10, q50, q90 = model.predict_quantiles(X_test)
                infer_sec = time.perf_counter() - t_infer

            elif m_key == "xgb":
                model = QuantileXGBEnsemble(n_estimators=50, max_depth=5)
                model.fit(X_train, y_train)
                train_sec = time.perf_counter() - t0

                t_infer = time.perf_counter()
                q10, q50, q90 = model.predict_quantiles(X_test)
                infer_sec = time.perf_counter() - t_infer

            elif m_key == "snaive":
                smodel = BaselineSeasonalNaive(season_length=7)
                smodel.fit(df_train, target_col=target_col, date_col=date_col)
                train_sec = time.perf_counter() - t0

                t_infer = time.perf_counter()
                q10, q50, q90 = smodel.predict_quantiles(df_test)
                infer_sec = time.perf_counter() - t_infer

            elif m_key == "rmean":
                rmodel = BaselineRollingMean(window_size=14)
                rmodel.fit(df_train, target_col=target_col, date_col=date_col)
                train_sec = time.perf_counter() - t0

                t_infer = time.perf_counter()
                q10, q50, q90 = rmodel.predict_quantiles(df_test)
                infer_sec = time.perf_counter() - t_infer

            tracking[name]["y_trues"].append(y_test)
            tracking[name]["q10s"].append(q10)
            tracking[name]["q50s"].append(q50)
            tracking[name]["q90s"].append(q90)
            tracking[name]["train_times"].append(train_sec)
            tracking[name]["infer_times"].append(infer_sec / max(1, len(X_test)) * 1000.0)

    # Aggregate results across all folds
    results: List[ModelBenchmarkResult] = []
    for name, _ in model_types:
        y_all = np.concatenate(tracking[name]["y_trues"]) if tracking[name]["y_trues"] else np.array([])
        q10_all = np.concatenate(tracking[name]["q10s"]) if tracking[name]["q10s"] else np.array([])
        q50_all = np.concatenate(tracking[name]["q50s"]) if tracking[name]["q50s"] else np.array([])
        q90_all = np.concatenate(tracking[name]["q90s"]) if tracking[name]["q90s"] else np.array([])

        if len(y_all) == 0:
            continue

        mae = float(mean_absolute_error(y_all, q50_all))
        rmse = float(root_mean_squared_error(y_all, q50_all))
        wape = calculate_wape(y_all, q50_all)
        p10 = calculate_pinball_loss(y_all, q10_all, 0.10)
        p50 = calculate_pinball_loss(y_all, q50_all, 0.50)
        p90 = calculate_pinball_loss(y_all, q90_all, 0.90)
        cov = calculate_coverage(y_all, q10_all, q90_all)
        mean_width = float(np.mean(q90_all - q10_all))
        avg_train_time = float(np.mean(tracking[name]["train_times"]))
        avg_infer_latency = float(np.mean(tracking[name]["infer_times"]))

        results.append(
            ModelBenchmarkResult(
                model_name=name,
                mae=mae,
                rmse=rmse,
                wape=wape,
                pinball_q10=p10,
                pinball_q50=p50,
                pinball_q90=p90,
                coverage_80=cov,
                mean_interval_width=mean_width,
                training_time_sec=avg_train_time,
                inference_latency_ms=avg_infer_latency,
                n_samples=len(y_all),
            )
        )

    # Format summary DataFrame
    summary_rows = []
    for res in results:
        summary_rows.append({
            "Model": res.model_name,
            "WAPE (%)": f"{res.wape * 100:.2f}%",
            "MAE": f"{res.mae:.2f}",
            "RMSE": f"{res.rmse:.2f}",
            "Pinball Loss (Avg)": f"{(res.pinball_q10 + res.pinball_q50 + res.pinball_q90)/3:.3f}",
            "80% Interval Coverage": f"{res.coverage_80 * 100:.1f}%",
            "Mean Interval Width": f"{res.mean_interval_width:.1f}",
            "Train Time (s)": f"{res.training_time_sec:.2f}s",
            "Latency (ms/obs)": f"{res.inference_latency_ms:.3f} ms",
        })

    summary_df = pd.DataFrame(summary_rows)
    return BenchmarkComparisonReport(
        results=results,
        summary_df=summary_df,
        n_folds=n_splits,
        test_horizon_days=test_horizon_days,
    )
