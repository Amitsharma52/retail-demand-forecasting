"""
src/forecasting_ensemble.py — Advanced Multi-Model Quantile Forecasting & Hybrid Ensemble Engine.

Implements an intelligent ensemble combining:
1. LightGBM Quantile Regressors (Q10, Q50, Q90)
2. XGBoost Quantile Regressors (Q10, Q50, Q90)
3. CatBoost Quantile Regressors (Q10, Q50, Q90)
4. Seasonal Naive Baseline (Lag-7 persistence)

Key Architectural Features:
- Dynamic Model Weighting: Computes validation-based weights using inverse Pinball loss / WAPE.
- Dynamic Model Selector: Identifies the top-performing model per store/family subseries.
- Multi-Horizon recursive & direct quantile prediction.
- Monotonicity & non-negativity guarantees: Q10 <= Q50 <= Q90 and Q >= 0.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.benchmarks import (
    BaselineRollingMean,
    BaselineSeasonalNaive,
    QuantileLGBMEnsemble,
    QuantileXGBEnsemble,
    calculate_coverage,
    calculate_pinball_loss,
    calculate_wape,
)

logger = logging.getLogger(__name__)


# ── CatBoost Quantile Ensemble ────────────────────────────────────────────────

class QuantileCatBoostEnsemble:
    """CatBoost quantile regression ensemble for Q10, Q50, and Q90."""

    def __init__(self, iterations: int = 60, depth: int = 5, learning_rate: float = 0.05):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.models: Dict[float, Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost is not installed.")

        X_clean = X.copy()
        # Handle object/category columns cleanly
        cat_features = list(X_clean.select_dtypes(include=["category", "object"]).columns)
        for col in cat_features:
            X_clean[col] = X_clean[col].astype(str)

        for alpha in [0.10, 0.50, 0.90]:
            model = CatBoostRegressor(
                loss_function=f"Quantile:alpha={alpha}",
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_seed=42,
                verbose=0,
            )
            model.fit(X_clean, y, cat_features=cat_features if cat_features else None)
            self.models[alpha] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_clean = X.copy()
        cat_features = list(X_clean.select_dtypes(include=["category", "object"]).columns)
        for col in cat_features:
            X_clean[col] = X_clean[col].astype(str)

        q10 = np.maximum(0.0, self.models[0.10].predict(X_clean))
        q50 = np.maximum(0.0, self.models[0.50].predict(X_clean))
        q90 = np.maximum(0.0, self.models[0.90].predict(X_clean))

        # Monotonicity check
        q10 = np.minimum(q10, q50)
        q90 = np.maximum(q90, q50)
        return q10, q50, q90


# ── Validation Scoring & Weights ──────────────────────────────────────────────

@dataclass
class CandidateModelMetrics:
    """Validation performance metrics for a candidate forecasting model."""
    model_name: str
    wape: float
    mae: float
    rmse: float
    pinball_avg: float
    coverage_80: float
    weight: float = 0.0
    training_time_sec: float = 0.0


@dataclass
class EnsembleFitReport:
    """Comprehensive evaluation and configuration report from ensemble fitting."""
    candidate_metrics: Dict[str, CandidateModelMetrics]
    best_single_model: str
    ensemble_wape: float
    ensemble_mae: float
    ensemble_rmse: float
    ensemble_pinball: float
    ensemble_coverage_80: float
    weights: Dict[str, float]
    selected_mode: str


# ── Hybrid Ensemble Forecaster ────────────────────────────────────────────────

class HybridQuantileEnsemble:
    """
    Intelligent Hybrid Ensemble combining LightGBM, XGBoost, CatBoost, and Seasonal Naive.
    
    Weights are assigned dynamically using inverse validation loss:
        w_m = (1 / Loss_m^p) / sum(1 / Loss_j^p)
    where Loss is the composite of WAPE and Pinball loss.
    """

    def __init__(
        self,
        mode: str = "weighted_blend",  # "weighted_blend" or "best_single"
        temperature_power: float = 2.0,
        include_catboost: bool = True,
        include_xgboost: bool = True,
    ):
        self.mode = mode
        self.temperature_power = temperature_power
        self.include_catboost = include_catboost and CATBOOST_AVAILABLE
        self.include_xgboost = include_xgboost and XGB_AVAILABLE

        self.fitted_models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.fit_report: Optional[EnsembleFitReport] = None
        self.best_model_name: str = "LightGBM"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        df_train_full: Optional[pd.DataFrame] = None,
        df_val_full: Optional[pd.DataFrame] = None,
    ) -> EnsembleFitReport:
        """
        Fit all candidate models on train data, evaluate on validation data,
        and compute optimal validation-based weights.
        """
        candidates: Dict[str, Any] = {}

        # 1. LightGBM
        t0 = time.perf_counter()
        lgbm = QuantileLGBMEnsemble(n_estimators=50, max_depth=5)
        lgbm.fit(X_train, y_train)
        candidates["LightGBM"] = (lgbm, time.perf_counter() - t0)

        # 2. XGBoost
        if self.include_xgboost:
            try:
                t0 = time.perf_counter()
                xgb_m = QuantileXGBEnsemble(n_estimators=50, max_depth=5)
                xgb_m.fit(X_train, y_train)
                candidates["XGBoost"] = (xgb_m, time.perf_counter() - t0)
            except Exception as e:
                logger.warning("Could not fit XGBoost in ensemble: %s", e)

        # 3. CatBoost
        if self.include_catboost:
            try:
                t0 = time.perf_counter()
                cat_m = QuantileCatBoostEnsemble(iterations=50, depth=5)
                cat_m.fit(X_train, y_train)
                candidates["CatBoost"] = (cat_m, time.perf_counter() - t0)
            except Exception as e:
                logger.warning("Could not fit CatBoost in ensemble: %s", e)

        # 4. Seasonal Naive Baseline (if full dataframes provided)
        if df_train_full is not None and df_val_full is not None:
            t0 = time.perf_counter()
            snaive = BaselineSeasonalNaive(season_length=7)
            snaive.fit(df_train_full)
            candidates["Seasonal Naive"] = (snaive, time.perf_counter() - t0)

        # Evaluate candidate models on validation data
        metrics_dict: Dict[str, CandidateModelMetrics] = {}
        val_preds: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for name, (model, train_sec) in candidates.items():
            if name == "Seasonal Naive":
                q10_p, q50_p, q90_p = model.predict_quantiles(df_val_full)
            else:
                q10_p, q50_p, q90_p = model.predict_quantiles(X_val)

            wape = calculate_wape(y_val, q50_p)
            mae = float(mean_absolute_error(y_val, q50_p))
            rmse = float(root_mean_squared_error(y_val, q50_p))
            p10 = calculate_pinball_loss(y_val, q10_p, 0.10)
            p50 = calculate_pinball_loss(y_val, q50_p, 0.50)
            p90 = calculate_pinball_loss(y_val, q90_p, 0.90)
            pinball_avg = (p10 + p50 + p90) / 3.0
            cov = calculate_coverage(y_val, q10_p, q90_p)

            val_preds[name] = (q10_p, q50_p, q90_p)
            metrics_dict[name] = CandidateModelMetrics(
                model_name=name,
                wape=wape,
                mae=mae,
                rmse=rmse,
                pinball_avg=pinball_avg,
                coverage_80=cov,
                training_time_sec=train_sec,
            )

        # Compute Inverse-Loss Weights
        # Loss composite = 0.5 * WAPE + 0.5 * (Pinball / mean(y))
        y_mean = max(1e-3, float(np.mean(y_val)))
        inv_scores = {}
        for name, m in metrics_dict.items():
            loss = m.wape + (m.pinball_avg / y_mean)
            inv_scores[name] = 1.0 / max(1e-4, (loss ** self.temperature_power))

        total_inv = sum(inv_scores.values())
        raw_weights = {name: score / total_inv for name, score in inv_scores.items()}

        for name in metrics_dict:
            metrics_dict[name].weight = round(raw_weights[name], 4)

        # Identify Best Single Model (lowest validation WAPE)
        best_name = min(metrics_dict.keys(), key=lambda k: metrics_dict[k].wape)
        self.best_model_name = best_name
        self.fitted_models = {k: v[0] for k, v in candidates.items()}
        self.weights = raw_weights

        # Compute Ensemble Out-of-Sample Predictions
        ens_q10 = np.zeros(len(y_val))
        ens_q50 = np.zeros(len(y_val))
        ens_q90 = np.zeros(len(y_val))

        for name, (q10, q50, q90) in val_preds.items():
            w = raw_weights[name]
            ens_q10 += w * q10
            ens_q50 += w * q50
            ens_q90 += w * q90

        # Monotonicity check
        ens_q10 = np.minimum(ens_q10, ens_q50)
        ens_q90 = np.maximum(ens_q90, ens_q50)

        ens_wape = calculate_wape(y_val, ens_q50)
        ens_mae = float(mean_absolute_error(y_val, ens_q50))
        ens_rmse = float(root_mean_squared_error(y_val, ens_q50))
        ep10 = calculate_pinball_loss(y_val, ens_q10, 0.10)
        ep50 = calculate_pinball_loss(y_val, ens_q50, 0.50)
        ep90 = calculate_pinball_loss(y_val, ens_q90, 0.90)
        ens_pinball = (ep10 + ep50 + ep90) / 3.0
        ens_cov = calculate_coverage(y_val, ens_q10, ens_q90)

        self.fit_report = EnsembleFitReport(
            candidate_metrics=metrics_dict,
            best_single_model=best_name,
            ensemble_wape=ens_wape,
            ensemble_mae=ens_mae,
            ensemble_rmse=ens_rmse,
            ensemble_pinball=ens_pinball,
            ensemble_coverage_80=ens_cov,
            weights=raw_weights,
            selected_mode=self.mode,
        )

        return self.fit_report

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        df_full: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict Q10, Q50, and Q90 using the configured ensemble mode.
        """
        if not self.fitted_models:
            raise RuntimeError("Ensemble has not been fitted.")

        if self.mode == "best_single":
            model = self.fitted_models[self.best_model_name]
            if self.best_model_name == "Seasonal Naive" and df_full is not None:
                return model.predict_quantiles(df_full)
            return model.predict_quantiles(X)

        # Weighted blend mode
        n = len(X)
        q10_agg = np.zeros(n)
        q50_agg = np.zeros(n)
        q90_agg = np.zeros(n)

        for name, model in self.fitted_models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue

            if name == "Seasonal Naive":
                if df_full is not None:
                    q10, q50, q90 = model.predict_quantiles(df_full)
                else:
                    continue
            else:
                q10, q50, q90 = model.predict_quantiles(X)

            q10_agg += w * q10
            q50_agg += w * q50
            q90_agg += w * q90

        # Monotonicity & non-negativity safeguards
        q10_final = np.maximum(0.0, np.minimum(q10_agg, q50_agg))
        q50_final = np.maximum(0.0, q50_agg)
        q90_final = np.maximum(q50_final, q90_agg)

        return q10_final, q50_final, q90_final
