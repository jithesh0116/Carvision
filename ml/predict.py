"""Inference: load the saved artefacts once and turn vehicle details into a price + interval.

Raises ModelNotReadyError (never fabricates a number) if the model has not been trained.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd

from ml import config as C
from ml import explainability
from ml.conformal import interval_from_log
from ml.feature_processing import FeatureProcessor


class ModelNotReadyError(RuntimeError):
    pass


def predict_log(booster, X: pd.DataFrame, best_iteration: int) -> np.ndarray:
    """Predictions on the log1p(price) scale, using only trees up to the best iteration."""
    import xgboost as xgb
    dm = xgb.DMatrix(X, enable_categorical=True)
    return booster.predict(dm, iteration_range=(0, best_iteration + 1))


class PricePredictor:
    def __init__(self):
        self._lock = threading.Lock()
        self._mtime = None
        self.booster = self.processor = self.meta = None

    def is_ready(self) -> bool:
        needed = (C.XGB_MODEL_PATH, C.PROCESSOR_PATH, C.META_PATH)
        if not all(p.exists() for p in needed):
            return False
        try:
            import xgboost  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self):
        if not self.is_ready():
            raise ModelNotReadyError("The price model has not been trained yet.")
        mtime = C.META_PATH.stat().st_mtime
        if self._mtime == mtime and self.booster is not None:
            return
        with self._lock:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(C.XGB_MODEL_PATH))
            self.booster = booster
            self.processor = FeatureProcessor.load()
            self.meta = json.loads(C.META_PATH.read_text(encoding="utf-8"))
            self._mtime = mtime

    def predict(self, inputs: dict) -> dict:
        self._ensure_loaded()
        df = pd.DataFrame([inputs])
        X = self.processor.transform(df)
        best = int(self.meta["best_iteration"])
        log_pred = float(predict_log(self.booster, X, best)[0])
        q = float(self.meta["conformal"]["quantile_log"])
        lo, hi = interval_from_log(np.array([log_pred]), q)
        local = explainability.local_contributions(self.booster, X, best)
        return {
            "estimate": float(np.expm1(log_pred)),
            "lower": float(lo[0]), "upper": float(hi[0]),
            "local_explanation": local,
            "warnings": self._warnings(inputs),
        }

    def _warnings(self, inputs: dict) -> list[str]:
        w = []
        p = self.processor
        ranges = p.state.get("ranges", {})
        yr = inputs.get("year")
        if yr is not None and "year" in ranges and not (ranges["year"][0] <= yr <= ranges["year"][1]):
            w.append("This manufacturing year is outside the range the model learned from, "
                     "so the estimate may be less reliable.")
        for f, label in (("km_driven", "kilometres driven"), ("engine_cc", "engine size"),
                         ("mileage", "mileage")):
            v = inputs.get(f)
            if v is not None and f in ranges and not (ranges[f][0] <= v <= ranges[f][1]):
                w.append(f"Your {label} is outside the range of vehicles the model learned from, "
                         "so the estimate may be less reliable.")
        for c in p.categorical:
            v = inputs.get(c)
            if v is not None and str(v) not in set(p.categories[c]):
                w.append(f"This exact {c.replace('_', ' ')} was not seen when the model was trained; "
                         "the estimate relies on similar vehicles.")
        return w


_predictor = PricePredictor()


def get_predictor() -> PricePredictor:
    return _predictor
