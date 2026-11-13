"""Explainability from the trained XGBoost model (no invented numbers).

* Global importance: XGBoost "gain" importance from the saved booster, normalised to shares.
* Local explanation: XGBoost's built-in exact tree SHAP (`pred_contribs=True`) for one car. Values
  live on the log-price scale and add up to the prediction, so they are summed per user-facing
  group (e.g. brand+model+variant -> "Make, model & variant") and converted to an approximate
  percentage effect relative to the average car in the training data.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ml import config as C

FEATURE_GROUP = {
    "vehicle_age": "age", "km_driven": "km", "km_per_year": "km",
    "brand": "vehicle", "model": "vehicle", "variant": "vehicle",
    "fuel_type": "fuel", "transmission": "transmission", "owner_count": "owners",
    "engine_cc": "engine", "mileage": "mileage", "location": "location",
    "vehicle_condition": "condition",
}
GROUP_LABEL = {
    "age": "Vehicle age", "km": "Kilometres driven", "vehicle": "Make, model & variant",
    "fuel": "Fuel type", "transmission": "Transmission", "owners": "Ownership history",
    "engine": "Engine", "mileage": "Mileage", "location": "Location", "condition": "Vehicle condition",
}


def _group(feature: str) -> str:
    return FEATURE_GROUP.get(feature, feature)


def _label(group: str) -> str:
    return GROUP_LABEL.get(group, group.replace("_", " ").title())


def compute_global_importance(booster, features) -> dict:
    """Gain-based importance per feature (normalised) and per user-facing group."""
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    per_feature = {f: float(gain.get(f, 0.0) / total) for f in features}
    groups: dict[str, float] = {}
    for f, v in per_feature.items():
        groups[_group(f)] = groups.get(_group(f), 0.0) + v
    grouped = [{"key": g, "label": _label(g), "share": round(100 * v, 2)}
               for g, v in sorted(groups.items(), key=lambda kv: -kv[1])]
    return {"method": "XGBoost gain importance", "per_feature": per_feature, "grouped": grouped}


def load_global_importance():
    if C.IMPORTANCE_PATH.exists():
        return json.loads(C.IMPORTANCE_PATH.read_text(encoding="utf-8"))["grouped"]
    return []


def local_contributions(booster, X, best_iteration: int):
    """Per-group contribution for one row, or None if unavailable."""
    import xgboost as xgb
    try:
        dm = xgb.DMatrix(X, enable_categorical=True)
        contribs = booster.predict(dm, pred_contribs=True, iteration_range=(0, best_iteration + 1))[0]
    except Exception:
        return None
    per_feature = dict(zip(list(X.columns), contribs[:-1]))   # last column is the bias/base value
    groups: dict[str, float] = {}
    for f, v in per_feature.items():
        groups[_group(f)] = groups.get(_group(f), 0.0) + float(v)
    total_abs = sum(abs(v) for v in groups.values()) or 1.0
    items = []
    for g, v in groups.items():
        items.append({
            "key": g, "label": _label(g),
            "direction": "up" if v > 0.005 else ("down" if v < -0.005 else "neutral"),
            "effect_pct": round(100 * (math.exp(v) - 1), 1),
            "share": round(100 * abs(v) / total_abs, 1),
        })
    return sorted(items, key=lambda d: -d["share"])


def plain_summary(items) -> str:
    if not items:
        return ("Your vehicle's estimated value is influenced by its age, kilometres driven, model, "
                "fuel type, condition, and ownership history.")
    top = [i["label"].lower() for i in items[:3]]
    joined = top[0] if len(top) == 1 else ", ".join(top[:-1]) + " and " + top[-1]
    return f"For your car, the estimate is influenced most by {joined}."
