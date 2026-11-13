"""Feature engineering + encoding shared by training and prediction.

Which features the model uses is decided from the real data (see `FeatureProcessor.fit`):
a field is used if it is filled in for >= MIN_FEATURE_COVERAGE of rows and is not constant.

Engineered features
-------------------
vehicle_age : reference_year - manufacturing year, where reference_year is the newest
              manufacturing year in the training data. Purpose: depreciation is driven by age,
              not by the calendar year itself. (It is a linear re-expression of `year`, so no
              information is added and no target information is used -> no leakage.)
km_per_year : km_driven / max(vehicle_age, 1). Purpose: usage intensity; a 3-year-old car with
              90,000 km is driven harder than a 9-year-old with 90,000 km.

Categorical fields use XGBoost's native categorical support (pandas `category` dtype with a
fixed, saved category list). Values not seen in training become "missing", which XGBoost
handles natively. Missing numerics are also left missing (no fabricated imputation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config as C

# order in which raw input fields are asked from the user
INPUT_FIELD_ORDER = ["brand", "model", "variant", "year", "km_driven", "fuel_type", "transmission",
                     "owner_count", "engine_cc", "mileage", "location", "vehicle_condition"]

ENGINEERED_FEATURES = {
    "vehicle_age": "Reference year minus manufacturing year; captures depreciation with age.",
    "km_per_year": "Kilometres driven per year of age; captures how intensively the car was used.",
}


class FeatureProcessor:
    def __init__(self, state: dict):
        self.state = state
        self.features: list[str] = state["features"]
        self.categorical: list[str] = state["categorical"]
        self.numeric: list[str] = state["numeric"]
        self.categories: dict = state["categories"]
        self.reference_year: int = state["reference_year"]

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, df: pd.DataFrame) -> "FeatureProcessor":
        n = len(df)
        for f in ("brand", "model", "year"):
            if f not in df.columns:
                raise ValueError(f"'{f}' is required to train the price model.")
        cats, nums, dropped = [], [], {}
        for c in C.CATEGORICAL_CANDIDATES:
            if c not in df.columns:
                dropped[c] = "not in dataset"
                continue
            cover = df[c].notna().mean()
            if c not in ("brand", "model") and cover < C.MIN_FEATURE_COVERAGE:
                dropped[c] = f"only {100 * cover:.0f}% filled (< {100 * C.MIN_FEATURE_COVERAGE:.0f}%)"
            elif c not in ("brand", "model") and df[c].nunique() < 2:
                dropped[c] = "constant"
            else:
                cats.append(c)
        for c in C.NUMERIC_CANDIDATES:
            if c not in df.columns:
                dropped[c] = "not in dataset"
                continue
            cover = df[c].notna().mean()
            if cover < C.MIN_FEATURE_COVERAGE:
                dropped[c] = f"only {100 * cover:.0f}% filled (< {100 * C.MIN_FEATURE_COVERAGE:.0f}%)"
            elif df[c].nunique() < 2:
                dropped[c] = "constant"
            else:
                nums.append(c)

        ref_year = int(df["year"].max())
        numeric = ["vehicle_age"] + (["km_driven", "km_per_year"] if "km_driven" in nums else []) \
            + [c for c in nums if c != "km_driven"]
        features = numeric + cats
        input_fields = [f for f in INPUT_FIELD_ORDER
                        if f == "year" or f in cats or f in nums]
        ranges = {}
        for c in ["year"] + [c for c in nums]:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            ranges[c] = [float(v.min()), float(v.max())]
        state = {
            "features": features, "categorical": cats, "numeric": numeric,
            "input_fields": input_fields, "reference_year": ref_year,
            "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cats},
            "ranges": ranges, "dropped_fields": dropped, "n_rows_fit": int(n),
            "engineered": {k: v for k, v in ENGINEERED_FEATURES.items() if k in features},
        }
        return cls(state)

    # ------------------------------------------------------------------ transforming
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        age = self.reference_year - pd.to_numeric(df["year"], errors="coerce")
        X["vehicle_age"] = age.astype(float)
        if "km_driven" in self.features:
            km = pd.to_numeric(df["km_driven"], errors="coerce").astype(float)
            X["km_driven"] = km
            X["km_per_year"] = km / age.clip(lower=1)
        for c in self.numeric:
            if c not in X.columns:
                X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in self.categorical:
            X[c] = pd.Categorical(df[c].astype(object).where(df[c].notna(), None).map(
                lambda v: None if v is None else str(v)), categories=self.categories[c])
        return X[self.features]

    # ------------------------------------------------------------------ persistence
    def save(self, path=C.PROCESSOR_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path=C.PROCESSOR_PATH) -> "FeatureProcessor":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
