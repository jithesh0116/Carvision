"""Model evaluation (MAE, RMSE, R², plus interval coverage) and optional baseline comparison.

    python -m ml.evaluate              # re-evaluate the saved XGBoost model on data/processed/splits/test.csv
    python -m ml.evaluate --compare    # also train/evaluate baseline models on the same splits

All numbers are computed here from real predictions and written to ml/models/*.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml import config as C
from ml.conformal import empirical_coverage, interval_from_log
from ml.feature_processing import FeatureProcessor


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "MAPE_pct": round(float(np.mean(np.abs(y_true - y_pred) / y_true) * 100), 2),
        "median_abs_error": round(float(np.median(np.abs(y_true - y_pred))), 2),
    }


def evaluate_saved_model(test_df: pd.DataFrame | None = None) -> dict:
    import xgboost as xgb
    from ml.predict import predict_log
    if test_df is None:
        test_df = pd.read_csv(C.SPLITS_DIR / "test.csv")
    proc = FeatureProcessor.load()
    meta = json.loads(C.META_PATH.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(C.XGB_MODEL_PATH))
    X = proc.transform(test_df)
    y = test_df["selling_price"].to_numpy(float)
    pred_log = predict_log(booster, X, int(meta["best_iteration"]))
    m = regression_metrics(y, np.expm1(pred_log))
    lo, hi = interval_from_log(pred_log, float(meta["conformal"]["quantile_log"]))
    m["interval_nominal_coverage"] = meta["conformal"]["nominal_coverage"]
    m["interval_coverage"] = round(empirical_coverage(y, lo, hi), 4)
    m["interval_median_width"] = round(float(np.median(hi - lo)), 2)
    m["n_test"] = int(len(test_df))
    C.METRICS_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def _codes(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric copy of X where categories become integer codes (-1 = missing)."""
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes.astype(float)
    return out


def compare_models(train, val, test, seed=42) -> dict:
    """Same features and same split for every model; metrics on the original price scale."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    proc = FeatureProcessor.load()
    fit_df = pd.concat([train, val])
    Xf, Xt = proc.transform(fit_df), proc.transform(test)
    yf, yt = np.log1p(fit_df["selling_price"].to_numpy(float)), test["selling_price"].to_numpy(float)
    cats, nums = proc.categorical, proc.numeric

    def onehot_frame(X):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
        return X

    linear = make_pipeline(
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", make_pipeline(SimpleImputer(strategy="median")), nums)]),
        LinearRegression())
    results = {}

    def run(name, model, Xa, Xb):
        model.fit(Xa, yf)
        results[name] = regression_metrics(yt, np.expm1(model.predict(Xb)))
        print(f"  {name:22}", results[name])

    print("\n=== BASELINE COMPARISON (test split, original price scale) ===")
    run("LinearRegression", linear, onehot_frame(Xf), onehot_frame(Xt))
    Cf, Ct = _codes(Xf), _codes(Xt)
    run("RandomForest", make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)),
        Cf, Ct)
    run("HistGradientBoosting", HistGradientBoostingRegressor(random_state=seed), Cf, Ct)
    results["XGBoost (saved model)"] = {k: v for k, v in json.loads(C.METRICS_PATH.read_text()).items()
                                        if k in ("MAE", "RMSE", "R2", "MAPE_pct", "median_abs_error")}
    print(f"  {'XGBoost (saved model)':22}", results["XGBoost (saved model)"])
    C.COMPARISON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if not C.META_PATH.exists():
        raise SystemExit("No trained model found. Run `python -m ml.train_xgboost` first.")
    print(json.dumps(evaluate_saved_model(), indent=2))
    if args.compare:
        s = C.SPLITS_DIR
        compare_models(pd.read_csv(s / "train.csv"), pd.read_csv(s / "val.csv"), pd.read_csv(s / "test.csv"))


if __name__ == "__main__":
    main()
