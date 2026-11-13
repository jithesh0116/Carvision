"""Train the XGBoost resale-price model on the cleaned dataset.

    python -m ml.train_xgboost                # sensible defaults
    python -m ml.train_xgboost --tune         # + randomized hyper-parameter search (slower)
    python -m ml.train_xgboost --compare      # + Linear/RandomForest/HistGB baselines

Steps: load cleaned CSV -> split (train/val/calibration/test) -> fit FeatureProcessor -> train on
log1p(price) with early stopping on the validation split -> conformal calibration -> save model ->
evaluate the SAVED model on the untouched test split -> write metrics/importance/meta.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config as C
from ml import evaluate as ev
from ml import explainability
from ml.conformal import conformal_quantile
from ml.feature_processing import FeatureProcessor
from ml.predict import predict_log


def make_splits(df, test_size, cal_size, val_size, seed):
    rest, test = train_test_split(df, test_size=test_size, random_state=seed)
    cal_frac = cal_size / (1 - test_size)
    rest, cal = train_test_split(rest, test_size=cal_frac, random_state=seed)
    val_frac = val_size / (1 - test_size - cal_size)
    train, val = train_test_split(rest, test_size=val_frac, random_state=seed)
    return train, val, cal, test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--cal-size", type=float, default=0.15, help="held-out set for conformal intervals")
    ap.add_argument("--val-size", type=float, default=0.10, help="early-stopping set")
    ap.add_argument("--alpha", type=float, default=0.10, help="miscoverage; 0.10 = 90%% intervals")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n-iter", type=int, default=20, help="random-search iterations for --tune")
    ap.add_argument("--compare", action="store_true", help="also train baseline models for the report")
    args = ap.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost is not installed. Run:  pip install -r requirements.txt")
    if not C.CLEANED_CSV.exists():
        raise SystemExit("Run `python -m ml.preprocessing` first (cleaned CSV not found).")

    df = pd.read_csv(C.CLEANED_CSV)
    print(f"Loaded {len(df):,} cleaned rows.")
    train, val, cal, test = make_splits(df, args.test_size, args.cal_size, args.val_size, args.seed)
    if len(cal) < 100:
        print(f"WARNING: only {len(cal)} calibration rows; the interval will be coarse.")
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("cal", cal), ("test", test)):
        part.to_csv(C.SPLITS_DIR / f"{name}.csv", index=False)
    print(f"Split -> train {len(train):,} | val {len(val):,} | calibration {len(cal):,} | test {len(test):,}")

    proc = FeatureProcessor.fit(df)   # category vocabulary/ranges only; never looks at the price
    print("Features used:", proc.features)
    if proc.state["dropped_fields"]:
        print("Fields not used:", proc.state["dropped_fields"])

    Xtr, Xva, Xca = (proc.transform(d) for d in (train, val, cal))
    ytr, yva, yca = (np.log1p(d["selling_price"].to_numpy(float)) for d in (train, val, cal))

    base = dict(tree_method="hist", enable_categorical=True, random_state=args.seed, n_jobs=-1)
    params = dict(n_estimators=3000, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0)
    if args.tune:
        from sklearn.model_selection import RandomizedSearchCV
        space = {"max_depth": [3, 4, 5, 6, 8, 10], "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                 "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                 "min_child_weight": [1, 3, 5, 10], "reg_lambda": [0.5, 1, 3, 10]}
        search = RandomizedSearchCV(
            xgb.XGBRegressor(n_estimators=400, **base), space, n_iter=args.n_iter, cv=3,
            scoring="neg_root_mean_squared_error", random_state=args.seed, n_jobs=1, verbose=1)
        search.fit(Xtr, ytr)
        params.update(search.best_params_)
        print("Best tuned params:", search.best_params_)

    model = xgb.XGBRegressor(**params, **base, early_stopping_rounds=50, eval_metric="rmse")
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_iter = int(model.best_iteration)
    booster = model.get_booster()
    print(f"Best iteration: {best_iter}")

    # conformal calibration on data the model never trained or early-stopped on
    resid = np.abs(yca - predict_log(booster, Xca, best_iter))
    q = conformal_quantile(resid, args.alpha)
    print(f"Conformal quantile (log scale): {q:.4f}  -> intervals ≈ x{np.expm1(q) + 1:.2f} / x{1 / (np.expm1(q) + 1):.2f}")

    # save artefacts
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(C.XGB_MODEL_PATH))
    proc.save()
    importance = explainability.compute_global_importance(booster, proc.features)
    C.IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "target": "selling_price", "target_transform": "log1p",
        "features": proc.features, "input_fields": proc.state["input_fields"],
        "dropped_fields": proc.state["dropped_fields"],
        "engineered_features": proc.state["engineered"],
        "reference_year": proc.reference_year,
        "best_iteration": best_iter, "hyperparameters": params, "tuned": bool(args.tune),
        "split": {"seed": args.seed, "n_total": len(df), "n_train": len(train), "n_val": len(val),
                  "n_calibration": len(cal), "n_test": len(test)},
        "conformal": {"method": "split conformal, absolute residuals on log1p(price)",
                      "alpha": args.alpha, "nominal_coverage": round(1 - args.alpha, 4),
                      "quantile_log": q, "n_calibration": len(cal)},
        "image_features_used_for_price": C.IMAGE_FEATURES_IN_PRICE_MODEL,
    }
    C.META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # evaluate the artefact that was actually saved, on the untouched test split
    metrics = ev.evaluate_saved_model(test)
    print("\n=== TEST-SET RESULTS (real, computed from the saved model) ===")
    for k, v in metrics.items():
        print(f"  {k:24} {v}")

    if args.compare:
        ev.compare_models(train, val, test, seed=args.seed)


if __name__ == "__main__":
    main()
