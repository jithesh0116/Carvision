"""Image upload (+ optional recognition) and resale-price prediction."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from flask import Blueprint, current_app, request, session, url_for

from database import queries as q
from database.db import execute
from ml import config as C
from ml import explainability
from ml.cnn_model import get_recognizer, match_label_to_catalog
from ml.predict import ModelNotReadyError, get_predictor
from ml.vehicle_validator import get_vehicle_validator
from routes.vehicle_routes import active_input_fields
from utils.format import round_price
from utils.images import ImageError, is_saved_name, save_image
from utils.model_state import model_metrics, model_meta
from utils.responses import fail, ok
from utils.session import current_sid
from utils.validation import validate_prediction_input

bp = Blueprint("prediction", __name__, url_prefix="/api")


@bp.post("/upload-image")
def upload_image():
    cfg = current_app.config
    file = request.files.get("image")
    if not file or not file.filename:
        return fail("Please upload a car image.", 400, "image")
    try:
        saved = save_image(file, cfg["UPLOAD_DIR"], cfg["ALLOWED_IMAGE_EXTENSIONS"],
                           int(cfg["MAX_UPLOAD_MB"] * 1024 * 1024))
    except ImageError as exc:
        return fail(str(exc), 400, "image")

    # Strict CNN Vehicle Validation
    validator = get_vehicle_validator()
    val_res = validator.validate(saved["path"], threshold=cfg.get("VEHICLE_VALIDATION_THRESHOLD", 0.30))
    if not val_res.is_vehicle:
        # Do not save the invalid image as a valid vehicle upload
        try:
            Path(saved["path"]).unlink(missing_ok=True)
            validator.evict_cached(saved["filename"])
        except Exception:
            pass
        return fail("No vehicle detected. Please upload a clear car image.", 400, "image",
                    vehicle_confidence=round(val_res.confidence, 4),
                    top_category=val_res.top_category)

    detection = None
    rec = get_recognizer()
    if rec.is_available():
        try:
            label, conf = rec.top_k(saved["path"], 1)[0]
            match = match_label_to_catalog(label, q.catalog())
            detection = {"label": label, "confidence": round(conf, 4),
                         "confident": conf >= cfg["CNN_CONFIDENCE_THRESHOLD"],
                         "brand": match[0] if match else None, "model": match[1] if match else None}
        except Exception:
            current_app.logger.exception("Image recognition failed")
    return ok(filename=saved["filename"], url=url_for("static", filename=f"uploads/{saved['filename']}"),
              detection=detection,
              vehicle_confidence=round(val_res.confidence, 4))


def _depreciation(clean, price, symbol_cfg):
    ref = q.original_price_reference(clean["brand"], clean["model"], clean.get("variant"), clean["year"])
    if not ref or ref["price"] <= 0:
        return None
    pct = (ref["price"] - price) / ref["price"] * 100
    if not 0 <= pct <= 100:
        return None
    return {"percent": round(pct, 1), "reference_price": round(ref["price"]), "n": ref["n"],
            "basis": ref["basis"],
            "explanation": ("Estimated value compared with the median original (new) price recorded in the "
                            f"dataset for the {ref['basis']} ({ref['n']} listings): "
                            "(original price - estimated value) / original price.")}


@bp.post("/predict")
def predict():
    cfg = current_app.config
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("Something went wrong while generating the estimate. Please try again.", 400)

    image = payload.get("image")
    if image:
        image_path = cfg["UPLOAD_DIR"] / image
        if not is_saved_name(image) or not image_path.exists():
            return fail("Please upload a car image.", 422, "image", errors={"image": "Please upload a car image."})
        validator = get_vehicle_validator()
        val_res = validator.validate(image_path, threshold=cfg.get("VEHICLE_VALIDATION_THRESHOLD", 0.30))
        if not val_res.is_vehicle:
            try:
                image_path.unlink(missing_ok=True)
                validator.evict_cached(image)
            except Exception:
                pass
            return fail("No vehicle detected. Please upload a clear car image.", 422, "image",
                        errors={"image": "No vehicle detected. Please upload a clear car image."})
    elif cfg["REQUIRE_IMAGE"]:
        return fail("Please upload a car image.", 422, "image", errors={"image": "Please upload a car image."})

    predictor = get_predictor()
    if not predictor.is_ready():
        return fail("Price estimates are not available yet because the price model has not been trained.", 503)

    fields = active_input_fields()
    clean, errors = validate_prediction_input(payload, fields)
    if errors:
        first = next(iter(errors))
        return fail(errors[first], 422, first, errors=errors)

    try:
        res = predictor.predict({k: v for k, v in clean.items() if v is not None})
    except ModelNotReadyError:
        return fail("Price estimates are not available yet because the price model has not been trained.", 503)
    except Exception:
        current_app.logger.exception("Prediction failed")
        return fail("Something went wrong while generating the estimate. Please try again.", 500)

    step = cfg["PRICE_ROUNDING"]
    est, lo, hi = (round_price(res[k], step) for k in ("estimate", "lower", "upper"))
    meta, metrics = model_meta() or {}, model_metrics() or {}
    local = res["local_explanation"]
    result = {
        "vehicle": {**{k: v for k, v in clean.items()},
                    "age_years": max(0, date.today().year - clean["year"])},
        "price": {"estimate": est, "lower": lo, "upper": hi,
                  "nominal_coverage": meta.get("conformal", {}).get("nominal_coverage"),
                  "empirical_coverage": metrics.get("interval_coverage"),
                  "method": meta.get("conformal", {}).get("method")},
        "depreciation": _depreciation(clean, est, cfg),
        "explanation": {"local": local, "overall": explainability.load_global_importance(),
                        "summary": explainability.plain_summary(local)},
        "warnings": res["warnings"],
        "image": image,
        "model": {"trained_at": meta.get("trained_at"), "MAE": metrics.get("MAE"), "RMSE": metrics.get("RMSE"),
                  "R2": metrics.get("R2"), "n_train": meta.get("split", {}).get("n_train"),
                  "n_test": metrics.get("n_test"), "image_used_for_price": C.IMAGE_FEATURES_IN_PRICE_MODEL},
    }
    pid = execute(
        "INSERT INTO predictions (session_id, brand, model, variant, year, km_driven, fuel_type, transmission, "
        "owner_count, engine_cc, mileage, location, vehicle_condition, predicted_price, lower_price, upper_price, "
        "image_path, result_json) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (current_sid(), clean["brand"], clean["model"], clean.get("variant"), clean["year"],
         clean.get("km_driven"), clean.get("fuel_type"), clean.get("transmission"), clean.get("owner_count"),
         clean.get("engine_cc"), clean.get("mileage"), clean.get("location"), clean.get("vehicle_condition"),
         est, lo, hi, image, json.dumps(result, ensure_ascii=False)))
    return ok(id=pid, redirect=url_for("pages.result", pid=pid))
