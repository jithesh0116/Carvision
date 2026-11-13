"""Dynamic, dataset-driven dropdown APIs."""
from flask import Blueprint, current_app, request

from database import queries as q
from ml.cnn_model import get_recognizer
from ml.predict import get_predictor
from utils.model_state import model_meta
from utils.responses import fail, ok

bp = Blueprint("vehicles", __name__, url_prefix="/api")

DETAIL_FIELDS = ["km_driven", "fuel_type", "transmission", "owner_count", "engine_cc", "mileage",
                 "location", "vehicle_condition"]


def _arg(name):
    v = request.args.get(name, "").strip()
    return v or None


def active_input_fields() -> list[str]:
    """Raw inputs the trained model needs; before training, the fields that have data in the DB."""
    meta = model_meta()
    if meta:
        return meta["input_fields"]
    from ml import config as C
    cov = q.field_coverage()
    fields = ["brand", "model"]
    if cov.get("variant", 0) >= C.MIN_FEATURE_COVERAGE:
        fields.append("variant")
    fields.append("year")
    fields += [f for f in DETAIL_FIELDS if cov.get(f, 0) >= C.MIN_FEATURE_COVERAGE]
    return fields


@bp.get("/status")
def status():
    total = q.count_vehicles()
    cfg = current_app.config
    return ok(vehicles=total, model_ready=get_predictor().is_ready(),
              cnn_ready=get_recognizer().is_available(),
              input_fields=active_input_fields() if total else [],
              max_upload_mb=cfg["MAX_UPLOAD_MB"], require_image=cfg["REQUIRE_IMAGE"])


@bp.get("/brands")
def brands():
    return ok(brands=q.list_brands())


@bp.get("/models")
def models():
    brand = _arg("brand")
    if not brand:
        return fail("Please select your car brand.", 400, "brand")
    brand = q.resolve_value("brand", brand)
    if not brand:
        return fail("This brand is not available. Please choose a brand from the list.", 404, "brand")
    return ok(brand=brand, models=q.list_models(brand))


@bp.get("/variants")
def variants():
    brand, model = _arg("brand"), _arg("model")
    if not brand:
        return fail("Please select your car brand.", 400, "brand")
    if not model:
        return fail("Please select a model.", 400, "model")
    brand = q.resolve_value("brand", brand)
    model = q.resolve_value("model", model, {"brand": brand}) if brand else None
    if not model:
        return fail("This model is not available for the selected brand.", 404, "model")
    return ok(brand=brand, model=model, variants=q.list_variants(brand, model))


@bp.get("/years")
def years():
    brand, model, variant = _arg("brand"), _arg("model"), _arg("variant")
    if not brand:
        return fail("Please select your car brand.", 400, "brand")
    if not model:
        return fail("Please select a model.", 400, "model")
    brand = q.resolve_value("brand", brand)
    model = q.resolve_value("model", model, {"brand": brand}) if brand else None
    if not model:
        return fail("This model is not available for the selected brand.", 404, "model")
    if variant:
        variant = q.resolve_value("variant", variant, {"brand": brand, "model": model})
        if not variant:
            return fail("This variant is not available for the selected model.", 404, "variant")
    return ok(years=q.list_years(brand, model, variant))


@bp.get("/options")
def options():
    """Fields the trained model needs + the values available for the (optionally) selected vehicle."""
    fields = active_input_fields()
    brand, model, variant = _arg("brand"), _arg("model"), _arg("variant")
    fuel, trans = _arg("fuel_type"), _arg("transmission")
    if brand:
        brand = q.resolve_value("brand", brand)
    if brand and model:
        model = q.resolve_value("model", model, {"brand": brand})
    else:
        model = None
    if model and variant:
        variant = q.resolve_value("variant", variant, {"brand": brand, "model": model})
    else:
        variant = None

    choices = {}
    for col in ("fuel_type", "transmission", "owner_count", "location", "vehicle_condition"):
        if col in fields:
            if col in ("location", "vehicle_condition", "owner_count"):
                choices[col] = q.choices_for(col)
            else:
                choices[col] = q.choices_for(col, brand, model, variant) if model else q.choices_for(col)
    suggestions = {}
    if model:
        for col in ("engine_cc", "mileage"):
            if col in fields:
                fuel_v = q.resolve_value("fuel_type", fuel, {"brand": brand, "model": model}) if fuel else None
                trans_v = q.resolve_value("transmission", trans, {"brand": brand, "model": model}) if trans else None
                t = q.typical_value(col, brand, model, variant, fuel_v, trans_v)
                if t:
                    t["value"] = round(t["value"]) if col == "engine_cc" else round(t["value"], 1)
                    suggestions[col] = t
    return ok(fields=fields, choices=choices, suggestions=suggestions)
