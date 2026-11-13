"""Server-side validation of a prediction request. Messages are written for normal users."""
from __future__ import annotations

import math
from datetime import date

from flask import current_app

from database import queries as q

CHOICE_MESSAGES = {
    "fuel_type": ("fuel type", "Please select the fuel type."),
    "transmission": ("transmission", "Please select the transmission."),
    "location": ("location", "Please select your location."),
    "vehicle_condition": ("vehicle condition", "Please select the vehicle condition."),
}


def _num(value):
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        v = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return "bad"
    return v if math.isfinite(v) else "bad"


def validate_prediction_input(payload: dict, fields: list[str]):
    """Returns (clean_inputs, errors). `fields` are the raw inputs the trained model needs."""
    cfg = current_app.config
    errors, clean = {}, {}

    def s(name):
        v = payload.get(name)
        return v.strip() if isinstance(v, str) else v

    # ---- Brand -> Model -> Variant -> Year (all checked against the database)
    brand = model = variant = None
    if not s("brand"):
        errors["brand"] = "Please select your car brand."
    else:
        brand = q.resolve_value("brand", s("brand"))
        if not brand:
            errors["brand"] = "This brand is not available. Please choose a brand from the list."
    if brand:
        if not s("model"):
            errors["model"] = "Please select a model."
        else:
            model = q.resolve_value("model", s("model"), {"brand": brand})
            if not model:
                errors["model"] = "This model is not available for the selected brand."
    if model and "variant" in fields:
        variants = q.list_variants(brand, model)
        if variants:
            if not s("variant"):
                errors["variant"] = "Please select a variant."
            else:
                variant = q.resolve_value("variant", s("variant"), {"brand": brand, "model": model})
                if not variant:
                    errors["variant"] = "This variant is not available for the selected model."
    clean.update(brand=brand, model=model, variant=variant)

    year = _num(s("year"))
    if year is None:
        errors["year"] = "Please select the manufacturing year."
    elif year == "bad" or year != int(year):
        errors["year"] = "Please select a valid manufacturing year."
    elif year > date.today().year:
        errors["year"] = "Year cannot be greater than the current year."
    elif brand and model and "variant" not in errors:
        if int(year) not in q.list_years(brand, model, variant):
            errors["year"] = "This year is not available for the selected vehicle."
    clean["year"] = int(year) if isinstance(year, float) and "year" not in errors else None

    # ---- numeric details
    def numeric(name, label_missing, label_negative, high, integer=False, positive=False):
        if name not in fields:
            return
        v = _num(s(name))
        if v is None:
            errors[name] = label_missing
        elif v == "bad":
            errors[name] = label_missing.replace("Please enter", "Please enter a valid number for")
        elif v < 0:
            errors[name] = label_negative
        elif positive and v == 0:
            errors[name] = label_missing
        elif v > high:
            errors[name] = "That value looks too high. Please check it and try again."
        else:
            clean[name] = int(round(v)) if integer else float(v)

    numeric("km_driven", "Please enter the kilometres driven.", "Kilometres driven cannot be negative.",
            cfg["MAX_KM"], integer=True)
    numeric("engine_cc", "Please enter the engine capacity.", "Engine capacity cannot be negative.",
            cfg["MAX_ENGINE_CC"], positive=True)
    numeric("mileage", "Please enter the mileage.", "Mileage cannot be negative.",
            cfg["MAX_MILEAGE"], positive=True)
    if "km_driven" in errors and errors["km_driven"].startswith("Please enter a valid number"):
        errors["km_driven"] = "Please enter a valid number of kilometres."

    # ---- Location (any city selected is valid)
    if "location" in fields:
        loc_val = s("location")
        if not loc_val:
            errors["location"] = "Please select your city."
        else:
            clean["location"] = str(loc_val).strip()

    # ---- Vehicle condition (any standard condition is valid)
    if "vehicle_condition" in fields:
        cond_val = s("vehicle_condition")
        if not cond_val:
            errors["vehicle_condition"] = "Please select the vehicle condition."
        else:
            conds = {c.lower(): c for c in ["Excellent", "Good", "Average", "Needs Repair"]}
            if str(cond_val).lower() in conds:
                clean["vehicle_condition"] = conds[str(cond_val).lower()]
            else:
                errors["vehicle_condition"] = "Please select a valid vehicle condition."

    # ---- Fuel type & Transmission
    for name, (label, missing) in [("fuel_type", ("fuel type", "Please select the fuel type.")),
                                    ("transmission", ("transmission", "Please select the transmission."))]:
        if name not in fields:
            continue
        val = s(name)
        if not val:
            errors[name] = missing
            continue
        allowed = q.choices_for(name, brand, model, variant) if brand and model else q.choices_for(name)
        match = next((a for a in allowed if str(a).lower() == str(val).lower()), None)
        if match is None:
            match = next((a for a in q.choices_for(name) if str(a).lower() == str(val).lower()), None)
        if match is None:
            errors[name] = f"Please select a valid {label}."
        else:
            clean[name] = match

    if "owner_count" in fields:
        v = _num(s("owner_count"))
        if v is None:
            errors["owner_count"] = "Please select the number of previous owners."
        elif v == "bad" or int(v) < 1:
            errors["owner_count"] = "Please select a valid number of previous owners."
        else:
            clean["owner_count"] = int(v)

    return clean, errors
