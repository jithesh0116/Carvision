"""Read-only catalogue queries. Every dropdown value in the UI comes from these, i.e. from the
imported dataset. Column names are whitelisted; values are always parameterised."""
from __future__ import annotations

import statistics

from database.db import query_all, query_one

FILTER_COLUMNS = {"brand", "model", "variant", "year", "fuel_type", "transmission"}
CHOICE_COLUMNS = {"fuel_type", "transmission", "owner_count", "location", "vehicle_condition"}
OPTIONAL_FIELDS = ["variant", "fuel_type", "transmission", "km_driven", "owner_count",
                   "engine_cc", "mileage", "location", "vehicle_condition"]


def _where(filters):
    clauses, params = [], []
    for col, val in (filters or {}).items():
        if val is None or val == "":
            continue
        if col not in FILTER_COLUMNS:
            raise ValueError(f"unsupported filter {col}")
        clauses.append(f"{col} = %s")
        params.append(val)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def count_vehicles() -> int:
    return int(query_one("SELECT COUNT(*) AS n FROM vehicles")["n"])


def field_coverage() -> dict:
    """Fraction of rows filled for each optional field (used before a model exists)."""
    row = query_one("SELECT COUNT(*) AS total, " +
                    ", ".join(f"COUNT({c}) AS {c}" for c in OPTIONAL_FIELDS) + " FROM vehicles")
    total = row["total"] or 1
    return {c: row[c] / total for c in OPTIONAL_FIELDS}


def list_brands():
    return [r["brand"] for r in query_all("SELECT DISTINCT brand FROM vehicles ORDER BY brand")]


def list_models(brand):
    return [r["model"] for r in query_all(
        "SELECT DISTINCT model FROM vehicles WHERE brand = %s ORDER BY model", (brand,))]


def list_variants(brand, model):
    return [r["variant"] for r in query_all(
        "SELECT DISTINCT variant FROM vehicles WHERE brand = %s AND model = %s "
        "AND variant IS NOT NULL ORDER BY variant", (brand, model))]


def list_years(brand, model, variant=None):
    where, params = _where({"brand": brand, "model": model, "variant": variant})
    return [int(r["year"]) for r in query_all(
        f"SELECT DISTINCT year FROM vehicles{where} ORDER BY year DESC", params)]


def catalog():
    return [(r["brand"], r["model"]) for r in query_all(
        "SELECT DISTINCT brand, model FROM vehicles")]


def resolve_value(column, value, filters=None):
    """Case-insensitive lookup that returns the spelling stored in the database (or None)."""
    if column not in {"brand", "model", "variant"} | CHOICE_COLUMNS:
        raise ValueError(column)
    where, params = _where(filters)
    join = " AND " if where else " WHERE "
    row = query_one(f"SELECT {column} AS v FROM vehicles{where}{join}{column} = %s LIMIT 1",
                    params + [value])
    return row["v"] if row else None


def choices_for(column, brand=None, model=None, variant=None):
    """Distinct values of a choice field for the selected vehicle; falls back to the whole dataset."""
    if column not in CHOICE_COLUMNS:
        raise ValueError(column)
    order = f"{column} + 0" if column == "owner_count" else column
    if column not in ("location", "vehicle_condition"):
        for flt in ({"brand": brand, "model": model, "variant": variant}, {"brand": brand, "model": model}):
            if not any(flt.values()) or not flt.get("brand"):
                continue
            where, params = _where(flt)
            join = " AND " if where else " WHERE "
            rows = query_all(f"SELECT DISTINCT {column} AS v FROM vehicles{where}{join}{column} IS NOT NULL "
                             f"ORDER BY {order}", params)
            if rows:
                return [r["v"] for r in rows]
    if column in ("location",):
        rows = query_all("SELECT DISTINCT location AS v FROM vehicles WHERE location IS NOT NULL "
                         "ORDER BY location")
    else:
        rows = query_all(f"SELECT DISTINCT {column} AS v FROM vehicles WHERE {column} IS NOT NULL "
                         f"ORDER BY {order}")
    return [r["v"] for r in rows]


def global_choices(column):
    return choices_for(column)


def typical_value(column, brand, model, variant=None, fuel_type=None, transmission=None):
    """Median of a numeric column among similar rows. Tries the most specific filter that has >= 3
    rows, otherwise the most specific non-empty one. Returns {value, n, basis} or None."""
    if column not in {"engine_cc", "mileage"}:
        raise ValueError(column)
    attempts = [
        ("variant, fuel and transmission", {"brand": brand, "model": model, "variant": variant,
                                            "fuel_type": fuel_type, "transmission": transmission}),
        ("variant and fuel", {"brand": brand, "model": model, "variant": variant, "fuel_type": fuel_type}),
        ("variant", {"brand": brand, "model": model, "variant": variant}),
        ("model and fuel", {"brand": brand, "model": model, "fuel_type": fuel_type}),
        ("model", {"brand": brand, "model": model}),
    ]
    fallback = None
    seen = set()
    for basis, flt in attempts:
        key = tuple(sorted((k, v) for k, v in flt.items() if v))
        if key in seen:
            continue
        seen.add(key)
        where, params = _where(flt)
        join = " AND " if where else " WHERE "
        vals = [float(r["v"]) for r in query_all(
            f"SELECT {column} AS v FROM vehicles{where}{join}{column} IS NOT NULL LIMIT 5000", params)]
        if not vals:
            continue
        cand = {"value": statistics.median(vals), "n": len(vals), "basis": basis}
        if len(vals) >= 3:
            return cand
        fallback = fallback or cand
    return fallback


def original_price_reference(brand, model, variant, year):
    """Median original (new) price recorded in the dataset for the same vehicle, if >= 3 rows exist."""
    for basis, flt in (("same variant and year", {"brand": brand, "model": model, "variant": variant, "year": year}),
                       ("same variant", {"brand": brand, "model": model, "variant": variant}),
                       ("same model", {"brand": brand, "model": model})):
        if basis != "same model" and not variant:
            continue
        where, params = _where(flt)
        join = " AND " if where else " WHERE "
        vals = [float(r["v"]) for r in query_all(
            f"SELECT original_price AS v FROM vehicles{where}{join}original_price IS NOT NULL LIMIT 5000", params)]
        if len(vals) >= 3:
            return {"price": statistics.median(vals), "n": len(vals), "basis": basis}
    return None
