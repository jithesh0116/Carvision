"""Data cleaning: raw CSV -> cleaned canonical table used by BOTH the database import and training.

    python -m ml.preprocessing

Pipeline: read -> apply column mapping -> (split combined name) -> parse numbers/text ->
drop invalid rows -> remove duplicates -> write data/processed/cleaned_vehicles.csv and
cleaning_report.json. Missing optional values stay missing (NULL in DB, native NaN handling in
XGBoost); nothing is invented or imputed with made-up values.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from ml import config as C
from ml.inspect_dataset import ensure_mapping, read_csv


class DatasetError(Exception):
    """Raised with a human-readable message when the dataset cannot be used."""


NA_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "unknown"}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ------------------------------------------------------------------ parsing helpers
def parse_number(s: pd.Series) -> pd.Series:
    """'45,000 km' -> 45000.0 ; '23.4 kmpl' -> 23.4 ; unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    txt = s.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(txt.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").astype(float)


def parse_price(s: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Numbers, or strings such as '5.5 Lakh' / '1.2 Crore'."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float) * multiplier
    txt = s.astype("string").str.lower().str.replace(",", "", regex=False)
    num = pd.to_numeric(txt.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce").astype(float)
    unit = pd.Series(1.0, index=s.index)
    unit[txt.str.contains(r"\b(?:lakh|lac|lakhs|lacs)\b", regex=True, na=False)] = 1e5
    unit[txt.str.contains(r"\b(?:crore|cr|crores)\b", regex=True, na=False)] = 1e7
    return num * unit * multiplier


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def parse_owner(s: pd.Series) -> pd.Series:
    """1 / '1st' / 'First Owner' / 'Fourth & Above Owner' -> integer; 'Test Drive Car' -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return np.nan
        t = str(v).lower()
        m = _NUM.search(t)
        if m:
            return float(m.group())
        for word, n in _ORDINALS.items():
            if word in t:
                return float(n)
        return np.nan

    return s.map(one).astype(float)


def unify_text(s: pd.Series) -> pd.Series:
    """Trim/collapse spaces, blank markers -> NA, and merge case variants ('diesel'/'Diesel')
    into the most frequent spelling."""
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    lower = s.str.lower()
    s = s.mask(lower.isin(NA_TOKENS))
    lower = s.str.lower()
    if s.notna().sum() == 0:
        return s
    canon = s.groupby(lower).agg(lambda x: x.mode().iat[0])
    return lower.map(canon).astype("string")


def _split_name(names: pd.Series, bt: int, mt: int):
    toks = names.astype("string").str.strip().str.split(r"\s+", regex=True)

    def part(t, a, b):
        if not isinstance(t, list):
            return pd.NA
        piece = " ".join(t[a:b]) if b is not None else " ".join(t[a:])
        return piece if piece else pd.NA

    brand = toks.map(lambda t: part(t, 0, bt))
    model = toks.map(lambda t: part(t, bt, bt + mt))
    variant = toks.map(lambda t: part(t, bt + mt, None))
    return brand, model, variant


# ------------------------------------------------------------------ main cleaning routine
def clean(raw: pd.DataFrame, mapping: dict):
    cols = mapping["columns"]
    rep = {"rows_raw": int(len(raw)), "removed": {}, "notes": []}
    out = pd.DataFrame(index=raw.index)

    for canon in C.CANONICAL_FIELDS:
        src = cols.get(canon)
        if src and src in raw.columns:
            out[canon] = raw[src]

    name_src = cols.get("name")
    if name_src and name_src in raw.columns and ("brand" not in out or "model" not in out):
        sp = mapping.get("name_split", {})
        bt, mt = int(sp.get("brand_tokens", 1)), int(sp.get("model_tokens", 1))
        b, m, v = _split_name(raw[name_src], bt, mt)
        if "brand" not in out:
            out["brand"] = b
        if "model" not in out:
            out["model"] = m
        if "variant" not in out:
            out["variant"] = v
        rep["notes"].append(f"brand/model/variant split from '{name_src}' "
                            f"(brand_tokens={bt}, model_tokens={mt})")

    # keep every unmapped column, untouched, for the DB's extra_json
    used = {v for v in cols.values() if v}
    extras = raw[[c for c in raw.columns if c not in used]]

    missing = [f for f in C.REQUIRED_FIELDS if f not in out.columns]
    if missing:
        raise DatasetError(
            f"The dataset is missing required field(s): {missing}. Edit "
            f"{C.MAPPING_JSON.name} so they point at the right CSV columns. CarVision needs "
            "brand, model, year and selling price (brand/model can come from a combined name "
            "column that starts with the brand).")

    # ---- parse
    mult = float(mapping.get("price_multiplier", 1.0))
    out["year"] = parse_number(out["year"]).round()
    out["selling_price"] = parse_price(out["selling_price"], mult)
    if "original_price" in out:
        out["original_price"] = parse_price(out["original_price"], mult)
    for c in ("km_driven", "engine_cc", "mileage"):
        if c in out:
            out[c] = parse_number(out[c])
    if "owner_count" in out:
        out["owner_count"] = parse_owner(out["owner_count"])
    if "image_path" in out:
        out["image_path"] = out["image_path"].astype("string").str.strip()
    for c in C.TEXT_FIELDS:
        if c in out:
            out[c] = unify_text(out[c])

    # engine given in litres (e.g. 1.5) -> cc
    if "engine_cc" in out and mapping.get("engine_unit", "auto") in ("auto", "litres"):
        med = out["engine_cc"].median()
        if mapping.get("engine_unit") == "litres" or (pd.notna(med) and med < 20):
            out["engine_cc"] = out["engine_cc"] * 1000
            rep["notes"].append("engine values looked like litres; converted to cc (x1000)")

    # ---- drop invalid rows (each reason counted)
    def drop(mask_bad, reason):
        nonlocal out
        n = int(mask_bad.sum())
        if n:
            rep["removed"][reason] = n
            out = out[~mask_bad]

    drop(out[["brand", "model", "year", "selling_price"]].isna().any(axis=1),
         "missing brand/model/year/selling_price")
    drop((out["selling_price"] <= 0), "selling_price <= 0")
    drop((out["year"] < C.MIN_VALID_YEAR) | (out["year"] > date.today().year + 1),
         "year outside valid range")

    # invalid optional numerics become missing (row is kept)
    for c, bad in (("km_driven", lambda x: x < 0), ("engine_cc", lambda x: x <= 0),
                   ("mileage", lambda x: x <= 0), ("owner_count", lambda x: x < 0),
                   ("original_price", lambda x: x <= 0)):
        if c in out:
            m = bad(out[c]).fillna(False)
            if m.any():
                rep["notes"].append(f"{int(m.sum())} invalid '{c}' values set to missing")
                out.loc[m, c] = np.nan

    if "mileage" in out and out["mileage"].median() > 200:
        rep["notes"].append(f"WARNING: 'mileage' median is {out['mileage'].median():,.0f}; it may be an "
                            "odometer reading rather than fuel economy. Check column_mapping.json.")
    if out["selling_price"].median() < 1000:
        rep["notes"].append(f"WARNING: median selling_price is {out['selling_price'].median():,.2f}. If prices "
                            "are in lakhs/thousands, set price_multiplier in column_mapping.json.")

    # ---- integer-typed columns
    out["year"] = out["year"].astype(int)
    for c in ("km_driven", "owner_count"):
        if c in out:
            out[c] = out[c].round().astype("Int64")

    # ---- extras + duplicates
    if extras.shape[1]:
        recs = json.loads(extras.loc[out.index].to_json(orient="records", date_format="iso"))
        out["extra_json"] = [json.dumps(r, ensure_ascii=False) for r in recs]
    else:
        out["extra_json"] = None
    before = len(out)
    out = out.drop_duplicates(subset=[c for c in out.columns if c != "extra_json"])
    if before - len(out):
        rep["removed"]["exact duplicate rows"] = int(before - len(out))

    ordered = [c for c in C.CANONICAL_FIELDS if c in out.columns] + ["extra_json"]
    out = out[ordered].reset_index(drop=True)

    rep["rows_clean"] = int(len(out))
    rep["fill_rate_pct"] = {c: round(100 * float(out[c].notna().mean()), 2)
                            for c in ordered if c != "extra_json"}
    rep["fields_absent_from_dataset"] = [f for f in C.CANONICAL_FIELDS if f not in out.columns]
    rep["selling_price_summary"] = {k: round(float(v), 2) for k, v in
                                    out["selling_price"].describe()[["min", "50%", "max"]].items()}
    rep["year_range"] = [int(out["year"].min()), int(out["year"].max())]
    rep["unique_counts"] = {c: int(out[c].nunique()) for c in C.TEXT_FIELDS if c in out.columns}
    return out, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", help="override the CSV path stored in column_mapping.json")
    args = ap.parse_args()
    mapping = ensure_mapping(args.csv)
    csv_path = args.csv or mapping["csv_path"]
    raw = read_csv(csv_path)
    try:
        df, rep = clean(raw, mapping)
    except DatasetError as exc:
        raise SystemExit(f"\nERROR: {exc}")
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEANED_CSV, index=False)
    C.CLEANING_REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nSaved {C.CLEANED_CSV}")


if __name__ == "__main__":
    main()
