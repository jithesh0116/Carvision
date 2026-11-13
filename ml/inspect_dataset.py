"""Inspect the real CSV and guess which column is which.

    python -m ml.inspect_dataset --csv data/raw/your_file.csv

Prints every column with dtype, fill rate and examples, proposes a mapping onto CarVision's
canonical fields, and saves it to data/processed/column_mapping.json. REVIEW THAT FILE and
edit it if a guess is wrong -- every later step (cleaning, DB import, training) follows it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

from ml import config as C


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def read_csv(path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # pragma: no cover


def guess_mapping(columns) -> dict:
    """canonical field -> CSV column name (or None). Exact alias match first, then prefix."""
    norm = {c: normalize(c) for c in columns}
    mapping, used = {}, set()
    for canon, aliases in C.COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            found = next((c for c, n in norm.items() if c not in used and n == alias), None)
            if found:
                break
        if not found:
            for alias in aliases:
                found = next((c for c, n in norm.items()
                              if c not in used and n.startswith(alias + "_")), None)
                if found:
                    break
        mapping[canon] = found
        if found:
            used.add(found)
    return mapping


def find_csv(explicit=None):
    if explicit:
        return explicit
    csvs = sorted(p for p in C.DATA_RAW.glob("*.csv"))
    if len(csvs) == 1:
        return str(csvs[0])
    sys.exit("Put exactly one CSV in data/raw/ or pass --csv path/to/file.csv "
             f"(found {len(csvs)} CSV files).")


def build_mapping(csv_path, df) -> dict:
    guess = guess_mapping(df.columns)
    return {
        "csv_path": str(csv_path),
        "columns": guess,
        "name_split": {"brand_tokens": 1, "model_tokens": 1},
        "price_multiplier": 1.0,
        "engine_unit": "auto",
        "_help": {
            "columns": "canonical field -> exact CSV column header (null if not in your data)",
            "name_split": "only used when brand/model are missing and a combined 'name' column "
                          "exists: first N tokens = brand, next M tokens = model, rest = variant",
            "price_multiplier": "multiply price values (e.g. 100000 if prices are in lakhs)",
            "engine_unit": "auto | cc | litres",
        },
    }


def ensure_mapping(csv_path=None, overwrite=False):
    """Return the mapping dict, creating and saving a guessed one if none exists."""
    if C.MAPPING_JSON.exists() and not overwrite:
        return json.loads(C.MAPPING_JSON.read_text(encoding="utf-8"))
    csv_path = find_csv(csv_path)
    df = read_csv(csv_path)
    mapping = build_mapping(csv_path, df)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    C.MAPPING_JSON.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", help="path to the vehicle price CSV (default: the only CSV in data/raw)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing column_mapping.json")
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    df = read_csv(csv_path)
    print(f"\nFile: {csv_path}\nRows: {len(df):,}   Columns: {df.shape[1]}\n")

    report = {"csv_path": str(csv_path), "rows": int(len(df)), "columns": []}
    print(f"{'column':32} {'dtype':10} {'filled':>7} {'unique':>8}  examples")
    for col in df.columns:
        s = df[col]
        filled = 100 * s.notna().mean()
        ex = [str(v)[:22] for v in s.dropna().unique()[:3]]
        print(f"{str(col)[:32]:32} {str(s.dtype):10} {filled:6.1f}% {s.nunique():8,}  {ex}")
        report["columns"].append({"name": str(col), "dtype": str(s.dtype),
                                  "filled_pct": round(filled, 2), "unique": int(s.nunique()),
                                  "examples": ex})

    guess = guess_mapping(df.columns)
    print("\nProposed mapping (canonical field -> CSV column):")
    for k, v in guess.items():
        print(f"  {k:18} -> {v}")
    missing = [f for f in C.REQUIRED_FIELDS
               if not guess.get(f) and not (f in ("brand", "model") and guess.get("name"))]
    if missing:
        print(f"\n!! Required field(s) not found: {missing}. Edit data/processed/column_mapping.json "
              "and point them at the right CSV columns.")
    if guess.get("name") and not (guess.get("brand") and guess.get("model")):
        print("\nNote: brand/model will be split from the combined name column "
              f"'{guess['name']}' (first token = brand, second = model, rest = variant). "
              "Change 'name_split' in the mapping if your brands or models have several words.")
    price = guess.get("selling_price")
    if price and pd.api.types.is_numeric_dtype(df[price]) and df[price].median() < 1000:
        print(f"\n!! Median of '{price}' is {df[price].median():,.2f}. If prices are in lakhs/thousands, "
              "set price_multiplier in the mapping (e.g. 100000).")
    unused = [c for c in df.columns if c not in guess.values()]
    if unused:
        print(f"\nColumns not mapped (kept in extra_json, not used by the model): {unused}")

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    C.DATASET_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if C.MAPPING_JSON.exists() and not args.overwrite:
        print(f"\n{C.MAPPING_JSON} already exists -- kept as is (use --overwrite to regenerate).")
    else:
        C.MAPPING_JSON.write_text(json.dumps(build_mapping(csv_path, df), indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nSaved {C.MAPPING_JSON} -- review it before continuing.")


if __name__ == "__main__":
    main()
