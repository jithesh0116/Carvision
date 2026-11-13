"""Import the cleaned dataset into MySQL:

    python -m database.seed [--csv data/processed/cleaned_vehicles.csv] [--truncate]

Run `python -m ml.preprocessing` first. The same cleaned file is what the model trains on, so the
dropdowns and the model always agree. Nothing is generated here -- rows are copied from your CSV.
"""
import argparse
import sys

import pandas as pd

from config import Config
from database.db import connect
from ml import config as C

COLUMNS = ["brand", "model", "variant", "year", "fuel_type", "transmission", "km_driven",
           "engine_cc", "mileage", "owner_count", "location", "vehicle_condition",
           "selling_price", "original_price", "image_path", "extra_json"]
INT_COLS = {"year", "km_driven", "owner_count"}


def py(col, v):
    if v is None or pd.isna(v):
        return None
    if col in INT_COLS:
        return int(round(float(v)))
    if col in {"engine_cc", "mileage", "selling_price", "original_price"}:
        return float(v)
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(C.CLEANED_CSV))
    ap.add_argument("--truncate", action="store_true", help="empty the vehicles table first")
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"{args.csv} not found. Run `python -m ml.preprocessing` first.")
    cols = [c for c in COLUMNS if c in df.columns]
    conn = connect(Config)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS n FROM vehicles")
        existing = cur.fetchone()["n"]
        if existing and not args.truncate:
            sys.exit(f"vehicles already has {existing:,} rows. Re-run with --truncate to replace them.")
        if args.truncate:
            cur.execute("TRUNCATE TABLE vehicles")
        sql = f"INSERT INTO vehicles ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
        rows = [tuple(py(c, v) for c, v in zip(cols, r)) for r in df[cols].itertuples(index=False, name=None)]
        for i in range(0, len(rows), args.batch):
            cur.executemany(sql, rows[i:i + args.batch])
        print(f"Imported {len(rows):,} vehicles ({len(cols)} columns: {cols}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
