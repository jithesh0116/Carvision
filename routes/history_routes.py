"""Prediction history (private to the browser session) and dashboard data."""
import json

from flask import Blueprint, request

from database import queries as q
from database.db import execute, query_all, query_one
from ml import config as C
from utils.model_state import cnn_metrics, model_meta, model_metrics
from utils.responses import fail, ok
from utils.session import current_sid

bp = Blueprint("history", __name__, url_prefix="/api")


def _row(r, full=False):
    d = {"id": r["id"], "created_at": r["created_at"].isoformat(), "brand": r["brand"], "model": r["model"],
         "variant": r["variant"], "year": r["year"], "predicted_price": float(r["predicted_price"]),
         "lower_price": float(r["lower_price"]), "upper_price": float(r["upper_price"])}
    if full:
        d["result"] = json.loads(r["result_json"]) if isinstance(r["result_json"], str) else r["result_json"]
    return d


@bp.get("/history")
def history():
    limit = min(max(request.args.get("limit", 50, type=int), 1), 100)
    offset = max(request.args.get("offset", 0, type=int), 0)
    sid = current_sid()
    rows = query_all("SELECT id, created_at, brand, model, variant, year, predicted_price, lower_price, upper_price "
                     "FROM predictions WHERE session_id = %s ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                     (sid, limit, offset))
    total = query_one("SELECT COUNT(*) AS n FROM predictions WHERE session_id = %s", (sid,))["n"]
    return ok(items=[_row(r) for r in rows], total=total)


@bp.get("/history/<int:pid>")
def history_item(pid):
    r = query_one("SELECT * FROM predictions WHERE id = %s AND session_id = %s", (pid, current_sid()))
    if not r:
        return fail("We couldn't find that estimate.", 404)
    return ok(item=_row(r, full=True))


@bp.delete("/history/<int:pid>")
def history_delete(pid):
    execute("DELETE FROM predictions WHERE id = %s AND session_id = %s", (pid, current_sid()))
    return ok()


@bp.get("/dashboard")
def dashboard():
    sid = current_sid()
    agg = query_one("SELECT COUNT(*) AS n, AVG(predicted_price) AS avg_price FROM predictions WHERE session_id = %s",
                    (sid,))
    brands = query_all("SELECT brand, COUNT(*) AS n FROM predictions WHERE session_id = %s "
                       "GROUP BY brand ORDER BY n DESC, brand LIMIT 6", (sid,))
    recent = query_all("SELECT id, created_at, brand, model, variant, year, predicted_price, lower_price, upper_price "
                       "FROM predictions WHERE session_id = %s ORDER BY created_at DESC, id DESC LIMIT 5", (sid,))
    metrics, meta = model_metrics(), model_meta()
    return ok(
        total_predictions=agg["n"],
        average_value=float(agg["avg_price"]) if agg["avg_price"] is not None else None,
        top_brand=brands[0]["brand"] if brands else None,
        brand_counts=[{"brand": b["brand"], "count": b["n"]} for b in brands],
        recent=[_row(r) for r in recent],
        dataset={"vehicles": q.count_vehicles(), "brands": len(q.list_brands())},
        model=None if not metrics else {
            "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"],
            "interval_coverage": metrics.get("interval_coverage"),
            "n_test": metrics.get("n_test"), "n_train": (meta or {}).get("split", {}).get("n_train"),
            "trained_at": (meta or {}).get("trained_at")},
        cnn=cnn_metrics(),
    )
