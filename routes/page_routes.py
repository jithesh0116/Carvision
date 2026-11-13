"""HTML pages."""
import json

from flask import Blueprint, render_template

from database.db import query_one
from ml import config as C
from utils.session import current_sid

bp = Blueprint("pages", __name__)


@bp.get("/home")
@bp.get("/")
def index():
    return render_template("index.html", image_in_price=C.IMAGE_FEATURES_IN_PRICE_MODEL)


@bp.get("/predict")
def predict_page():
    return render_template("predict.html")


@bp.get("/result/<int:pid>")
def result(pid):
    row = query_one("SELECT id, created_at, result_json FROM predictions WHERE id = %s AND session_id = %s",
                    (pid, current_sid()))
    if not row:
        return render_template("error.html", title="Estimate not found",
                               message="We couldn't find that estimate. It may belong to a different browser session."), 404
    r = json.loads(row["result_json"]) if isinstance(row["result_json"], str) else row["result_json"]
    return render_template("result.html", r=r, pid=row["id"], created_at=row["created_at"])


@bp.get("/history")
def history_page():
    return render_template("history.html")


@bp.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")
