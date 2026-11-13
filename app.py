"""CarVision -- Flask application factory.   Run:  python app.py"""
import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from config import Config
from database import db
from database.db import DatabaseUnavailable
from ml import config as mlc
from utils.format import format_money


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY is not set. Copy .env.example to .env and set SECRET_KEY "
                           "(python -c \"import secrets; print(secrets.token_hex(32))\").")
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    db.init_app(app)

    from routes.history_routes import bp as history_bp
    from routes.page_routes import bp as pages_bp
    from routes.prediction_routes import bp as prediction_bp
    from routes.vehicle_routes import bp as vehicle_bp
    for bp in (pages_bp, vehicle_bp, prediction_bp, history_bp):
        app.register_blueprint(bp)

    app.jinja_env.filters["money"] = lambda v: format_money(
        v, app.config["CURRENCY_SYMBOL"], app.config["CURRENCY_LOCALE"])
    app.jinja_env.filters["nice_date"] = lambda d: d.strftime("%d %b %Y") if d else ""

    @app.context_processor
    def inject_config():
        cfg = app.config
        return {"cv_config": {"currency": cfg["CURRENCY_SYMBOL"], "locale": cfg["CURRENCY_LOCALE"],
                              "requireImage": cfg["REQUIRE_IMAGE"], "maxUploadMb": cfg["MAX_UPLOAD_MB"]}}

    def wants_json():
        return request.path.startswith("/api/")

    def respond(message, status):
        if wants_json():
            return jsonify({"ok": False, "error": message}), status
        return render_template("error.html", title="Something went wrong", message=message), status

    @app.errorhandler(DatabaseUnavailable)
    def _db(exc):
        app.logger.error("Database problem: %s", exc)
        return respond("We couldn't reach the vehicle database right now. Please try again in a moment.", 503)

    @app.errorhandler(413)
    def _too_big(_exc):
        return respond(f"That photo is too large. Please upload an image under {int(app.config['MAX_UPLOAD_MB'])} MB.", 413)

    @app.errorhandler(404)
    def _404(_exc):
        return respond("We couldn't find that page.", 404)

    @app.errorhandler(405)
    def _405(_exc):
        return respond("That action isn't available here.", 405)

    @app.errorhandler(Exception)
    def _any(exc):
        if isinstance(exc, HTTPException):
            return respond(exc.description if exc.code < 500 else "Something went wrong. Please try again.", exc.code)
        app.logger.exception("Unhandled error")
        return respond("Something went wrong. Please try again.", 500)

    return app


app = create_app()   # fails with a clear message if SECRET_KEY is missing

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
