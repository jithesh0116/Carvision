"""Flask configuration. Secrets and credentials come from environment variables / .env only."""
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY")

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("DB_PORT", "3306"))
    DB_USER = os.environ.get("DB_USER", "")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME = os.environ.get("DB_NAME", "carvision_db")

    UPLOAD_DIR = BASE_DIR / "static" / "uploads"
    MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "8"))
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024) + 1024 * 64
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    REQUIRE_IMAGE = _bool("REQUIRE_IMAGE", True)

    CNN_CONFIDENCE_THRESHOLD = float(os.environ.get("CNN_CONFIDENCE_THRESHOLD", "0.60"))
    VEHICLE_VALIDATION_THRESHOLD = float(os.environ.get("VEHICLE_VALIDATION_THRESHOLD", "0.30"))

    CURRENCY_SYMBOL = os.environ.get("CURRENCY_SYMBOL", "₹")
    CURRENCY_LOCALE = os.environ.get("CURRENCY_LOCALE", "en-IN")
    PRICE_ROUNDING = int(os.environ.get("PRICE_ROUNDING", "1000"))

    # Input sanity limits (validation only)
    MAX_KM = 1_500_000
    MAX_ENGINE_CC = 20_000
    MAX_MILEAGE = 200

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(days=365)
