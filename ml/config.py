"""Central paths and constants for the CarVision ML pipeline.

Nothing in here is vehicle data. Brands, models, variants, years, etc. always come from the
CSV you provide (via the database).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED / "splits"
IMAGES_DIR = DATA_RAW / "images"
MODELS_DIR = ROOT / "ml" / "models"

MAPPING_JSON = DATA_PROCESSED / "column_mapping.json"
DATASET_REPORT = DATA_PROCESSED / "dataset_report.json"
CLEANED_CSV = DATA_PROCESSED / "cleaned_vehicles.csv"
CLEANING_REPORT = DATA_PROCESSED / "cleaning_report.json"

XGB_MODEL_PATH = MODELS_DIR / "xgb_price_model.json"
PROCESSOR_PATH = MODELS_DIR / "feature_processor.json"
META_PATH = MODELS_DIR / "model_meta.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
CNN_CLASSES_PATH = MODELS_DIR / "cnn_classes.json"
CNN_METRICS_PATH = MODELS_DIR / "cnn_metrics.json"

# The CNN and the price model are separate components. Flip this to True ONLY if you add a
# genuinely paired image->price training pipeline. The UI text reads this flag.
IMAGE_FEATURES_IN_PRICE_MODEL = False

# Canonical fields CarVision understands (order = order of columns in the cleaned CSV).
CANONICAL_FIELDS = [
    "brand", "model", "variant", "year", "fuel_type", "transmission", "km_driven",
    "engine_cc", "mileage", "owner_count", "location", "vehicle_condition",
    "selling_price", "original_price", "image_path",
]
REQUIRED_FIELDS = ["brand", "model", "year", "selling_price"]
TEXT_FIELDS = ["brand", "model", "variant", "fuel_type", "transmission", "location", "vehicle_condition"]

# Header aliases (after normalising: lower-case, non-alphanumerics -> "_"). Priority = list order.
# "name" is a combined "Brand Model Variant" column, used only if brand/model are absent.
COLUMN_ALIASES = {
    "brand": ["brand", "make", "company", "manufacturer", "car_brand", "car_make"],
    "model": ["model", "car_model", "model_name"],
    "variant": ["variant", "trim", "version", "sub_model", "submodel", "car_variant"],
    "name": ["name", "car_name", "full_name", "title", "car_title"],
    "year": ["year", "manufacturing_year", "manufacture_year", "mfg_year", "model_year",
             "registration_year", "yr"],
    "fuel_type": ["fuel_type", "fuel", "fueltype"],
    "transmission": ["transmission", "transmission_type", "gearbox", "gear_type"],
    "km_driven": ["km_driven", "kms_driven", "kilometers_driven", "kilometres_driven",
                  "driven_kms", "odometer", "kms", "km"],
    "engine_cc": ["engine_cc", "engine", "engine_capacity", "displacement", "cc"],
    "mileage": ["mileage", "mileage_kmpl", "fuel_economy", "kmpl", "fuel_efficiency"],
    "owner_count": ["owner_count", "owner", "owners", "owner_type", "previous_owners",
                    "no_of_owners", "ownership", "number_of_owners"],
    "location": ["location", "city", "registered_city", "region", "state"],
    "vehicle_condition": ["vehicle_condition", "condition", "car_condition"],
    "selling_price": ["selling_price", "resale_price", "sale_price", "price", "price_inr",
                      "asking_price", "target"],
    "original_price": ["original_price", "new_price", "present_price", "ex_showroom_price",
                       "showroom_price"],
    "image_path": ["image_path", "image", "img", "photo", "image_url", "img_path"],
}

# Model feature selection
CATEGORICAL_CANDIDATES = ["brand", "model", "variant", "fuel_type", "transmission",
                          "location", "vehicle_condition"]
NUMERIC_CANDIDATES = ["km_driven", "owner_count", "engine_cc", "mileage"]
MIN_FEATURE_COVERAGE = 0.80     # a field must be filled in for >= 80% of rows to be used
MIN_VALID_YEAR = 1980
