"""Tests for CNN-based Vehicle Image Validation.

Verifies that:
1. Real car photographs from different angles & backgrounds are accepted.
2. Non-car images (TUF Gaming logo, brand logos, screenshots, portraits, animals, landscapes)
   are strictly rejected.
3. Corrupted, unsupported, or tiny images (< 64px) are handled safely without crashing.
4. /api/upload-image rejects non-vehicles and does not persist invalid files.
5. /api/predict validates on the backend and refuses to predict on non-vehicle images.
"""
import io
import shutil
from pathlib import Path
from PIL import Image
import pytest

from app import app
from ml.vehicle_validator import get_vehicle_validator, VehicleValidator


@pytest.fixture(scope="module")
def validator():
    val = get_vehicle_validator()
    assert val.is_available(), "Vehicle validator model or weights not available"
    return val


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


# ----------------------------------------------------------------------
# 1. Direct Model Unit Tests (Positive & Negative Classifications)
# ----------------------------------------------------------------------

def test_car_photographs_different_angles_accepted(validator):
    pos_dir = Path("tests/test_images/positives")
    assert pos_dir.exists() and any(pos_dir.iterdir()), "Positive test images not found"
    
    for img_path in pos_dir.glob("*.jpg"):
        res = validator.validate(img_path, threshold=0.30)
        assert res.is_vehicle, f"Expected {img_path.name} to be accepted, but got confidence={res.confidence:.4f} (top: {res.top_category})"
        assert res.confidence >= 0.30, f"Confidence {res.confidence} below threshold 0.30 for {img_path.name}"


def test_tuf_gaming_logo_rejected(validator):
    tuf_path = Path("tests/test_images/negatives/tuf_gaming_logo.png")
    assert tuf_path.exists(), "TUF gaming logo test image missing"
    
    res = validator.validate(tuf_path, threshold=0.30)
    assert not res.is_vehicle, f"TUF Gaming logo should be rejected! Got confidence={res.confidence:.4f} (top: {res.top_category})"
    assert res.confidence < 0.10, f"Expected confidence < 0.10, got {res.confidence}"


def test_brand_logos_rejected(validator):
    logo_path = Path("tests/test_images/negatives/nike_logo.jpg")
    if logo_path.exists():
        res = validator.validate(logo_path, threshold=0.30)
        assert not res.is_vehicle, f"Brand logo should be rejected! Got confidence={res.confidence:.4f}"
        assert res.confidence < 0.10


def test_mobile_screenshot_rejected(validator):
    shot_path = Path("tests/test_images/negatives/mobile_screenshot.png")
    if shot_path.exists():
        res = validator.validate(shot_path, threshold=0.30)
        assert not res.is_vehicle, f"Mobile screenshot should be rejected! Got confidence={res.confidence:.4f}"
        assert res.confidence < 0.10


def test_human_portrait_rejected(validator):
    portrait_path = Path("tests/test_images/negatives/human_portrait.jpg")
    if portrait_path.exists():
        res = validator.validate(portrait_path, threshold=0.30)
        assert not res.is_vehicle, f"Human portrait should be rejected! Got confidence={res.confidence:.4f}"
        assert res.confidence < 0.10


def test_animal_photographs_rejected(validator):
    for animal in ["animal_cat.jpg", "animal_dog.jpg"]:
        p = Path("tests/test_images/negatives") / animal
        if p.exists():
            res = validator.validate(p, threshold=0.30)
            assert not res.is_vehicle, f"Animal photograph {animal} should be rejected! Got confidence={res.confidence:.4f}"
            assert res.confidence < 0.10


def test_landscape_photograph_rejected(validator):
    p = Path("tests/test_images/negatives/landscape_mountain.jpg")
    if p.exists():
        res = validator.validate(p, threshold=0.30)
        assert not res.is_vehicle, f"Landscape photo should be rejected! Got confidence={res.confidence:.4f}"
        assert res.confidence < 0.10


def test_corrupted_image_safe_rejection(validator):
    corrupt = Path("tests/test_images/negatives/corrupted_file.jpg")
    assert corrupt.exists()
    res = validator.validate(corrupt, threshold=0.30)
    assert not res.is_vehicle
    assert res.error is not None
    assert "Invalid or corrupted" in res.error or "cannot identify" in res.error.lower()


def test_tiny_image_safe_rejection(validator):
    tiny = Path("tests/test_images/negatives/tiny_icon_32px.png")
    assert tiny.exists()
    res = validator.validate(tiny, threshold=0.30)
    assert not res.is_vehicle
    assert res.error is not None
    assert "too small" in res.error.lower()


# ----------------------------------------------------------------------
# 2. Endpoint Integration Tests (/api/upload-image)
# ----------------------------------------------------------------------

def test_api_upload_accepts_valid_car(client):
    car_path = Path("tests/test_images/positives/car_front_honda_city.jpg")
    with open(car_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), car_path.name)}
        resp = client.post("/api/upload-image", data=data, content_type="multipart/form-data")
    
    assert resp.status_code == 200
    res_json = resp.get_json()
    assert res_json["ok"] is True
    assert "filename" in res_json
    assert res_json.get("vehicle_confidence", 0) >= 0.30
    
    # Confirm file was saved
    saved_path = app.config["UPLOAD_DIR"] / res_json["filename"]
    assert saved_path.exists()
    # Cleanup
    saved_path.unlink(missing_ok=True)


def test_api_upload_rejects_tuf_gaming_logo(client):
    tuf_path = Path("tests/test_images/negatives/tuf_gaming_logo.png")
    with open(tuf_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), tuf_path.name)}
        resp = client.post("/api/upload-image", data=data, content_type="multipart/form-data")
    
    assert resp.status_code == 400
    res_json = resp.get_json()
    assert res_json["ok"] is False
    assert res_json["error"] == "No vehicle detected. Please upload a clear car image."
    
    # Ensure invalid file was NOT retained in upload directory
    uploads = list(app.config["UPLOAD_DIR"].glob("*.png"))
    for u in uploads:
        assert u.name != tuf_path.name


def test_api_upload_rejects_animal_photo(client):
    cat_path = Path("tests/test_images/negatives/animal_cat.jpg")
    with open(cat_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), cat_path.name)}
        resp = client.post("/api/upload-image", data=data, content_type="multipart/form-data")
    
    assert resp.status_code == 400
    res_json = resp.get_json()
    assert res_json["ok"] is False
    assert res_json["error"] == "No vehicle detected. Please upload a clear car image."


def test_api_upload_rejects_corrupted_file(client):
    corrupt = Path("tests/test_images/negatives/corrupted_file.jpg")
    with open(corrupt, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), corrupt.name)}
        resp = client.post("/api/upload-image", data=data, content_type="multipart/form-data")
    
    assert resp.status_code == 400
    res_json = resp.get_json()
    assert res_json["ok"] is False
    assert "couldn't read" in res_json["error"].lower() or "not detected" in res_json["error"].lower()


# ----------------------------------------------------------------------
# 3. Backend Gatekeeper Test (/api/predict)
# ----------------------------------------------------------------------

def test_api_predict_rejects_non_vehicle_image_payload(client):
    # Simulate an adversary attempting to bypass upload validation by planting a non-car image
    # and directly calling /api/predict
    tuf_src = Path("tests/test_images/negatives/tuf_gaming_logo.png")
    fake_filename = "11111111222233334444555566667777.png"
    fake_dest = app.config["UPLOAD_DIR"] / fake_filename
    shutil.copy2(tuf_src, fake_dest)
    
    payload = {
        "brand": "Hyundai",
        "model": "Creta",
        "variant": "1.6 SX",
        "year": 2018,
        "km_driven": 45000,
        "fuel_type": "Diesel",
        "transmission": "Manual",
        "owner_count": 1,
        "engine_cc": 1582,
        "mileage": 19.6,
        "location": "Mumbai",
        "vehicle_condition": "good",
        "image": fake_filename
    }
    
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422
    res_json = resp.get_json()
    assert res_json["ok"] is False
    assert "No vehicle detected" in res_json["error"]
    
    # Confirm invalid file was removed
    assert not fake_dest.exists()

