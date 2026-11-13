"""Unit tests for the parsing / statistics code. They use tiny in-memory examples of *formats*
(e.g. '45,000 km'); no vehicle data is bundled with the project."""
import numpy as np
import pandas as pd
import pytest

from ml.cnn_model import match_label_to_catalog
from ml.conformal import conformal_quantile, empirical_coverage, interval_from_log
from ml.inspect_dataset import guess_mapping
from ml.preprocessing import DatasetError, clean, parse_number, parse_owner, parse_price, unify_text
from utils.format import format_money


def test_parse_number():
    out = parse_number(pd.Series(["45,000 km", "23.4 kmpl", "abc", None]))
    assert out.iloc[0] == 45000 and out.iloc[1] == 23.4 and np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])


def test_parse_price_units():
    out = parse_price(pd.Series(["5.5 Lakh", "1.2 Crore", "450000"]))
    assert list(out) == [550000.0, 12000000.0, 450000.0]


def test_parse_owner():
    out = parse_owner(pd.Series(["First Owner", "Second Owner", "Fourth & Above Owner", "Test Drive Car", "3"]))
    assert list(out[:3]) == [1, 2, 4] and np.isnan(out[3]) and out[4] == 3


def test_unify_text_merges_case_variants():
    out = unify_text(pd.Series(["Diesel", "diesel", " Diesel ", "Petrol", "n/a"]))
    assert out.iloc[:3].nunique() == 1 and pd.isna(out.iloc[4])


def test_guess_mapping_and_name_split():
    m = guess_mapping(["Name", "Year", "Selling_Price", "Kms_Driven", "Fuel", "Owner"])
    assert m["name"] == "Name" and m["selling_price"] == "Selling_Price" and m["km_driven"] == "Kms_Driven"
    raw = pd.DataFrame({"Name": ["Acme Roadster GT", "Acme Roadster GT", "Zed Cruiser"],
                        "Year": [2015, 2016, 2014], "Selling_Price": [100, 120, 90],
                        "Kms_Driven": ["10,000 km", "20,000 km", "5,000 km"]})
    mapping = {"columns": {**m, "name": "Name"}, "name_split": {"brand_tokens": 1, "model_tokens": 1}}
    df, rep = clean(raw, mapping)
    assert list(df["brand"]) == ["Acme", "Acme", "Zed"] and df["variant"].iloc[0] == "GT"
    assert df["km_driven"].iloc[0] == 10000


def test_missing_required_fields_gives_friendly_error():
    with pytest.raises(DatasetError):
        clean(pd.DataFrame({"a": [1]}), {"columns": {}})


def test_conformal_quantile_and_coverage():
    rng = np.random.default_rng(0)
    resid = np.abs(rng.normal(size=999))
    q = conformal_quantile(resid, 0.1)
    assert q == np.sort(resid)[int(np.ceil(1000 * 0.9)) - 1]
    with pytest.raises(ValueError):
        conformal_quantile([0.1, 0.2], 0.1)
    lo, hi = interval_from_log(np.log1p(np.array([100.0])), 0.1)
    assert lo[0] < 100 < hi[0] and empirical_coverage([100], lo, hi) == 1.0


def test_label_matching():
    cat = [("Hyundai", "Elantra"), ("Hyundai", "Elan"), ("Mercedes-Benz", "C-Class")]
    assert match_label_to_catalog("Hyundai Elantra Sedan 2007", cat) == ("Hyundai", "Elantra")
    assert match_label_to_catalog("mercedes_benz_c_class", cat) == ("Mercedes-Benz", "C-Class")
    assert match_label_to_catalog("Unknown Thing", cat) is None



def test_money_format():
    assert format_money(845000) == "₹8,45,000"
    assert format_money(845000, "$", "en-US") == "$845,000"


def test_dhash_and_hamming():
    from PIL import Image
    from ml.validate_dataset import compute_dhash, hamming_distance
    img1 = Image.new("RGB", (250, 250), color="white")
    img2 = Image.new("RGB", (250, 250), color="white")
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert hamming_distance(h1, h2) == 0


def test_parse_vehicle_id():
    from ml.validate_dataset import parse_vehicle_id
    assert parse_vehicle_id("maruti_swift_v02_front.jpg") == "v02"
    assert parse_vehicle_id("creta_car15_side.png") == "car15"
    assert parse_vehicle_id("standalone_car.jpg") is None


def test_image_dimension_validation(tmp_path):
    from PIL import Image
    from ml.validate_dataset import validate_image_file
    tiny = tmp_path / "tiny.jpg"
    Image.new("RGB", (100, 100), color="red").save(tiny)
    valid, err, dim, _ = validate_image_file(tiny)
    assert not valid and "Resolution too low" in err

    proper = tmp_path / "proper.jpg"
    Image.new("RGB", (300, 300), color="green").save(proper)
    valid, err, dim, dhash = validate_image_file(proper)
    assert valid and err is None and dim == (300, 300) and dhash is not None

