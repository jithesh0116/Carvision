import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app import app

client = app.test_client()

def test_routes():
    print("--- Testing Routes ---")
    routes = ["/", "/home", "/predict", "/history", "/dashboard"]
    for r in routes:
        resp = client.get(r)
        assert resp.status_code == 200, f"Route {r} returned {resp.status_code}"
        print(f"Route {r}: HTTP {resp.status_code} OK ({len(resp.data)} bytes)")

def test_homepage_elements():
    print("\n--- Testing Homepage Elements ---")
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Required strings
    checks = [
        ("Branding", "Car<b>Vision</b>"),
        ("Eyebrow", "AI-POWERED VEHICLE VALUATION"),
        ("Primary Heading", "Discover Your Car&#39;s True Resale Value" if "&#39;" in html else "Discover Your Car's True Resale Value"),
        ("Hero Subtitle", "AI-Powered Vehicle Resale Price Prediction Using CNN and XGBoost."),
        ("Lead Paragraph", "precision valuation powered by dual machine learning models"),
        ("Predict CTA", 'href="/predict"'),
        ("Predict Button Text", "Predict My Car's Value"),
        ("How It Works CTA", 'href="#how-it-works"'),
        ("Get Started Button", "Get Started"),
        ("CNN Chip", "CNN Visual Recognition"),
        ("XGBoost Chip", "XGBoost Regression"),
        ("Intervals Chip", "Calibrated Price Intervals"),
        ("Telemetry Card Header", "CarVision Telemetry"),
        ("Telemetry Row 1", "Vehicle Recognition"),
        ("Telemetry Row 1 Value", "Transfer Learning CNN"),
        ("Telemetry Row 2", "Resale Price Engine"),
        ("Telemetry Row 2 Value", "Gradient Boosted Trees (XGBoost)"),
        ("Telemetry Row 3", "Uncertainty Bounds"),
        ("Telemetry Row 3 Value", "Split Conformal Calibration"),
        ("Feature 1", "AI Image Analysis"),
        ("Feature 2", "Smart Price Prediction"),
        ("Feature 3", "Dynamic Vehicle Selection"),
        ("Feature 4", "Easy-to-Understand Results"),
        ("Arch Pillar 1", "AI-Powered Vehicle Recognition"),
        ("Arch Pillar 2", "XGBoost Price Prediction"),
        ("Arch Pillar 3", "Statistically Calibrated Price Ranges"),
        ("Step 1", "Upload Photo"),
        ("Step 2", "Select Vehicle"),
        ("Step 3", "Enter Specifications"),
        ("Step 4", "Instant Valuation"),
        ("Dashboard Teaser", "Explore Real-Time Market Analytics &amp; Dashboard" if "&amp;" in html else "Explore Real-Time Market Analytics & Dashboard"),
        ("Bottom CTA Heading", "Ready to Discover Your Car&#39;s True Resale Value?" if "&#39;" in html else "Ready to Discover Your Car's True Resale Value?"),
    ]

    for label, needle in checks:
        if needle in html:
            print(f" [PASS] {label}: Found '{needle[:45]}...'")
        else:
            print(f" [FAIL] {label}: NOT found '{needle}'")
            sys.exit(1)

    # Prohibited claims
    prohibited = ["Active Pipeline", "Know Your Car's Value", "Ready to find out what your car is worth?"]
    for ban in prohibited:
        if ban in html:
            print(f" [FAIL] Prohibited/Duplicate text found: '{ban}'")
            sys.exit(1)
        else:
            print(f" [PASS] Clean: '{ban}' not present.")

def test_css():
    print("\n--- Testing CSS Integrity ---")
    with open("static/css/style.css", "r", encoding="utf-8") as f:
        css = f.read()

    # Check media queries
    queries = ["max-width: 1024px", "max-width: 768px", "max-width: 576px", "max-width: 380px"]
    for q in queries:
        assert q in css, f"Media query {q} missing from style.css"
        print(f" [PASS] Media query present: {q}")

    # Check for unclosed braces in CSS
    open_b = css.count("{")
    close_b = css.count("}")
    assert open_b == close_b, f"Brace mismatch: {open_b} {{ vs {close_b} }}"
    print(f" [PASS] CSS Braces match: {open_b} pairs perfectly balanced.")

if __name__ == "__main__":
    test_routes()
    test_homepage_elements()
    test_css()
    print("\nALL FRONTEND VERIFICATION CHECKS PASSED SUCCESSFULLY!")
