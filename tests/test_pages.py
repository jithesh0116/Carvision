"""Tests for page routes and navigation bar."""
import re
from app import app


def test_home_page_status_and_content():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Must feature CarVision branding
    assert "Car<b>Vision</b>" in html or "CarVision" in html
    # Must feature exact required heading
    assert "Discover Your Car&#39;s True Resale Value" in html or "Discover Your Car's True Resale Value" in html
    # Must feature exact required subtitle
    assert "AI-Powered Vehicle Resale Price Prediction Using CNN and XGBoost." in html
    # Must feature prominent Get Started button linking to predict
    assert 'href="/predict"' in html
    assert "Get Started" in html
    # Must explain the 3 core pillars
    assert "AI-Powered Vehicle Recognition" in html
    assert "XGBoost Price Prediction" in html
    assert "Statistically Calibrated Price Ranges" in html
    # Must include automotive-themed hero section
    assert "hero" in html


def test_navbar_links():
    client = app.test_client()
    for route in ["/", "/predict", "/history", "/dashboard"]:
        resp = client.get(route)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        # Verify navigation bar contains Home, Predict, History, Dashboard, How It Works
        assert 'href="/"' in html or 'href="/home"' in html
        assert ">Home<" in html
        assert 'href="/predict"' in html
        assert ">Predict<" in html
        assert 'href="/history"' in html
        assert ">History<" in html
        assert 'href="/dashboard"' in html
        assert ">Dashboard<" in html
        assert 'href="/#how-it-works"' in html or 'href="#how-it-works"' in html
        assert ">How It Works<" in html


def test_home_does_not_redirect_to_dashboard():
    client = app.test_client()
    resp = client.get("/", follow_redirects=False)
    # Status code must be 200 OK directly, not 301/302 redirect
    assert resp.status_code == 200
    assert resp.location is None


def test_dashboard_accessible_directly():
    client = app.test_client()
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Dashboard" in html


def test_predict_accessible():
    client = app.test_client()
    resp = client.get("/predict", follow_redirects=False)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Upload Your Car Photo" in html


def test_history_accessible():
    client = app.test_client()
    resp = client.get("/history", follow_redirects=False)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Prediction History" in html

