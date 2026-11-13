import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app import app
import subprocess

client = app.test_client()
resp = client.get("/")
assert resp.status_code == 200
html = resp.get_data(as_text=True)

# Replace relative static path with absolute or relative path to static folder
html = html.replace('/static/css/style.css', '../static/css/style.css')
html = html.replace('/static/js/common.js', '../static/js/common.js')

rendered_path = os.path.abspath("scratch/rendered_home.html")
with open(rendered_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved rendered HTML to", rendered_path)

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

viewports = [
    ("desktop_1440", 1440, 900),
    ("laptop_1366", 1366, 768),
    ("small_laptop_1024", 1024, 768),
    ("tablet_768", 768, 1024),
    ("mobile_430", 430, 932),
    ("small_mobile_375", 375, 667),
    ("narrow_mobile_320", 320, 568),
]

screenshots_dir = os.path.abspath("scratch/screenshots")
os.makedirs(screenshots_dir, exist_ok=True)

file_url = "file:///" + rendered_path.replace("\\", "/")

for name, w, h in viewports:
    out_png = os.path.join(screenshots_dir, f"{name}.png")
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={w},{h}",
        f"--screenshot={out_png}",
        file_url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
        print(f" [PASS] Viewport {w}x{h} ({name}): Screenshot captured ({os.path.getsize(out_png)} bytes)")
    else:
        print(f" [FAIL] Viewport {w}x{h} ({name}): Failed. Output: {res.stderr}")
