import os
import sys
import subprocess
import shutil
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from app import app

client = app.test_client()
resp = client.get("/")
assert resp.status_code == 200
html = resp.get_data(as_text=True)

# Replace static links with relative paths
html = html.replace('/static/css/style.css', '../static/css/style.css')
html = html.replace('/static/js/common.js', '../static/js/common.js')

rendered_path = os.path.abspath("scratch/rendered_home.html")
with open(rendered_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Updated rendered_home.html at", rendered_path)

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
file_url = "file:///" + rendered_path.replace("\\", "/")

viewports = [
    ("viewport_1440_desktop", 1440, 900),
    ("viewport_1366_laptop", 1366, 768),
    ("viewport_1024_small_laptop", 1024, 768),
    ("viewport_768_tablet", 768, 1024),
    ("viewport_430_mobile", 430, 932),
    ("viewport_375_small_mobile", 375, 667),
    ("viewport_320_narrow_mobile", 320, 568),
]

out_dir = os.path.abspath("scratch/screenshots")
os.makedirs(out_dir, exist_ok=True)

profile_dir = os.path.abspath("scratch/edge_temp_profile")
if os.path.exists(profile_dir):
    shutil.rmtree(profile_dir, ignore_errors=True)

results = []

for name, w, h in viewports:
    out_png = os.path.join(out_dir, f"{name}.png")
    if os.path.exists(out_png):
        os.remove(out_png)
    
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={profile_dir}",
        f"--window-size={w},{h}",
        f"--screenshot={out_png}",
        file_url
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(out_png):
        im = Image.open(out_png)
        print(f"Captured {name}: {im.size} ({os.path.getsize(out_png)} bytes)")
        results.append((name, w, h, im.size, os.path.getsize(out_png)))
    else:
        print(f"FAILED to capture {name}")

print("\nAll", len(results), "screenshots generated successfully.")
