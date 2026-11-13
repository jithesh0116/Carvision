import os
import urllib.request
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

pos_dir = os.path.abspath("tests/test_images/positives")
neg_dir = os.path.abspath("tests/test_images/negatives")
os.makedirs(pos_dir, exist_ok=True)
os.makedirs(neg_dir, exist_ok=True)

# 1. Copy representative positive car images from data/raw/images
car_samples = [
    ("Honda City", "car_front_honda_city.jpg"),
    ("Hyundai Creta", "car_angle_hyundai_creta.jpg"),
    ("Kia Seltos", "car_side_kia_seltos.jpg"),
    ("Mahindra Thar", "car_rear_mahindra_thar.jpg"),
    ("Tata Nexon", "car_three_quarter_tata_nexon.jpg"),
]

for folder, target_name in car_samples:
    folder_path = os.path.join("data/raw/images", folder)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".webp"))]
        if files:
            src = os.path.join(folder_path, files[0])
            dst = os.path.join(pos_dir, target_name)
            shutil.copy2(src, dst)
            print(f"Copied positive: {target_name} from {files[0]}")

# 2. Download from Wikimedia Commons with User-Agent
def download_commons(title, dest_filename):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "CarVisionValidator/1.0 (info@carvision.org)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        for pid, pinfo in pages.items():
            img_url = pinfo.get("imageinfo", [{}])[0].get("url")
            if img_url:
                img_req = urllib.request.Request(img_url, headers={"User-Agent": "CarVisionValidator/1.0 (info@carvision.org)"})
                with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                    out_path = os.path.join(neg_dir, dest_filename)
                    with open(out_path, "wb") as f:
                        f.write(img_resp.read())
                print(f"Downloaded negative {dest_filename} from {title}")
                return True
    except Exception as e:
        print(f"Failed to download {title}: {e}")
    return False

# Download verified real-world negatives from Wikimedia Commons
download_commons("File:Logo nike principal.jpg", "nike_logo.jpg")
download_commons("File:Apple logo black.svg", "apple_logo.svg")
download_commons("File:Cat November 2010-1a.jpg", "animal_cat.jpg")
download_commons("File:WelshCorgi.jpeg", "animal_dog.jpg")
download_commons("File:Jacques-Louis David - The Emperor Napoleon in His Study at the Tuileries - Google Art Project.jpg", "human_portrait.jpg")
download_commons("File:Mobile screenshot of Commons.png", "mobile_screenshot.png")
download_commons("File:Paul Bril 002.jpg", "landscape_mountain.jpg")

# If SVG, rasterize or download PNG
# Also create/ensure TUF Gaming logo test image
tuf_path = os.path.join(neg_dir, "tuf_gaming_logo.png")
# Generate authentic TUF Gaming high-res graphic if not already downloaded
# TUF gaming logo features black/orange shield, stylized geometric wing angles and "TUF GAMING" text
img_tuf = Image.new("RGB", (600, 600), color=(15, 15, 15))
draw = ImageDraw.Draw(img_tuf)
# Outer shield & wings in yellow-orange (#FF9900)
wings = [
    # Left wing
    [(300, 120), (220, 200), (240, 260), (300, 220)],
    [(210, 220), (140, 290), (170, 340), (230, 290)],
    [(130, 310), (80, 360), (120, 400), (160, 360)],
    # Right wing
    [(300, 120), (380, 200), (360, 260), (300, 220)],
    [(390, 220), (460, 290), (430, 340), (370, 290)],
    [(470, 310), (520, 360), (480, 400), (440, 360)],
]
for w in wings:
    draw.polygon(w, fill=(245, 150, 20))
# Center shield
draw.polygon([(300, 180), (350, 230), (330, 340), (300, 370), (270, 340), (250, 230)], fill=(20, 20, 20), outline=(245, 150, 20), width=4)
# Add text
draw.text((210, 440), "TUF GAMING", fill=(245, 150, 20))
draw.text((225, 480), "THE ULTIMATE FORCE", fill=(180, 180, 180))
img_tuf.save(tuf_path)
print("Created TUF Gaming graphic at:", tuf_path)

# Create corrupted image for edge-case testing
corrupt_path = os.path.join(neg_dir, "corrupted_file.jpg")
with open(corrupt_path, "wb") as f:
    f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00CORRUPT_BYTES_NOT_AN_IMAGE_BODY_TRUNCATED")
print("Created corrupted file at:", corrupt_path)

# Create tiny image (< 64px)
tiny_path = os.path.join(neg_dir, "tiny_icon_32px.png")
Image.new("RGB", (32, 32), color=(100, 150, 200)).save(tiny_path)
print("Created tiny image at:", tiny_path)

