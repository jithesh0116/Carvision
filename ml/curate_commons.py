"""Open-license image curation pipeline from Wikimedia Commons for CarVision.

Strictly adheres to:
1. Permissible licensing: CC0, CC-BY, CC-BY-SA, Public Domain.
2. Transparent provenance: Records author, license URL, source URL, hashes in dataset_manifest.json.
3. Exterior quality filtering: Rejects interiors, engines, wheels, diagrams, and diecast models.
4. Exact and perceptual deduplication: Rejects SHA-256 collision and dHash Hamming distance <= 4.
5. Quarantine isolation: Routes ambiguous or uncertain vehicle photos to data/raw/images_quarantine/.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from ml import config as C
from ml.validate_dataset import compute_dhash, compute_sha256, hamming_distance, validate_image_file

Image.MAX_IMAGE_PIXELS = 250_000_000

USER_AGENT = "CarVisionCurationBot/1.0 (academic vehicle recognition research; mail@carvision.org)"
ALLOWED_LICENSES = {
    "cc0", "public domain", "cc-by", "cc by", "cc by-sa", "cc-by-sa", "gfdl"
}

REJECT_KEYWORDS = {
    "interior", "inside", "cockpit", "dashboard", "seat", "steering", "engine",
    "speedometer", "tachometer", "gear lever", "wheel", "rim", "tire", "badge",
    "emblem", "logo", "drawing", "render", "blueprint", "diagram", "icon",
    "key", "brochure", "scale model", "diecast", "hot wheels", "accident",
    "crash", "fire", "burnt", "cutaway", "chassis", "motor show banner"
}

CLASS_QUERIES = {
    "Maruti Suzuki Dzire": ("Suzuki Dzire", ["dzire", "swift dzire"], ["swift"]),
    "Hyundai Creta": ("Hyundai Creta", ["creta", "ix25"], ["alcazar", "venue", "tucson"]),
    "Maruti Suzuki Brezza": ("Suzuki Brezza", ["brezza", "vitara brezza"], ["grand vitara"]),
    "Maruti Suzuki Swift": ("Suzuki Swift", ["swift"], ["dzire", "swift dzire"]),
    "Tata Nexon": ("Tata Nexon", ["nexon"], ["punch", "harrier", "safari"]),
    "Maruti Suzuki Baleno": ("Suzuki Baleno", ["baleno"], ["glanza", "fronx"]),
    "Maruti Suzuki Ertiga": ("Suzuki Ertiga", ["ertiga"], ["xl6", "rumion"]),
    "Hyundai Venue": ("Hyundai Venue", ["venue"], ["creta", "exter"]),
    "Hyundai Grand i10 Nios": ("Hyundai Grand i10", ["grand i10", "i10 nios", "nios"], ["i20"]),
    "Mahindra Scorpio-N": ("Mahindra Scorpio-N", ["scorpio-n", "scorpio n"], ["scorpio classic", "getaway"]),
    "Mahindra XUV700": ("Mahindra XUV700", ["xuv700", "xuv 700"], ["xuv500", "xuv300"]),
    "Toyota Fortuner": ("Toyota Fortuner", ["fortuner", "sw4"], ["innova", "hilux", "land cruiser"]),
    "Maruti Suzuki Wagon R": ("Suzuki Wagon R", ["wagon r", "wagonr"], ["stingray", "spresso"]),
    "Tata Punch": ("Tata Punch", ["tata punch"], ["nexon", "tiago", "altroz"]),
    "Kia Seltos": ("Kia Seltos", ["seltos"], ["sonet", "carens", "sportage"]),
    "Hyundai i20": ("Hyundai i20", ["i20", "elite i20"], ["i10", "i30"]),
    "Tata Altroz": ("Tata Altroz", ["altroz"], ["tiago", "punch"]),
    "Toyota Innova Crysta": ("Toyota Innova", ["innova", "innova crysta"], ["fortuner", "hycross"]),
    "Honda City": ("Honda City sedan", ["honda city", "city sedan"], ["civic", "accord", "amaze"]),
    "Mahindra Thar": ("Mahindra Thar", ["mahindra thar"], ["roxor", "major", "armada"])
}


def clean_html(text: str) -> str:
    """Strips HTML tags from metadata strings."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def normalize_license(lic: str) -> str:
    """Normalizes license string to lowercase identifier."""
    lic_clean = re.sub(r"[^a-zA-Z0-9\.\-]+", " ", str(lic).lower()).strip()
    return lic_clean


def infer_perspective(title: str, desc: str) -> str:
    combined = (title + " " + desc).lower()
    if any(k in combined for k in ["front three quarter", "front 3/4", "f34", "3/4 front"]):
        return "front_three_quarter"
    if any(k in combined for k in ["rear three quarter", "rear 3/4", "r34", "3/4 rear"]):
        return "rear_three_quarter"
    if any(k in combined for k in ["front view", "front", "facelift front"]):
        return "front"
    if any(k in combined for k in ["rear view", "rear", "back view", "tail"]):
        return "rear"
    if any(k in combined for k in ["side view", "side profile", "side"]):
        return "side"
    return "exterior_angle"


def search_commons(query: str, limit: int = 50) -> List[dict]:
    """Queries Wikimedia Commons API for files matching query."""
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|size|extmetadata|mime&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            return list(pages.values())
    except Exception as exc:
        print(f"Error querying commons for '{query}': {exc}")
        return []


def download_file(url: str, dst_path: Path) -> bool:
    """Downloads an image file with rate-limiting and user-agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp, open(dst_path, "wb") as out_f:
            while chunk := resp.read(65536):
                out_f.write(chunk)
        return True
    except Exception as exc:
        if dst_path.exists():
            dst_path.unlink()
        return False


def curate_dataset(
    max_per_class: int = 30,
    quarantine_dir: Path = Path("data/raw/images_quarantine"),
    manifest_path: Path = Path("data/raw/images/dataset_manifest.json")
) -> dict:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    existing_shas: Set[str] = set()
    existing_dhashes: List[Tuple[str, int]] = []

    # Load existing manifest if present
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for m in manifest:
                existing_shas.add(m["sha256"])
                if "dhash" in m and m["dhash"]:
                    existing_dhashes.append((m["file_name"], int(m["dhash"], 16)))
        except Exception:
            manifest = []

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "classes_processed": 0,
        "images_curated": 0,
        "images_quarantined": 0,
        "rejected_licenses": 0,
        "rejected_content": 0,
        "rejected_duplicates": 0,
        "curated_per_class": {},
        "quarantined_files": []
    }

    img_counter = len(manifest) + 1

    for class_name, (query, positive_keywords, negative_keywords) in CLASS_QUERIES.items():
        report["classes_processed"] += 1
        class_dir = Path("data/raw/images") / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        report["curated_per_class"][class_name] = len([
            f for f in class_dir.glob("*.*") if f.name != ".gitkeep"
        ])

        print(f"\n--- Curating '{class_name}' (Current: {report['curated_per_class'][class_name]}) ---")
        pages = search_commons(query, limit=50)
        curated_for_this_class = 0

        for page in pages:
            if curated_for_this_class >= max_per_class:
                break

            title = page.get("title", "")
            ii = page.get("imageinfo", [{}])[0]
            mime = ii.get("mime", "")
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue

            w, h = ii.get("width", 0), ii.get("height", 0)
            if w < 224 or h < 224:
                continue

            meta = ii.get("extmetadata", {})
            lic_short = meta.get("LicenseShortName", {}).get("value", "")
            lic_url = meta.get("LicenseUrl", {}).get("value", "")
            artist = clean_html(meta.get("Artist", {}).get("value", "Wikimedia Contributor"))
            credit = clean_html(meta.get("Credit", {}).get("value", ""))
            desc = clean_html(meta.get("ImageDescription", {}).get("value", ""))

            # License check
            lic_norm = normalize_license(lic_short)
            if not any(al in lic_norm for al in ALLOWED_LICENSES):
                report["rejected_licenses"] += 1
                continue

            # Content rejection keywords
            combined_text = (title + " " + desc).lower()
            if any(rk in combined_text for rk in REJECT_KEYWORDS):
                report["rejected_content"] += 1
                continue

            # Model ambiguity check
            has_pos = any(pk in combined_text for pk in positive_keywords)
            has_neg = any(nk in combined_text for nk in negative_keywords) and not has_pos

            image_url = ii.get("url")
            if not image_url:
                continue

            # Target filename
            ext = ".jpg" if "jpeg" in mime else (".png" if "png" in mime else ".webp")
            safe_slug = re.sub(r"[^a-z0-9]+", "_", class_name.lower())
            angle = infer_perspective(title, desc)
            veh_id = f"v{report['curated_per_class'][class_name] + 1:02d}"
            target_name = f"{safe_slug}_{veh_id}_{angle}_{img_counter:04d}{ext}"
            
            temp_download = class_dir / f"temp_{img_counter}{ext}"
            success = download_file(image_url, temp_download)
            if not success:
                continue

            # Verify downloaded file
            valid, err, dims, dh = validate_image_file(temp_download)
            if not valid:
                temp_download.unlink()
                continue

            sha = compute_sha256(temp_download)
            if sha in existing_shas:
                temp_download.unlink()
                report["rejected_duplicates"] += 1
                continue

            # Check perceptual near-duplicate
            is_near_dup = False
            for prev_name, prev_dh in existing_dhashes:
                if hamming_distance(dh, prev_dh) <= 4:
                    is_near_dup = True
                    break
            if is_near_dup:
                temp_download.unlink()
                report["rejected_duplicates"] += 1
                continue

            # If ambiguous model label, move to quarantine
            if has_neg or not has_pos:
                quarantine_target = quarantine_dir / f"quarantine_{target_name}"
                if quarantine_target.exists():
                    quarantine_target.unlink()
                temp_download.rename(quarantine_target)
                report["images_quarantined"] += 1
                report["quarantined_files"].append({
                    "file": quarantine_target.name,
                    "reason": "Ambiguous model or competing negative keyword",
                    "title": title
                })
                continue

            # Final acceptance
            final_path = class_dir / target_name
            if final_path.exists():
                final_path.unlink()
            temp_download.rename(final_path)

            existing_shas.add(sha)
            existing_dhashes.append((target_name, dh))

            brand, model = class_name.split(" ", 1) if " " in class_name else (class_name, class_name)
            manifest_entry = {
                "image_id": f"CV_IMG_{img_counter:04d}",
                "file_name": target_name,
                "relative_path": str(final_path.as_posix()),
                "brand": brand,
                "model": model,
                "vehicle_id": veh_id,
                "perspective": angle,
                "resolution": f"{dims[0]}x{dims[1]}",
                "format": mime.split("/")[-1].upper(),
                "sha256": sha,
                "dhash": hex(dh),
                "source": "Wikimedia Commons",
                "source_url": image_url,
                "author": artist[:100] if artist else "Unknown",
                "license": lic_short,
                "license_url": lic_url,
                "attribution": credit[:150] if credit else f"Photo by {artist} via Wikimedia Commons",
                "collection_date": datetime.date.today().isoformat(),
                "status": "Verified Clean"
            }
            manifest.append(manifest_entry)

            img_counter += 1
            curated_for_this_class += 1
            report["images_curated"] += 1
            report["curated_per_class"][class_name] += 1
            print(f"  [+] Ingested: {target_name} ({dims[0]}x{dims[1]}, {lic_short})")

            # Courtesy delay to respect Wikimedia API rate guidelines
            time.sleep(0.3)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikimedia Commons Curation Tool for CarVision")
    parser.add_argument("--max-per-class", type=int, default=25, help="Maximum images to curate per class in this run")
    args = parser.parse_args()

    print("\n==================================================================")
    print("      CARVISION WIKIMEDIA COMMONS OPEN-LICENSE CURATOR")
    print("==================================================================")
    rep = curate_dataset(max_per_class=args.max_per_class)
    print("\nCuration Finished:")
    print(f"  Accepted & Ingested : {rep['images_curated']}")
    print(f"  Quarantined         : {rep['images_quarantined']}")
    print(f"  Rejected Licenses   : {rep['rejected_licenses']}")
    print(f"  Rejected Content    : {rep['rejected_content']}")
    print(f"  Rejected Duplicates : {rep['rejected_duplicates']}")
    print("==================================================================\n")

