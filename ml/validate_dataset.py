"""Comprehensive dataset validation and quality control for CarVision CNN.

Validates image datasets before CNN training, checking:
1. Image file corruption and decoding.
2. Image formats and extensions.
3. Cryptographic duplicates (SHA-256).
4. Near-duplicates via perceptual difference hashing (dHash).
5. Directory name compliance with database catalog.
6. Class balance and minimum image thresholds.
7. Image resolution and aspect ratio quality.
8. Vehicle-level grouping to prevent train/val/test data leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 250_000_000

from ml import config as C
from ml.cnn_model import match_label_to_catalog

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMG_DIMENSION = 224
DHASH_SIZE = 8  # 8x8 comparison -> 64-bit hash
DEFAULT_DHASH_THRESHOLD = 4  # Hamming distance <= 4 considered near-duplicate


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of an image file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = DHASH_SIZE) -> int:
    """Computes a 64-bit difference hash (dHash) for perceptual similarity comparison."""
    # Resize to (width + 1, height) in grayscale
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.int16)
    diff = arr[:, :-1] > arr[:, 1:]
    
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Returns Hamming distance between two 64-bit integer hashes."""
    return (h1 ^ h2).bit_count()


def parse_vehicle_id(file_name: str) -> Optional[str]:
    """Extracts a vehicle group ID from filename if present (e.g. 'swift_veh01_front.jpg' -> 'veh01')."""
    patterns = [
        r"(?:v|veh|vehicle|car|unit)[-_]?(\d+)",
        r"_([A-Za-z0-9]+)_(?:front|rear|side|f34|r34|34)",
    ]
    for pat in patterns:
        m = re.search(pat, file_name, re.IGNORECASE)
        if m:
            return m.group(0).lower()
    return None


def get_catalog_pairs() -> Set[Tuple[str, str]]:
    """Loads valid (brand, model) catalog pairs from processed CSV or fallback."""
    if C.CLEANED_CSV.exists():
        import pandas as pd
        df = pd.read_csv(C.CLEANED_CSV, usecols=["brand", "model"])
        return set(zip(df["brand"], df["model"]))
    return set()


def validate_image_file(path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[int]]:
    """Validates an image file: corruption, dimensions, and calculates dHash.
    Returns: (is_valid, error_msg, (width, height), dhash_int)
    """
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported extension: {path.suffix}", None, None

    try:
        with Image.open(path) as img:
            img.verify()
        
        # Re-open for dimension inspection and dHash computation
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_IMG_DIMENSION or h < MIN_IMG_DIMENSION:
                return False, f"Resolution too low ({w}x{h} < {MIN_IMG_DIMENSION}x{MIN_IMG_DIMENSION})", (w, h), None
            
            # Check aspect ratio
            aspect = w / float(h)
            if aspect < 0.35 or aspect > 2.8:
                return False, f"Extreme aspect ratio ({aspect:.2f})", (w, h), None

            dhash = compute_dhash(img)
            return True, None, (w, h), dhash
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, f"Corrupted or invalid image: {exc}", None, None


def simulate_grouped_split(
    samples_by_class: Dict[str, List[dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Partitions samples into train, val, test subsets, grouping by vehicle_id when available."""
    rng = np.random.default_rng(seed)
    train_samples, val_samples, test_samples = [], [], []

    for cls_name, samples in samples_by_class.items():
        # Group by vehicle_id
        groups: Dict[str, List[dict]] = defaultdict(list)
        for s in samples:
            vid = s.get("vehicle_id") or s["path"]
            groups[vid].append(s)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_val = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0
        n_test = max(1, int(round(n_groups * (1.0 - train_ratio - val_ratio)))) if n_groups >= 3 else 0
        n_train = n_groups - n_val - n_test

        train_keys = set(group_keys[:n_train])
        val_keys = set(group_keys[n_train:n_train + n_val])
        test_keys = set(group_keys[n_train + n_val:])

        for k in train_keys:
            train_samples.extend(groups[k])
        for k in val_keys:
            val_samples.extend(groups[k])
        for k in test_keys:
            test_samples.extend(groups[k])

    return train_samples, val_samples, test_samples


def run_dataset_audit(
    data_dir: Path,
    min_images_per_class: int = 100,
    target_images_per_class: int = 150,
    dhash_threshold: int = DEFAULT_DHASH_THRESHOLD
) -> dict:
    """Executes a full audit of the image dataset in data_dir."""
    catalog = get_catalog_pairs()
    report = {
        "data_dir": str(data_dir.resolve()),
        "total_classes_found": 0,
        "valid_catalog_classes": 0,
        "invalid_catalog_classes": [],
        "total_images_scanned": 0,
        "valid_images": 0,
        "corrupted_images": [],
        "invalid_format_images": [],
        "low_resolution_images": [],
        "exact_duplicates": [],
        "near_duplicates": [],
        "class_counts": {},
        "classes_meeting_minimum": [],
        "classes_needing_images": [],
        "imbalance_ratio": None,
        "split_simulation": {},
        "split_leakage_detected": []
    }

    if not data_dir.exists():
        report["error"] = f"Directory {data_dir} does not exist."
        return report

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    report["total_classes_found"] = len(class_dirs)

    # Check directory catalog compliance
    for cdir in class_dirs:
        matched = match_label_to_catalog(cdir.name, catalog) if catalog else (cdir.name, cdir.name)
        if not matched:
            report["invalid_catalog_classes"].append(cdir.name)
        else:
            report["valid_catalog_classes"] += 1

    # Scan image files
    sha256_map: Dict[str, List[str]] = defaultdict(list)
    dhash_list: List[Tuple[str, int, str]] = []  # (path, dhash, class)
    samples_by_class: Dict[str, List[dict]] = defaultdict(list)

    for cdir in class_dirs:
        class_name = cdir.name
        img_files = [f for f in cdir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        report["class_counts"][class_name] = 0

        for f in img_files:
            report["total_images_scanned"] += 1
            is_valid, err, dim, dhash = validate_image_file(f)

            if not is_valid:
                if "Unsupported extension" in (err or ""):
                    report["invalid_format_images"].append({"file": str(f), "error": err})
                elif "Resolution too low" in (err or "") or "aspect ratio" in (err or ""):
                    report["low_resolution_images"].append({"file": str(f), "error": err})
                else:
                    report["corrupted_images"].append({"file": str(f), "error": err})
                continue

            # Exact duplicate check via SHA-256
            sha = compute_sha256(f)
            sha256_map[sha].append(str(f))

            dhash_list.append((str(f), dhash, class_name))
            vid = parse_vehicle_id(f.name)
            samples_by_class[class_name].append({
                "path": str(f),
                "class": class_name,
                "vehicle_id": vid,
                "dhash": dhash,
                "sha256": sha
            })
            report["class_counts"][class_name] += 1
            report["valid_images"] += 1

    # Collate exact duplicates
    for sha, paths in sha256_map.items():
        if len(paths) > 1:
            report["exact_duplicates"].append({"sha256": sha, "files": paths})

    # Collate near-duplicates within same class and across classes
    n_dhash = len(dhash_list)
    for i in range(n_dhash):
        p1, h1, c1 = dhash_list[i]
        for j in range(i + 1, min(i + 100, n_dhash)):  # compare neighborhood
            p2, h2, c2 = dhash_list[j]
            dist = hamming_distance(h1, h2)
            if dist <= dhash_threshold:
                report["near_duplicates"].append({
                    "file_a": p1,
                    "file_b": p2,
                    "hamming_dist": dist,
                    "same_class": (c1 == c2)
                })

    # Class balance evaluation
    counts = [c for c in report["class_counts"].values() if c > 0]
    for cname, cnt in report["class_counts"].items():
        if cnt >= min_images_per_class:
            report["classes_meeting_minimum"].append(cname)
        else:
            report["classes_needing_images"].append({"class": cname, "count": cnt, "deficit": min_images_per_class - cnt})

    if counts:
        report["imbalance_ratio"] = round(max(counts) / min(counts), 2)
    else:
        report["imbalance_ratio"] = None

    # Split simulation & leakage testing
    if report["valid_images"] >= 6:
        train_s, val_s, test_s = simulate_grouped_split(samples_by_class)
        report["split_simulation"] = {
            "n_train": len(train_s),
            "n_val": len(val_s),
            "n_test": len(test_s),
            "train_ratio": round(len(train_s) / max(1, report["valid_images"]), 3),
            "val_ratio": round(len(val_s) / max(1, report["valid_images"]), 3),
            "test_ratio": round(len(test_s) / max(1, report["valid_images"]), 3)
        }

        # Check vehicle-level leakage between test and train/val
        test_vehicles = {s["vehicle_id"] for s in test_s if s.get("vehicle_id")}
        train_vehicles = {s["vehicle_id"] for s in train_s if s.get("vehicle_id")}
        val_vehicles = {s["vehicle_id"] for s in val_s if s.get("vehicle_id")}

        overlap_tv = test_vehicles.intersection(train_vehicles)
        overlap_tvl = test_vehicles.intersection(val_vehicles)
        if overlap_tv:
            report["split_leakage_detected"].append(f"Vehicle IDs shared between test and train: {overlap_tv}")
        if overlap_tvl:
            report["split_leakage_detected"].append(f"Vehicle IDs shared between test and val: {overlap_tvl}")

    return report


def main():
    parser = argparse.ArgumentParser(description="CarVision CNN Dataset Validator and Quality Auditor")
    parser.add_argument("--data-dir", default=str(C.IMAGES_DIR), help="Root directory of raw class images")
    parser.add_argument("--min-images", type=int, default=100, help="Minimum images required per class")
    parser.add_argument("--target-images", type=int, default=150, help="Target images per class")
    parser.add_argument("--dhash-threshold", type=int, default=DEFAULT_DHASH_THRESHOLD, help="Max Hamming distance for near-duplicates")
    parser.add_argument("--out-report", default=str(C.DATA_PROCESSED / "dataset_validation_report.json"), help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"\n==================================================================")
    print(f"      CARVISION CNN DATASET QUALITY CONTROL & VALIDATION")
    print(f"==================================================================")
    print(f"Scanning target directory: {data_dir}")
    print(f"Minimum threshold per class: {args.min_images} | Target: {args.target_images}")
    print(f"Near-duplicate Hamming threshold: <= {args.dhash_threshold}\n")

    report = run_dataset_audit(data_dir, args.min_images, args.target_images, args.dhash_threshold)

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Directory Classes Found     : {report['total_classes_found']}")
    print(f"Catalog Conforming Classes  : {report['valid_catalog_classes']}")
    print(f"Non-Conforming Classes      : {len(report['invalid_catalog_classes'])}")
    print(f"Total Images Scanned        : {report['total_images_scanned']}")
    print(f"Valid High-Quality Images   : {report['valid_images']}")
    print(f"Corrupted Images            : {len(report['corrupted_images'])}")
    print(f"Low Resolution / Bad Aspect : {len(report['low_resolution_images'])}")
    print(f"Exact Duplicates (SHA-256)  : {len(report['exact_duplicates'])}")
    print(f"Near Duplicates (dHash)     : {len(report['near_duplicates'])}")
    print(f"Classes Meeting Target (>= {args.min_images}): {len(report['classes_meeting_minimum'])}/{report['total_classes_found']}")

    if report["invalid_catalog_classes"]:
        print(f"\n[!] WARNING: Non-conforming class folders:")
        for bad in report["invalid_catalog_classes"]:
            print(f"    - '{bad}'")

    if report["imbalance_ratio"]:
        print(f"Class Imbalance Ratio       : {report['imbalance_ratio']}:1")

    print(f"\nDetailed validation report saved to: {out_path}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    main()
