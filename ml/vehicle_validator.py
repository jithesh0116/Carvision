"""CNN-based vehicle validation pipeline using MobileNetV2.

Determines whether an uploaded image genuinely contains a motor vehicle/automobile
before allowing progression to vehicle resale price prediction.
Rejects non-vehicle content including logos, gaming graphics, screenshots, documents,
human portraits, animals, and landscapes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ml import config as C

logger = logging.getLogger(__name__)

# Core vehicle synsets from ImageNet-1k relevant to road passenger & utility motor vehicles
VEHICLE_SYNSET_NAMES = {
    # Passenger cars & automobiles
    "sports_car",
    "convertible",
    "minivan",
    "pickup",  # pickup truck
    "limousine",
    "cab",  # taxi / hack
    "jeep",  # SUV / off-road vehicle
    "beach_wagon",  # station wagon / estate car
    "racer",  # race car
    # Commercial & utility passenger vehicles
    "police_van",
    "ambulance",
    "moving_van",
    "tow_truck",
    "trailer_truck",
    "fire_engine",
    "garbage_truck",
    "recreational_vehicle",  # RV / motorhome
    "minibus",
    "school_bus",
    # Key automotive structural components
    "grille",  # radiator grille of an automobile
    "car_wheel",
    "snowplow",
}


@dataclass
class VehicleValidationResult:
    is_vehicle: bool
    confidence: float
    top_category: str
    top_vehicle_category: str
    error: str | None = None


class VehicleValidator:
    """Thread-safe validator using MobileNetV2 pre-trained on ImageNet."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._vehicle_indices = None
        self._class_index = None
        self._validated_cache: dict[str, float] = {}

    def is_available(self) -> bool:
        """Check whether weights and TensorFlow are present."""
        weights_file = C.MODELS_DIR / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
        class_file = C.MODELS_DIR / "imagenet_class_index.json"
        if not (weights_file.exists() and class_file.exists()):
            return False
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        with self._lock:
            if self._model is not None:
                return

            weights_file = C.MODELS_DIR / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
            class_file = C.MODELS_DIR / "imagenet_class_index.json"

            if not weights_file.exists():
                raise FileNotFoundError(f"MobileNetV2 weights not found at {weights_file}")
            if not class_file.exists():
                raise FileNotFoundError(f"ImageNet class index not found at {class_file}")

            import tensorflow as tf

            # Suppress excessive C++ logging
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

            self._model = tf.keras.applications.MobileNetV2(
                weights=str(weights_file),
                input_shape=(224, 224, 3)
            )

            with open(class_file, "r", encoding="utf-8") as f:
                self._class_index = json.loads(f.read())

            self._vehicle_indices = [
                int(k) for k, (synset, name) in self._class_index.items()
                if name in VEHICLE_SYNSET_NAMES
            ]

    def validate(
        self,
        image_input: Union[str, Path, Image.Image],
        threshold: float = 0.30
    ) -> VehicleValidationResult:
        """Validate whether an image contains a vehicle.

        Parameters:
            image_input: File path (str/Path) or PIL Image instance.
            threshold: Minimum aggregate vehicle probability (default 0.30).

        Returns:
            VehicleValidationResult with decision and metrics.
        """
        # Check cache if image_input is a path filename
        if isinstance(image_input, (str, Path)):
            fname = Path(image_input).name
            with self._lock:
                if fname in self._validated_cache:
                    cached_conf = self._validated_cache[fname]
                    return VehicleValidationResult(
                        is_vehicle=cached_conf >= threshold,
                        confidence=cached_conf,
                        top_category="cached_vehicle",
                        top_vehicle_category="cached_vehicle",
                        error=None
                    )

        # 1. Image loading and safety verification
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    return VehicleValidationResult(
                        is_vehicle=False, confidence=0.0, top_category="none",
                        top_vehicle_category="none", error="Image file does not exist."
                    )
                img = Image.open(img_path)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return VehicleValidationResult(
                    is_vehicle=False, confidence=0.0, top_category="none",
                    top_vehicle_category="none", error="Invalid image input type."
                )

            # Check image dimensions
            width, height = img.size
            if width < 64 or height < 64:
                return VehicleValidationResult(
                    is_vehicle=False, confidence=0.0, top_category="none",
                    top_vehicle_category="none",
                    error="Image is too small. Please upload a clearer, larger photo."
                )

            # Transpose EXIF orientation and convert to RGB
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            resized = img.resize((224, 224), Image.Resampling.BILINEAR)

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning(f"Failed to decode image during validation: {exc}")
            return VehicleValidationResult(
                is_vehicle=False, confidence=0.0, top_category="corrupt",
                top_vehicle_category="none",
                error="Invalid or corrupted image format."
            )

        # 2. Forward inference
        try:
            self._load()
            import tensorflow as tf

            arr = np.asarray(resized, dtype=np.float32)[None, ...]
            preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            preds = self._model.predict(preprocessed, verbose=0)[0]

            # Aggregate probability mass assigned to vehicle synsets
            vehicle_conf = float(np.sum(preds[self._vehicle_indices]))

            # Top class overall
            top_idx = int(np.argmax(preds))
            top_category = self._class_index[str(top_idx)][1]

            # Top vehicle class
            top_veh_idx = self._vehicle_indices[int(np.argmax(preds[self._vehicle_indices]))]
            top_vehicle_category = self._class_index[str(top_veh_idx)][1]

            is_veh = bool(vehicle_conf >= threshold)

            # Cache validation result if input was a path
            if isinstance(image_input, (str, Path)) and is_veh:
                with self._lock:
                    self._validated_cache[Path(image_input).name] = vehicle_conf

            return VehicleValidationResult(
                is_vehicle=is_veh,
                confidence=vehicle_conf,
                top_category=top_category,
                top_vehicle_category=top_vehicle_category,
                error=None
            )

        except Exception as exc:
            logger.exception(f"Unexpected error during vehicle validation: {exc}")
            return VehicleValidationResult(
                is_vehicle=False, confidence=0.0, top_category="error",
                top_vehicle_category="none", error=f"Validation failed: {str(exc)}"
            )

    def cache_validated(self, filename: str, confidence: float):
        """Manually mark a filename as validated."""
        with self._lock:
            self._validated_cache[filename] = confidence

    def evict_cached(self, filename: str):
        """Remove a filename from the validation cache."""
        with self._lock:
            self._validated_cache.pop(filename, None)


_validator = VehicleValidator()


def get_vehicle_validator() -> VehicleValidator:
    """Return the global VehicleValidator instance."""
    return _validator

