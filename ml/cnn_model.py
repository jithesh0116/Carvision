"""Image recognition (inference side). TensorFlow is imported lazily so the web app and the price
model work without it.

The CNN only recognises what it was trained on (class-per-folder labels such as "Hyundai Creta").
A detection is offered to the user only if (1) confidence >= the configured threshold and
(2) the label matches a brand+model that exists in the vehicle database. The user can always
change it manually. In this version the CNN output is NOT an input to the price model.
"""
from __future__ import annotations

import json
import re
import threading

import numpy as np

from ml import config as C


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def match_label_to_catalog(label: str, catalog):
    """catalog: iterable of (brand, model). Returns (brand, model) or None.

    A label matches when it starts with the brand and the following words start with the model
    (longest model wins), e.g. 'Hyundai Elantra Sedan 2007' -> ('Hyundai', 'Elantra')."""
    lab = normalise(label)
    best = None
    for brand, model in catalog:
        nb, nm = normalise(brand), normalise(model)
        if not nb or not nm or not (lab == nb or lab.startswith(nb + " ")):
            continue
        rest = lab[len(nb):].strip()
        if rest == nm or rest.startswith(nm + " "):
            if best is None or len(nm) > len(normalise(best[1])):
                best = (brand, model)
    return best


class CnnRecognizer:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._classes = None
        self._img_size = 224

    def is_available(self) -> bool:
        if not (C.CNN_MODEL_PATH.exists() and C.CNN_CLASSES_PATH.exists()):
            return False
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        with self._lock:
            if self._model is None:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(str(C.CNN_MODEL_PATH), compile=False)
                info = json.loads(C.CNN_CLASSES_PATH.read_text(encoding="utf-8"))
                self._classes = info["classes"]
                self._img_size = int(info.get("img_size", 224))

    def top_k(self, image_path: str, k: int = 3):
        self._load()
        from PIL import Image
        img = Image.open(image_path).convert("RGB").resize((self._img_size, self._img_size))
        arr = np.asarray(img, dtype="float32")[None, ...]        # preprocessing layers live inside the model
        probs = self._model.predict(arr, verbose=0)[0]
        idx = np.argsort(probs)[::-1][:k]
        return [(self._classes[i], float(probs[i])) for i in idx]


_recognizer = CnnRecognizer()


def get_recognizer() -> CnnRecognizer:
    return _recognizer
