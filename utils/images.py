"""Safe image upload handling: validate, strip metadata by re-encoding, save with a random name."""
from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
SAVED_NAME = re.compile(r"^[a-f0-9]{32}\.(jpg|png|webp)$")
BAD_TYPE = "Please upload a JPG, PNG, or WEBP image."


class ImageError(Exception):
    """Message is safe to show to the user."""


def save_image(file_storage, upload_dir: Path, allowed_ext: set, max_bytes: int) -> dict:
    name = secure_filename(file_storage.filename or "")
    if Path(name).suffix.lower() not in allowed_ext:
        raise ImageError(BAD_TYPE)
    data = file_storage.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ImageError(f"That photo is too large. Please upload an image under {max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise ImageError("We couldn't read this image. Please try another photo.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise ImageError("We couldn't read this image. Please try another photo.")
    if fmt not in FORMAT_EXT:
        raise ImageError(BAD_TYPE)
    if min(img.size) < 64:
        raise ImageError("This photo is too small. Please upload a clearer, larger photo.")
    img = ImageOps.exif_transpose(img)
    if fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    filename = uuid.uuid4().hex + FORMAT_EXT[fmt]
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    img.save(path, format=fmt, **({"quality": 90} if fmt in ("JPEG", "WEBP") else {}))
    return {"filename": filename, "path": str(path)}


def is_saved_name(name) -> bool:
    return isinstance(name, str) and bool(SAVED_NAME.match(name))
