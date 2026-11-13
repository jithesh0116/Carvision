"""Read-only access to the saved model's metadata/metrics (cached by file mtime)."""
import json

from ml import config as C

_cache = {}


def _read(path):
    if not path.exists():
        return None
    m = path.stat().st_mtime
    hit = _cache.get(path)
    if hit and hit[0] == m:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    _cache[path] = (m, data)
    return data


def model_meta():
    return _read(C.META_PATH)


def model_metrics():
    return _read(C.METRICS_PATH)


def cnn_metrics():
    return _read(C.CNN_METRICS_PATH)
