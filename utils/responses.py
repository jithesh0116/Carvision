from flask import jsonify


def ok(**payload):
    return jsonify({"ok": True, **payload})


def fail(message, status=400, field=None, **extra):
    body = {"ok": False, "error": message, **extra}
    if field:
        body["field"] = field
    return jsonify(body), status
