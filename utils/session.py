import uuid

from flask import session


def current_sid() -> str:
    """Anonymous per-browser id, so each visitor only sees their own history."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        session.permanent = True
    return sid
