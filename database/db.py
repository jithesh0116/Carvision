"""MySQL access via PyMySQL. All queries elsewhere use parameterised statements."""
from __future__ import annotations

import pymysql
from flask import current_app, g
from pymysql.cursors import DictCursor


class DatabaseUnavailable(Exception):
    """The database could not be reached (shown to users as a friendly message)."""


def connect(cfg=None, with_database=True):
    """Open a connection. `cfg` is a mapping/Config-like object; defaults to the Flask app config."""
    if cfg is None:
        cfg = current_app.config
    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    kwargs = dict(host=get("DB_HOST"), port=int(get("DB_PORT")), user=get("DB_USER"),
                  password=get("DB_PASSWORD"), charset="utf8mb4", cursorclass=DictCursor,
                  autocommit=True, connect_timeout=5)
    if with_database:
        kwargs["database"] = get("DB_NAME")
    return pymysql.connect(**kwargs)


def get_db():
    if "db" not in g:
        try:
            g.db = connect()
        except pymysql.MySQLError as exc:
            raise DatabaseUnavailable(str(exc)) from exc
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)


def _run(sql, params):
    try:
        cur = get_db().cursor()
        cur.execute(sql, params)
        return cur
    except pymysql.MySQLError as exc:
        # table missing / connection dropped etc.
        raise DatabaseUnavailable(str(exc)) from exc


def query_all(sql, params=()):
    return _run(sql, params).fetchall()


def query_one(sql, params=()):
    return _run(sql, params).fetchone()


def execute(sql, params=()):
    cur = _run(sql, params)
    return cur.lastrowid
