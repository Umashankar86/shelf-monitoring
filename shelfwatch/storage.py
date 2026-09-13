import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def utcnow():
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS catalog (sku TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS plans (name TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY, source TEXT NOT NULL, slot TEXT NOT NULL,
                    state TEXT NOT NULL, expected TEXT, observed TEXT,
                    created TEXT NOT NULL, updated TEXT NOT NULL,
                    resolved TEXT, acknowledged TEXT, note TEXT DEFAULT ''
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_open_alert
                ON alerts(source, slot) WHERE resolved IS NULL;
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def catalog(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM catalog ORDER BY sku")]

    def put_product(self, sku, name):
        with self.connect() as db:
            db.execute(
                "INSERT INTO catalog VALUES (?, ?) ON CONFLICT(sku) DO UPDATE SET name=excluded.name",
                (sku, name),
            )

    def plan(self, name):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM plans WHERE name=?", (name,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_plan(self, name, payload):
        with self.connect() as db:
            db.execute(
                "INSERT INTO plans VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET payload=excluded.payload",
                (name, json.dumps(payload)),
            )
            db.execute(
                "UPDATE alerts SET resolved=?, updated=? WHERE source=? AND resolved IS NULL",
                (utcnow(), utcnow(), name),
            )

    def observe(self, source, slot, state, expected, observed):
        if state in ("UNKNOWN", "PENDING"):
            return
        now = utcnow()
        with self.connect() as db:
            old = db.execute(
                "SELECT * FROM alerts WHERE source=? AND slot=? AND resolved IS NULL", (source, slot)
            ).fetchone()
            if old and (state == "OK" or old["state"] != state or old["observed"] != observed):
                db.execute("UPDATE alerts SET resolved=?, updated=? WHERE id=?", (now, now, old["id"]))
                old = None
            if state == "OK":
                return
            if old:
                db.execute("UPDATE alerts SET updated=? WHERE id=?", (now, old["id"]))
            else:
                db.execute(
                    "INSERT INTO alerts(source,slot,state,expected,observed,created,updated) VALUES(?,?,?,?,?,?,?)",
                    (source, slot, state, expected, observed, now, now),
                )

    def alerts(self, source=None):
        with self.connect() as db:
            if source:
                rows = db.execute("SELECT * FROM alerts WHERE source=? ORDER BY id DESC LIMIT 500", (source,))
            else:
                rows = db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 500")
            return [dict(r) for r in rows]

    def acknowledge(self, alert_id, note):
        with self.connect() as db:
            cur = db.execute(
                "UPDATE alerts SET acknowledged=?, note=? WHERE id=?", (utcnow(), note, alert_id)
            )
            if not cur.rowcount:
                raise ValueError("Alert does not exist.")
