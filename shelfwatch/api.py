import csv
import io
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, load_config
from .domain import validate_planogram
from .models import build_index, bundle_status
from .monitor import Monitor
from .shelvesense import valid_sku
from .storage import Store
from .vision import generate_plan


class StartRequest(BaseModel):
    mode: str = "demo"
    source: str = ""
    plan: str = "main"


class AckRequest(BaseModel):
    note: str = Field(default="", max_length=1000)


def safe_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        raise ValueError("Use 1–64 letters, digits, underscores, or hyphens for the shelf name.")
    return value


def create_app(root=ROOT):
    cfg = load_config(root)
    store = Store(cfg["database"])
    monitor = Monitor(cfg, store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        monitor.stop()

    app = FastAPI(title="Proglint Shelfwatch", lifespan=lifespan)
    app.state.monitor, app.state.store, app.state.cfg = monitor, store, cfg

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def local_mutations(request, call_next):
        # Loopback-only app; reject cross-origin browser writes to local files/cameras.
        origin = request.headers.get("origin")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and origin != str(request.base_url).rstrip("/")
        ):
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=403, content={"detail": "Cross-origin changes are not allowed."})
        return await call_next(request)

    @app.get("/api/status")
    def status():
        return {
            "monitor": monitor.snapshot(),
            "files": bundle_status(cfg),
            "artifact_path": str(cfg["artifacts"]),
            "reference_path": str(cfg["references"]),
        }

    @app.get("/api/results")
    def results():
        return monitor.snapshot()

    @app.get("/api/frame")
    def frame():
        with monitor.lock:
            if monitor.jpeg is None:
                return Response(status_code=204)
            return Response(
                content=monitor.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
            )

    @app.post("/api/monitor/start")
    def start(body: StartRequest):
        safe_id(body.plan)
        with monitor.operation:
            monitor.start(body.mode, body.source, body.plan)
        return {"ok": True}

    @app.post("/api/monitor/stop")
    def stop():
        with monitor.operation:
            monitor.stop()
        return {"ok": True}

    @app.post("/api/demo/{phase}")
    def phase(phase: str):
        if phase not in {"stocked", "missing", "misplaced", "unknown"}:
            raise ValueError("Unknown demo scenario.")
        monitor.demo_phase = phase
        return {"ok": True}

    def sample_paths():
        relative = cfg.get("samples", {}).get("directory")
        if not relative:
            raise ValueError("No supplied sample media are configured.")
        return cfg["root"] / relative

    def import_sample_plan():
        folder = sample_paths()
        if store.plan("sample") is not None:
            return
        original = json.loads((folder / "planogram.json").read_text(encoding="utf-8"))
        width, height = original["image_width"], original["image_height"]
        plan = {"name": "sample", "width": width, "height": height, "slots": []}
        for sku, product in original["products"].items():
            plan["slots"].append(
                {
                    "id": f"R{product['row'] + 1}-C{product['column'] + 1}",
                    "row": product["row"] + 1,
                    "expected_sku": sku,
                    "bbox": [
                        v / scale for v, scale in zip(product["bbox_xyxy"], [width, height, width, height])
                    ],
                }
            )
        validate_planogram(plan)
        image = cv2.imread(str(folder / "correctly_stocked_shelf.jpg"))
        if image is None or not cv2.imwrite(str(cfg["planograms"] / "sample.jpg"), image):
            raise ValueError("Cannot load the supplied reference shelf image.")
        (cfg["planograms"] / "sample.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        store.save_plan("sample", plan)

    @app.post("/api/samples/{kind}")
    def run_sample(kind: str):
        if kind not in {"image", "video"}:
            raise ValueError("Choose the sample image or video.")
        with monitor.operation:
            monitor.ensure_stopped()
            import_sample_plan()
            if kind == "video":
                video = sample_paths() / "shelf_test_video.mp4"
                if not video.is_file():
                    raise ValueError("Supplied test video is missing.")
                monitor.start("video", str(video), "sample")
                return {"ok": True, "message": "Running the supplied synthetic shelf video with your models."}
            image = cv2.imread(str(sample_paths() / "correctly_stocked_shelf.jpg"))
            if image is None:
                raise ValueError("Supplied shelf image is unreadable.")
            return monitor.analyze(image, "sample")

    @app.post("/api/analyze")
    def analyze(file: Annotated[UploadFile, File()], plan: Annotated[str, Form()] = "main"):
        safe_id(plan)
        data = file.file.read(25 * 1024 * 1024 + 1)
        if len(data) > 25 * 1024 * 1024:
            raise ValueError("Use an image smaller than 25 MB.")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Upload a readable JPEG or PNG image.")
        with monitor.operation:
            return monitor.analyze(image, plan)

    @app.post("/api/media")
    def upload_media(file: Annotated[UploadFile, File()]):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise ValueError("Choose an MP4, AVI, MOV, or MKV video.")
        path = cfg["media"] / (uuid.uuid4().hex + suffix)
        total = 0
        try:
            with path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 1024 * 1024 * 1024:
                        raise ValueError(
                            "Video uploads are limited to 1 GB. Use a local path for larger files."
                        )
                    target.write(chunk)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {"path": str(path)}

    @app.get("/api/catalog")
    def catalog():
        registered = {p["sku"]: p["name"] for p in store.catalog()}
        for folder in cfg["references"].iterdir():
            if folder.is_dir() and valid_sku(folder.name):
                registered.setdefault(folder.name, "SKU " + folder.name)
        return [
            {
                "sku": sku,
                "name": name,
                "images": sum(
                    p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in (cfg["references"] / sku).glob("*")
                ),
            }
            for sku, name in sorted(registered.items())
        ]

    @app.post("/api/catalog")
    def register(
        sku: Annotated[str, Form()], name: Annotated[str, Form()], files: Annotated[list[UploadFile], File()]
    ):
        if not valid_sku(sku) or not name.strip() or len(name) > 120:
            raise ValueError(
                "Use a SKU ID with letters, digits, underscores or hyphens and a product name up to 120 characters."
            )
        with monitor.operation:
            monitor.ensure_stopped()
            decoded = []
            for file in files:
                content = file.file.read(10 * 1024 * 1024 + 1)
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("Each reference image must be smaller than 10 MB.")
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Every reference must be a readable product image.")
                decoded.append(image)
            if not decoded:
                raise ValueError("Select at least one reference image.")
            folder = cfg["references"] / sku
            folder.mkdir(exist_ok=True)
            for image in decoded:
                if not cv2.imwrite(str(folder / f"{uuid.uuid4().hex}.jpg"), image):
                    raise ValueError("Unable to save reference image.")
            store.put_product(sku, name.strip())
        return {
            "sku": sku,
            "added": len(decoded),
            "message": "References saved. Rebuild the index to activate them.",
        }

    @app.post("/api/catalog/rebuild")
    def rebuild():
        with monitor.operation:
            monitor.ensure_stopped()
            result = build_index(cfg)
            monitor.recognizer = None
        return {
            **result,
            "message": "Index rebuilt. Models reload on the next monitoring start. Recheck unknown-product rejection after catalog changes.",
        }

    @app.post("/api/models/validate")
    def validate_models():
        with monitor.operation:
            monitor.ensure_stopped()
            monitor.load_models()
        return {
            "ok": True,
            "message": "Models, preprocessing, reference index, and threshold loaded successfully.",
        }

    @app.get("/api/planogram/{name}")
    def get_plan(name: str):
        safe_id(name)
        from .demo import planogram

        return planogram() if name == "demo" else store.plan(name)

    @app.post("/api/planogram/{name}/generate")
    def draft_plan(name: str):
        safe_id(name)
        if name == "demo":
            raise ValueError("The synthetic demo planogram is read-only.")
        with monitor.operation:
            monitor.ensure_stopped()
            with monitor.lock:
                if monitor.raw_frame is None or monitor.snapshot()["mode"] == "demo":
                    raise ValueError("Analyze your correctly stocked shelf image first.")
                h, w = monitor.raw_frame.shape[:2]
                return generate_plan(monitor.detections, w, h, name)

    @app.put("/api/planogram/{name}")
    def save_plan(name: str, plan: dict):
        safe_id(name)
        if name == "demo":
            raise ValueError("The synthetic demo planogram is read-only.")
        with monitor.operation:
            monitor.ensure_stopped()
            validate_planogram(plan)
            plan["name"] = name
            if (
                monitor.raw_frame is not None
                and monitor.snapshot()["mode"] != "demo"
                and not cv2.imwrite(str(cfg["planograms"] / f"{name}.jpg"), monitor.raw_frame)
            ):
                raise ValueError("Could not save the reference shelf image.")
            (cfg["planograms"] / f"{name}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            store.save_plan(name, plan)
        return {"ok": True, "message": "Planogram saved. Earlier alerts for this shelf were closed."}

    @app.get("/api/alerts")
    def alerts(source: str | None = None):
        return store.alerts(source)

    @app.post("/api/alerts/{alert_id}/acknowledge")
    def ack(alert_id: int, body: AckRequest):
        store.acknowledge(alert_id, body.note)
        return {"ok": True}

    @app.get("/api/export/results")
    def export_results():
        return Response(
            json.dumps(monitor.snapshot(), indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="shelf-results.json"'},
        )

    @app.get("/api/export/alerts")
    def export_alerts():
        rows = store.alerts()
        output = io.StringIO()
        fields = [
            "id",
            "source",
            "slot",
            "state",
            "expected",
            "observed",
            "created",
            "updated",
            "resolved",
            "acknowledged",
            "note",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        # Escape spreadsheet formula prefixes in user-entered notes/labels.
        for row in rows:
            writer.writerow(
                {
                    k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
                    for k, v in row.items()
                }
            )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="shelf-alerts.csv"'},
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def home():
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app
