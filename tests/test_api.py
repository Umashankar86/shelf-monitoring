import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from shelfwatch.api import create_app


def image_bytes(color=(0, 0, 255)):
    return cv2.imencode(".png", np.full((64, 64, 3), color, np.uint8))[1].tobytes()


def test_empty_install_reports_missing_models_and_never_fakes_real_inference(cfg):
    with TestClient(create_app(cfg["root"])) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert all(not f["present"] for f in client.get("/api/status").json()["files"])
        response = client.post("/api/monitor/start", json={"mode": "camera", "source": "0"})
        assert response.status_code == 400 and "Missing model" in response.json()["detail"]
        assert not client.get("/api/results").json()["running"]


def test_catalog_upload_validation_and_path_safety(cfg):
    with TestClient(create_app(cfg["root"])) as client:
        response = client.post(
            "/api/catalog",
            data={"sku": "../x", "name": "bad"},
            files={"files": ("a.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 400
        response = client.post(
            "/api/catalog",
            data={"sku": "101", "name": "New product"},
            files={"files": ("a.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert client.get("/api/catalog").json()[0]["images"] == 1
        assert client.post("/api/catalog/rebuild").status_code == 400


def test_real_image_pipeline_plan_and_no_single_frame_alerts(bundle):
    with TestClient(create_app(bundle["root"])) as client:
        assert client.post("/api/models/validate").status_code == 200
        response = client.post("/api/analyze", files={"file": ("shelf.png", image_bytes(), "image/png")})
        assert response.status_code == 200
        assert response.json()["detections"][0]["sku"] == "101"
        plan = client.post("/api/planogram/main/generate").json()
        assert client.put("/api/planogram/main", json=plan).status_code == 200
        assert client.get("/api/planogram/main").json()["slots"][0]["expected_sku"] == "101"
        assert client.get("/api/frame").headers["content-type"] == "image/jpeg"
        assert client.get("/api/alerts").json() == []


def test_demo_creates_and_resolves_alert_without_models(cfg):
    app = create_app(cfg["root"])
    app.state.cfg["monitor"].update(
        min_inference_interval=0.01, consecutive_observations=2, persistence_seconds=0.04
    )

    def wait_for(client, predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if predicate(client.get("/api/alerts").json()):
                return
            time.sleep(0.05)
        raise AssertionError("Expected demo alert transition did not occur")

    with TestClient(app) as client:
        client.post("/api/demo/missing")
        assert client.post("/api/monitor/start", json={"mode": "demo"}).status_code == 200
        wait_for(client, lambda rows: len(rows) == 1 and rows[0]["source"] == "demo")
        assert client.post("/api/catalog/rebuild").status_code == 400
        client.post("/api/demo/stocked")
        wait_for(client, lambda rows: rows and rows[0]["resolved"] is not None)
        assert client.post("/api/monitor/stop").status_code == 200
        assert not client.get("/api/results").json()["running"]


def test_cross_origin_writes_are_rejected(cfg):
    with TestClient(create_app(cfg["root"])) as client:
        response = client.post("/api/demo/missing", headers={"Origin": "https://unrelated.example"})
        assert response.status_code == 403
