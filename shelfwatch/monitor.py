import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import demo
from .models import Detector, Recognizer
from .tracking import Consensus, Tracker
from .vision import align_to_reference, annotate, compare_plan

log = logging.getLogger(__name__)


class Capture:
    """Read continuously into a single latest-frame slot, avoiding a growing RTSP queue."""

    def __init__(self, source):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest = None
        self.sequence = 0
        self.finished = False
        self.error = None
        self.source = source
        self.thread = threading.Thread(target=self.run, daemon=True, name="shelf-capture")
        self.thread.start()

    def run(self):
        cap = None
        try:
            source = int(self.source) if str(self.source).isdecimal() else self.source
            if isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
                cap = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
                )
            else:
                cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError("Cannot open the selected camera or video. Check its path and connection.")
            local_file = isinstance(source, str) and Path(source).is_file()
            fps = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / (fps if np.isfinite(fps) and 1 <= fps <= 240 else 25) if local_file else 0
            while not self.stop_event.is_set():
                start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if not local_file:
                        self.error = (
                            "Camera stream disconnected. Alerts are paused; restart after reconnecting."
                        )
                    break
                with self.lock:
                    self.latest = frame
                    self.sequence += 1
                self.stop_event.wait(max(0, delay - (time.monotonic() - start)))
        except Exception as exc:
            log.exception("Capture failed")
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            self.finished = True

    def read(self):
        with self.lock:
            return self.sequence, self.latest.copy() if self.latest is not None else None

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


class Monitor:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self.lock = threading.RLock()
        self.operation = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.raw_frame = None
        self.detections = []
        self.jpeg = None
        self.result = {
            "mode": "idle",
            "running": False,
            "detections": [],
            "cells": [],
            "message": "Ready when you are.",
        }
        self.demo_phase = "stocked"
        self.detector = None
        self.recognizer = None
        self.model_message = None

    def snapshot(self):
        with self.lock:
            return dict(self.result)

    def ensure_stopped(self):
        if self.thread and self.thread.is_alive():
            raise ValueError("Stop monitoring before changing models, references, or the planogram.")

    def load_models(self, require_recognition=True):
        detector = Detector(self.cfg)
        recognizer = None
        self.model_message = None
        try:
            recognizer = Recognizer(self.cfg)
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            if require_recognition:
                raise ValueError(str(exc)) from exc
            self.model_message = str(exc)
        self.detector, self.recognizer = detector, recognizer

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=8)
            if self.thread.is_alive():
                raise ValueError("The camera is still closing. Wait a few seconds and try again.")
        with self.lock:
            self.result["running"] = False

    def start(self, mode, source="", plan_name="main"):
        self.ensure_stopped()
        if mode not in {"demo", "camera", "video"}:
            raise ValueError("Choose demo, camera, or video.")
        if mode != "demo":
            if not str(source).strip():
                raise ValueError("Enter a camera index, stream URL, or video path.")
            self.load_models(require_recognition=True)
        self.stop_event.clear()
        with self.lock:
            self.raw_frame, self.jpeg = None, None
            self.detections = []
        self.result = {
            "running": True,
            "mode": mode,
            "plan": "demo" if mode == "demo" else plan_name,
            "detections": [],
            "cells": [],
            "message": "Opening source…",
        }
        self.thread = threading.Thread(
            target=self.run, args=(mode, source, plan_name), daemon=True, name="shelf-inference"
        )
        self.thread.start()

    def publish(self, image, detections, cells, mode, message, elapsed, plan_name, running):
        ok, encoded = cv2.imencode(".jpg", annotate(image, detections, cells), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Unable to encode the processed frame.")
        with self.lock:
            self.raw_frame = image.copy()
            self.detections = detections
            self.jpeg = encoded.tobytes()
            self.result = {
                "mode": mode,
                "running": running,
                "plan": plan_name,
                "detections": [d.to_dict() for d in detections],
                "cells": cells,
                "message": message,
                "inference_ms": round(elapsed * 1000, 1),
                "updated": time.time(),
                "width": image.shape[1],
                "height": image.shape[0],
            }

    def run(self, mode, source, plan_name):
        capture = None
        try:
            opts = self.cfg["monitor"]
            plan_name = "demo" if mode == "demo" else plan_name
            plan = demo.planogram() if mode == "demo" else self.store.plan(plan_name)
            tracker = Tracker(opts["tracking_iou"], opts["tracking_max_missed"])
            consensus = Consensus(opts["consecutive_observations"], opts["persistence_seconds"])
            reference = None
            if mode != "demo" and opts["alignment"]:
                reference = cv2.imread(str(self.cfg["planograms"] / f"{plan_name}.jpg"))
                if reference is None:
                    raise ValueError(
                        "Alignment is enabled but no reference shelf image exists. Save a planogram first."
                    )
            if mode != "demo":
                capture = Capture(source)
            last_seq, last_inference, previous = -1, 0.0, None
            while not self.stop_event.is_set():
                if mode == "demo":
                    image, detections = demo.scene(self.demo_phase)
                else:
                    seq, image = capture.read()
                    if capture.finished and (image is None or seq == last_seq):
                        if capture.error:
                            raise ValueError(capture.error)
                        break
                    if image is None or seq == last_seq:
                        self.stop_event.wait(0.04)
                        continue
                    last_seq = seq
                now = time.monotonic()
                if now - last_inference < opts["min_inference_interval"]:
                    self.stop_event.wait(0.02)
                    continue
                small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 48))
                change = (
                    float(np.abs(small.astype(float) - previous.astype(float)).mean())
                    if previous is not None
                    else 255
                )
                if (
                    mode != "demo"
                    and change < opts["change_threshold"]
                    and now - last_inference < opts["max_inference_interval"]
                ):
                    continue
                previous, last_inference = small, now
                start = time.monotonic()
                if reference is not None:
                    aligned, error = align_to_reference(image, reference, opts["alignment_min_inliers"])
                    if error:
                        consensus.history.clear()
                        self.publish(image, [], [], mode, error + " Alerts paused.", 0, plan_name, True)
                        continue
                    image = aligned
                if mode != "demo":
                    detections = self.recognizer.identify(image, self.detector.detect(image))
                detections = tracker.update(detections)
                cells = (
                    compare_plan(plan, detections, image.shape[1], image.shape[0], opts["slot_min_overlap"])
                    if plan
                    else []
                )
                cells = consensus.update(cells, now)
                for cell in cells:
                    if cell["stable"]:
                        self.store.observe(
                            plan_name, cell["id"], cell["state"], cell["expected_sku"], cell["observed_sku"]
                        )
                message = (
                    "Synthetic demo — no trained models used."
                    if mode == "demo"
                    else (
                        "Monitoring shelf positions."
                        if plan
                        else "Detection and recognition active. Save a planogram to enable alerts."
                    )
                )
                self.publish(
                    image, detections, cells, mode, message, time.monotonic() - start, plan_name, True
                )
                self.stop_event.wait(0.08)
        except Exception as exc:
            log.exception("Monitoring stopped")
            with self.lock:
                self.result["message"] = str(exc)
                self.result["error"] = True
        finally:
            if capture:
                capture.stop()
            with self.lock:
                self.result["running"] = False

    def analyze(self, image, plan_name="main"):
        self.ensure_stopped()
        self.load_models(require_recognition=False)
        start = time.monotonic()
        detections = self.detector.detect(image)
        if self.recognizer:
            detections = self.recognizer.identify(image, detections)
        plan = self.store.plan(plan_name)
        cells = (
            compare_plan(
                plan, detections, image.shape[1], image.shape[0], self.cfg["monitor"]["slot_min_overlap"]
            )
            if plan
            else []
        )
        for cell in cells:
            cell["stable"] = False
        message = "Single image analyzed. Persistent alerts require a live/video sequence."
        if self.model_message:
            message += " Detection only: " + self.model_message
        self.publish(image, detections, cells, "image", message, time.monotonic() - start, plan_name, False)
        return self.snapshot()
