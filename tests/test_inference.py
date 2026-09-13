import json

import cv2
import numpy as np
import pytest

from shelfwatch.models import Detector, Embedder, Recognizer, build_index


def test_real_onnx_runtime_and_index_recognize_and_reject(bundle):
    detector, recognizer = Detector(bundle), Recognizer(bundle)
    red = np.full((64, 64, 3), (0, 0, 255), np.uint8)
    detections = recognizer.identify(red, detector.detect(red))
    assert len(detections) == 1 and detections[0].sku == "101"
    blue = np.full((64, 64, 3), (255, 0, 0), np.uint8)
    detections = recognizer.identify(blue, detector.detect(blue))
    assert detections[0].sku is None


def test_rebuild_adds_new_sku_without_changing_model(bundle):
    before = Embedder(bundle).signature
    folder = bundle["references"] / "102"
    folder.mkdir()
    cv2.imwrite(str(folder / "blue.jpg"), np.full((30, 30, 3), (255, 0, 0), np.uint8))
    report = build_index(bundle)
    assert report["products"] == 2
    assert Embedder(bundle).signature == before
    blue = np.full((64, 64, 3), (255, 0, 0), np.uint8)
    assert Recognizer(bundle).identify(blue, Detector(bundle).detect(blue))[0].sku == "102"


def test_mixed_model_bundle_is_rejected(bundle):
    path = bundle["artifacts"] / "index_metadata.json"
    data = json.loads(path.read_text())
    data["model_sha256"] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another embedding"):
        Recognizer(bundle)


def test_changed_labels_are_rejected(bundle):
    np.save(bundle["artifacts"] / "sku_index.labels.npy", np.array(["999"]))
    with pytest.raises(ValueError, match="incomplete or changed"):
        Recognizer(bundle)


def test_uncalibrated_threshold_is_rejected(bundle):
    path = bundle["artifacts"] / "thresholds.json"
    data = json.loads(path.read_text())
    data["calibrated"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Calibrate"):
        Recognizer(bundle)


def test_colab_helper_preprocessing_agrees_with_runtime(bundle):
    from scripts.colab_export import build_reference_index, encoder

    path = bundle["references"] / "101" / "red.jpg"
    colab = encoder(bundle["artifacts"])(path)
    local = Embedder(bundle).embed([cv2.imread(str(path))])
    assert np.allclose(colab, local)
    build_reference_index(bundle["references"], bundle["artifacts"])
    assert Recognizer(bundle).index.ntotal == 1
