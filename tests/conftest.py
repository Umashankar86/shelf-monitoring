import json
import shutil

import cv2
import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from shelfwatch.config import ROOT, load_config
from shelfwatch.models import build_index, sha256


@pytest.fixture
def cfg(tmp_path):
    shutil.copy2(ROOT / "config.yaml", tmp_path / "config.yaml")
    settings = yaml.safe_load((tmp_path / "config.yaml").read_text())
    settings["references"] = "data/references"
    settings["detector"]["file"] = "detector.onnx"
    settings["embedding"] = {"file": "embedding.onnx", "config": "embedding_config.json"}
    settings["recognition"].pop("format", None)
    settings.pop("samples", None)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    return load_config(tmp_path)


def save_graph(path, nodes, inputs, outputs):
    model = helper.make_model(
        helper.make_graph(nodes, "test_fixture", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture
def bundle(cfg):
    """Tiny deterministic ONNX graphs exercise the real runtime, never shipped as user models."""
    folder = cfg["artifacts"]
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 16, 16])
    out = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3])
    save_graph(
        folder / "embedding.onnx",
        [helper.make_node("ReduceMean", ["images"], ["features"], axes=[2, 3], keepdims=0)],
        [inp],
        [out],
    )
    settings = {
        "input_size": 16,
        "color": "RGB",
        "resize": "stretch",
        "interpolation": "linear",
        "layout": "NCHW",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "l2_normalize": True,
    }
    (folder / "embedding_config.json").write_text(json.dumps(settings))
    product = cfg["references"] / "101"
    product.mkdir()
    cv2.imwrite(str(product / "red.jpg"), np.full((30, 30, 3), (0, 0, 255), np.uint8))
    build_index(cfg)
    threshold = {
        "metric": "squared_l2",
        "max_distance": 0.05,
        "calibrated": True,
        "model_sha256": sha256(folder / "embedding.onnx"),
        "preprocessing_sha256": sha256(folder / "embedding_config.json"),
    }
    (folder / "thresholds.json").write_text(json.dumps(threshold))
    boxes = np.zeros((1, 5, 10), np.float32)
    boxes[0, :, 0] = [32, 32, 40, 40, 0.95]
    tensor = helper.make_tensor("boxes", TensorProto.FLOAT, boxes.shape, boxes.flatten())
    save_graph(
        folder / "detector.onnx",
        [helper.make_node("Constant", [], ["boxes"], value=tensor)],
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 10])],
    )
    return cfg
