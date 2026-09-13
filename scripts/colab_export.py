"""Upload this file to Colab. See docs/COLAB_EXPORT.md. No training happens here."""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export_models(best_pt, output_dir, embedding_model=None):
    """Export an existing trained YOLO plus pooled MobileNetV3-Large features.

    Supply your own embedding_model if fine-tuned; it must return [B,D], not class logits.
    Otherwise explicitly download and use torchvision's ImageNet-pretrained features.
    """
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
    from ultralytics import YOLO

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detector_path = YOLO(str(best_pt)).export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        half=False,
        nms=False,
        opset=17,
        simplify=False,
        device="cpu",
    )
    target = output / "detector.onnx"
    if Path(detector_path).resolve() != target.resolve():
        shutil.copy2(detector_path, target)
    if embedding_model is None:
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        embedding_model = torch.nn.Sequential(base.features, base.avgpool, torch.nn.Flatten(1))
    embedding_model = embedding_model.cpu().eval()
    with torch.no_grad():
        test = embedding_model(torch.zeros(1, 3, 224, 224))
        if test.ndim != 2 or test.shape[0] != 1:
            raise ValueError("Embedding model must return [batch, feature_dimension].")
        torch.onnx.export(
            embedding_model,
            torch.zeros(1, 3, 224, 224),
            str(output / "embedding.onnx"),
            input_names=["images"],
            output_names=["embeddings"],
            opset_version=17,
            dynamo=False,
        )
    settings = {
        "input_size": 224,
        "color": "RGB",
        "resize": "stretch",
        "interpolation": "linear",
        "layout": "NCHW",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "l2_normalize": True,
    }
    (output / "embedding_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return str(output)


def signature(folder):
    return {
        "model_sha256": digest(folder / "embedding.onnx"),
        "preprocessing_sha256": digest(folder / "embedding_config.json"),
    }


def encoder(folder):
    import onnxruntime as ort

    settings = json.loads((folder / "embedding_config.json").read_text())
    session = ort.InferenceSession(str(folder / "embedding.onnx"), providers=["CPUExecutionProvider"])
    size = settings["input_size"]
    mean, std = np.array(settings["mean"], np.float32), np.array(settings["std"], np.float32)

    def embed(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        image = cv2.cvtColor(
            cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
        )
        tensor = ((image.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
        vector = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(tensor)})[0]
        lengths = np.linalg.norm(vector, axis=1, keepdims=True)
        if not np.isfinite(vector).all() or np.any(lengths < 1e-8):
            raise ValueError("Invalid embedding vector.")
        return np.ascontiguousarray(vector / lengths, np.float32)

    return embed


def image_files(folder):
    return sorted(p for p in Path(folder).glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def build_reference_index(reference_dir, output_dir):
    import faiss

    folder = Path(output_dir)
    embed = encoder(folder)
    files = image_files(reference_dir)
    if not files or any(not p.parent.name.isdecimal() for p in files):
        raise ValueError("Use numeric SKU folders containing product reference images.")
    vectors = np.concatenate([embed(path) for path in files])
    labels = np.array([path.parent.name for path in files], dtype=str)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(folder / "sku_index.faiss"))
    np.save(folder / "sku_index.labels.npy", labels, allow_pickle=False)
    np.save(
        folder / "sku_index.paths.npy",
        np.array([p.relative_to(reference_dir).as_posix() for p in files], dtype=str),
        allow_pickle=False,
    )
    metadata = {
        **signature(folder),
        "count": len(files),
        "dimension": index.d,
        "index_sha256": digest(folder / "sku_index.faiss"),
        "labels_sha256": digest(folder / "sku_index.labels.npy"),
    }
    (folder / "index_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"images": len(files), "products": len(set(labels)), "dimension": index.d}


def calibrate_threshold(known_dir, unknown_dir, output_dir, max_unknown_acceptance=0.05):
    """Choose a rejection threshold on validation crops, not on reference/training images.

    known_dir/<numeric SKU>/*.jpg; unknown_dir/<any product ID>/*.jpg.
    Independent test data are still needed for an unbiased final accuracy report.
    """
    import faiss

    if not 0 <= max_unknown_acceptance < 1:
        raise ValueError("max_unknown_acceptance must be between 0 inclusive and 1 exclusive.")
    folder = Path(output_dir)
    embed = encoder(folder)
    index = faiss.read_index(str(folder / "sku_index.faiss"))
    labels = np.load(folder / "sku_index.labels.npy", allow_pickle=False).astype(str)
    known, unknown = image_files(known_dir), image_files(unknown_dir)
    if len(known) < 5 or len(unknown) < 5:
        raise ValueError(
            "Provide at least five held-out known and five unknown crops; more varied samples are better."
        )
    if any(p.parent.name not in set(labels) for p in known):
        raise ValueError("Every known validation SKU must already be in the reference index.")

    def search(files):
        d, ids = index.search(np.concatenate([embed(p) for p in files]), 1)
        return d[:, 0], labels[ids[:, 0]]

    kd, predicted = search(known)
    ud, _ = search(unknown)
    correct = predicted == np.array([p.parent.name for p in known])
    candidates = np.unique(np.r_[0.0, kd, np.nextafter(ud, -np.inf)])
    candidates = candidates[(candidates >= 0) & (candidates <= 4)]
    feasible = [t for t in candidates if np.mean(ud <= t) <= max_unknown_acceptance]
    if not feasible:
        raise ValueError(
            "No threshold meets the unknown-rejection target. Improve reference data or the embedding model."
        )
    threshold = max(feasible, key=lambda t: (float(np.mean(correct & (kd <= t))), -float(t)))
    report = {
        "metric": "squared_l2",
        "max_distance": float(threshold),
        "calibrated": True,
        **signature(folder),
        "known_samples": len(known),
        "unknown_samples": len(unknown),
        "known_correct_acceptance": float(np.mean(correct & (kd <= threshold))),
        "unknown_false_acceptance": float(np.mean(ud <= threshold)),
        "note": "Validation calibration, not independent test accuracy. Recheck after catalog changes.",
    }
    (folder / "thresholds.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
