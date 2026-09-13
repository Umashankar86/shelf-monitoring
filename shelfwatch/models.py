"""Explicit model contracts. No model download and no demo fallback during real inference."""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .domain import Detection
from .vision import letterbox, nms


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite [batch, feature] matrix.")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Embedding model produced a zero vector.")
    return np.ascontiguousarray(vectors / lengths)


class Runtime:
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Missing model: {path.name}. Place your Colab export in {path.parent}.")
        self.kind = path.suffix.lower()
        if self.kind == ".onnx":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self.model = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            inputs = self.model.get_inputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise ValueError("Export a single-input float32 model (half=False).")
            self.input = inputs[0].name
            self.shape = inputs[0].shape
        elif self.kind == ".xml":
            try:
                import openvino as ov
            except ImportError as exc:
                raise ValueError("Install the openvino extra to load .xml models.") from exc
            self.model = ov.Core().compile_model(str(path), "CPU")
            self.shape = list(self.model.input(0).shape)
        else:
            raise ValueError("Use an ONNX or OpenVINO model; export detector weights from Colab first.")
        if len(self.shape) != 4:
            raise ValueError("Model input must be NCHW with four dimensions.")

    def infer(self, tensor):
        tensor = np.ascontiguousarray(tensor, np.float32)
        if self.kind == ".onnx":
            return self.model.run(None, {self.input: tensor})[0]
        return np.asarray(self.model([tensor])[self.model.output(0)])


def decode_yolo(output, shape, ratio, left, top, confidence=0.35, nms_iou=0.45, limit=1000):
    """Decode raw YOLOv8/11/12 [1,4+classes,N] or exported NMS [1,N,6]."""
    output = np.asarray(output)
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f"Unsupported detector output {output.shape}; export a YOLO detection model.")
    raw = output[0]
    if raw.shape[1] == 6 and raw.shape[0] > 6:
        boxes, scores = raw[:, :4].copy(), raw[:, 4]
    else:
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        if raw.shape[1] < 5:
            raise ValueError(f"Unsupported raw detector output {output.shape}.")
        scores = raw[:, 4:].max(axis=1)
        center, size = raw[:, :2], raw[:, 2:4]
        boxes = np.concatenate([center - size / 2, center + size / 2], axis=1)
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= confidence)
    boxes, scores = boxes[valid], scores[valid]
    if not len(boxes):
        return []
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / ratio
    h, w = shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    keep = nms(boxes, scores, nms_iou, limit)
    return [Detection(boxes[i].tolist(), float(scores[i])) for i in keep]


class Detector:
    def __init__(self, cfg):
        self.settings = cfg["detector"]
        self.runtime = Runtime(cfg["artifacts"] / self.settings["file"])
        h, w = self.runtime.shape[-2:]
        if isinstance(h, int) and isinstance(w, int) and h != w:
            raise ValueError("Export the detector with a square input.")
        self.size = h if isinstance(h, int) else self.settings["image_size"]

    def detect(self, image):
        tensor, ratio, left, top = letterbox(image, self.size)
        return decode_yolo(
            self.runtime.infer(tensor),
            image.shape,
            ratio,
            left,
            top,
            self.settings["confidence"],
            self.settings["nms_iou"],
            self.settings["max_detections"],
        )


class Embedder:
    def __init__(self, cfg):
        self.path = cfg["artifacts"] / cfg["embedding"]["file"]
        self.config_path = cfg["artifacts"] / cfg["embedding"]["config"]
        if not self.config_path.exists():
            raise ValueError("Missing embedding_config.json. Use the supplied Colab export helper.")
        self.settings = json.loads(self.config_path.read_text(encoding="utf-8"))
        expected = {
            "color": "RGB",
            "resize": "stretch",
            "interpolation": "linear",
            "layout": "NCHW",
            "l2_normalize": True,
        }
        if any(self.settings.get(k) != v for k, v in expected.items()):
            raise ValueError("Embedding preprocessing does not match this app. See docs/COLAB_EXPORT.md.")
        self.size = int(self.settings["input_size"])
        self.mean = np.asarray(self.settings["mean"], np.float32).reshape(1, 1, 3)
        self.std = np.asarray(self.settings["std"], np.float32).reshape(1, 1, 3)
        if self.size < 16 or self.size > 2048 or np.any(self.std <= 0):
            raise ValueError("Invalid embedding image size or normalization.")
        self.runtime = Runtime(self.path)
        for dim in self.runtime.shape[-2:]:
            if isinstance(dim, int) and dim != self.size:
                raise ValueError("Embedding model dimensions do not match embedding_config.json.")
        self.signature = {"model_sha256": sha256(self.path), "preprocessing_sha256": sha256(self.config_path)}

    def embed(self, images):
        result = []
        for image in images:
            if image is None or image.size == 0:
                raise ValueError("Cannot embed an empty product crop.")
            rgb = cv2.cvtColor(
                cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
            )
            tensor = ((rgb.astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)[None]
            output = self.runtime.infer(tensor)
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("Export the pooled embedding model with output [batch, feature_dimension].")
            result.append(output[0])
        return normalize(result)


class Recognizer:
    def __init__(self, cfg, embedder=None):
        self.notebook = None
        if cfg["recognition"].get("format") == "shelvesense":
            from .shelvesense import NotebookRecognizer

            self.notebook = NotebookRecognizer(cfg)
            self.index, self.labels = self.notebook.index, self.notebook.labels
            self.embedder, self.threshold = self.notebook.embedder, self.notebook.threshold
            return
        import faiss

        self.embedder = embedder or Embedder(cfg)
        folder, spec = cfg["artifacts"], cfg["recognition"]
        needed = [spec[k] for k in ("index", "labels", "metadata", "thresholds")]
        missing = [name for name in needed if not (folder / name).is_file()]
        if missing:
            raise ValueError("Missing recognition files: " + ", ".join(missing))
        metadata = json.loads((folder / spec["metadata"]).read_text(encoding="utf-8"))
        if any(metadata.get(k) != v for k, v in self.embedder.signature.items()):
            raise ValueError(
                "Reference index belongs to another embedding model/config. Rebuild it in Products."
            )
        self.index = faiss.read_index(str(folder / spec["index"]))
        if not isinstance(self.index, faiss.IndexFlatL2):
            raise ValueError("Reference index must use normalized vectors and FAISS IndexFlatL2.")  # noqa: TRY004 -- invalid artifact, not a caller type error
        self.labels = np.load(folder / spec["labels"], allow_pickle=False)
        if self.labels.ndim != 1 or len(self.labels) != self.index.ntotal or not self.index.ntotal:
            raise ValueError("Reference index and labels are empty or inconsistent.")
        if self.labels.dtype.kind not in "iuUS":
            raise ValueError("SKU labels must be integer or Unicode arrays, never pickled objects.")
        if metadata.get("index_sha256") != sha256(folder / spec["index"]) or metadata.get(
            "labels_sha256"
        ) != sha256(folder / spec["labels"]):
            raise ValueError(
                "Reference index bundle is incomplete or changed. Rebuild or copy the entire bundle."
            )
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError("Reference vectors are not L2 normalized. Rebuild the index.")
        thresholds = json.loads((folder / spec["thresholds"]).read_text(encoding="utf-8"))
        if thresholds.get("metric") != "squared_l2" or thresholds.get("calibrated") is not True:
            raise ValueError("Calibrate thresholds.json on held-out known and unknown products first.")
        if any(thresholds.get(k) != v for k, v in self.embedder.signature.items()):
            raise ValueError("Recognition threshold belongs to a different embedding model/config.")
        self.threshold = float(thresholds["max_distance"])
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 4:
            raise ValueError("Normalized squared-L2 threshold must be between 0 and 4.")

    def identify(self, image, detections):
        if self.notebook is not None:
            return self.notebook.identify(image, detections)
        if not detections:
            return detections
        crops = []
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            crops.append(image[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)])
        embeddings = self.embedder.embed(crops)
        if embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the FAISS index.")
        distances, indices = self.index.search(embeddings, 1)
        for d, distance, idx in zip(detections, distances[:, 0], indices[:, 0]):
            d.distance = float(distance)
            d.sku = (
                str(self.labels[idx]) if 0 <= idx < len(self.labels) and distance <= self.threshold else None
            )
        return detections


def build_index(cfg):
    """Build complete catalog with matching preprocessing; commit metadata last."""
    if cfg["recognition"].get("format") == "shelvesense":
        from .shelvesense import rebuild

        return rebuild(cfg)
    import faiss

    embedder = Embedder(cfg)
    files = sorted(p for p in cfg["references"].glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError("Add product reference images before building the index.")
    vectors, labels, paths = [], [], []
    for path in files:
        if not path.parent.name.isdecimal():
            raise ValueError(f"Reference folder must be a numeric SKU ID: {path.parent.name}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable reference image: {path.name}")
        vectors.append(embedder.embed([image])[0])
        labels.append(path.parent.name)
        paths.append(path.relative_to(cfg["references"]).as_posix())
    matrix = normalize(vectors)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    folder, spec = cfg["artifacts"], cfg["recognition"]
    index_path, labels_path = folder / spec["index"], folder / spec["labels"]
    temp = folder / "index.building.faiss"
    faiss.write_index(index, str(temp))
    os.replace(temp, index_path)
    np.save(labels_path, np.asarray(labels, dtype=str), allow_pickle=False)
    np.save(folder / "sku_index.paths.npy", np.asarray(paths, dtype=str), allow_pickle=False)
    metadata = {
        **embedder.signature,
        "count": len(labels),
        "dimension": index.d,
        "index_sha256": sha256(index_path),
        "labels_sha256": sha256(labels_path),
    }
    (folder / spec["metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"images": len(labels), "products": len(set(labels)), "dimension": index.d}


def bundle_status(cfg):
    if cfg["recognition"].get("format") == "shelvesense":
        names = {
            "Detector": cfg["detector"]["file"],
            "Embedding model": cfg["embedding"]["file"],
            "Embedding weights": cfg["embedding"]["file"] + ".data",
            "Preprocessing": cfg["embedding"]["config"],
            "Reference index": cfg["recognition"]["index"],
            "SKU mapping": cfg["recognition"]["mapping"],
            "Unknown threshold": cfg["recognition"]["thresholds"],
        }
        return [
            {"name": name, "file": file, "present": (cfg["artifacts"] / file).is_file()}
            for name, file in names.items()
        ]
    names = {
        "Detector": cfg["detector"]["file"],
        "Embedding model": cfg["embedding"]["file"],
        "Preprocessing": cfg["embedding"]["config"],
        "Reference index": cfg["recognition"]["index"],
        "SKU labels": cfg["recognition"]["labels"],
        "Index metadata": cfg["recognition"]["metadata"],
        "Unknown threshold": cfg["recognition"]["thresholds"],
    }
    return [
        {"name": name, "file": file, "present": (cfg["artifacts"] / file).is_file()}
        for name, file in names.items()
    ]
