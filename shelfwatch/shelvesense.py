"""Adapter for the user's existing MobileNet/FAISS Colab exports. No model training."""

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
from PIL import Image

from .models import Runtime, normalize


def valid_sku(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value) is not None


def preprocess(image):
    """Torchvision MobileNetV3 ImageNet V2 PIL: short side 232, center crop 224."""
    if image is None or image.size == 0:
        raise ValueError("Cannot recognize an empty product crop.")
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = pil.size
    resized = (232, int(232 * height / width)) if width <= height else (int(232 * width / height), 232)
    pil = pil.resize(resized, Image.Resampling.BILINEAR)
    width, height = pil.size
    left, top = round((width - 224) / 2), round((height - 224) / 2)
    pixels = np.array(pil.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255
    pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def vote(scores, indices, labels):
    """Preserve the notebook's majority -> mean similarity -> nearest-order tie break."""
    votes = defaultdict(list)
    for score, idx in zip(scores, indices):
        if 0 <= idx < len(labels) and np.isfinite(score):
            votes[str(labels[idx])].append(float(score))
    if not votes:
        return None, None
    sku = max(votes, key=lambda label: (len(votes[label]), float(np.mean(votes[label]))))
    return sku, max(votes[sku])


class NotebookEmbedder:
    def __init__(self, cfg):
        self.cfg = cfg
        settings = json.loads((cfg["artifacts"] / cfg["embedding"]["config"]).read_text(encoding="utf-8"))
        if cfg["embedding"].get("preprocessing") != "torchvision_v2":
            raise ValueError("The supplied notebook requires torchvision_v2 preprocessing.")
        if settings.get("input_size") != [224, 224] or settings.get("color_format") != "RGB":
            raise ValueError("The supplied embedding preprocessing configuration is incompatible.")
        if (
            settings.get("mean") != [0.485, 0.456, 0.406]
            or settings.get("std") != [0.229, 0.224, 0.225]
            or settings.get("pixel_scale") != "0_to_1"
            or settings.get("normalization") != "L2"
        ):
            raise ValueError("Embedding normalization does not match the supplied notebook.")
        self.dimension = int(settings["embedding_dimension"])
        self.runtime = Runtime(cfg["artifacts"] / cfg["embedding"]["file"])

    def embed(self, images):
        result = []
        for image in images:
            output = self.runtime.infer(preprocess(image))
            if output.shape != (1, self.dimension):
                raise ValueError(f"Expected embedding shape (1, {self.dimension}), got {output.shape}.")
            result.append(output[0])
        return normalize(result)


class NotebookRecognizer:
    def __init__(self, cfg):
        self.embedder = NotebookEmbedder(cfg)
        folder, spec = cfg["artifacts"], cfg["recognition"]
        self.index = faiss.read_index(str(folder / spec["index"]))
        if not isinstance(self.index, faiss.IndexFlatIP) or self.index.d != self.embedder.dimension:
            raise ValueError("This bundle requires a cosine IndexFlatIP with matching embedding dimensions.")
        mapping = json.loads((folder / spec["mapping"]).read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(self.index.ntotal)}:
            raise ValueError("SKU mapping entries must match every index position exactly.")
        self.labels = np.array([mapping[str(i)]["sku_id"] for i in range(self.index.ntotal)])
        if not len(self.labels) or not all(valid_sku(label) for label in self.labels):
            raise ValueError("The reference index is empty or contains invalid SKU identifiers.")
        vectors = self.index.reconstruct_n(0, self.index.ntotal)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError("Cosine recognition requires normalized reference embeddings.")
        self.settings = json.loads((folder / spec["thresholds"]).read_text(encoding="utf-8"))
        required = {
            "metric": "cosine_similarity",
            "accept_condition": "score >= threshold",
            "retrieval_rule": "majority_vote_then_mean_similarity",
            "score_rule": "maximum_similarity_for_winning_sku",
        }
        if any(self.settings.get(k) != v for k, v in required.items()):
            raise ValueError("Recognition rules do not match the notebook threshold calibration.")
        self.threshold = float(self.settings["threshold"])
        self.top_k = int(self.settings["retrieval_top_k"])
        if not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1 or self.top_k < 1:
            raise ValueError("Invalid cosine threshold or retrieval count.")
        # Catch a mismatched embedding export/index using available original references.
        if cfg["references"].exists():
            for idx in sorted({0, self.index.ntotal // 2, self.index.ntotal - 1}):
                name = Path(mapping[str(idx)].get("reference_image", "")).name
                path = cfg["references"] / str(self.labels[idx]) / name
                if path.is_file():
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError(f"Unreadable reference: {path}")
                    similarity = float(self.embedder.embed([image])[0] @ vectors[idx])
                    if similarity < 0.995:
                        raise ValueError(
                            "The embedding model/preprocessing does not reproduce the reference index. Rebuild the index before monitoring."
                        )

    def predict_embeddings(self, embeddings):
        scores, indices = self.index.search(embeddings, min(self.top_k, self.index.ntotal))
        return [vote(s, i, self.labels) for s, i in zip(scores, indices)]

    def identify(self, image, detections):
        if not detections:
            return detections
        crops = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            crops.append(image[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)])
        predictions = self.predict_embeddings(self.embedder.embed(crops))
        for detection, (sku, score) in zip(detections, predictions):
            detection.similarity = score
            detection.sku = sku if score is not None and score >= self.threshold else None
        return detections


def rebuild(cfg):
    """Preserve cosine metric, SKU IDs and transforms when adding local references."""
    import shutil
    import uuid

    embedder = NotebookEmbedder(cfg)
    paths = sorted(p for p in cfg["references"].glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths or any(not valid_sku(p.parent.name) for p in paths):
        raise ValueError("Add reference images in valid SKU folders first.")
    vectors, mapping = [], {}
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable reference: {path.name}")
        vectors.append(embedder.embed([image])[0])
        mapping[str(i)] = {"sku_id": path.parent.name, "reference_image": str(path)}
    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(normalize(vectors))
    folder, spec = cfg["artifacts"], cfg["recognition"]
    backup = folder / "index_backups" / uuid.uuid4().hex
    backup.mkdir(parents=True)
    for key in ("index", "mapping"):
        current = folder / spec[key]
        if current.exists():
            shutil.copy2(current, backup / current.name)
    temporary_index = folder / "index.building.faiss"
    temporary_mapping = folder / "mapping.building.json"
    faiss.write_index(index, str(temporary_index))
    temporary_mapping.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    temporary_index.replace(folder / spec["index"])
    temporary_mapping.replace(folder / spec["mapping"])
    return {"images": len(paths), "products": len({p.parent.name for p in paths}), "dimension": index.d}
