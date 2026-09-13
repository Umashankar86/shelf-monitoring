import json

import faiss
import numpy as np
import pytest

from shelfwatch.domain import Detection
from shelfwatch.models import Recognizer
from shelfwatch.shelvesense import NotebookEmbedder, preprocess, valid_sku, vote


def test_voting_preserves_notebook_rule_not_nearest_neighbor():
    labels = ["sku_a", "sku_b", "sku_b", "sku_b", "sku_a"]
    sku, score = vote([0.99, 0.85, 0.83, 0.81, 0.8], [0, 1, 2, 3, 4], labels)
    assert sku == "sku_b" and score == 0.85


def test_voting_breaks_equal_counts_by_mean_similarity():
    sku, score = vote([0.9, 0.8, 0.79, 0.7], [0, 1, 2, 3], ["a", "b", "b", "a"])
    assert sku == "a" and score == 0.9


def test_faiss_missing_neighbors_do_not_select_negative_label():
    assert vote([float("-inf")], [-1], ["sku_a"]) == (None, None)


def test_preprocessing_center_crops_instead_of_stretching():
    image = np.zeros((224, 700, 3), dtype=np.uint8)
    image[:, 200:500, 2] = 255
    result = preprocess(image)
    assert result.shape == (1, 3, 224, 224)
    assert np.allclose(result[0, 0], (1 - 0.485) / 0.229)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("value", ["../x", "", "a/b", "a\\b", ".", ".."])
def test_sku_paths_are_rejected(value):
    assert not valid_sku(value)


def test_notebook_bundle_votes_rejects_and_preserves_sku_names(bundle, monkeypatch):
    # Exercise actual FAISS + JSON mapping through the application adapter.
    cfg = bundle
    cfg["recognition"].update(format="shelvesense", mapping="sku_mapping.json")
    vectors = np.array([[1, 0, 0], [0.99, 0.01, 0], [0.98, 0.02, 0], [0, 1, 0], [0, 0.99, 0.01]], np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    faiss.write_index(index, str(cfg["artifacts"] / "sku_index.faiss"))
    mapping = {str(i): {"sku_id": "sku_red" if i < 3 else "sku_green"} for i in range(5)}
    (cfg["artifacts"] / "sku_mapping.json").write_text(json.dumps(mapping))
    settings = {
        "metric": "cosine_similarity",
        "accept_condition": "score >= threshold",
        "threshold": 0.8,
        "retrieval_top_k": 5,
        "retrieval_rule": "majority_vote_then_mean_similarity",
        "score_rule": "maximum_similarity_for_winning_sku",
    }
    (cfg["artifacts"] / "thresholds.json").write_text(json.dumps(settings))
    monkeypatch.setattr(NotebookEmbedder, "__init__", lambda self, cfg: setattr(self, "dimension", 3))
    monkeypatch.setattr(NotebookEmbedder, "embed", lambda self, images: np.array([[1, 0, 0]], np.float32))
    recognizer = Recognizer(cfg)
    image = np.zeros((10, 10, 3), np.uint8)
    detected = recognizer.identify(image, [Detection([0, 0, 10, 10], 0.9)])[0]
    assert detected.sku == "sku_red" and detected.similarity == 1
    monkeypatch.setattr(NotebookEmbedder, "embed", lambda self, images: np.array([[0, 0, 1]], np.float32))
    assert recognizer.identify(image, [Detection([0, 0, 10, 10], 0.9)])[0].sku is None


def test_notebook_threshold_metric_cannot_be_confused_with_l2(bundle, monkeypatch):
    cfg = bundle
    cfg["recognition"].update(format="shelvesense", mapping="sku_mapping.json")
    index = faiss.IndexFlatIP(3)
    index.add(np.array([[1, 0, 0]], np.float32))
    faiss.write_index(index, str(cfg["artifacts"] / "sku_index.faiss"))
    (cfg["artifacts"] / "sku_mapping.json").write_text(json.dumps({"0": {"sku_id": "sku_red"}}))
    monkeypatch.setattr(NotebookEmbedder, "__init__", lambda self, cfg: setattr(self, "dimension", 3))
    with pytest.raises(ValueError, match="rules"):
        Recognizer(cfg)
