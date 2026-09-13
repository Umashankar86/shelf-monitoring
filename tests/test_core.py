import numpy as np
import pytest

from shelfwatch.domain import Detection, validate_planogram
from shelfwatch.models import decode_yolo
from shelfwatch.storage import Store
from shelfwatch.tracking import Consensus, Tracker
from shelfwatch.vision import align_to_reference, compare_plan, letterbox


def plan():
    return {
        "slots": [
            {"id": "a", "bbox": [0, 0, 0.5, 1], "expected_sku": "101"},
            {"id": "b", "bbox": [0.5, 0, 1, 1], "expected_sku": "102"},
        ]
    }


def test_unknown_occupies_slot_and_missing_is_distinct():
    cells = compare_plan(plan(), [Detection([0, 0, 50, 100], 0.9)], 100, 100)
    assert [c["state"] for c in cells] == ["UNKNOWN", "OOS"]


def test_swapped_products_are_both_misplaced():
    cells = compare_plan(
        plan(), [Detection([0, 0, 50, 100], 0.9, "102"), Detection([50, 0, 100, 100], 0.9, "101")], 100, 100
    )
    assert [c["state"] for c in cells] == ["MISPLACED", "MISPLACED"]


def test_one_detection_cannot_fill_two_slots():
    cells = compare_plan(plan(), [Detection([0, 0, 100, 100], 0.9, "101")], 100, 100)
    assert sum(c["state"] == "OOS" for c in cells) == 1


def test_tracker_keeps_identity_but_never_returns_ghost_stock():
    tracker = Tracker()
    first = tracker.update([Detection([0, 0, 20, 20], 0.9)])[0].track_id
    assert tracker.update([Detection([1, 0, 21, 20], 0.9)])[0].track_id == first
    assert tracker.update([]) == []


def test_consensus_requires_time_count_and_same_observed_identity():
    consensus = Consensus(3, 2)

    def update(t, state="MISPLACED", sku="102"):
        return consensus.update([{"id": "a", "state": state, "observed_sku": sku}], t)[0]["stable"]

    assert not update(0)
    assert not update(0.1)
    assert not update(0.2)
    assert update(2.1)
    assert not update(3, sku="103")
    assert not update(4, "UNKNOWN", None)
    assert not update(5, sku="103")


def test_alert_dedup_acknowledge_and_resolution(tmp_path):
    store = Store(tmp_path / "alerts.sqlite")
    store.observe("main", "a", "OOS", "101", None)
    store.observe("main", "a", "OOS", "101", None)
    rows = store.alerts()
    assert len(rows) == 1
    store.acknowledge(rows[0]["id"], "Checked")
    assert store.alerts()[0]["acknowledged"]
    store.observe("main", "a", "UNKNOWN", "101", None)
    assert not store.alerts()[0]["resolved"]
    store.observe("main", "a", "OK", "101", "101")
    assert store.alerts()[0]["resolved"]
    store.observe("main", "a", "OOS", "101", None)
    assert len(store.alerts()) == 2


def test_letterbox_decode_restores_original_coordinates_and_nms():
    image = np.zeros((100, 200, 3), np.uint8)
    _, ratio, left, top = letterbox(image, 640)
    raw = np.zeros((1, 5, 10), np.float32)
    raw[0, :, 0] = [320, 320, 320, 160, 0.9]
    raw[0, :, 1] = [320, 320, 320, 160, 0.8]
    detections = decode_yolo(raw, image.shape, ratio, left, top)
    assert len(detections) == 1
    assert np.allclose(detections[0].bbox, [50, 25, 150, 75])


def test_blank_camera_alignment_fails_closed():
    blank = np.zeros((200, 300, 3), np.uint8)
    aligned, error = align_to_reference(blank, blank)
    assert aligned is None and error


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 2, 1], [0, 0, float("nan"), 1]])
def test_invalid_planogram_boxes_rejected(box):
    p = plan()
    p["slots"][0]["bbox"] = box
    with pytest.raises(ValueError):
        validate_planogram(p)
