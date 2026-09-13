import math
from dataclasses import asdict, dataclass


@dataclass
class Detection:
    bbox: list[float]
    confidence: float
    sku: str | None = None
    distance: float | None = None
    track_id: int | None = None
    similarity: float | None = None

    def to_dict(self):
        return asdict(self)


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return area / max(total - area, 1e-8)


def validate_planogram(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("slots"), list) or not plan["slots"]:
        raise ValueError("A planogram needs at least one slot.")
    ids = set()
    for slot in plan["slots"]:
        sid = str(slot.get("id", "")).strip()
        if not sid or sid in ids:
            raise ValueError("Slot IDs must be nonempty and unique.")
        ids.add(sid)
        box = slot.get("bbox", [])
        if len(box) != 4 or not all(
            isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in box
        ):
            raise ValueError("Slot boxes must contain four normalized coordinates between 0 and 1.")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Slot boxes must have positive width and height.")
        if not str(slot.get("expected_sku", "")).strip():
            raise ValueError("Assign an expected SKU to every slot before saving.")
    return plan
