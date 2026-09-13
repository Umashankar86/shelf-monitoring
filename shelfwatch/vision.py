"""Image geometry, conservative alignment, and one-to-one shelf assignment."""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import iou


def letterbox(image, size):
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = round(w * ratio), round(h * ratio)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh))
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, ratio, left, top


def nms(boxes, scores, threshold, limit):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size and len(keep) < limit:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[idx, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[idx, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[idx, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[idx, 3], boxes[rest, 3])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        overlap = intersection / np.maximum(area[idx] + area[rest] - intersection, 1e-8)
        order = rest[overlap <= threshold]
    return keep


def align_to_reference(frame, reference, min_inliers=12):
    orb = cv2.ORB_create(nfeatures=2500)
    a, da = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    b, db = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(db) < 2:
        return None, "Not enough shelf features to align the camera."
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    if len(good) < min_inliers:
        return None, "Camera alignment has too few reliable matches."
    src = np.float32([a[m.queryIdx].pt for m in good])
    dst = np.float32([b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or not np.isfinite(matrix).all() or mask.sum() < min_inliers or mask.mean() < 0.45:
        return None, "Camera alignment failed its quality check."
    h, w = reference.shape[:2]
    coverage = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), matrix, (w, h))
    if (coverage > 0).mean() < 0.9:
        return None, "Camera view does not cover enough of the reference shelf."
    return cv2.warpPerspective(frame, matrix, (w, h)), None


def generate_plan(detections, width, height, name):
    if not detections:
        raise ValueError("No detected products. Analyze a correctly stocked shelf first.")
    # Cluster by vertical center relative to product height; order rows left-to-right.
    median_height = float(np.median([d.bbox[3] - d.bbox[1] for d in detections]))
    rows = []
    for d in sorted(detections, key=lambda x: (x.bbox[1] + x.bbox[3]) / 2):
        center = (d.bbox[1] + d.bbox[3]) / 2
        if (
            not rows
            or abs(center - np.mean([(x.bbox[1] + x.bbox[3]) / 2 for x in rows[-1]])) > median_height * 0.5
        ):
            rows.append([])
        rows[-1].append(d)
    slots = []
    for row_idx, row in enumerate(rows, 1):
        for col_idx, d in enumerate(sorted(row, key=lambda x: x.bbox[0]), 1):
            slots.append(
                {
                    "id": f"R{row_idx}-C{col_idx}",
                    "row": row_idx,
                    "bbox": [float(v / s) for v, s in zip(d.bbox, [width, height, width, height])],
                    "expected_sku": d.sku or "",
                }
            )
    return {"name": name, "width": width, "height": height, "slots": slots}


def compare_plan(plan, detections, width, height, min_overlap=0.2):
    slots = plan["slots"]
    boxes = [[v * s for v, s in zip(slot["bbox"], [width, height, width, height])] for slot in slots]
    matches = {}
    if detections and slots:
        scores = np.array([[iou(b, d.bbox) for d in detections] for b in boxes])
        rows, cols = linear_sum_assignment(-scores)
        matches = {int(r): int(c) for r, c in zip(rows, cols) if scores[r, c] >= min_overlap}
    results = []
    for i, slot in enumerate(slots):
        d = detections[matches[i]] if i in matches else None
        state = (
            "OOS"
            if d is None
            else (
                "UNKNOWN" if d.sku is None else ("OK" if d.sku == str(slot["expected_sku"]) else "MISPLACED")
            )
        )
        results.append(
            {
                "id": slot["id"],
                "expected_sku": str(slot["expected_sku"]),
                "observed_sku": d.sku if d else None,
                "state": state,
                "bbox": boxes[i],
            }
        )
    return results


def annotate(image, detections, cells):
    out = image.copy()
    colors = {
        "OK": (101, 175, 39),
        "OOS": (83, 83, 230),
        "MISPLACED": (27, 172, 239),
        "UNKNOWN": (160, 150, 140),
    }
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = colors["OK"] if d.sku else colors["UNKNOWN"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.sku or 'unknown'}  #{d.track_id or '-'}"
        cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell["bbox"])
        color = colors[cell["state"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{cell['id']} {cell['state']}",
            (x1, min(out.shape[0] - 5, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
