"""Clearly labeled synthetic scenario. Never used as a real model fallback."""

import cv2
import numpy as np

from .domain import Detection
from .vision import generate_plan

PRODUCTS = {
    "101": "Oat milk",
    "102": "Granola",
    "103": "Green tea",
    "104": "Pasta",
    "105": "Tomato soup",
    "106": "Coffee",
}
COLORS = [(184, 197, 128), (104, 171, 222), (147, 179, 92), (125, 201, 227), (102, 118, 204), (143, 120, 99)]


def scene(phase="stocked"):
    frame = np.full((600, 1080, 3), (228, 233, 235), np.uint8)
    cv2.putText(frame, "DEMO / SYNTHETIC SHELF", (35, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (95, 100, 100), 2)
    detections = []
    for row in range(2):
        y = 92 + row * 240
        cv2.rectangle(frame, (28, y + 173), (1052, y + 187), (149, 160, 165), -1)
        for col in range(6):
            if phase == "missing" and row == 0 and col == 2:
                continue
            x = 65 + col * 170
            sku_idx = col if phase != "misplaced" or (row, col) != (1, 4) else 1
            sku = str(101 + sku_idx)
            color = COLORS[sku_idx]
            cv2.rectangle(frame, (x, y), (x + 112, y + 168), color, -1)
            cv2.rectangle(frame, (x + 8, y + 40), (x + 104, y + 116), (244, 246, 244), -1)
            cv2.putText(
                frame,
                PRODUCTS[sku].split()[0],
                (x + 14, y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (50, 64, 65),
                1,
            )
            cv2.putText(frame, sku, (x + 30, y + 99), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 64, 65), 2)
            detections.append(
                Detection(
                    [x, y, x + 112, y + 168],
                    0.99,
                    None if phase == "unknown" and (row, col) == (0, 0) else sku,
                )
            )
    return frame, detections


def planogram():
    image, detections = scene()
    return generate_plan(detections, image.shape[1], image.shape[0], "demo")
