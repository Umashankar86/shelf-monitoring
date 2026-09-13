"""Kalman prediction + Hungarian IoU association. Only current detections are returned."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .domain import iou


class Tracker:
    def __init__(self, min_iou=0.25, max_missed=3):
        self.min_iou, self.max_missed = min_iou, max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4)
        observation = np.eye(4, 8)
        for track in self.tracks.values():
            track["x"] = transition @ track["x"]
            track["p"] = transition @ track["p"] @ transition.T + np.eye(8)
            track["missed"] += 1
        ids = list(self.tracks)
        matched = set()
        if ids and detections:
            costs = np.array([[1 - iou(self.tracks[t]["x"][:4], d.bbox) for d in detections] for t in ids])
            rows, cols = linear_sum_assignment(costs)
            for r, c in zip(rows, cols):
                if 1 - costs[r, c] < self.min_iou:
                    continue
                track = self.tracks[ids[r]]
                cov = observation @ track["p"] @ observation.T + np.eye(4) * 4
                gain = np.linalg.solve(cov, observation @ track["p"]).T
                track["x"] += gain @ (np.asarray(detections[c].bbox) - observation @ track["x"])
                track["p"] = (np.eye(8) - gain @ observation) @ track["p"]
                track["missed"] = 0
                detections[c].track_id = ids[r]
                matched.add(c)
        for i, d in enumerate(detections):
            if i not in matched:
                d.track_id = self.next_id
                self.tracks[self.next_id] = {
                    "x": np.r_[d.bbox, np.zeros(4)],
                    "p": np.eye(8) * 10,
                    "missed": 0,
                }
                self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items() if v["missed"] <= self.max_missed}
        return detections


class Consensus:
    def __init__(self, observations=3, seconds=2):
        self.observations, self.seconds = observations, seconds
        self.history = {}

    def update(self, cells, timestamp):
        for cell in cells:
            key = cell["id"]
            value = (cell["state"], cell["observed_sku"])
            old = self.history.get(key)
            if cell["state"] == "UNKNOWN":
                self.history.pop(key, None)
                cell["stable"] = False
                continue
            if old is None or old["value"] != value:
                old = {"value": value, "count": 0, "since": timestamp}
            old["count"] += 1
            self.history[key] = old
            cell["stable"] = old["count"] >= self.observations and timestamp - old["since"] >= self.seconds
        return cells
