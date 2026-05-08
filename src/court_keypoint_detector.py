"""
Court keypoint detection using a YOLO-pose model (best_court.pt).

The model detects 24 handball-court keypoints per frame (goal posts, field
corners, 7m spot, 9m arc / sideline crossings, 6m arc / goal-line crossings).

Usage in pipeline
-----------------
Calibrate: run on ~30 sampled frames to collect stable mean positions.
Visualise:  call draw_keypoints(frame, detected_kpts) in Pass 2 to show
            coloured dots for each recognised court landmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DET_CONF_THRESH = 0.40
_KPT_CONF_THRESH = 0.50

# 24 distinct BGR colours — one per keypoint index
_KPT_COLORS: List[Tuple[int, int, int]] = [
    (50,  50,  255), (50,  200, 50),  (255, 50,  50),  (255, 255, 0),
    (255, 0,   255), (0,   255, 255), (0,   150, 255), (150, 255, 0),
    (255, 100, 0),   (0,   255, 150), (200, 0,   200), (0,   200, 200),
    (255, 180, 0),   (180, 255, 0),   (0,   180, 255), (180, 0,   255),
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255), (220, 220, 0),
    (220, 0,   220), (0,   220, 220), (200, 150, 100), (100, 200, 150),
]

# Short human-readable labels for the most commonly detected keypoints.
# Fill in as ground truth is confirmed; others show "kNN".
_KPT_LABELS: Dict[int, str] = {
    3:  "6m-Front L",
    4:  "6m-Torlinie L",
    8:  "Mittelkreis L",
    9:  "Mittelpunkt",
    6:  "Mittelkreis oben",
    15: "6m-Front R",
}


class CourtKeypointDetector:
    """Detect handball court keypoints; visualise as coloured dots."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self._model_path = Path(model_path)
        self._device = device
        self._model = None
        # Mean pixel position per keypoint index, filled by calibrate()
        self._kpt_means: Dict[int, Tuple[float, float]] = {}
        self._frame_w: int = 0
        self._frame_h: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load(self) -> bool:
        if not self._model_path.exists():
            logger.warning("Court-Modell nicht gefunden: %s", self._model_path)
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info("Court-Keypoint-Modell geladen: %s", self._model_path.name)
            return True
        except Exception as exc:
            logger.warning("Court-Modell konnte nicht geladen werden: %s", exc)
            self._model = None
            return False

    # ------------------------------------------------------------------
    # Per-frame detection — returns (x_px, y_px, conf, kpt_index)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray
               ) -> List[Tuple[int, int, float, int]]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                frame, task="pose", verbose=False,
                device=self._device, conf=_DET_CONF_THRESH,
            )
        except Exception as exc:
            logger.debug("Court-Erkennung fehlgeschlagen: %s", exc)
            return []

        out: List[Tuple[int, int, float, int]] = []
        for r in results:
            if r.keypoints is None:
                continue
            for det in r.keypoints.data:
                for ki, kpt in enumerate(det):
                    x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
                    if conf >= _KPT_CONF_THRESH and x > 0 and y > 0:
                        out.append((int(x), int(y), float(conf), ki))
        return out

    # ------------------------------------------------------------------
    # Calibration — collects mean keypoint positions for reference
    # ------------------------------------------------------------------

    def calibrate(self, video_path: str, n_frames: int = 30) -> bool:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        from collections import defaultdict
        accum: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        for i in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n_frames))
            ret, frame = cap.read()
            if not ret:
                break
            for x, y, _c, ki in self.detect(frame):
                accum[ki].append((float(x), float(y)))

        cap.release()

        for ki, pts in accum.items():
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                self._kpt_means[ki] = (float(np.mean(xs)), float(np.mean(ys)))

        logger.info(
            "Court-Kalibrierung: %d stabile Keypoints aus %d Frames",
            len(self._kpt_means), n_frames,
        )
        return len(self._kpt_means) >= 2

    # ------------------------------------------------------------------
    # hull_available — always False; hull was removed
    # ------------------------------------------------------------------

    def hull_available(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def draw_keypoints(self, frame: np.ndarray,
                       keypoints: List[Tuple[int, int, float, int]]
                       ) -> np.ndarray:
        """
        Draw per-frame detected keypoints as coloured circles.
        Each keypoint index gets a unique colour defined in _KPT_COLORS.
        """
        out = frame.copy()
        for x, y, conf, ki in keypoints:
            col = _KPT_COLORS[ki % len(_KPT_COLORS)]
            radius = max(5, int(conf * 9))
            cv2.circle(out, (x, y), radius, col, -1)
            cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1)
            label = _KPT_LABELS.get(ki, f"k{ki}")
            cv2.putText(out, label, (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 2)
            cv2.putText(out, label, (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        col, 1)
        return out

    def draw_overlay(self, frame: np.ndarray,
                     keypoints: Optional[List] = None) -> np.ndarray:
        """Backward-compat wrapper — draws per-keypoint dots if available."""
        if keypoints and len(keypoints) > 0:
            if len(keypoints[0]) == 4:
                return self.draw_keypoints(frame, keypoints)
        return frame.copy()
