"""
Ball detection and shot tracking.

BallDetector  — YOLO-based ball detection (model/ball_detection.pt).
detect_shots  — identifies shot events from tracked ball positions.

A shot is detected when:
  • The ball is in the attacking half (x > 0.5 or x < 0.5 depending on direction).
  • The ball has a high lateral velocity toward the goal (|vx| > SHOT_VX_THRESHOLD).
  • The ball position is within the plausible shooting range (6m to 15m from goal).

The function returns a list of (x_norm, y_norm) positions from which shots were taken,
normalised to [0,1] on the full-court coordinate system.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum x-velocity (per analysed frame, normalised 0-1) to count as a shot.
# At frame_skip=5 and 25 fps that's 5 frames → ~0.04/frame ≈ 1.6 m per frame step.
_SHOT_VX_THRESHOLD = 0.025

# Shooting zone in normalised court x: between 6m (~0.15) and 15m (~0.375) from goal.
_SHOT_X_MIN = 0.12   # player must be at least this far from goal
_SHOT_X_MAX = 0.42   # player must not be further than this (half-court + buffer)


class BallDetector:
    """YOLO-based ball detector with frame-to-frame position tracking."""

    def __init__(self, model_path: str, confidence: float = 0.25,
                 device: str = "cpu") -> None:
        self._available = False
        self._model_path = model_path
        try:
            import ultralytics
            self._model = ultralytics.YOLO(model_path)
            self._model.to(device)
            self._conf = confidence
            self._available = True
            logger.info("BallDetector bereit: %s  device=%s  conf=%.2f",
                        model_path, device, confidence)
        except Exception as exc:
            logger.warning("BallDetector nicht verfügbar (%s). Keine Torschuss-Erkennung.", exc)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray,
               frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float]]:
        """Run YOLO on frame and return (x_norm, y_norm, confidence) of best ball detection."""
        if not self._available:
            return None
        try:
            results = self._model.predict(frame, conf=self._conf,
                                          verbose=False, imgsz=640)
            best_conf = 0.0
            best_xy: Optional[Tuple[float, float]] = None
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    c = float(box.conf[0])
                    if c > best_conf:
                        best_conf = c
                        xyxy = box.xyxy[0].cpu().numpy()
                        cx = float((xyxy[0] + xyxy[2]) / 2) / frame_w
                        cy = float((xyxy[1] + xyxy[3]) / 2) / frame_h
                        best_xy = (cx, cy)
            if best_xy:
                return (best_xy[0], best_xy[1], best_conf)
        except Exception as exc:
            logger.debug("BallDetector.detect failed: %s", exc)
        return None


def detect_shots(ball_tracks: List[Tuple[int, float, float]],
                 halftime_frame: Optional[int] = None,
                 ) -> List[Tuple[float, float]]:
    """
    Analyse ball positions over time and return estimated shot positions.

    Parameters
    ----------
    ball_tracks : list of (frame_idx, x_norm, y_norm) — sorted by frame_idx.
    halftime_frame : if set, x-coords are flipped for frames >= halftime_frame
                     so that the left goal is always "own goal" after normalisation.

    Returns
    -------
    List of (x_norm, y_norm) positions where shots were initiated.
    These are in FULL-COURT coordinates (0 = left goal, 1 = right goal).
    """
    if len(ball_tracks) < 2:
        return []

    shots: List[Tuple[float, float]] = []
    prev_frame, prev_x, prev_y = ball_tracks[0]

    for frame_idx, x, y in ball_tracks[1:]:
        # Apply halftime x-flip so analysis is always in consistent orientation
        eff_x = (1.0 - x) if (halftime_frame and frame_idx >= halftime_frame) else x
        eff_prev_x = (1.0 - prev_x) if (halftime_frame and prev_frame >= halftime_frame) else prev_x

        vx = eff_x - eff_prev_x
        abs_vx = abs(vx)

        if abs_vx > _SHOT_VX_THRESHOLD:
            # Ball moving toward left goal (vx < 0) or right goal (vx > 0)
            if vx < 0:
                # Shot toward left goal: ball must be in left half
                dist_from_goal = eff_prev_x  # distance of shot origin from left goal
            else:
                # Shot toward right goal: ball must be in right half
                dist_from_goal = 1.0 - eff_prev_x

            if _SHOT_X_MIN <= dist_from_goal <= _SHOT_X_MAX:
                shots.append((prev_x, prev_y))  # record original (unflipped) position
                logger.debug("Torschuss erkannt: frame=%d  x=%.3f  y=%.3f  vx=%.3f",
                             prev_frame, prev_x, prev_y, vx)

        prev_frame, prev_x, prev_y = frame_idx, x, y

    logger.info("Torschüsse erkannt: %d", len(shots))
    return shots
