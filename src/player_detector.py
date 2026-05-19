"""
Player detection module.

Primary:  YOLO11 + ByteTrack (ported from wels-monorepo PersonDetector).
          Uses model.track() with persist=True so ByteTrack assigns stable
          track_ids across frames — this means the same player keeps the same
          cluster assignment throughout the video.

Fallback: MOG2 background subtraction + contour filtering.
          No model download required; works on static-camera handball footage.

Key differences from original YOLOv8 approach
----------------------------------------------
- Model:      yolo11n.pt  (or yolo11m.pt for higher accuracy on GPU)
- Confidence: 0.3  (vs 0.5 — catches more distant/occluded players)
- imgsz:      1280 (vs 640 — critical for small players in wide-angle shots)
- Tracking:   ByteTrack via model.track(persist=True) — stable player IDs
- Device:     configurable ("cpu" or "cuda")
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class Detection:
    __slots__ = ("bbox", "confidence", "track_id", "class_id", "class_name")

    def __init__(self, bbox: tuple, confidence: float = 1.0, track_id: int = -1,
                 class_id: int = 0, class_name: str = "player"):
        self.bbox = bbox
        self.confidence = confidence
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name


# ---------------------------------------------------------------------------
# YOLO11 + ByteTrack detector (from wels-monorepo PersonDetector)
# ---------------------------------------------------------------------------

class YOLODetector:
    def __init__(self, model_name: str = "player_detection.pt",
                 confidence: float = 0.3,
                 min_height: int = 50,
                 imgsz: int = 1280,
                 device: str = "cpu",
                 max_persons: int = 20):
        self._available = False
        try:
            import ultralytics
            # Fallback: look in models/ subdirectory if not found at given path
            if not Path(model_name).exists() and not Path(model_name).is_absolute():
                candidate = Path("models") / model_name
                if candidate.exists():
                    model_name = str(candidate)
            self._model = ultralytics.YOLO(model_name)
            self._model.to(device)
            self._conf = confidence
            self._min_h = min_height
            self._imgsz = imgsz
            self._max = max_persons
            self._half = (device != "cpu")

            # Auto-detect which class IDs to use:
            # Standard COCO models (80 classes) → filter to "person" class 0.
            # Custom sports models (≤10 classes without "person") → detect all classes.
            self._class_names: dict = getattr(self._model, "names", {0: "player"})
            if any(v == "person" for v in self._class_names.values()):
                person_ids = [k for k, v in self._class_names.items() if v == "person"]
                self._detect_classes: Optional[List[int]] = person_ids
            else:
                self._detect_classes = None  # detect all classes

            self._available = True
            logger.info("YOLO + ByteTrack bereit: %s  device=%s  imgsz=%d  conf=%.2f  classes=%s",
                        model_name, device, imgsz, confidence,
                        list(self._class_names.values()))
        except Exception as exc:
            logger.warning("YOLO nicht verfügbar (%s). Nutze MOG2-Fallback.", exc)

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO11 + ByteTrack. Returns detections sorted by confidence desc."""
        if not self._available:
            return []

        results = self._model.track(
            frame,
            classes=self._detect_classes,
            conf=self._conf,
            imgsz=self._imgsz,
            half=self._half,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                if (y2 - y1) < self._min_h:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                class_id = int(box.cls[0]) if box.cls is not None else 0
                class_name = self._class_names.get(class_id, "player")
                out.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                ))

        out.sort(key=lambda d: d.confidence, reverse=True)
        return out[: self._max]


# ---------------------------------------------------------------------------
# MOG2 fallback detector
# ---------------------------------------------------------------------------

class MOG2Detector:
    """Background-subtraction based person detector.

    Works best on a mostly static camera. Requires a warm-up period (first ~50
    frames) to build a stable background model.
    """

    def __init__(self, min_height: int = 50, min_area: int = 1_200):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=48, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._min_h = min_height
        self._min_area = min_area
        logger.info("MOG2-Fallback-Detektor initialisiert")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        fg = self._bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
        fg = cv2.dilate(fg, self._kernel, iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fh, fw = frame.shape[:2]
        out: List[Detection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = h / max(w, 1)
            if ratio < 1.2 or ratio > 5.5:
                continue
            if h < self._min_h:
                continue
            if x < 4 or y < 4 or x + w > fw - 4 or y + h > fh - 4:
                continue
            conf = min(area / 6_000.0, 1.0)
            out.append(Detection((x, y, x + w, y + h), conf, track_id=-1))

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_detector(config) -> "YOLODetector | MOG2Detector":
    """Return YOLO11 detector if available, else MOG2 fallback."""
    if config.use_yolo:
        yolo = YOLODetector(
            model_name=config.yolo_model,
            confidence=config.yolo_confidence,
            min_height=config.min_bbox_height,
            imgsz=config.yolo_imgsz,
            device=config.yolo_device,
            max_persons=config.max_persons,
        )
        if yolo.available:
            return yolo
    return MOG2Detector(min_height=config.min_bbox_height)
