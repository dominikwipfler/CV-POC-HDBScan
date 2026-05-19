"""
Per-frame colour correction and court-ROI isolation.

Ported from wels-monorepo/packages/ingestion/src/ingestion/pipeline/utils/.
Two corrections are applied per frame (in order):
  1. White balance  — Gray-World algorithm removes colour cast from artificial lighting
  2. CLAHE          — boosts local contrast so dark jerseys vs. bright floor differ more

Why this matters for HDBSCAN clustering
-----------------------------------------
Without correction, the same red jersey looks completely different at the left baseline
(warm spotlight) vs. the right baseline (cooler daylight). That variance spreads feature
points that should cluster tightly, making HDBSCAN either merge teams or inflate noise.

Flicker correction (temporal brightness smoothing) is available but requires a
calibration pass to compute target brightness — omitted here for simplicity.

ROI isolation
-------------
The court-floor detector samples HSV values from the lower centre of the frame to
estimate the floor colour, then builds a mask and finds the largest contiguous region.
The result is a (y_top, y_bottom) crop that removes tribunes, scoreboards and any
off-court area.  Player detection and feature extraction run on the cropped frame;
the full frame is still written to the output video.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour correction (from wels-monorepo color_correction.py)
# ---------------------------------------------------------------------------

def correct_white_balance(frame: np.ndarray,
                           roi_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Gray-World white balance.

    If roi_bounds=(y_top, y_bottom) is given the channel means are computed on
    that strip only (avoids over-correction from dominant jersey colours).
    """
    b, g, r = cv2.split(frame.astype(np.float32))
    if roi_bounds is not None:
        y0, y1 = roi_bounds
        avg_b = b[y0:y1].mean()
        avg_g = g[y0:y1].mean()
        avg_r = r[y0:y1].mean()
    else:
        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()

    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b < 1 or avg_g < 1 or avg_r < 1:
        return frame  # avoid division by zero on black frames

    b = np.clip(b * (avg_gray / avg_b), 0, 255)
    g = np.clip(g * (avg_gray / avg_g), 0, 255)
    r = np.clip(r * (avg_gray / avg_r), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the LAB lightness channel — preserves hue."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Court ROI detection (from wels-monorepo isolate_roi.py)
# ---------------------------------------------------------------------------

def detect_court_roi(frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """Detect court floor bounds from a single frame.

    Returns ((x, y, w, h), confidence) where confidence ∈ [0, 1].
    Returns (None, 0.0) if detection fails.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    img_h, img_w = frame.shape[:2]

    # Sample HSV from lower-centre strip (floor area)
    h_vals, s_vals, v_vals = [], [], []
    for y0_frac, y1_frac in [(0.4, 0.55), (0.55, 0.7), (0.7, 0.85)]:
        y0 = int(img_h * y0_frac)
        y1 = int(img_h * y1_frac)
        x0 = int(img_w * 0.3)
        x1 = int(img_w * 0.7)
        region = hsv[y0:y1, x0:x1]
        if region.size > 0:
            h_vals.extend(region[:, :, 0].flatten().tolist())
            s_vals.extend(region[:, :, 1].flatten().tolist())
            v_vals.extend(region[:, :, 2].flatten().tolist())

    if not h_vals:
        return None, 0.0

    mh = int(np.percentile(h_vals, 50))
    ms = int(np.percentile(s_vals, 50))
    mv = int(np.percentile(v_vals, 50))

    lower = np.array([max(0, mh - 15), max(0, ms - 50), max(0, mv - 50)])
    upper = np.array([min(179, mh + 15), min(255, ms + 50), min(255, mv + 50)])
    court_mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    frame_area = img_h * img_w
    x, y, w, h = cv2.boundingRect(largest)

    mask_pix = int(np.sum(court_mask > 0))
    confidence = float(np.sqrt((mask_pix / frame_area) * (contour_area / frame_area)))
    return (x, y, w, h), confidence


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------

class FramePreprocessor:
    """Applies white balance and CLAHE to each frame.

    Optionally detects the court ROI once during calibration and exposes
    roi_bounds so that downstream code (detection, feature extraction) can
    restrict processing to the court area.
    """

    def __init__(self, config):
        self._enabled: bool = config.use_preprocessing
        self.roi_bounds: Optional[Tuple[int, int]] = None  # (y_top, y_bottom)
        self._wb_roi: Optional[Tuple[int, int]] = None     # same, used for white balance

    def calibrate_roi(self, video_path: str, n_samples: int = 30) -> None:
        """Sample n_samples frames to estimate a stable court ROI."""
        if not self._enabled:
            return

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Jump directly to evenly-spaced frames instead of reading sequentially
        sample_indices = [int(i * total / n_samples) for i in range(n_samples)]
        logger.info("ROI-Kalibrierung: %d Frames werden gesampelt …", n_samples)

        y_tops, y_bottoms, confs = [], [], []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            roi, conf = detect_court_roi(frame)
            if roi is not None:
                _, y, _, h = roi
                y_tops.append(y)
                y_bottoms.append(y + h)
                confs.append(conf)
        cap.release()

        if not y_tops:
            logger.warning("ROI-Kalibrierung fehlgeschlagen — nutze ganzes Bild")
            return

        y_top = max(0, int(np.median(y_tops)) - int(height * 0.08))
        y_bottom = min(height, int(np.percentile(y_bottoms, 90)) + 15)
        self.roi_bounds = (y_top, y_bottom)
        self._wb_roi = (y_top, y_bottom)
        logger.info("Court-ROI erkannt: y=[%d, %d]  (Konfidenz Ø %.2f)",
                    y_top, y_bottom, float(np.mean(confs)))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply white balance + CLAHE. Returns corrected frame (same size)."""
        if not self._enabled:
            return frame
        frame = correct_white_balance(frame, roi_bounds=self._wb_roi)
        frame = apply_clahe(frame)
        return frame
