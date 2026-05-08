"""
Main processing pipeline.

Two-pass approach
-----------------
Pass 0  (calibrate) : Sample video to detect court ROI bounds (optional but recommended).
Pass 1  (extract)   : Read video, preprocess frames, detect players, extract jersey features.
                      Store all detections + features indexed by frame number.
Pass 2  (render)    : Re-read video frame by frame.
                      Annotate processed frames with cluster labels from Pass 1.
                      Write output video(s).

Why two passes?
  Storing all raw frames in RAM is impractical (>1 GB for a 30 s video).
  Two passes keep memory usage low while still producing a fully labelled output.

Preprocessing (new, ported from wels-monorepo)
-----------------------------------------------
  Each frame is colour-corrected (white balance + CLAHE) BEFORE detection and
  feature extraction.  This makes jersey colours consistent across lighting zones,
  which directly improves HDBSCAN cluster quality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .clustering import ClusteringEngine
from .court_keypoint_detector import CourtKeypointDetector
from .feature_extractor import FeatureExtractor
from .frame_preprocessor import FramePreprocessor
from .player_detector import get_detector
from .reporter import Reporter
from .tactical_analyzer import analyze_formations, analyze_shot_zones, analyze_trajectories
from .visualizer import Visualizer

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, config):
        self._cfg = config
        self._out = Path(config.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

        logger.info("Initialisiere Pipeline …")
        self._preprocessor = FramePreprocessor(config)
        self._detector = get_detector(config)
        self._extractor = FeatureExtractor(config)
        self._clusterer = ClusteringEngine(config)
        self._vis = Visualizer(config)
        self._reporter = Reporter(config)

        # Court keypoint detector (optional — disabled if model file absent)
        self._court = CourtKeypointDetector(
            config.court_model_path, device=config.yolo_device)
        self._court_active = bool(config.court_model_path) and self._court.load()
        if self._court_active:
            logger.info("Court-Keypoint-Detektor aktiv")
        else:
            logger.info("Court-Keypoint-Detektor deaktiviert (Modell fehlt oder deaktiviert)")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, video_path: str) -> None:
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {vpath}")

        # Pass 0: ROI calibration (samples ~30 frames, fast)
        if self._cfg.use_preprocessing and self._cfg.use_roi_calibration:
            logger.info("Pass 0 – ROI-Kalibrierung …")
            self._preprocessor.calibrate_roi(str(vpath))

        # Pass 0b: Court keypoint calibration (builds convex-hull boundary)
        if self._court_active:
            logger.info("Pass 0b – Court-Kalibrierung …")
            ok = self._court.calibrate(str(vpath), self._cfg.court_hull_frames)
            if not ok:
                logger.warning("Court-Kalibrierung fehlgeschlagen — alle Detektionen werden akzeptiert")
                self._court_active = False

        # Pass 1: extract
        all_features, frame_data = self._pass_extract(vpath)
        n_samples = len(all_features)
        logger.info("Pass 1 fertig: %d Spieler-Samples aus %d Frames",
                    n_samples, len(frame_data))

        if n_samples < 4:
            logger.error(
                "Zu wenige Spieler erkannt (%d). Prüfe Videodatei, "
                "Konfidenz (--yolo-confidence) oder --frame-skip.", n_samples)
            return

        # Clustering
        hdb, km = self._clusterer.fit(all_features)

        # Write labels back into frame_data (dicts are mutable → in-place)
        for fdata in frame_data.values():
            for det in fdata["detections"]:
                fi = det["feature_idx"]
                det["label"] = int(hdb.labels[fi])
                det["hdbscan_role"] = hdb.role_map.get(int(hdb.labels[fi]), "Unbekannt")
                det["probability"] = float(hdb.probabilities[fi])
                det["kmeans_label"] = int(km.labels[fi]) if km else -1

        # Pass 2: render
        self._pass_render(vpath, frame_data, hdb, km)

        # Scatter plot
        features_2d = self._clusterer.get_2d_representation(all_features)
        self._vis.save_scatter_plot(features_2d, hdb, km, self._out)

        # Reports
        self._reporter.save_reports(frame_data, hdb, km, self._out)
        self._reporter.print_summary(hdb, km, n_samples)

        # Tactical analyses (formations, shot zones, trajectories)
        cap_tmp = cv2.VideoCapture(str(vpath))
        frame_w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()

        # Derive the actual role name (jersey colour) for Team A and Team B
        team_colors = hdb.team_colors  # {"Team A": "Grün", "Team B": "Orange"}
        teams = (
            team_colors.get("Team A", "Team A"),
            team_colors.get("Team B", "Team B"),
        )
        logger.info("Trikotfarben: Team A=%s  Team B=%s", teams[0], teams[1])

        logger.info("Starte taktische Analysen …")
        try:
            analyze_formations(frame_data, hdb.role_map,
                               frame_w, frame_h, self._out,
                               teams=teams, team_colors=team_colors)
        except Exception as exc:
            logger.warning("Formations-Analyse fehlgeschlagen: %s", exc)
        try:
            analyze_shot_zones(frame_data, hdb.role_map, frame_w, frame_h,
                               self._out, teams=teams, team_colors=team_colors)
        except Exception as exc:
            logger.warning("Wurfzonen-Analyse fehlgeschlagen: %s", exc)
        try:
            analyze_trajectories(frame_data, frame_w, frame_h, self._out,
                                 role_map=hdb.role_map)
        except Exception as exc:
            logger.warning("Trajektorien-Analyse fehlgeschlagen: %s", exc)

        logger.info("Fertig! Alle Ergebnisse in: %s", self._out.resolve())

    # ------------------------------------------------------------------
    # Pass 1: preprocessing + detection + feature extraction
    # ------------------------------------------------------------------

    def _pass_extract(self, vpath: Path) -> Tuple[np.ndarray, Dict]:
        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Video: %d Frames @ %.1f fps", total, fps)

        all_features: List[np.ndarray] = []
        frame_data: Dict[int, dict] = {}
        feat_idx = 0
        frame_cnt = 0
        debug_saved = 0
        max_frames = self._cfg.max_frames or total

        pbar = tqdm(total=min(total, max_frames) // self._cfg.frame_skip,
                    desc="Pass 1 – Feature-Extraktion", unit="frame")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_cnt >= max_frames:
                break

            if frame_cnt % self._cfg.frame_skip == 0:
                # Colour correction — improves jersey colour consistency
                processed = self._preprocessor.process(frame)

                # Optionally crop to court ROI for detection
                # (bboxes are in cropped-frame space; we translate back below)
                roi = self._preprocessor.roi_bounds
                if roi is not None:
                    y0, y1 = roi
                    detect_frame = processed[y0:y1, :]
                    y_offset = y0
                else:
                    detect_frame = processed
                    y_offset = 0

                detections = self._detector.detect(detect_frame)

                det_records: List[dict] = []
                rois: List[Optional[np.ndarray]] = []
                fh, fw = frame.shape[:2]  # full-frame dims for position normalisation

                for di, det in enumerate(detections):
                    # Translate bbox back to full-frame coordinates
                    x1, y1_d, x2, y2_d = det.bbox
                    full_bbox = (x1, y1_d + y_offset, x2, y2_d + y_offset)

                    feats = self._extractor.extract(processed, full_bbox, fw, fh)
                    if feats is None:
                        continue

                    roi_img = self._extractor.get_torso_roi(processed, full_bbox)
                    rois.append(roi_img)

                    det_records.append({
                        "bbox": full_bbox,
                        "confidence": det.confidence,
                        "track_id": det.track_id,
                        "det_idx": di,
                        "feature_idx": feat_idx,
                        "features": feats.tolist(),
                        "label": -1,
                        "probability": 0.0,
                        "kmeans_label": -1,
                    })
                    all_features.append(feats)
                    feat_idx += 1

                frame_data[frame_cnt] = {"detections": det_records}

                # Save a few pre-clustering debug frames
                if (self._cfg.save_debug_frames
                        and debug_saved < 3
                        and det_records
                        and frame_cnt % (self._cfg.debug_frame_interval
                                         * self._cfg.frame_skip) == 0):
                    self._vis.save_debug_frame(
                        processed, det_records, rois, frame_cnt, self._out,
                        {-1: "Noch nicht geclustert"})
                    debug_saved += 1

                pbar.update(1)

            frame_cnt += 1

        cap.release()
        pbar.close()

        if not all_features:
            return np.empty((0, FeatureExtractor.FEATURE_DIM), dtype=np.float32), frame_data
        return np.stack(all_features), frame_data

    # ------------------------------------------------------------------
    # Pass 2: render annotated output
    # ------------------------------------------------------------------

    def _pass_render(self, vpath: Path, frame_data: Dict,
                     hdb_result, kmeans_result) -> None:
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        hdb_writer = self._vis.create_writer(self._out / "output_hdbscan.mp4", fps, w, h)

        comp_writer = None
        if kmeans_result:
            try:
                comp_writer = self._vis.create_writer(
                    self._out / "output_comparison.mp4", fps, w * 2 + 4, h)
            except Exception as exc:
                logger.warning("Vergleichs-Video konnte nicht erstellt werden: %s", exc)

        divider = np.full((h, 4, 3), 180, dtype=np.uint8)
        frame_cnt = 0
        debug_label_saved = 0
        last_dets: list = []   # carry-forward: last sampled frame's detections

        with tqdm(total=total, desc="Pass 2 – Video rendern", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Update carry-forward on sampled frames
                if frame_cnt in frame_data:
                    last_dets = frame_data[frame_cnt]["detections"]

                dets = last_dets  # always annotate with latest known detections

                # Base frame
                ann_base = frame.copy()

                # Draw court ROI boundary for orientation (if detected)
                if self._preprocessor.roi_bounds is not None:
                    cy0, cy1 = self._preprocessor.roi_bounds
                    cv2.line(ann_base, (0, cy0), (w, cy0), (0, 200, 200), 1)
                    cv2.line(ann_base, (0, cy1), (w, cy1), (0, 200, 200), 1)

                # HDBSCAN annotated
                ann_hdb = self._vis.draw_frame(ann_base, dets, hdb_result.role_map)
                ann_hdb = self._vis.add_legend(ann_hdb, hdb_result.role_map)
                ann_hdb = self._vis.add_frame_info(
                    ann_hdb, frame_cnt, "HDBSCAN", len(dets))
                hdb_writer.write(ann_hdb)

                # Labelled debug frames (first 3 with actual cluster labels)
                if (self._cfg.save_debug_frames and debug_label_saved < 3
                        and dets and frame_cnt in frame_data):
                    rois = [self._extractor.get_torso_roi(frame, d["bbox"])
                            for d in dets]
                    self._vis.save_debug_frame(
                        frame, dets, rois, frame_cnt, self._out,
                        hdb_result.role_map)
                    debug_label_saved += 1

                # Comparison video
                if comp_writer and kmeans_result:
                    km_dets = [{**d, "label": d["kmeans_label"]} for d in dets]
                    ann_km = self._vis.draw_frame(ann_base, km_dets,
                                                  kmeans_result.role_map)
                    ann_km = self._vis.add_frame_info(
                        ann_km, frame_cnt, "K-Means", len(dets))

                    cv2.putText(ann_hdb, "HDBSCAN", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(ann_km, "K-Means", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    comp_writer.write(np.hstack([ann_hdb, divider, ann_km]))

                frame_cnt += 1
                pbar.update(1)

        cap.release()
        hdb_writer.release()
        if comp_writer:
            comp_writer.release()
        logger.info("Video-Rendering abgeschlossen.")
