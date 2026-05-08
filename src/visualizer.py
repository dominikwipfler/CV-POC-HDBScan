"""
Visualisation: OpenCV frame annotation + matplotlib scatter plots.

Colour scheme
-------------
Roles use either a fixed colour (Schiedsrichter, Torwart, Noise) or
a jersey-colour-derived colour (team display names are the German colour
word, e.g. "Grün", "Orange").  Role names that are not in the static dict
fall back to a hue derived from the German colour name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Static roles: fixed colour regardless of jersey colour
_COLORS_BGR: Dict[str, tuple] = {
    # Fixed roles
    "Schiedsrichter":  (0,   220, 220),   # yellow
    "Torwart A":       (200, 230,   0),   # cyan-green
    "Torwart B":       (200,   0, 200),   # magenta
    "Sonstige":        (120, 120, 120),   # grey
    "Noise / Unklar":  (0,     0, 170),   # dark red
    # Jersey-colour names → approximate BGR hue
    "Weiß":            (230, 230, 230),
    "Dunkel":          (60,   60,  60),
    "Grau":            (160, 160, 160),
    "Rot":             (0,    0,  220),
    "Orange-Rot":      (0,   70,  230),
    "Orange":          (0,  140,  255),
    "Gelb":            (0,  220,  220),
    "Grün":            (50,  200,  50),
    "Türkis":          (200, 200,   0),
    "Blau":            (220,  60,  60),
    "Lila":            (190,  50, 150),
}
_COLORS_MPL: Dict[str, str] = {
    "Schiedsrichter":  "#DDDD00",
    "Torwart A":       "#00E6B4",
    "Torwart B":       "#CC00CC",
    "Sonstige":        "#787878",
    "Noise / Unklar":  "#AA0000",
    "Weiß":            "#E6E6E6",
    "Dunkel":          "#3C3C3C",
    "Grau":            "#A0A0A0",
    "Rot":             "#DC0000",
    "Orange-Rot":      "#E64600",
    "Orange":          "#FF8C00",
    "Gelb":            "#DCDC00",
    "Grün":            "#32C832",
    "Türkis":          "#00C8C8",
    "Blau":            "#3C3CDC",
    "Lila":            "#9632BE",
}
_DEFAULT_BGR = (128, 128, 128)
_DEFAULT_MPL = "#888888"


class Visualizer:
    def __init__(self, config):
        self._thick = config.bbox_thickness
        self._fscale = config.font_scale

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def draw_frame(self, frame: np.ndarray, detections: List[dict],
                   role_map: Dict[int, str]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det.get("label", -1)
            prob = det.get("probability", 1.0)
            role = role_map.get(label, "Unbekannt")
            color = _COLORS_BGR.get(role, _DEFAULT_BGR)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, self._thick)

            text = role if prob >= 0.99 else f"{role} {prob:.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self._fscale, 1)
            # label background
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, self._fscale, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return out

    def add_legend(self, frame: np.ndarray, role_map: Dict[int, str]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        roles = sorted(set(role_map.values()))
        lh = 24
        pad = 10
        legend_w = 210
        legend_h = len(roles) * lh + 2 * pad
        lx = w - legend_w - 8
        ly = 8
        # semi-transparent background
        overlay = out.copy()
        cv2.rectangle(overlay, (lx - pad, ly), (lx + legend_w, ly + legend_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        cv2.rectangle(out, (lx - pad, ly), (lx + legend_w, ly + legend_h),
                      (180, 180, 180), 1)
        for i, role in enumerate(roles):
            color = _COLORS_BGR.get(role, _DEFAULT_BGR)
            y = ly + pad + i * lh + 12
            cv2.rectangle(out, (lx, y - 10), (lx + 18, y + 4), color, -1)
            cv2.putText(out, role, (lx + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return out

    def add_frame_info(self, frame: np.ndarray, frame_idx: int,
                       method: str, n_det: int) -> np.ndarray:
        out = frame.copy()
        h = out.shape[0]
        text = f"Frame {frame_idx} | {method} | {n_det} Spieler"
        cv2.putText(out, text, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------

    def save_scatter_plot(self, features_2d: np.ndarray,
                          hdb_result, kmeans_result,
                          output_dir: Path) -> None:
        results = [r for r in (hdb_result, kmeans_result) if r is not None]
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
        if n == 1:
            axes = [axes]
        fig.suptitle("Trikot-Feature-Clustering: HDBSCAN vs K-Means", fontsize=13)

        for ax, result in zip(axes, results):
            for lbl in sorted(set(result.labels)):
                role = result.role_map.get(lbl, f"Cluster {lbl}")
                color = _COLORS_MPL.get(role, _DEFAULT_MPL)
                mask = result.labels == lbl
                marker = "x" if role == "Noise / Unklar" else "o"
                alpha = 0.35 if role == "Noise / Unklar" else 0.75
                size = 25 if role == "Noise / Unklar" else 45
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=color, alpha=alpha, s=size, marker=marker,
                           label=f"{role} (n={int(mask.sum())})")

            title = f"{result.method}  |  {result.n_clusters} Cluster"
            if result.n_noise:
                title += f", {result.n_noise} Noise"
            if result.silhouette is not None:
                title += f"\nSilhouette = {result.silhouette:.3f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("UMAP-Dim 1")
            ax.set_ylabel("UMAP-Dim 2")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        path = output_dir / "cluster_scatter.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Scatter-Plot gespeichert: %s", path)

    # ------------------------------------------------------------------
    # Debug frame saving
    # ------------------------------------------------------------------

    def save_debug_frame(self, frame: np.ndarray, detections: List[dict],
                         rois: List[Optional[np.ndarray]],
                         frame_idx: int, output_dir: Path,
                         role_map: Dict[int, str]) -> None:
        debug_dir = output_dir / "debug_frames"
        debug_dir.mkdir(exist_ok=True)

        prefix = debug_dir / f"frame_{frame_idx:05d}"
        cv2.imwrite(str(prefix) + "_1_original.jpg", frame)

        annotated = self.draw_frame(frame, detections, role_map)
        cv2.imwrite(str(prefix) + "_2_detections.jpg", annotated)

        valid_rois = [(i, r) for i, r in enumerate(rois) if r is not None]
        if valid_rois:
            size = (56, 80)
            grid = np.zeros((size[1], len(valid_rois) * size[0], 3), dtype=np.uint8)
            for col, (_, roi) in enumerate(valid_rois[:12]):
                grid[:, col * size[0]: (col + 1) * size[0]] = cv2.resize(roi, size)
            cv2.imwrite(str(prefix) + "_3_torso_rois.jpg", grid)

    # ------------------------------------------------------------------
    # Video writer helper
    # ------------------------------------------------------------------

    @staticmethod
    def create_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
        for codec in ("mp4v", "avc1", "H264", "XVID"):
            ext = ".avi" if codec == "XVID" else ".mp4"
            out_path = path.with_suffix(ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            if writer.isOpened():
                logger.info("Video-Writer: %s  codec=%s", out_path, codec)
                return writer
        raise RuntimeError("Kein funktionierender Video-Codec gefunden.")
