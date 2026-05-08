"""
Standalone tactical analysis — reads outputs/clustering_results.json
and produces formation, shot-zone, and trajectory plots WITHOUT
re-running YOLO detection.

Usage:
    python run_tactical.py
    python run_tactical.py --json outputs/clustering_results.json --output outputs
"""

import argparse
import json
import logging
from pathlib import Path

from src.tactical_analyzer import (
    analyze_formations,
    analyze_shot_zones,
    analyze_trajectories,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _load_frame_data(json_path: Path):
    """Reconstruct frame_data dict from clustering_results.json."""
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    # role_map: int key (from hdbscan_label) → role name
    raw_role_map = payload["metadata"]["hdbscan"].get("role_map", {})
    role_map = {int(k): v for k, v in raw_role_map.items()}

    frame_w = frame_h = None  # estimated from bbox later

    frame_data: dict = {}
    for det in payload["detections"]:
        frame_idx = det["frame"]
        if frame_idx not in frame_data:
            frame_data[frame_idx] = {"detections": []}

        bbox = det["bbox"]
        frame_data[frame_idx]["detections"].append({
            "bbox": bbox,
            "track_id": det.get("track_id", -1),
            "label": det.get("hdbscan_label", -1),
            "hdbscan_role": det.get("hdbscan_role", ""),
            "confidence": det.get("detection_confidence", 1.0),
            "feature_idx": det.get("feature_index", -1),
        })

        # Estimate frame dimensions from max bbox coords
        if frame_w is None or bbox[2] > frame_w:
            frame_w = int(bbox[2]) + 1
        if frame_h is None or bbox[3] > frame_h:
            frame_h = int(bbox[3]) + 1

    # Fallback dimensions if not determinable
    frame_w = frame_w or 1920
    frame_h = frame_h or 1080

    return frame_data, role_map, frame_w, frame_h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="outputs/clustering_results.json")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Lade JSON: %s", json_path)
    frame_data, role_map, frame_w, frame_h = _load_frame_data(json_path)
    log.info("Frames: %d  frame_w=%d  frame_h=%d", len(frame_data), frame_w, frame_h)
    log.info("Rollen: %s", role_map)

    log.info("── Formations-Analyse ──────────────────")
    try:
        analyze_formations(frame_data, role_map, frame_w, frame_h, out_dir)
    except Exception as exc:
        log.error("Formations-Analyse fehlgeschlagen: %s", exc)

    log.info("── Wurfzonen-Analyse ───────────────────")
    try:
        analyze_shot_zones(frame_data, role_map, frame_w, frame_h, out_dir)
    except Exception as exc:
        log.error("Wurfzonen-Analyse fehlgeschlagen: %s", exc)

    log.info("── Trajektorien-Analyse ────────────────")
    try:
        analyze_trajectories(frame_data, frame_w, frame_h, out_dir)
    except Exception as exc:
        log.error("Trajektorien-Analyse fehlgeschlagen: %s", exc)

    log.info("Fertig! Outputs in: %s", out_dir.resolve())
    print(f"\n  Formations Team A : {out_dir}/tactical_formations_Team_A.png")
    print(f"  Formations Team B : {out_dir}/tactical_formations_Team_B.png")
    print(f"  Wurfzonen         : {out_dir}/tactical_shot_zones.png")
    print(f"  Laufwege          : {out_dir}/tactical_trajectories.png\n")


if __name__ == "__main__":
    main()
