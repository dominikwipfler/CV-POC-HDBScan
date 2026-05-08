#!/usr/bin/env python3
"""
HDBSCAN Handball Team Clustering — POC Entry Point

Usage examples
--------------
  python main.py --video C:/Users/domwipfler/Downloads/test30sec.mp4
  python main.py --video C:/Users/domwipfler/Downloads/test30sec.mp4 --device cuda
  python main.py --video input/test30sec.mp4 --frame-skip 3 --min-cluster-size 4
  python main.py --video input/test30sec.mp4 --no-yolo --verbose
  python main.py --video input/test30sec.mp4 --max-frames 200  # quick smoke-test
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import Config
from src.video_processor import VideoProcessor


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HDBSCAN-basierte Spieler-Team-Erkennung fuer Handball-Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", "-v", required=True,
                   help="Pfad zum Eingangs-Video")
    p.add_argument("--output", "-o", default="outputs",
                   help="Ausgabeverzeichnis  [Standard: outputs]")

    # Video sampling
    p.add_argument("--frame-skip", type=int, default=5,
                   help="Jeden N-ten Frame verarbeiten  [Standard: 5]")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Maximale Frame-Anzahl (schneller Testlauf)")

    # HDBSCAN
    p.add_argument("--min-cluster-size", type=int, default=80,
                   help="HDBSCAN min_cluster_size  [Standard: 80 mit UMAP+epsilon]")
    p.add_argument("--min-samples", type=int, default=5,
                   help="HDBSCAN min_samples  [Standard: 5]")
    p.add_argument("--kmeans-k", type=int, default=4,
                   help="K-Means Cluster-Anzahl  [Standard: 4 = TeamA/B + Torwart + Schiri]")

    # YOLO11 detection (same model as wels-monorepo)
    p.add_argument("--yolo-model",
                   default="yolo11m.pt",
                   choices=["yolo11m.pt", "yolo11n.pt", "yolo11s.pt",
                            "yolov8n.pt", "yolov8m.pt"],
                   help="YOLO-Modell  [Standard: yolo11m.pt — wie wels-monorepo]")
    p.add_argument("--yolo-confidence", type=float, default=0.3,
                   help="YOLO Konfidenzschwelle  [Standard: 0.3]")
    p.add_argument("--yolo-imgsz", type=int, default=1280,
                   help="YOLO Input-Bildgroesse  [Standard: 1280]")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inferenz-Geraet  [Standard: cpu]")
    p.add_argument("--no-yolo", action="store_true",
                   help="YOLO deaktivieren -> MOG2-Fallback")

    # Preprocessing (from wels-monorepo color_correction + isolate_roi)
    p.add_argument("--no-preprocessing", action="store_true",
                   help="Farbkorrektur (White Balance + CLAHE) deaktivieren")
    p.add_argument("--no-roi", action="store_true",
                   help="Automatische Court-ROI-Erkennung deaktivieren")

    # Court keypoint model
    p.add_argument("--court-model",
                   default="best_court.pt",
                   help="YOLO-Pose Modell fuer Court-Keypoints  [Standard: best_court.pt]")
    p.add_argument("--no-court-model", action="store_true",
                   help="Court-Keypoint-Modell deaktivieren")

    # Misc
    p.add_argument("--no-debug", action="store_true",
                   help="Debug-Bilder nicht speichern")
    p.add_argument("--verbose", action="store_true",
                   help="Ausfuehrliche Logging-Ausgabe")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    # Auto-organise: outputs/<video-stem>/  so each run has its own folder
    video_stem = Path(args.video).stem
    output_dir = str(Path(args.output) / video_stem)

    config = Config(
        video_path=args.video,
        output_dir=output_dir,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        use_yolo=not args.no_yolo,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_imgsz=args.yolo_imgsz,
        yolo_device=args.device,
        use_preprocessing=not args.no_preprocessing,
        use_roi_calibration=not args.no_roi,
        hdbscan_min_cluster_size=args.min_cluster_size,
        hdbscan_min_samples=args.min_samples,
        kmeans_k=args.kmeans_k,
        save_debug_frames=not args.no_debug,
        court_model_path="" if args.no_court_model else args.court_model,
    )

    log.info("=== HDBSCAN Handball-Team-Clustering POC ===")
    log.info("Video      : %s", config.video_path)
    log.info("Output     : %s", Path(output_dir).resolve())
    log.info("Detektor   : %s  conf=%.2f  imgsz=%d  device=%s",
             config.yolo_model, config.yolo_confidence,
             config.yolo_imgsz, config.yolo_device)
    log.info("HDBSCAN    : min_cluster_size=%d  min_samples=%d  K-Means-k=%d",
             config.hdbscan_min_cluster_size, config.hdbscan_min_samples, config.kmeans_k)
    log.info("Preprocessing: wb+clahe=%s  roi=%s",
             config.use_preprocessing, config.use_roi_calibration)
    log.info("Court-Modell : %s",
             config.court_model_path if config.court_model_path else "deaktiviert")

    try:
        VideoProcessor(config).process(config.video_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Abgebrochen.")
        sys.exit(0)
    except Exception as exc:
        log.exception("Unerwarteter Fehler: %s", exc)
        sys.exit(1)

    out = Path(output_dir)
    print("\n  Verarbeitung abgeschlossen!")
    print(f"     Ausgabe-Ordner     : {out.resolve()}")
    print(f"     HDBSCAN-Video      : {out / 'output_hdbscan.mp4'}")
    print(f"     Scatter-Plot       : {out / 'cluster_scatter.png'}")
    print(f"     Abwehr-Formationen  : {out / 'tactical_abwehr_*.png'}")
    print(f"     Angriffs-Muster     : {out / 'tactical_angriff_*.png'}")
    print(f"     Wurfzonen           : {out / 'tactical_shot_zones.png'}")
    print(f"     Laufwege            : {out / 'tactical_trajectories.png'}")
    print(f"     Debug-Bilder       : {out / 'debug_frames/'}\n")


if __name__ == "__main__":
    main()
