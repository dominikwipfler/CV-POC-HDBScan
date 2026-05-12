from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Input / Output
    video_path: str = ""
    output_dir: str = "outputs"

    # Video sampling
    frame_skip: int = 5          # Process every Nth frame
    max_frames: Optional[int] = None  # Cap total frames (quick test mode)

    use_yolo: bool = True
    yolo_model: str = "player_detection.pt"  # Spezialisiertes Spieler-Erkennungsmodell
    yolo_confidence: float = 0.3      # Lower than YOLOv8 default — catches more players
    yolo_imgsz: int = 1280            # Larger input = better small/distant player detection
    yolo_device: str = "cpu"          # "cuda" on GPU machines
    max_persons: int = 25             # Handball: 7v7 + GK + referees = up to ~20 persons
    min_bbox_height: int = 40         # Slightly lower to catch distant players

    # Frame preprocessing (ported from wels-monorepo color_correction + isolate_roi)
    use_preprocessing: bool = True    # White balance + CLAHE per frame
    use_roi_calibration: bool = True  # Auto-detect court bounds before processing

    # Feature extraction — torso crop (fraction of bbox height)
    torso_top: float = 0.15      # Skip head (top 15%)
    torso_bottom: float = 0.60   # Skip legs (below 60%)

    # HDBSCAN — with UMAP preprocessing, much lower values work (tight embedded clusters)
    # Rule of thumb with UMAP: min_cluster_size ≈ 0.3–1% of total samples
    # For 6000 samples → 20–60; without UMAP use 100–200
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 5

    # K-Means comparison — 4 classes: Team A, Team B, Torwart, Schiedsrichter
    kmeans_k: int = 5

    # Visualisation
    bbox_thickness: int = 2
    font_scale: float = 0.50

    # Debug output
    save_debug_frames: bool = True
    debug_frame_interval: int = 75
