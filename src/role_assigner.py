"""
Track-aware role assigner for handball.

Why a dedicated module instead of HDBSCAN?
------------------------------------------
HDBSCAN finds *arbitrary* density-based clusters — the wrong tool here:

• Teams dominate by player count (14 field players vs ≤ 4 GK/refs).
  k-Means(k=2) on colour forces exactly two clusters; teams pull the
  centroids reliably even when a few refs/GKs are mixed in.

• White-jersey players have undefined hue → flat hue histogram →
  HDBSCAN scatters them into many small clusters that never consolidate.
  k-Means(k=2) forces all of them into one group.

• Goalkeeper and referee detection based on a single frame's colour is
  fragile.  ByteTrack assigns stable track_ids across frames; aggregating
  a track's x-position over the entire video makes goalkeeper detection
  trivial: goalkeepers stay within the 6 m zone throughout the game.

Algorithm
---------
1. Build per-track statistics (mean x-position + mean jersey colour).
2. Identify goalkeepers:
     a. YOLO class labels ("goalkeeper") — highest priority.
     b. Position: tracks whose mean_x is within the 6 m zone (≤ 0.15 or
        ≥ 0.85) AND whose x-std is low (< 0.10).  At most 2.
3. k-Means(k=2) on colour features of all non-GK detections.
     → Team A centroid, Team B centroid.
4. Referee detection: per-detection distance to nearest team centroid.
   Detections far from BOTH centroids (MAD-based outlier threshold)
   → referee candidates.  A track with ≥ 45 % outlier detections
   → Schiedsrichter.
5. Apply roles; fall back to nearest-centroid colour for short/no tracks.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Feature layout (must match feature_extractor.py) ────────────────────────
_N_COLOR    = 19
_IDX_WHITE  = 6
_IDX_DARK   = 7
_IDX_COLOR  = 8
_IDX_HUE    = 9
_HUE_NAMES  = ["Rot", "Orange-Rot", "Orange", "Gelb",
                "Grün", "Türkis", "Blau", "Lila"]

# Synthetic integer labels (must match visualiser's colour lookup)
_LABEL = {"Team A": 0, "Team B": 1, "Torwart": 2,
          "Schiedsrichter": 3, "Sonstige": 4, "Noise / Unklar": -1}


def _jersey_color_name(centroid: np.ndarray) -> str:
    white = float(centroid[_IDX_WHITE])
    dark  = float(centroid[_IDX_DARK])
    col   = float(centroid[_IDX_COLOR])
    if white > 0.45:
        return "Weiß"
    if dark > 0.50 and col < 0.20:
        return "Dunkel"
    if col < 0.15:
        return "Grau"
    return _HUE_NAMES[int(np.argmax(centroid[_IDX_HUE: _IDX_HUE + 8]))]


# ── Compatibility shim so video_processor.py needs minimal changes ───────────

class RoleResult:
    """Drop-in replacement for ClusterResult used by visualiser / reporter."""

    def __init__(self, labels: np.ndarray, role_map: Dict[int, str],
                 team_colors: Dict[str, str], n_samples: int):
        self.labels       = labels
        self.probabilities = np.ones(n_samples, dtype=np.float32)
        self.method       = "k-Means(k=2) + Track-Positionen"
        self.n_clusters   = len({v for v in role_map.values()
                                  if v != "Noise / Unklar"})
        self.n_noise      = 0
        self.n_reassigned = 0
        self.silhouette   = None
        self.role_map     = role_map
        self.team_colors  = team_colors

    def get_role(self, label: int) -> str:
        return self.role_map.get(label, "Sonstige")

    # Stubs so old call-sites don't crash
    def map_roles(self):       return self.role_map
    def map_roles_smart(self, _): return self.role_map


# ── Main class ───────────────────────────────────────────────────────────────

class RoleAssigner:
    """Assigns handball roles using tracking trajectories + k-Means(k=2)."""

    GK_ZONE_X      = 0.15   # 6 m / 40 m ≈ 0.15 of normalised court length
    GK_X_STD_MAX   = 0.10   # keepers have low lateral variance
    GK_MAX         = 2      # at most one per goal
    MIN_TRACK_OBS  = 4      # minimum frames for a track to be trusted
    REF_MAD_K      = 2.5    # MAD multiplier for outlier threshold
    REF_TRACK_FRAC = 0.45   # fraction of track detections that must be outliers

    _GK_YOLO  = frozenset({"goalkeeper", "torwart", "keeper", "gk"})
    _REF_YOLO = frozenset({"referee", "schiedsrichter", "ref"})

    def __init__(self, config=None):
        self._km:      Optional[KMeans]         = None
        self._scaler:  Optional[StandardScaler] = None
        self._centers: Optional[np.ndarray]     = None  # (2, _N_COLOR)

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        frame_data:     Dict,
        all_features:   np.ndarray,
        frame_w:        int,
        frame_h:        int,
        halftime_frame: Optional[int] = None,
    ) -> RoleResult:
        """
        Assign roles to every detection in frame_data (in-place).

        Sets det["hdbscan_role"], det["label"], det["probability"].
        Returns a RoleResult compatible with the existing visualiser.
        """
        n = len(all_features)
        if n < 4:
            return RoleResult(np.full(n, -1, np.int32),
                              {-1: "Noise / Unklar"}, {}, n)

        # 1. Per-track statistics
        tracks = self._build_tracks(frame_data, all_features,
                                    frame_w, frame_h, halftime_frame)

        # 2. YOLO-class overrides (highest confidence)
        yolo_gk, yolo_ref = self._yolo_overrides(frame_data)

        # 3. Position-based goalkeeper detection
        pos_gk = self._position_gk(tracks, exclude=yolo_gk | yolo_ref)
        gk_tids = yolo_gk | pos_gk
        logger.info("Torwart-Tracks: %d gesamt (YOLO:%d Pos:%d)",
                    len(gk_tids), len(yolo_gk), len(pos_gk))

        # 4. k-Means(k=2) on non-GK colour features → team centroids
        if not self._fit_kmeans(frame_data, all_features, gk_tids):
            logger.warning("k-Means fehlgeschlagen — alle → Sonstige")
            labels = np.full(n, _LABEL["Sonstige"], np.int32)
            self._write_roles(frame_data, all_features,
                              gk_tids, set(), labels)
            return RoleResult(labels, {_LABEL["Sonstige"]: "Sonstige"}, {}, n)

        # 5. Referee detection disabled — yellow/unusual jerseys were falsely flagged.
        #    YOLO-labelled refs are still respected; all others go to nearest team.
        ref_tids = yolo_ref

        # 6. Apply roles and build output
        labels = self._write_roles(
            frame_data, all_features, gk_tids, ref_tids, np.full(n, -1, np.int32))

        team_colors = {}
        for i, name in enumerate(("Team A", "Team B")):
            team_colors[name] = _jersey_color_name(self._centers[i])

        role_map = {
            _LABEL["Team A"]:         "Team A",
            _LABEL["Team B"]:         "Team B",
            _LABEL["Torwart"]:        "Torwart",
            _LABEL["Schiedsrichter"]: "Schiedsrichter",
            _LABEL["Sonstige"]:       "Sonstige",
            -1:                        "Noise / Unklar",
        }

        logger.info("Rollenzuweisung abgeschlossen: %s",
                    {r: int(np.sum(labels == l)) for l, r in role_map.items()
                     if l != -1})
        return RoleResult(labels, role_map, team_colors, n)

    # ── Internal steps ────────────────────────────────────────────────────────

    def _build_tracks(
        self, frame_data: Dict, all_features: np.ndarray,
        frame_w: int, frame_h: int, halftime_frame: Optional[int]
    ) -> Dict[int, dict]:
        tracks: Dict[int, dict] = {}
        for frame_idx, fdata in frame_data.items():
            for det in fdata["detections"]:
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2.0 / frame_w
                # Flip after halftime so each GK always appears near the same goal
                if halftime_frame and frame_idx >= halftime_frame:
                    cx = 1.0 - cx
                fi = det.get("feature_idx", -1)
                t  = tracks.setdefault(tid, {"xs": [], "colors": []})
                t["xs"].append(cx)
                if 0 <= fi < len(all_features):
                    t["colors"].append(all_features[fi, :_N_COLOR])

        for tid, t in tracks.items():
            xs = np.asarray(t["xs"])
            t["n"]      = len(xs)
            t["mean_x"] = float(xs.mean())
            t["std_x"]  = float(xs.std()) if len(xs) > 1 else 0.0
            cols = np.asarray(t["colors"]) if t["colors"] else np.empty((0, _N_COLOR))
            t["mean_color"] = cols.mean(axis=0) if len(cols) else None

        return tracks

    def _yolo_overrides(self, frame_data: Dict) -> Tuple[Set[int], Set[int]]:
        gk: Set[int] = set()
        ref: Set[int] = set()
        for fdata in frame_data.values():
            for det in fdata["detections"]:
                cls = det.get("yolo_class", "").lower()
                tid = det.get("track_id", -1)
                if tid < 0:
                    continue
                if cls in self._GK_YOLO:
                    gk.add(tid)
                elif cls in self._REF_YOLO:
                    ref.add(tid)
        if gk or ref:
            logger.info("YOLO-Klassen: %d GK-Tracks, %d Ref-Tracks", len(gk), len(ref))
        return gk, ref

    def _position_gk(self, tracks: Dict, exclude: Set[int]) -> Set[int]:
        """
        Select goalkeeper tracks by sustained near-goal position.

        A goalkeeper:
          • mean_x < GK_ZONE_X  (left goal)  OR  mean_x > 1 – GK_ZONE_X  (right)
          • std_x  < GK_X_STD_MAX  (low lateral movement)
          • n ≥ MIN_TRACK_OBS

        At most one per goal side.
        """
        left:  List[Tuple[float, int]] = []   # (n_obs, tid) near left goal
        right: List[Tuple[float, int]] = []   # near right goal

        for tid, t in tracks.items():
            if tid in exclude or t["n"] < self.MIN_TRACK_OBS:
                continue
            if t["std_x"] >= self.GK_X_STD_MAX:
                continue
            if t["mean_x"] < self.GK_ZONE_X:
                left.append((t["n"], tid))
            elif t["mean_x"] > 1.0 - self.GK_ZONE_X:
                right.append((t["n"], tid))

        gk: Set[int] = set()
        for side in (left, right):
            if side:
                best_n, best_tid = max(side)
                gk.add(best_tid)
                logger.info("Torwart erkannt: Track %d  mean_x=%.3f  n=%d",
                            best_tid, tracks[best_tid]["mean_x"], best_n)
        return gk

    def _fit_kmeans(
        self, frame_data: Dict, all_features: np.ndarray, gk_tids: Set[int]
    ) -> bool:
        """Fit k-Means(k=2) on colour features of non-GK detections."""
        colors: List[np.ndarray] = []
        for fdata in frame_data.values():
            for det in fdata["detections"]:
                if det.get("track_id", -1) in gk_tids:
                    continue
                fi = det.get("feature_idx", -1)
                if 0 <= fi < len(all_features):
                    colors.append(all_features[fi, :_N_COLOR])

        if len(colors) < 4:
            return False

        X = np.asarray(colors)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self._km = KMeans(n_clusters=2, n_init=25, random_state=42, max_iter=500)
        self._km.fit(Xs)
        self._centers = self._scaler.inverse_transform(self._km.cluster_centers_)

        for i, name in enumerate(("Team A", "Team B")):
            logger.info("k-Means %s → Trikot: %s  (centroid white=%.2f dark=%.2f colorful=%.2f)",
                        name, _jersey_color_name(self._centers[i]),
                        float(self._centers[i][_IDX_WHITE]),
                        float(self._centers[i][_IDX_DARK]),
                        float(self._centers[i][_IDX_COLOR]))
        return True

    def _detect_refs(
        self, frame_data: Dict, all_features: np.ndarray, exclude_tids: Set[int]
    ) -> Set[int]:
        """
        Flag referee tracks as colour outliers.

        Per detection: compute min(dist_to_A, dist_to_B) in scaled colour space.
        Threshold = median + REF_MAD_K * MAD.
        A track is a referee if ≥ REF_TRACK_FRAC of its detections exceed threshold.
        """
        assert self._km is not None and self._scaler is not None

        # Collect (feature_idx, min_dist) for all non-excluded detections
        fi_list:   List[int]   = []
        dist_list: List[float] = []

        for fdata in frame_data.values():
            for det in fdata["detections"]:
                if det.get("track_id", -1) in exclude_tids:
                    continue
                fi = det.get("feature_idx", -1)
                if fi < 0 or fi >= len(all_features):
                    continue
                Xs = self._scaler.transform(all_features[fi, :_N_COLOR].reshape(1, -1))
                dists = self._km.transform(Xs)[0]
                fi_list.append(fi)
                dist_list.append(float(np.min(dists)))

        if not dist_list:
            return set()

        arr      = np.asarray(dist_list)
        median_d = float(np.median(arr))
        mad_d    = float(np.median(np.abs(arr - median_d))) + 1e-8
        threshold = median_d + self.REF_MAD_K * mad_d
        logger.info("Schiedsrichter Outlier-Schwelle: %.3f  (median=%.3f  MAD=%.3f)",
                    threshold, median_d, mad_d)

        outlier_fi = {fi for fi, d in zip(fi_list, dist_list) if d > threshold}

        # Per-track: count what fraction of detections are outliers
        track_total: Dict[int, int] = {}
        track_out:   Dict[int, int] = {}
        for fdata in frame_data.values():
            for det in fdata["detections"]:
                tid = det.get("track_id", -1)
                if tid < 0 or tid in exclude_tids:
                    continue
                fi = det.get("feature_idx", -1)
                if fi < 0:
                    continue
                track_total[tid] = track_total.get(tid, 0) + 1
                if fi in outlier_fi:
                    track_out[tid] = track_out.get(tid, 0) + 1

        ref_tids: Set[int] = set()
        for tid, total in track_total.items():
            n_out = track_out.get(tid, 0)
            frac  = n_out / max(total, 1)
            if total >= self.MIN_TRACK_OBS and frac >= self.REF_TRACK_FRAC:
                ref_tids.add(tid)
                logger.info("Schiedsrichter-Track %d: %d/%d Detektionen Farbausreißer (%.0f%%)",
                            tid, n_out, total, frac * 100)

        return ref_tids

    def _write_roles(
        self,
        frame_data:   Dict,
        all_features: np.ndarray,
        gk_tids:      Set[int],
        ref_tids:     Set[int],
        labels:       np.ndarray,
    ) -> np.ndarray:
        """Write role strings and integer labels to every detection."""
        for fdata in frame_data.values():
            for det in fdata["detections"]:
                tid = det.get("track_id", -1)
                fi  = det.get("feature_idx", -1)

                if tid in gk_tids:
                    role = "Torwart"
                elif tid in ref_tids:
                    role = "Schiedsrichter"
                elif (self._km is not None and self._scaler is not None
                      and 0 <= fi < len(all_features)):
                    Xs  = self._scaler.transform(
                        all_features[fi, :_N_COLOR].reshape(1, -1))
                    km_lbl = int(self._km.predict(Xs)[0])
                    role = "Team A" if km_lbl == 0 else "Team B"
                else:
                    role = "Sonstige"

                det["hdbscan_role"] = role
                det["label"]        = _LABEL.get(role, _LABEL["Sonstige"])
                det["probability"]  = 1.0
                det["kmeans_label"] = -1

                int_lbl = det["label"]
                if 0 <= fi < len(labels):
                    labels[fi] = int_lbl

        return labels
