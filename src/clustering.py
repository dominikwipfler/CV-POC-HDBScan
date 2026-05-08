"""
Clustering engine: HDBSCAN (primary) + K-Means (comparison).

Design decisions
----------------
- Features are StandardScaler-normalised before clustering.
- UMAP reduces the 19-dim feature space to 10D before HDBSCAN.
- HDBSCAN runs with cluster_selection_epsilon > 0 to merge nearby micro-clusters
  (important for achromatic jerseys like white that spread in feature space).
- Noise points are assigned to the nearest centroid in UMAP space.
- Role assignment is colour-aware:
    1. Darkest cluster (dark_frac) → Schiedsrichter
    2. Two largest remaining    → Team A, Team B
    3. Smaller remaining        → Torwart, Weitere
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, List

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Feature indices (must match feature_extractor.py)
# Colour features — first 19 dims
_IDX_WHITE     = 6   # white_frac
_IDX_DARK      = 7   # dark_frac
_IDX_COLORFUL  = 8   # colorful_frac
_IDX_HUE_START = 9   # first of 8 hue-histogram bins (OpenCV hue 0–180)
_IDX_MEAN_V    = 17  # mean HSV brightness
# Position features — dims 19-20 (added by feature_extractor)
_IDX_POS_X     = 19  # normalised bbox-centre x  (0 = left, 1 = right)
_IDX_POS_Y     = 20  # normalised bbox-centre y  (0 = top,  1 = bottom)

# German colour names for the 8 hue bins (OpenCV hue 0–180, 22.5° per bin)
_HUE_NAMES = ["Rot", "Orange-Rot", "Orange", "Gelb",
               "Grün", "Türkis", "Blau", "Lila"]

# Fallback role names ordered by cluster size (used only by map_roles())
_ROLE_NAMES = ["Team A", "Team B", "Schiedsrichter", "Torwart A", "Torwart B", "Sonstige"]


def _infer_jersey_color(centroid: np.ndarray) -> str:
    """Return a human-readable German colour name for a cluster centroid."""
    white_frac = float(centroid[_IDX_WHITE])
    dark_frac  = float(centroid[_IDX_DARK])
    colorful   = float(centroid[_IDX_COLORFUL])

    if white_frac > 0.45:
        return "Weiß"
    if dark_frac > 0.50 and colorful < 0.20:
        return "Dunkel"
    if colorful < 0.15:
        return "Grau"

    hue_hist = centroid[_IDX_HUE_START: _IDX_HUE_START + 8]
    return _HUE_NAMES[int(np.argmax(hue_hist))]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ClusterResult:
    def __init__(self, labels: np.ndarray, probabilities: np.ndarray,
                 method: str, n_clusters: int, n_noise: int,
                 silhouette: Optional[float] = None,
                 n_reassigned: int = 0):
        self.labels = labels
        self.probabilities = probabilities
        self.method = method
        self.n_clusters = n_clusters
        self.n_noise = n_noise
        self.n_reassigned = n_reassigned
        self.silhouette = silhouette
        self.role_map: Dict[int, str] = {}
        self.team_colors: Dict[str, str] = {}  # e.g. {"Team A": "Grün", "Team B": "Orange"}

    def map_roles(self) -> Dict[int, str]:
        """Fallback: assign roles by descending cluster size."""
        unique = sorted(l for l in set(self.labels) if l != -1)
        counts = {l: int(np.sum(self.labels == l)) for l in unique}
        sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.role_map = {-1: "Noise / Unklar"}
        for i, (lbl, _) in enumerate(sorted_pairs):
            self.role_map[lbl] = _ROLE_NAMES[i] if i < len(_ROLE_NAMES) else f"Cluster {lbl}"
        return self.role_map

    def map_roles_smart(self, raw_features: np.ndarray) -> Dict[int, str]:
        """
        Role assignment with at most 5 named groups.

        1. Two LARGEST clusters → Team A, Team B (colour-agnostic).
           Jersey colour names are inferred and stored in self.team_colors.
        2. Cluster with max combined L2-distance from Team A and Team B
           centroids → Schiedsrichter (works for orange, red, any colour).
        3. Next two by size → Torwart A, Torwart B.
        4. All others → Sonstige.
        """
        unique: List[int] = sorted(l for l in set(self.labels) if l != -1)
        self.role_map   = {-1: "Noise / Unklar"}
        self.team_colors = {}

        if not unique:
            return self.role_map

        if raw_features.shape[1] <= max(_IDX_DARK, _IDX_COLORFUL):
            return self.map_roles()

        centroids = {l: raw_features[self.labels == l].mean(axis=0) for l in unique}
        counts    = {l: int(np.sum(self.labels == l)) for l in unique}

        # Step 1 — Teams: two largest clusters
        by_size   = sorted(unique, key=lambda l: counts[l], reverse=True)
        remaining = list(by_size)
        team_lbls: List[int] = []
        for team_name in ("Team A", "Team B"):
            if not remaining:
                break
            lbl = remaining.pop(0)
            color = _infer_jersey_color(centroids[lbl])
            display_name = color          # e.g. "Grün", "Orange", "Weiß"
            self.role_map[lbl]         = display_name
            self.team_colors[team_name] = color
            team_lbls.append(lbl)
            logger.debug("%s (%s) ← Cluster %d  size=%d",
                         team_name, color, lbl, counts[lbl])

        # Step 2 — Schiedsrichter: pick the cluster that is most "unlike both
        # teams" weighted by its smallness.  Referees are always few people, so
        # among similarly-distant clusters we prefer the smaller one.
        #
        # score = max(dist_to_A, dist_to_B) / sqrt(size)
        #   • max(dist) rewards being very different from at least ONE team
        #     (works even when referee orange ≈ team orange)
        #   • /sqrt(size) penalises large clusters so we don't mislabel a
        #     big field-player cluster as Schiri
        if remaining:
            team_centroids = np.array([centroids[l] for l in team_lbls])
            def _schiri_score(lbl: int) -> float:
                c = centroids[lbl][:19].reshape(1, -1)   # colour dims only
                tc = team_centroids[:, :19]
                dists = pairwise_distances(c, tc, metric="euclidean")[0]
                return float(dists.max()) / float(counts[lbl] ** 0.5 + 1e-8)
            best = max(remaining, key=_schiri_score)
            self.role_map[best] = "Schiedsrichter"
            remaining.remove(best)
            logger.debug("Schiedsrichter ← Cluster %d  score=%.3f  color=%s",
                         best, _schiri_score(best),
                         _infer_jersey_color(centroids[best]))

        # Step 3 — Torwart A / Torwart B: prefer clusters whose mean x-position
        # is closest to a goal line (x ≈ 0 or x ≈ 1).  Goalkeepers rarely
        # leave their goal area, so their cluster centroid x is extreme.
        # Falls back to size order if position features are unavailable.
        if remaining and raw_features.shape[1] > _IDX_POS_X:
            remaining = sorted(
                remaining,
                key=lambda l: float(np.minimum(
                    centroids[l][_IDX_POS_X],
                    1.0 - centroids[l][_IDX_POS_X]
                ))
            )
        for tw_name in ("Torwart A", "Torwart B"):
            if not remaining:
                break
            lbl = remaining.pop(0)
            self.role_map[lbl] = tw_name
            logger.debug("%s ← Cluster %d  mean_x=%.2f",
                         tw_name, lbl,
                         centroids[lbl][_IDX_POS_X]
                         if raw_features.shape[1] > _IDX_POS_X else float("nan"))

        # Step 4 — Everything else → Sonstige
        for lbl in remaining:
            self.role_map[lbl] = "Sonstige"

        logger.info("Rollenzuweisung: %s", self.role_map)
        logger.info("Trikotfarben:    %s", self.team_colors)
        return self.role_map

    def get_role(self, label: int) -> str:
        return self.role_map.get(label, f"Cluster {label}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ClusteringEngine:
    def __init__(self, config):
        self._min_cluster_size = config.hdbscan_min_cluster_size
        self._min_samples      = config.hdbscan_min_samples
        self._kmeans_k         = config.kmeans_k
        self._scaler           = StandardScaler()
        self._features_scaled: Optional[np.ndarray] = None
        self._umap_2d:         Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray,
            raw_features: Optional[np.ndarray] = None
            ) -> Tuple[ClusterResult, Optional[ClusterResult]]:
        _raw = raw_features if raw_features is not None else features
        logger.info("Clustering von %d Feature-Vektoren …", len(features))

        # Scale colour features (first 19) and position (last 2) separately.
        # Position is kept raw (already in [0,1]) so its natural std (~0.25)
        # gives it roughly 25 % weight relative to the unit-scaled colour dims.
        n_color = min(19, features.shape[1])
        color_scaled = self._scaler.fit_transform(features[:, :n_color])
        if features.shape[1] > n_color:
            self._features_scaled = np.concatenate(
                [color_scaled, features[:, n_color:]], axis=1)
        else:
            self._features_scaled = color_scaled

        # UMAP 2D for scatter-plot
        if _UMAP_AVAILABLE:
            logger.info("UMAP 2D für Visualisierung …")
            umap_vis = UMAP(n_components=2,
                            n_neighbors=min(30, max(5, len(features) // 20)),
                            min_dist=0.3, random_state=42, metric="euclidean")
            self._umap_2d = umap_vis.fit_transform(self._features_scaled)

        # Auto-scale min_cluster_size to dataset size.
        # Small datasets  → may reduce to avoid requiring impossibly large clusters.
        # Large datasets  → scale UP so we don't get dozens of micro-clusters.
        # Target: each cluster must hold at least ~1.5 % of all samples.
        auto_cs = max(self._min_cluster_size, int(len(features) * 0.015))
        min_cs  = min(auto_cs, max(2, len(features) // 8))
        if min_cs != self._min_cluster_size:
            logger.info("min_cluster_size angepasst: %d → %d  (n=%d)",
                        self._min_cluster_size, min_cs, len(features))

        hdb = self._fit_hdbscan(self._features_scaled, min_cs)
        hdb.map_roles_smart(_raw)

        km = self._fit_kmeans(self._features_scaled)
        if km:
            km.map_roles_smart(_raw)

        return hdb, km

    def get_2d_representation(self, features: np.ndarray) -> np.ndarray:
        if self._umap_2d is not None:
            return self._umap_2d
        scaled = self._features_scaled if self._features_scaled is not None \
            else self._scaler.transform(features)
        return PCA(n_components=2, random_state=42).fit_transform(scaled)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit_hdbscan(self, scaled: np.ndarray, min_cs: int) -> ClusterResult:

        # ── UMAP 10D embedding ──────────────────────────────────────────
        if _UMAP_AVAILABLE and scaled.shape[0] >= 100:
            n_comp = min(10, scaled.shape[1])
            # High n_neighbors (50-100) gives a global view: similar-coloured
            # players from different frames/zones converge to the same region.
            n_nb   = max(15, min(100, scaled.shape[0] // 20))
            logger.info("UMAP %dD → %dD  n_neighbors=%d …",
                        scaled.shape[1], n_comp, n_nb)
            reducer = UMAP(n_components=n_comp, n_neighbors=n_nb,
                           min_dist=0.0, random_state=42, metric="euclidean")
            umap_data = reducer.fit_transform(scaled)
            logger.info("UMAP fertig")
        else:
            umap_data = scaled

        # ── HDBSCAN ─────────────────────────────────────────────────────
        clusterer = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=self._min_samples,
            cluster_selection_epsilon=0.0,
            metric="euclidean",
            cluster_selection_method="eom",
            store_centers="centroid",
            copy=True,
        )
        labels = clusterer.fit_predict(umap_data)
        probs  = clusterer.probabilities_.copy()

        unique_real  = [l for l in set(labels) if l != -1]
        n_clusters   = len(unique_real)
        n_noise_orig = int(np.sum(labels == -1))

        logger.info("HDBSCAN → %d Cluster, %d Noise (vor Neuzuweisung)",
                    n_clusters, n_noise_orig)

        # ── Noise reassignment ──────────────────────────────────────────
        # Noise points are reassigned to the nearest cluster centroid in
        # UMAP space.  Probability reflects distance to that centroid.
        n_reassigned = 0
        if n_noise_orig > 0 and n_clusters >= 1:
            centroids = np.array([umap_data[labels == l].mean(axis=0)
                                  for l in unique_real])
            noise_idx = np.where(labels == -1)[0]
            dists     = pairwise_distances(umap_data[noise_idx], centroids,
                                           metric="euclidean")
            nearest   = np.argmin(dists, axis=1)

            labels = labels.copy()
            labels[noise_idx] = np.array(unique_real)[nearest]

            min_dists   = dists[np.arange(len(noise_idx)), nearest]
            ref_radius  = np.array([umap_data[clusterer.labels_ == l].std() + 1e-8
                                    for l in unique_real])[nearest] * 2
            probs[noise_idx] = np.clip(1.0 - min_dists / ref_radius, 0.05, 1.0)

            n_reassigned = n_noise_orig
            logger.info("%d Noise-Punkte dem nächsten Cluster zugewiesen",
                        n_reassigned)

        n_noise_final = int(np.sum(labels == -1))

        sil = None
        if n_clusters >= 2:
            try:
                non_noise = labels != -1
                sil = float(silhouette_score(umap_data[non_noise],
                                             labels[non_noise]))
            except Exception:
                pass

        tag = "UMAP+HDBSCAN" if _UMAP_AVAILABLE else "HDBSCAN"
        logger.info("%s → %d Cluster, Silhouette=%s, %d neu zugewiesen",
                    tag, n_clusters,
                    f"{sil:.3f}" if sil is not None else "n/a",
                    n_reassigned)

        return ClusterResult(labels, probs, tag, n_clusters,
                             n_noise_final, sil, n_reassigned)

    def _fit_kmeans(self, scaled: np.ndarray) -> Optional[ClusterResult]:
        try:
            km     = KMeans(n_clusters=self._kmeans_k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            sil    = None
            if len(set(labels)) >= 2:
                sil = float(silhouette_score(scaled, labels))
            probs = np.ones(len(labels), dtype=np.float32)
            logger.info("K-Means (k=%d) → Silhouette=%s",
                        self._kmeans_k,
                        f"{sil:.3f}" if sil is not None else "n/a")
            return ClusterResult(labels, probs,
                                 f"K-Means (k={self._kmeans_k})",
                                 self._kmeans_k, 0, sil)
        except Exception as exc:
            logger.warning("K-Means fehlgeschlagen: %s", exc)
            return None
