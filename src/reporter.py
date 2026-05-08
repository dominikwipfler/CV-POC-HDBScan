"""
Report generation: JSON, CSV, and console summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, config):
        self._output_dir = Path(config.output_dir)

    def save_reports(self, frame_data: Dict, hdb_result, kmeans_result,
                     output_dir: Path) -> None:
        records = self._build_records(frame_data, hdb_result, kmeans_result)

        # --- JSON (full data including raw features) ---
        payload = {
            "metadata": {
                "total_samples": len(records),
                "frames_processed": len(frame_data),
                "hdbscan": {
                    "n_clusters": hdb_result.n_clusters,
                    "n_noise": hdb_result.n_noise,
                    "silhouette": hdb_result.silhouette,
                    "role_map": {str(k): v for k, v in hdb_result.role_map.items()},
                },
                "kmeans": {
                    "n_clusters": kmeans_result.n_clusters if kmeans_result else None,
                    "silhouette": kmeans_result.silhouette if kmeans_result else None,
                    "role_map": (
                        {str(k): v for k, v in kmeans_result.role_map.items()}
                        if kmeans_result else None
                    ),
                },
            },
            "detections": records,
        }
        json_path = output_dir / "clustering_results.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.info("JSON-Bericht gespeichert: %s", json_path)

        # --- CSV (no raw features — easy to open in Excel) ---
        df = pd.DataFrame(records)
        csv_cols = [c for c in df.columns if c != "features"]
        df[csv_cols].to_csv(output_dir / "clustering_results.csv", index=False)
        logger.info("CSV-Bericht gespeichert: %s", output_dir / "clustering_results.csv")

    def print_summary(self, hdb_result, kmeans_result,
                      total_samples: int) -> None:
        sep = "=" * 62
        print(f"\n{sep}")
        print("  HDBSCAN Handball-Analyse  —  Zusammenfassung")
        print(sep)
        print(f"\n  Verarbeitete Spieler-Samples : {total_samples}")

        # HDBSCAN block
        print(f"\n  ── HDBSCAN ──────────────────────────────────────────")
        print(f"  Gefundene Cluster    : {hdb_result.n_clusters}")
        noise_pct = hdb_result.n_noise / max(total_samples, 1) * 100
        print(f"  Noise / Ausreißer    : {hdb_result.n_noise}  ({noise_pct:.1f} %)")
        print(f"  Stabile Samples      : {total_samples - hdb_result.n_noise}")
        if hdb_result.silhouette is not None:
            print(f"  Silhouette-Score     : {hdb_result.silhouette:.3f}")

        print(f"\n  Cluster-Zuordnung (HDBSCAN):")
        for lbl, role in sorted(hdb_result.role_map.items()):
            n = int(np.sum(hdb_result.labels == lbl))
            bar = "█" * min(n // max(total_samples // 40, 1), 32)
            print(f"    {role:<32} {n:5d}  {bar}")

        # K-Means block
        if kmeans_result:
            print(f"\n  ── K-Means (k={kmeans_result.n_clusters}) ──────────────────────────────")
            if kmeans_result.silhouette is not None:
                print(f"  Silhouette-Score     : {kmeans_result.silhouette:.3f}")
            for lbl, role in sorted(kmeans_result.role_map.items()):
                if lbl == -1:
                    continue
                n = int(np.sum(kmeans_result.labels == lbl))
                bar = "█" * min(n // max(total_samples // 40, 1), 32)
                print(f"    {role:<32} {n:5d}  {bar}")

        # Evaluation verdict
        print(f"\n  ── Bewertung ────────────────────────────────────────")
        nc = hdb_result.n_clusters
        if nc == 2:
            print("  ✓ HDBSCAN hat genau 2 Teams gefunden – ideal!")
        elif nc > 2:
            print(f"  ~ {nc} Cluster gefunden. Torwart/Schiri evtl. als eigene Gruppe.")
        elif nc == 1:
            print("  ! Nur 1 Cluster. Teamfarben möglicherweise zu ähnlich.")
        else:
            print("  ! Keine stabilen Cluster. min_cluster_size reduzieren.")

        if noise_pct > 35:
            print("  ! Hohe Noise-Rate (>35 %). --min-cluster-size verringern oder")
            print("    --frame-skip erhöhen für mehr Samples.")

        if (kmeans_result and hdb_result.silhouette is not None
                and kmeans_result.silhouette is not None):
            winner = "HDBSCAN" if hdb_result.silhouette >= kmeans_result.silhouette \
                else "K-Means"
            print(f"  → Besserer Silhouette-Score: {winner}")

        print(f"\n{sep}\n")

    # ------------------------------------------------------------------

    @staticmethod
    def _build_records(frame_data: Dict, hdb_result, kmeans_result) -> List[dict]:
        records = []
        for frame_idx, fdata in frame_data.items():
            for det in fdata.get("detections", []):
                fi = det.get("feature_idx", -1)
                h_label = int(hdb_result.labels[fi]) if fi >= 0 else -1
                h_prob = float(hdb_result.probabilities[fi]) if fi >= 0 else 0.0
                h_role = hdb_result.role_map.get(h_label, "Unbekannt")

                km_label: Optional[int] = None
                km_role: Optional[str] = None
                if kmeans_result and fi >= 0:
                    km_label = int(kmeans_result.labels[fi])
                    km_role = kmeans_result.role_map.get(km_label, "Unbekannt")

                records.append({
                    "frame": int(frame_idx),
                    "detection_index": det.get("det_idx", 0),
                    "track_id": det.get("track_id", -1),
                    "bbox": list(det["bbox"]),
                    "detection_confidence": round(det.get("confidence", 1.0), 3),
                    "feature_index": fi,
                    "hdbscan_label": h_label,
                    "hdbscan_role": h_role,
                    "hdbscan_probability": round(h_prob, 3),
                    "is_noise": h_label == -1,
                    "kmeans_label": km_label,
                    "kmeans_role": km_role,
                    "features": [round(float(f), 4) for f in det.get("features", [])],
                })
        return records
