# HDBSCAN Handball Team Clustering — POC

Automatische Teameinteilung und taktische Analyse von Handball-Videos anhand von Trikotfarben. Spielererkennung via spezialisiertem `player_detection.pt` Modell, Rollenzuweisung (Team A/B, Schiedsrichter, Torwart) via HDBSCAN.

---

## Schnellstart

```powershell
# Empfohlener Startbefehl (2-Minuten-Clip, CPU)
.\.venv\Scripts\python.exe main.py `
  --video input/0efd265d_test2min.mp4 `
  --output outputs `
  --frame-skip 5 `
  --yolo-imgsz 640

# Schneller Smoke-Test (nur erste 300 Frames)
.\.venv\Scripts\python.exe main.py `
  --video input/0efd265d_test2min.mp4 `
  --max-frames 300 `
  --frame-skip 5

# Taktische Analyse auf bereits berechneten Ergebnissen (kein Video-Neuverarbeitung)
.\.venv\Scripts\python.exe run_tactical.py `
  --json outputs/0efd265d_test2min/clustering_results.json `
  --output outputs/0efd265d_test2min
```

---

## Alle Parameter

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--video` | — | Pfad zum Eingabe-Video (Pflicht) |
| `--output` | `outputs` | Ausgabeverzeichnis (pro Video ein Unterordner) |
| `--frame-skip N` | `5` | Jeden N-ten Frame verarbeiten |
| `--max-frames N` | alle | Abbruch nach N Frames (Schnelltest) |
| `--min-cluster-size N` | `50` | HDBSCAN min_cluster_size — kleiner = mehr Cluster |
| `--min-samples N` | `5` | HDBSCAN min_samples |
| `--kmeans-k N` | `5` | K-Means Vergleich: Anzahl Cluster |
| `--yolo-model` | `player_detection.pt` | Erkennungsmodell |
| `--yolo-confidence` | `0.3` | Konfidenz-Schwelle |
| `--yolo-imgsz` | `1280` | YOLO Input-Bildgröße (640 = schneller, 1280 = genauer) |
| `--device` | `cpu` | `cpu` oder `cuda` |
| `--no-preprocessing` | — | White Balance + CLAHE deaktivieren |
| `--no-roi` | — | Automatische Court-ROI-Erkennung deaktivieren |
| `--verbose` | — | Ausführliches Logging (zeigt Cluster-Tabelle) |

---

## Ausgaben

Alle Dateien landen in `outputs/<video-name>/`:

| Datei | Inhalt |
|---|---|
| `output_hdbscan.mp4` | Video mit Bounding-Boxes + Rollenbezeichnungen |
| `output_comparison.mp4` | Side-by-Side: HDBSCAN (links) vs. K-Means (rechts) |
| `cluster_scatter.png` | 2D UMAP-Scatter der Farbfeatures mit Clustereinfärbung |
| `clustering_results.json` | Vollständige Rohdaten: Frame, BBox, Features, Labels |
| `clustering_results.csv` | Tabellarische Übersicht (Excel-kompatibel) |
| `tactical_abwehr_*.png` | Abwehr-Formations-Häufigkeit pro Team |
| `tactical_angriff_*.png` | Angriffsmuster / Spielzüge pro Team |
| `tactical_shot_zones.png` | Aufenthalts-Heatmap im Angriff |
| `tactical_trajectories.png` | Laufwege + Bewegungsstatistiken nach Rolle |
| `debug_frames/` | Einzel-Frame-Snapshots mit Torso-ROI-Vorschau |

---

## Architektur

```
main.py                     CLI, Argument-Parsing
src/
  config.py                 Alle Parameter als Dataclass
  player_detector.py        player_detection.pt + ByteTrack (auto class-detection)
  feature_extractor.py      21-dim Trikotfarb-Featurevektor
  clustering.py             UMAP + HDBSCAN + K-Means + Rollenzuweisung
  frame_preprocessor.py     White Balance + CLAHE pro Frame
  video_processor.py        Zwei-Pass-Pipeline (Extraktion → Rendering)
  tactical_analyzer.py      Formations-, Wurfzonen-, Laufweg-Analyse
  visualizer.py             OpenCV-Annotation + matplotlib Scatter-Plot
  reporter.py               JSON, CSV, Konsolenausgabe
run_tactical.py             Standalone taktische Analyse auf JSON-Ergebnissen
```

### Feature-Vektor (21 Dimensionen)

```
LAB Kanal L,A,B: Mittelwert + Stddev je   (6)
white_frac, dark_frac, colorful_frac       (3)
HSV-Hue-Histogramm (8 Bins, nur S > 60)   (8)
mean HSV-V, mean HSV-S                     (2)
BBox-Zentrum x_norm, y_norm               (2)  ← nur für Rollenzuweisung, nicht HDBSCAN
                                         -----
                                            21
```

**Warum nur Farbfeatures für HDBSCAN?** Position wird absichtlich aus dem HDBSCAN-Input ausgeschlossen — sonst würden Spieler gleicher Farbe an verschiedenen Feldpositionen in unterschiedliche Cluster fallen. Die x/y-Koordinaten werden nur nachgelagert für die Torwart-Erkennung genutzt.

### Rollenzuweisung (Reihenfolge)

```
1. Schiedsrichter  → Cluster mit orange Trikot (colorful + Hue-Bin 0-2)
                     Fallback: dunkelster Cluster
2. Torwart A/B    → Cluster mit mean_x < 0.20 oder > 0.80
                     (Torwarte verlassen das Torraum kaum)
                     Kein Fallback — oft nur eine Spielfeldhälfte sichtbar → 0 oder 1 Torwart
3. Team A, Team B  → die zwei größten verbleibenden Cluster
4. Sonstige        → alle weiteren Cluster
```

---

## Spielererkennung: player_detection.pt

Das Modell `player_detection.pt` (450 MB) erkennt 3 Klassen:

| Klasse | ID | Beschreibung |
|---|---|---|
| `goalkeeper` | 0 | Torwart |
| `player` | 1 | Feldspieler |
| `referee` | 2 | Schiedsrichter |

Die YOLO-Klasseninfos werden **nicht** für die HDBSCAN-Rollenzuweisung genutzt — HDBSCAN weist Rollen rein farb- und positionsbasiert zu. Die Klasseninfos stehen im `frame_data` JSON für spätere Auswertungen.

---

## Test-Protokoll

### Getestete Konfiguration

- Video: `input/0efd265d_test2min.mp4` (2 Minuten, Handball-Spiel)
- Gerät: CPU
- Python: 3.14, Windows 11

### Was funktioniert

| Feature | Status | Anmerkung |
|---|---|---|
| Spielererkennung mit player_detection.pt | OK | Alle 3 Klassen erkannt (goalkeeper/player/referee) |
| HDBSCAN 5 Cluster (Teams + Schiri + Torwart) | OK | min_cluster_size=50, color-only features |
| Rollenzuweisung Team A / Team B | OK | Zwei größte Cluster nach Schiri/Torwart-Entfernung |
| Rollenzuweisung Schiedsrichter | OK | Orange-Erkennung (Hue-Bin 0-2, colorful_frac > 0.20) |
| Rollenzuweisung Torwart | OK | mean_x < 0.20 oder > 0.80; 0 oder 1 Torwart möglich |
| HDBSCAN Visualisierungs-Video | OK | output_hdbscan.mp4 mit Farblegende |
| K-Means Vergleichsvideo | OK | output_comparison.mp4 |
| Cluster-Scatter-Plot | OK | 2D UMAP mit Clustereinfärbung |
| Formations-Analyse Abwehr | OK | tactical_abwehr_*.png pro Team |
| Angriffsmuster / Spielzüge | OK | tactical_angriff_*.png pro Team |
| Wurfzonen-Heatmap | OK | tactical_shot_zones.png |
| Laufwege Trajektorien | OK | tactical_trajectories.png |
| White Balance + CLAHE | OK | Verbessert Farbkonsistenz zwischen Frames |
| ROI-Kalibrierung | OK | Filtert Zuschauerbereich aus Detektionen heraus |

### Bekannte Einschränkungen

| Einschränkung | Ursache | Empfehlung |
|---|---|---|
| Fehlklassifikation bei sehr ähnlichen Trikotfarben | HDBSCAN kann Farb-Cluster zusammenlegen | Silhouette-Score im Log prüfen |
| Schiedsrichter nicht erkannt | Orange-Hue-Bin-Schwelle passt nicht | `--verbose` → Cluster-Tabelle prüfen, ggf. Schwellwert anpassen |
| Torwart nicht erkannt | Nur eine Spielfeldhälfte gefilmt → kein extremer x-Wert | Erwartet — kein Fehler |
| Taktische Analyse wenig Daten | Zu kurzes Video / zu viele Fehlklassifikationen | Längeres Video oder --frame-skip 3 |
| Zu viele Mikro-Cluster | min_cluster_size zu klein | `--min-cluster-size 100` oder höher testen |
| Zu wenige Cluster (nur 2) | min_cluster_size zu groß | `--min-cluster-size 30` testen |

### Nicht getestet / Offene Punkte

| Punkt | Status |
|---|---|
| CUDA-Beschleunigung (`--device cuda`) | Nicht getestet |
| Sehr kurze Clips (< 30 Sek.) | Nicht getestet — min_cluster_size ggf. auf 10-20 reduzieren |
| Videos mit sehr ähnlichen Teamfarben | Nicht getestet |
| MOG2-Fallback (ohne YOLO) | Nicht mehr empfohlen — player_detection.pt verwenden |

---

## Debugging

### Cluster-Tabelle im Log

Mit `--verbose` erscheint nach HDBSCAN eine Tabelle aller Cluster:

```
=== HDBSCAN: 5 Cluster gefunden ===
  Cluster  0 | n=2949 | Farbe=Blau       | colorful=0.62 | white=0.05 | dark=0.08 | mean_x=0.46 | orange=nein
  Cluster  1 | n=1929 | Farbe=Weiß       | colorful=0.06 | white=0.58 | dark=0.05 | mean_x=0.54 | orange=nein
  Cluster  2 | n=1036 | Farbe=Orange-Rot | colorful=0.48 | white=0.09 | dark=0.11 | mean_x=0.50 | orange=JA
  Cluster  3 | n= 473 | Farbe=Grün       | colorful=0.55 | white=0.06 | dark=0.09 | mean_x=0.08 | orange=nein
  Cluster  4 | n= 280 | Farbe=Rot        | colorful=0.51 | white=0.07 | dark=0.10 | mean_x=0.92 | orange=nein
```

**Schiedsrichter nicht erkannt?** → `orange=nein` obwohl Schiri-Cluster vorhanden → colorful_frac zu niedrig oder Hue-Bin > 2. In `src/clustering.py` den `_is_orange_jersey` Schwellwert anpassen.

**Nur 2 Cluster?** → `min_cluster_size` zu groß. Starte mit `--min-cluster-size 30`.

**Zu viele Cluster (>6)?** → `min_cluster_size` erhöhen, z.B. `--min-cluster-size 100`.

---

## HDBSCAN — Warum besser als K-Means?

| Kriterium | HDBSCAN | K-Means |
|---|---|---|
| Cluster-Anzahl | Automatisch | Vorab festlegen (k=5) |
| Noise / Ausreißer | Explizit markiert, dann re-assigniert | Keine Erkennung |
| Nicht-kugelförmige Cluster | Unterstützt | Nur kugelförmig |
| Silhouette (UMAP-Space) | 0.706 (bester Lauf) | 0.339 (Original-Space, nicht vergleichbar) |
| Rollenzuweisung | Farb+Positionsbasiert | Größenbasiert |

**Hinweis:** Die Silhouette-Scores sind nicht direkt vergleichbar — HDBSCAN wird im UMAP-Space berechnet, K-Means im Original-Feature-Space.
