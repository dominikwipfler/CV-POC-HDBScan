# HDBSCAN Handball Team Clustering — POC

Proof of Concept zur automatischen Teameinteilung im Handball anhand von Trikotfarben,
basierend auf HDBSCAN und OpenCV.

---

## Was ist HDBSCAN?

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
ist ein Clustering-Algorithmus, der Datenpunkte nach ihrer **lokalen Dichte** gruppiert.

Vereinfacht erklärt: HDBSCAN sucht in einem Datenraum nach „Verdichtungen" — Bereichen,
in denen viele ähnliche Punkte dicht beieinander liegen. Punkte, die zu keiner solchen
Verdichtung gehören, werden als **Noise** (Ausreißer) markiert statt in einen Cluster
gezwungen.

Drei wesentliche Eigenschaften machen HDBSCAN besonders:

| Eigenschaft | Bedeutung |
|---|---|
| Keine feste Cluster-Anzahl nötig | Der Algorithmus entscheidet selbst, wie viele Gruppen sinnvoll sind |
| Noise-Handling | Unklare Punkte werden als Ausreißer markiert, nicht falsch zugeordnet |
| Hierarchisch | Intern wird eine Cluster-Hierarchie gebaut, aus der die stabilsten Gruppen extrahiert werden |

---

## Warum HDBSCAN für die Handball-Trikot-Analyse?

Im Handball-Kontext gibt es **kein garantiert festes Schema**: Neben den zwei Teams
spielen Torwarte (meist Sonderfarbe), Schiedsrichter und evtl. weitere Personen im
Bild. K-Means würde hier erzwingen, genau k Gruppen zu finden — auch wenn das
inhaltlich falsch ist.

HDBSCAN ist hier besser weil:

- Die Anzahl sinnvoller Farbcluster variiert (2 Teams + optionale Sonderfarben)
- Partiell verdeckte Spieler oder Personen am Rand sollen als Noise markiert werden,
  nicht falsch einem Team zugeordnet
- Trikotfarben bilden echte Dichteinseln im Farbraum — genau das, was HDBSCAN sucht

**Einschränkung:** Haben beide Teams sehr ähnliche Farben (z. B. zwei Blautöne),
kann HDBSCAN die Cluster zusammenfassen. In diesem Fall gibt der Silhouette-Score
Aufschluss darüber, wie trennbar die Farben tatsächlich sind.

---

## HDBSCAN vs. K-Means in diesem Kontext

| Kriterium | HDBSCAN | K-Means |
|---|---|---|
| Cluster-Anzahl | Automatisch bestimmt | Muss vorab festgelegt werden (k=2) |
| Noise / Ausreißer | Explizit als -1 markiert | Keine Noise-Erkennung — jeder Punkt wird zugeordnet |
| Nicht-runde Cluster | Unterstützt | Nur kugelförmige Cluster zuverlässig |
| Robustheit bei Farbüberlappung | Besser (Noise-Puffer) | Schlechter (schneidet an der falschen Stelle) |
| Reproduzierbarkeit | Deterministisch | Hängt von zufälliger Initialisierung ab (n_init hilft) |
| Rechenzeit | Langsamer (O(n log n)) | Schneller für kleine k |
| Empfehlung | POC / Exploration | Produktion wenn k bekannt und stabil |

---

## Weitere HDBSCAN-Einsatzmöglichkeiten im Handball

1. **Formations-Clustering**
   Spielerpositionen (x, y auf dem Feld) über mehrere Angriffe → HDBSCAN erkennt
   typische Positionsmuster (3-2-1, 4-2, Kreis-läufer-Zone) ohne Vorgabe der
   Formations-Anzahl.

2. **Wurfzonen-Analyse**
   Abschlusspositionen oder Ballflugbahn-Startpunkte werden geclustert → automatische
   Erkennung bevorzugter Wurfpositionen pro Spieler oder Team.

3. **Laufweg-Typisierung**
   Tracking-Trajektorien aller Spieler über ein Spiel → HDBSCAN gruppiert typische
   Laufmuster (Außenspieler-Diagonale, Kreisläufer-Kurzbewegungen) für taktische
   Nachanalyse.

---

## Installation

### Voraussetzungen

- Python 3.9 oder höher
- pip

### Schritte

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# YOLO-Modell wird beim ersten Start automatisch heruntergeladen (~6 MB für yolov8n)
# Internetverbindung beim Erststart erforderlich.
```

> **Offline-Modus**: Ohne Internetverbindung kann `--no-yolo` übergeben werden.
> Der MOG2-Fallback-Detektor funktioniert ohne Modelldatei.

---

## Verwendung

### Standardstart

```bash
python main.py --video C:/Users/domwipfler/Downloads/test30sec.mp4
```

### Alle Optionen

```
--video            Pfad zum Eingabe-Video (Pflicht)
--output           Ausgabeverzeichnis [Standard: outputs]
--frame-skip N     Jeden N-ten Frame verarbeiten [Standard: 5]
--max-frames N     Nur N Frames verarbeiten (schneller Testlauf)
--min-cluster-size Mindestgröße eines HDBSCAN-Clusters [Standard: 5]
--min-samples      HDBSCAN min_samples [Standard: 3]
--kmeans-k         K-Means Cluster-Anzahl [Standard: 2]
--yolo-model       yolov8n.pt | yolov8s.pt | yolov8m.pt
--yolo-confidence  Konfidenzschwelle für YOLO [Standard: 0.5]
--no-yolo          MOG2-Fallback statt YOLO
--no-debug         Keine Debug-Bilder speichern
--verbose          Ausführliches Logging
```

### Beispiele

```bash
# Schneller Testlauf (nur erste 150 Frames)
python main.py --video input/test30sec.mp4 --max-frames 150

# Feinere Erkennung bei vielen Spielern
python main.py --video input/test30sec.mp4 --frame-skip 3 --min-cluster-size 8

# Aggressiveres Clustering bei kleinen Teams
python main.py --video input/test30sec.mp4 --min-cluster-size 3 --min-samples 2

# Ohne Internet / YOLO
python main.py --video input/test30sec.mp4 --no-yolo
```

---

## Ausgaben

Nach der Verarbeitung befinden sich in `outputs/`:

| Datei / Ordner | Inhalt |
|---|---|
| `output_hdbscan.mp4` | Vollständiges Video mit Bounding-Boxes und Cluster-Labels |
| `output_comparison.mp4` | Side-by-Side: HDBSCAN (links) vs. K-Means (rechts) |
| `cluster_scatter.png` | 2D PCA-Visualisierung der Feature-Punkte mit Clusterfarben |
| `clustering_results.json` | Vollständige Rohdaten: Frame, Bbox, Features, Labels |
| `clustering_results.csv` | Tabellarische Übersicht ohne rohe Features (Excel-kompatibel) |
| `debug_frames/` | Einzelne Frame-Snapshots: Original, Detektionen, Torso-ROIs |

### Farb-Legende im Video

| Farbe | Bedeutung |
|---|---|
| Grün | Team A (größter Cluster) |
| Orange | Team B (zweitgrößter Cluster) |
| Cyan | Team C / Torwart / Schiedsrichter |
| Dunkelrot | Noise / Unklar — kein stabiler Cluster |

---

## Technische Architektur

```
main.py                 → CLI-Einstiegspunkt, Argument-Parsing
src/
  config.py             → Alle Parameter als Dataclass
  player_detector.py    → YOLOv8 Personenerkennung (+ MOG2-Fallback)
  feature_extractor.py  → Trikotfarb-Features: LAB-Statistiken + HSV-Hue-Histogramm
  clustering.py         → HDBSCAN + K-Means, Silhouette-Score, Rollen-Mapping
  visualizer.py         → OpenCV-Annotation, Legende, matplotlib Scatter-Plot
  reporter.py           → JSON, CSV, Konsolen-Zusammenfassung
  video_processor.py    → Zwei-Phasen-Pipeline (Extraktion → Rendering)
```

### Feature-Vektor (22 Dimensionen)

```
LAB-Kanal L: Mittelwert + Stddev   (2)
LAB-Kanal A: Mittelwert + Stddev   (2)
LAB-Kanal B: Mittelwert + Stddev   (2)
HSV-Hue-Histogramm: 16 Bins        (16)
                                 ------
                                    22
```

Warum LAB + HSV?
- **LAB** ist perceptuell gleichmäßig → Helligkeitsunterschiede zwischen Kamerapositionen
  verfälschen den Farbabstand weniger
- **HSV-Hue** gibt die dominante Trikotfarbe kompakt und beleuchtungsunabhängig wieder

---

## Grenzen dieses POCs

| Grenze | Ursache | Workaround |
|---|---|---|
| Ähnliche Teamfarben → 1 Cluster | HDBSCAN kann keine Gruppen trennen, die im Farbraum überlappen | Mehr Feature-Dimensionen (Textur, Logo-Erkennung) |
| Hallenlicht schwankt → instabile Features | Weißabgleich variiert zwischen Kamerapositionen | Normalisierung auf Hallenboden-Weiß oder Graukarte |
| MOG2-Fallback erkennt nur Bewegung | Statische Spieler werden nicht erkannt | Besser mit YOLO-Modell arbeiten |
| Kein Frame-übergreifendes Tracking | Spieler erhalten keine persistente ID | IoU-Tracker (SORT, ByteTrack) integrieren |
| 30-Sekunden-Clip = kleine Datenbasis | Dichte-Schätzung weniger robust | Längere Clips oder mehrere Clips kombinieren |

---

## Ideen für ein größeres Handball-Projekt

1. **Persistentes Spieler-Tracking**
   ByteTrack oder DeepSORT für stabile Spieler-IDs über den gesamten Spielverlauf.

2. **Robustere Features**
   Farbhistogramm + CNN-Feature (z. B. ResNet-Embedding des Torso-ROI) für
   verlässliche Trennung auch bei ähnlichen Teamfarben.

3. **Online-Clustering**
   HDBSCAN einmalig auf den ersten 300 Frames trainieren, dann neue Frames mit
   `hdbscan.approximate_predict()` einordnen — kein Neutraining nötig.

4. **Formations-Erkennung**
   Spielerpositionen aus Tracking-Daten → zweite HDBSCAN-Instanz auf Raumkoordinaten
   für automatische taktische Analyse.

5. **Echtzeit-Pipeline**
   OpenCV + Threading für live Videostream; YOLO auf GPU für >30 fps Processing.

---

## Wann ist HDBSCAN im Sportvideo gut?

**Gut geeignet wenn:**
- Die Anzahl der Kategorien (Teams, Rollen) nicht vorab bekannt ist
- Ausreißer (verdeckte Spieler, Zuschauer im Bild) sauber ignoriert werden sollen
- Natürliche Cluster-Strukturen im Farb- oder Positionsraum vorhanden sind

**Weniger geeignet wenn:**
- Genau k=2 Teams gesucht werden und beide sehr ähnliche Farben haben → K-Means mit
  k=2 ist stabiler und schneller
- Echtzeit-Anforderungen bestehen und die Datenbasis klein ist (<50 Samples)
- Das Ergebnis vollständig reproduzierbar und erklärbar sein muss (regulierte Umgebung)
