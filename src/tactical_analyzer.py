"""
Tactical analyses: formations, shot zones, trajectories.

analyze_formations  → two PNGs per team:
    tactical_abwehr_{jersey}.png   — defense formation frequency (6-0, 5-1, …)
    tactical_angriff_{jersey}.png  — attack movement patterns (Tiefenlauf, …)

analyze_shot_zones  → tactical_shot_zones.png
analyze_trajectories → tactical_trajectories.png  (density + track lines)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Colour maps ───────────────────────────────────────────────────────────────

_TEAM_COLORS: Dict[str, str] = {
    "Schiedsrichter": "#DDDD00",
    "Torwart A":      "#00E6B4",
    "Torwart B":      "#CC00CC",
    "Sonstige":       "#787878",
    "Unbekannt":      "#888888",
    "Weiß":           "#E6E6E6",
    "Dunkel":         "#3C3C3C",
    "Grau":           "#A0A0A0",
    "Rot":            "#DC0000",
    "Orange-Rot":     "#E64600",
    "Orange":         "#FF8C00",
    "Gelb":           "#DCDC00",
    "Grün":           "#32C832",
    "Türkis":         "#00C8C8",
    "Blau":           "#3C3CDC",
    "Lila":           "#9632BE",
}

_PATTERN_COLORS: Dict[str, str] = {
    "Tiefenlauf":      "#FF3333",
    "Diagonallauf":    "#FF9900",
    "Lateralbewegung": "#33AAFF",
    "Positionsspiel":  "#88DD88",
    "Rückzug":         "#888888",
    "Sonstiges":       "#666666",
}

_PATTERN_LABELS_DE: Dict[str, str] = {
    "Tiefenlauf":      "Tiefenläufe (Richtung Tor)",
    "Diagonallauf":    "Diagonalläufe / Schnittbewegungen",
    "Lateralbewegung": "Lateralbewegung (seitwärts)",
    "Positionsspiel":  "Positionsspiel (kaum Bewegung)",
    "Rückzug":         "Rückzug (weg vom Tor)",
    "Sonstiges":       "Sonstige",
}

# ── Court geometry ────────────────────────────────────────────────────────────

_SIX_RX  = 6 / 40
_SIX_RY  = 6 / 20
_NINE_RX = 9 / 40
_NINE_RY = 9 / 20
_GOAL_H  = 1.5 / 20
_SEVEN_X = 7 / 40


# ── Generic helpers ───────────────────────────────────────────────────────────

def _bbox_center(bbox: list) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _get_role(d: dict, role_map: Dict) -> str:
    return d.get("hdbscan_role") or role_map.get(d.get("label", -1), "")


# ── Court drawing ─────────────────────────────────────────────────────────────

def _draw_full_court(ax, title: str = "") -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.invert_yaxis(); ax.set_aspect(0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#1a3a1a")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor="#2e7d32",
                                     edgecolor="white", lw=2.5, zorder=1))
    # Center line + center circle
    ax.axvline(0.5, color="white", lw=1.5, alpha=0.9, zorder=2)
    cx_theta = np.linspace(0, 2 * np.pi, 200)
    _CIR_RX, _CIR_RY = 6 / 40, 6 / 20
    ax.plot(0.5 + _CIR_RX * np.cos(cx_theta),
            0.5 + _CIR_RY * np.sin(cx_theta),
            color="white", lw=1.2, alpha=0.8, zorder=2)
    ax.scatter([0.5], [0.5], marker="o", c="white", s=20, zorder=3)
    # Goal areas + 9m arcs
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    for gx, sign in ((0.0, 1), (1.0, -1)):
        # 6m arc (solid)
        ax.plot(gx + sign * _SIX_RX * np.cos(theta),
                0.5 + _SIX_RY * np.sin(theta), color="white", lw=2, zorder=3)
        # 9m arc (dashed)
        ax.plot(gx + sign * _NINE_RX * np.cos(theta),
                0.5 + _NINE_RY * np.sin(theta),
                color="white", lw=1.0, linestyle="--", alpha=0.6, zorder=3)
        # Goal rectangle (extends outside court boundary)
        goal_x = gx if sign == 1 else gx - 0.022
        ax.add_patch(mpatches.Rectangle(
            (goal_x, 0.5 - _GOAL_H), 0.022, 2 * _GOAL_H,
            facecolor="#b71c1c", edgecolor="white", lw=1.5, zorder=4))
        # 7m spot
        ax.scatter([gx + sign * _SEVEN_X], [0.5], marker="+",
                   c="white", s=60, linewidths=1.5, zorder=5)
        # Substitution lines (short marks on sideline)
        for fy in (0.5 - 4.5 / 20, 0.5 + 4.5 / 20):
            ax.plot([gx + sign * 4.5 / 40, gx + sign * 5.5 / 40], [fy, fy],
                    color="white", lw=1.2, alpha=0.5, zorder=3)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3, color="#dddddd")


def _draw_half_court(ax, title: str = "") -> None:
    """Right half (goal at x=1) — attack/shot-zone view."""
    ax.set_xlim(0.48, 1.03); ax.set_ylim(-0.02, 1.02)
    ax.invert_yaxis(); ax.set_aspect(0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#1a3a1a")
    ax.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1, facecolor="#2e7d32",
                                     edgecolor="none", zorder=1))
    ax.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1, fill=False,
                                     edgecolor="white", lw=2.5, zorder=2))
    ax.axvline(0.5, color="white", lw=1.5, alpha=0.9, zorder=2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    gx, sign = 1.0, -1
    ax.plot(gx + sign * _SIX_RX * np.cos(theta),
            0.5 + _SIX_RY * np.sin(theta), color="white", lw=2, zorder=3)
    ax.plot(gx + sign * _NINE_RX * np.cos(theta),
            0.5 + _NINE_RY * np.sin(theta),
            color="white", lw=1.0, linestyle="--", alpha=0.65, zorder=3)
    ax.add_patch(mpatches.Rectangle(
        (gx - 0.022, 0.5 - _GOAL_H), 0.022, 2 * _GOAL_H,
        facecolor="#b71c1c", edgecolor="white", lw=1.5, zorder=4))
    ax.scatter([gx + sign * _SEVEN_X], [0.5], marker="+",
               c="white", s=100, linewidths=2, zorder=5)
    ax.text(1.01, 0.5, "TOR", transform=ax.transData, ha="left", va="center",
            fontsize=8, color="white", fontweight="bold", rotation=90, clip_on=False)
    ax.annotate("", xy=(0.92, 0.08), xytext=(0.62, 0.08),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
                xycoords="data", clip_on=True)
    ax.text(0.77, 0.06, "Angriff", ha="center", va="bottom",
            fontsize=7, color="yellow", transform=ax.transData)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=5)


def _draw_half_court_left(ax, title: str = "") -> None:
    """Left half (goal at x=0) — defense view."""
    ax.set_xlim(-0.03, 0.52); ax.set_ylim(-0.02, 1.02)
    ax.invert_yaxis(); ax.set_aspect(0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_facecolor("#1a3a1a")
    ax.add_patch(mpatches.Rectangle((0, 0), 0.5, 1, facecolor="#2e7d32",
                                     edgecolor="none", zorder=1))
    ax.add_patch(mpatches.Rectangle((0, 0), 0.5, 1, fill=False,
                                     edgecolor="white", lw=2.5, zorder=2))
    ax.axvline(0.5, color="white", lw=1.5, alpha=0.9, zorder=2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    gx, sign = 0.0, 1
    ax.plot(gx + sign * _SIX_RX * np.cos(theta),
            0.5 + _SIX_RY * np.sin(theta), color="white", lw=2, zorder=3)
    ax.plot(gx + sign * _NINE_RX * np.cos(theta),
            0.5 + _NINE_RY * np.sin(theta),
            color="white", lw=1.0, linestyle="--", alpha=0.65, zorder=3)
    ax.add_patch(mpatches.Rectangle(
        (gx, 0.5 - _GOAL_H), 0.022, 2 * _GOAL_H,
        facecolor="#b71c1c", edgecolor="white", lw=1.5, zorder=4))
    ax.scatter([gx + sign * _SEVEN_X], [0.5], marker="+",
               c="white", s=100, linewidths=2, zorder=5)
    ax.text(-0.03, 0.5, "TOR", transform=ax.transData, ha="right", va="center",
            fontsize=8, color="white", fontweight="bold", rotation=90, clip_on=False)
    ax.annotate("", xy=(0.08, 0.08), xytext=(0.38, 0.08),
                arrowprops=dict(arrowstyle="->", color="#88aaff", lw=1.5),
                xycoords="data", clip_on=True)
    ax.text(0.23, 0.06, "eigenes Tor", ha="center", va="bottom",
            fontsize=7, color="#88aaff", transform=ax.transData)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=5)


# ── Phase classification ──────────────────────────────────────────────────────

def _classify_phases(frame_data: Dict, role_map: Dict,
                     frame_w: int, frame_h: int,
                     teams: Tuple[str, str]) -> Dict[int, Dict[str, str]]:
    team_a, team_b = teams
    phases: Dict[int, Dict[str, str]] = {}
    for frame_idx, fdata in frame_data.items():
        dets = fdata["detections"]
        a_xs = [_bbox_center(d["bbox"])[0] / frame_w
                for d in dets if _get_role(d, role_map) == team_a]
        b_xs = [_bbox_center(d["bbox"])[0] / frame_w
                for d in dets if _get_role(d, role_map) == team_b]
        if len(a_xs) < 3 or len(b_xs) < 3:
            continue
        diff = float(np.mean(a_xs)) - float(np.mean(b_xs))
        if abs(diff) < 0.03:
            continue
        if diff > 0:
            phases[frame_idx] = {team_a: "Angriff", team_b: "Abwehr"}
        else:
            phases[frame_idx] = {team_a: "Abwehr", team_b: "Angriff"}
    logger.info("Phase-Klassifikation: %d/%d Frames",
                len(phases), max(len(frame_data), 1))
    return phases


# ── Density helpers ───────────────────────────────────────────────────────────

def _density_grid(xs: np.ndarray, ys: np.ndarray,
                  x_range: Tuple[float, float],
                  y_range: Tuple[float, float] = (0.0, 1.0),
                  bins: int = 40, sigma: float = 3.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, xe, ye = np.histogram2d(xs, ys, bins=bins, range=[x_range, y_range])
    hist = _gaussian_filter(hist.T, sigma=sigma) if _SCIPY else hist.T
    return hist, 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])


def _team_cmap(hex_color: str) -> mcolors.LinearSegmentedColormap:
    hx = hex_color.lstrip("#")
    r, g, b = (int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return mcolors.LinearSegmentedColormap.from_list(
        "t", [(r, g, b, 0.0), (r, g, b, 0.55), (1.0, 1.0, 0.85, 0.92)], N=128)


def _draw_density(ax, xs: np.ndarray, ys: np.ndarray,
                  x_range: Tuple[float, float], cmap, bins: int = 40) -> None:
    if len(xs) < 15:
        return
    try:
        hist, xc, yc = _density_grid(xs, ys, x_range, bins=bins)
        if hist.max() < 1e-9:
            return
        XX, YY = np.meshgrid(xc, yc)
        levels = np.linspace(hist.max() * 0.05, hist.max(), 12)
        ax.contourf(XX, YY, hist, levels=levels, cmap=cmap, zorder=4)
    except Exception as exc:
        logger.debug("density contourf failed: %s", exc)


# ── Track collector ───────────────────────────────────────────────────────────

def _collect_tracks(frame_data: Dict, frame_w: int, frame_h: int,
                    role_map: Dict) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = {}
    for frame_idx, fdata in sorted(frame_data.items()):
        for d in fdata["detections"]:
            tid = d.get("track_id", -1)
            if tid < 0:
                continue
            tracks.setdefault(tid, []).append({
                "frame": frame_idx,
                "x":     _bbox_center(d["bbox"])[0] / frame_w,
                "y":     _bbox_center(d["bbox"])[1] / frame_h,
                "role":  _get_role(d, role_map),
            })
    return tracks


# ── Formation helpers ─────────────────────────────────────────────────────────

def _classify_defense_formation(xs: List[float]) -> str:
    """
    Given x-positions of defenders (mirrored, own goal at x=0),
    return handball formation label: '6-0', '5-1', '3-2-1', etc.
    """
    if not xs:
        return "?"
    n = len(xs)
    if n <= 2:
        return f"{n}-0"
    sorted_xs = sorted(xs)
    diffs = np.diff(sorted_xs)
    # Adaptive gap threshold, minimum 0.07 ≈ 2.8 m on a 40 m court
    thresh = max(float(np.mean(diffs)) + 0.7 * float(np.std(diffs)), 0.07)
    split_idx = np.where(diffs > thresh)[0] + 1
    rows = [r for r in np.split(np.array(sorted_xs), split_idx) if len(r) > 0]
    counts = [len(r) for r in rows]
    if len(counts) == 1:
        return f"{counts[0]}-0"   # flat defense (all in one line)
    return "-".join(str(c) for c in counts)


def _formation_dots(form_label: str) -> List[Tuple[float, float]]:
    """
    Idealized player positions for a defense formation diagram.
    Left half-court coordinates: x=0 = own goal, x=0.5 = centre line.
    Rows ordered from most defensive (small x) to most advanced (larger x).
    """
    try:
        parts = [int(s) for s in form_label.split("-") if s.isdigit()]
    except ValueError:
        parts = [6]
    parts = [p for p in parts if p > 0]
    if not parts:
        return []

    n_rows = len(parts)
    # Place rows between 9m line (~0.225) and near centre (~0.43)
    # Single-row (6-0): all at the 9m line (~0.225)
    if n_rows == 1:
        row_xs = [0.22]
    else:
        row_xs = [0.20 + 0.22 * r / (n_rows - 1) for r in range(n_rows)]

    dots: List[Tuple[float, float]] = []
    for rx, count in zip(row_xs, parts):
        ys = np.linspace(0.10, 0.90, count) if count > 1 else [0.5]
        for y in ys:
            dots.append((rx, float(y)))
    return dots


def _classify_attack_movement(xs: np.ndarray, ys: np.ndarray) -> str:
    """Classify a player's attack track into a movement pattern."""
    if len(xs) < 4:
        return "Sonstiges"
    dx = float(xs[-1] - xs[0])
    dy = float(ys[-1] - ys[0])
    dist = float(np.sqrt(dx ** 2 + dy ** 2))
    if dist < 0.04:
        return "Positionsspiel"
    if dx > 0.07:
        if abs(dy) < dx * 0.50:
            return "Tiefenlauf"
        return "Diagonallauf"
    if abs(dy) > 0.07 and abs(dx) < 0.04:
        return "Lateralbewegung"
    if dx < -0.05:
        return "Rückzug"
    return "Sonstiges"


def _count_crossings(tracks: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """
    Count player pairs that swapped lateral (y) position during attack.
    Requires both players to start and end with a clear y-separation that flips.
    """
    count = 0
    n = len(tracks)
    for i in range(n):
        xa, ya = tracks[i]
        for j in range(i + 1, n):
            xb, yb = tracks[j]
            if len(ya) < 2 or len(yb) < 2:
                continue
            ya0, ya1 = float(ya[0]),  float(ya[-1])
            yb0, yb1 = float(yb[0]),  float(yb[-1])
            # Require clear lateral separation at start AND end
            if abs(ya0 - yb0) < 0.07 or abs(ya1 - yb1) < 0.07:
                continue
            # Require x-ranges to overlap (both in same court zone)
            xa_lo, xa_hi = float(xa.min()), float(xa.max())
            xb_lo, xb_hi = float(xb.min()), float(xb.max())
            if min(xa_hi, xb_hi) < max(xa_lo, xb_lo):
                continue
            if np.sign(ya0 - yb0) != np.sign(ya1 - yb1):
                count += 1
    return count


# ── Private figure builders ───────────────────────────────────────────────────

def _save_abwehr_formations(formation_counts: Dict[str, int],
                             jersey: str, tcolor: str,
                             output_dir: Path,
                             formation_positions: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                             ) -> None:
    if not formation_counts:
        logger.warning("Keine Abwehr-Formationsdaten für %s", jersey)
        return

    total = max(sum(formation_counts.values()), 1)
    top_fms = sorted(formation_counts.items(), key=lambda x: x[1], reverse=True)
    n_courts = min(len(top_fms), 3)

    fig = plt.figure(figsize=(5 + n_courts * 5.2, 7))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"{jersey}  —  Abwehr-Formationen",
                 fontsize=15, fontweight="bold", color=tcolor, y=1.02)

    # ── Bar chart ────────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(1, 1 + n_courts, 1)
    ax_bar.set_facecolor("#1a1a2a")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#444455")

    labels = [f[0] for f in top_fms]
    pcts   = [100.0 * f[1] / total for f in top_fms]
    y_pos  = np.arange(len(labels))
    bars   = ax_bar.barh(y_pos, pcts, color=tcolor, alpha=0.85,
                         edgecolor="#ffffff22", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="white", fontsize=15, fontweight="bold")
    ax_bar.set_xlabel("% der Abwehrphasen", color="#aaaaaa", fontsize=10)
    ax_bar.tick_params(axis="x", colors="#aaaaaa")
    ax_bar.set_title("Formationshäufigkeit", fontsize=12,
                     fontweight="bold", color="white", pad=8)
    ax_bar.grid(axis="x", color="#333344", alpha=0.7, linestyle="--")
    ax_bar.invert_yaxis()
    for bar_obj, pct in zip(bars, pcts):
        ax_bar.text(bar_obj.get_width() + 0.8,
                   bar_obj.get_y() + bar_obj.get_height() / 2,
                   f"{pct:.0f} %", va="center", ha="left",
                   color="white", fontsize=11, fontweight="bold")

    # ── Court diagrams for top formations ────────────────────────────────────
    for i, (form_label, count) in enumerate(top_fms[:n_courts]):
        ax = fig.add_subplot(1, 1 + n_courts, 2 + i)
        pct = 100.0 * count / total
        _draw_half_court_left(ax, title=f"{form_label}   ({pct:.0f} %)")

        # Layer 1: actual observed positions as density / scatter
        actual = (formation_positions or {}).get(form_label, [])
        if len(actual) >= 10:
            ax_xs = np.array([p[0] for p in actual])
            ax_ys = np.array([p[1] for p in actual])
            if _SCIPY:
                try:
                    hist, xe, ye = np.histogram2d(
                        ax_xs, ax_ys, bins=18,
                        range=[[0.0, 0.5], [0.0, 1.0]])
                    hist = _gaussian_filter(hist.T, sigma=1.8)
                    if hist.max() > 1e-9:
                        xc = 0.5 * (xe[1:] + xe[:-1])
                        yc = 0.5 * (ye[1:] + ye[:-1])
                        XX, YY = np.meshgrid(xc, yc)
                        lev = np.linspace(hist.max() * 0.15, hist.max(), 8)
                        hx = int(tcolor.lstrip("#")[0:2], 16) / 255
                        hy = int(tcolor.lstrip("#")[2:4], 16) / 255
                        hz = int(tcolor.lstrip("#")[4:6], 16) / 255
                        cmap_heat = mcolors.LinearSegmentedColormap.from_list(
                            "fh", [(hx, hy, hz, 0.0), (hx, hy, hz, 0.55)], N=64)
                        ax.contourf(XX, YY, hist, levels=lev,
                                    cmap=cmap_heat, zorder=5)
                except Exception:
                    pass
            else:
                ax.scatter(ax_xs, ax_ys, s=12, c=[tcolor], alpha=0.20, zorder=5)

        # Layer 2: idealized formation dots + row connections
        dots = _formation_dots(form_label)
        parts = [int(s) for s in form_label.split("-") if s.isdigit() and int(s) > 0]
        idx = 0
        for count_in_row in parts:
            row_dots = dots[idx: idx + count_in_row]
            if len(row_dots) >= 2:
                rx = [p[0] for p in row_dots]
                ry = [p[1] for p in row_dots]
                ax.plot(rx, ry, color=tcolor, alpha=0.55, lw=2.0, zorder=7,
                        linestyle="--")
            idx += count_in_row
        for px, py in dots:
            ax.scatter(px, py, s=500, c=[tcolor], zorder=8,
                       edgecolors="white", linewidths=2.5)

    plt.tight_layout()
    path = output_dir / f"tactical_abwehr_{jersey}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Abwehr-Formations-Plot: %s", path)


def _save_angriff_patterns(
        pattern_counts: Dict[str, int],
        pattern_tracks: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
        crossing_count: int,
        jersey: str, tcolor: str,
        output_dir: Path) -> None:
    if not pattern_counts:
        logger.warning("Keine Angriffsmuster-Daten für %s", jersey)
        return

    total = max(sum(pattern_counts.values()), 1)

    fig, (ax_court, ax_stats) = plt.subplots(
        1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"{jersey}  —  Angriffsmuster  (Spielzüge)",
                 fontsize=15, fontweight="bold", color=tcolor, y=1.02)

    # ── Court: movement arrows per pattern ───────────────────────────────────
    _draw_half_court(ax_court, title="Bewegungsmuster im Angriff")

    legend_patches = []
    for pat, tracks in pattern_tracks.items():
        pcolor = _PATTERN_COLORS.get(pat, "#aaaaaa")
        sample = tracks[:10]
        for xs, ys in sample:
            if len(xs) < 2:
                continue
            ax_court.plot(xs, ys, color=pcolor, alpha=0.30, lw=1.2, zorder=4)
            # Arrow head at end of track
            n = len(xs)
            tail = max(0, n - max(3, n // 5))
            if xs[tail] != xs[-1] or ys[tail] != ys[-1]:
                ax_court.annotate(
                    "", xy=(xs[-1], ys[-1]), xytext=(xs[tail], ys[tail]),
                    arrowprops=dict(arrowstyle="->", color=pcolor, lw=1.8,
                                   alpha=0.75),
                    zorder=5)
        legend_patches.append(
            mpatches.Patch(color=pcolor,
                           label=f"{_PATTERN_LABELS_DE.get(pat, pat)}"
                                 f"  ({pattern_counts[pat]})"))

    ax_court.legend(handles=legend_patches, fontsize=8, loc="lower left",
                    facecolor="#2a2a2a", labelcolor="white", framealpha=0.9)

    # ── Stats bar chart ───────────────────────────────────────────────────────
    ax_stats.set_facecolor("#1a1a2a")
    for sp in ax_stats.spines.values():
        sp.set_edgecolor("#444455")

    ordered = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    bar_labels = [_PATTERN_LABELS_DE.get(p, p) for p, _ in ordered]
    pcts       = [100.0 * c / total for _, c in ordered]
    colors     = [_PATTERN_COLORS.get(p, "#aaaaaa") for p, _ in ordered]
    y_pos      = np.arange(len(bar_labels))

    bars = ax_stats.barh(y_pos, pcts, color=colors, alpha=0.85,
                         edgecolor="#ffffff22", linewidth=0.5)
    ax_stats.set_yticks(y_pos)
    ax_stats.set_yticklabels(bar_labels, color="white", fontsize=10)
    ax_stats.set_xlabel("% der Angriffs-Tracks", color="#aaaaaa", fontsize=10)
    ax_stats.tick_params(axis="x", colors="#aaaaaa")
    ax_stats.set_title("Häufigkeit der Bewegungsmuster",
                       fontsize=12, fontweight="bold", color="white", pad=8)
    ax_stats.grid(axis="x", color="#333344", alpha=0.7, linestyle="--")
    ax_stats.invert_yaxis()
    for bar_obj, pct in zip(bars, pcts):
        ax_stats.text(bar_obj.get_width() + 0.5,
                     bar_obj.get_y() + bar_obj.get_height() / 2,
                     f"{pct:.0f} %", va="center", ha="left",
                     color="white", fontsize=10, fontweight="bold")

    # Crossing count badge
    ax_stats.text(0.97, 0.04,
                 f"Kreuzläufe erkannt: {crossing_count}",
                 transform=ax_stats.transAxes, ha="right", va="bottom",
                 fontsize=12, color="#FFDD44", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="#333322",
                           alpha=0.85, edgecolor="#FFDD44"))

    plt.tight_layout()
    path = output_dir / f"tactical_angriff_{jersey}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Angriffs-Muster-Plot: %s", path)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_formations(frame_data: Dict, role_map: Dict[int, str],
                       frame_w: int, frame_h: int,
                       output_dir: Path,
                       teams: Tuple[str, str] = ("Team A", "Team B"),
                       team_colors: Optional[Dict[str, str]] = None,
                       min_players: int = 5) -> None:
    """
    Per team:
      • tactical_abwehr_{jersey}.png  — defense formation frequencies
      • tactical_angriff_{jersey}.png — attack movement patterns + crossing count
    """
    logger.info("Formations-Analyse …")
    phases = _classify_phases(frame_data, role_map, frame_w, frame_h, teams)
    _tc = team_colors or {}
    all_tracks = _collect_tracks(frame_data, frame_w, frame_h, role_map)

    for team_key, team in zip(("Team A", "Team B"), teams):
        jersey = _tc.get(team_key, team)
        tcolor = _TEAM_COLORS.get(team, "#888888")

        # ── ABWEHR: classify formation per frame ──────────────────────────
        formation_counts:    Dict[str, int] = {}
        formation_positions: Dict[str, List[Tuple[float, float]]] = {}

        for frame_idx, fdata in sorted(frame_data.items()):
            if phases.get(frame_idx, {}).get(team) != "Abwehr":
                continue
            dets = [d for d in fdata["detections"]
                    if _get_role(d, role_map) == team]
            if len(dets) < min_players:
                continue

            xs = [_bbox_center(d["bbox"])[0] / frame_w for d in dets]
            ys = [_bbox_center(d["bbox"])[1] / frame_h for d in dets]
            mirror = float(np.mean(xs)) > 0.5
            if mirror:
                xs = [1.0 - x for x in xs]

            label = _classify_defense_formation(xs)
            formation_counts[label] = formation_counts.get(label, 0) + 1
            pts = formation_positions.setdefault(label, [])
            for x, y in zip(xs, ys):
                pts.append((x, y))

        _save_abwehr_formations(formation_counts, jersey, tcolor, output_dir,
                                formation_positions=formation_positions)

        # ── ANGRIFF: classify track movement patterns ─────────────────────
        pattern_counts: Dict[str, int] = {}
        pattern_tracks: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        attack_tracks_xy: List[Tuple[np.ndarray, np.ndarray]] = []

        for tid, track_pts in all_tracks.items():
            att_pts = [p for p in track_pts
                      if p["role"] == team
                      and phases.get(p["frame"], {}).get(team) == "Angriff"]
            if len(att_pts) < 8:
                continue

            xs = np.array([p["x"] for p in att_pts])
            ys = np.array([p["y"] for p in att_pts])
            if float(np.mean(xs)) < 0.5:
                xs = 1.0 - xs

            pat = _classify_attack_movement(xs, ys)
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
            pattern_tracks.setdefault(pat, []).append((xs, ys))
            attack_tracks_xy.append((xs, ys))

        crossing_count = _count_crossings(attack_tracks_xy)
        logger.info("%s: %d Kreuzläufe erkannt", jersey, crossing_count)

        _save_angriff_patterns(pattern_counts, pattern_tracks,
                               crossing_count, jersey, tcolor, output_dir)


def analyze_shot_zones(frame_data: Dict,
                       role_map: Dict,
                       frame_w: int, frame_h: int,
                       output_dir: Path,
                       teams: Tuple[str, str] = ("Team A", "Team B"),
                       team_colors: Optional[Dict[str, str]] = None) -> None:
    """
    Smooth heatmap of player positions during attacking phases (half-court).
    NOT shot tracking — shows WHERE players stand while attacking.
    """
    logger.info("Wurfzonen-Analyse …")
    phases = _classify_phases(frame_data, role_map, frame_w, frame_h, teams)
    _tc = team_colors or {}
    team_positions: Dict[str, List[Tuple[float, float]]] = {t: [] for t in teams}

    for frame_idx, fdata in frame_data.items():
        fp = phases.get(frame_idx, {})
        for d in fdata["detections"]:
            role = _get_role(d, role_map)
            if role not in teams or fp.get(role) != "Angriff":
                continue
            cx, cy = _bbox_center(d["bbox"])
            xn, yn = cx / frame_w, cy / frame_h
            team_xs = [_bbox_center(dd["bbox"])[0] / frame_w
                       for dd in fdata["detections"]
                       if _get_role(dd, role_map) == role]
            if team_xs and float(np.mean(team_xs)) < 0.5:
                xn = 1 - xn
            if xn >= 0.5:
                team_positions[role].append((xn, yn))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        "Aufenthalts-Heatmap im Angriff\n"
        "(Spielerpositionen während Angriffsphasen  —  keine Torschuss-Erkennung)",
        fontsize=13, fontweight="bold", color="white", y=1.03)

    for ax, (team_key, team) in zip(axes, zip(("Team A", "Team B"), teams)):
        positions = team_positions.get(team, [])
        jersey    = _tc.get(team_key, team)
        tcolor    = _TEAM_COLORS.get(jersey, "#888888")
        _draw_half_court(ax)

        if len(positions) < 20:
            ax.text(0.75, 0.5, "Zu wenige\nDaten",
                    ha="center", va="center", fontsize=13,
                    color="white", alpha=0.7)
            ax.set_title(f"{jersey}  —  {len(positions)} Positionen",
                         fontsize=12, fontweight="bold", color=tcolor, pad=8)
            continue

        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])

        hist, xc, yc = _density_grid(xs, ys, x_range=(0.5, 1.0),
                                     bins=45, sigma=3.5)
        if hist.max() > 1e-9:
            XX, YY = np.meshgrid(xc, yc)
            levels = np.linspace(hist.max() * 0.04, hist.max(), 18)
            ax.contourf(XX, YY, hist, levels=levels, cmap="hot",
                        alpha=0.82, zorder=4)

            sm = plt.cm.ScalarMappable(
                cmap="hot", norm=mcolors.Normalize(0, hist.max()))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
            cb.set_label("Aufenthaltsdichte", fontsize=9, color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
            cb.outline.set_edgecolor("white")

            peak = np.unravel_index(np.argmax(hist), hist.shape)
            ax.scatter(xc[peak[1]], yc[peak[0]], s=400, marker="*",
                       c="#ffff00", zorder=7, edgecolors="white",
                       linewidths=1.5, label="Häufigste Zone")
            ax.legend(fontsize=9, facecolor="#2a2a2a", labelcolor="white",
                      loc="upper left", framealpha=0.85)

        ax.set_title(
            f"{jersey}  —  {len(positions)} Positionen im Angriff",
            fontsize=12, fontweight="bold", color=tcolor, pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "tactical_shot_zones.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Wurfzonen-Plot: %s", path)


def analyze_trajectories(frame_data: Dict,
                         frame_w: int, frame_h: int,
                         output_dir: Path,
                         role_map: Optional[Dict] = None,
                         min_track_len: int = 15) -> None:
    """
    Left:  full court — density heatmap per role + individual track lines on top.
    Right: horizontal movement-statistics bars per role.
    """
    logger.info("Trajektorien-Analyse …")
    if role_map is None:
        role_map = {}

    _SKIP = {"Noise / Unklar", "Sonstige", "Unbekannt"}

    # Collect per-role: aggregated point clouds AND individual tracks
    role_pts:    Dict[str, Tuple[List[float], List[float]]] = {}
    role_tracks_list: Dict[str, List[Dict]]                 = {}
    role_stats:  Dict[str, Dict[str, List[float]]]          = {}
    track_count: Dict[str, int]                             = {}

    for tid, pts_raw in _collect_tracks(
            frame_data, frame_w, frame_h, role_map).items():
        pts_raw.sort(key=lambda p: p["frame"])
        if len(pts_raw) < min_track_len:
            continue
        role = pts_raw[len(pts_raw) // 2]["role"] or "Unbekannt"
        if role in _SKIP:
            continue

        xs = np.array([p["x"] for p in pts_raw])
        ys = np.array([p["y"] for p in pts_raw])

        role_pts.setdefault(role, ([], []))
        role_pts[role][0].extend(xs.tolist())
        role_pts[role][1].extend(ys.tolist())
        role_tracks_list.setdefault(role, []).append({"xs": xs, "ys": ys})
        track_count[role] = track_count.get(role, 0) + 1

        dx, dy = np.diff(xs), np.diff(ys)
        steps = np.sqrt(dx ** 2 + dy ** 2)
        if len(steps) == 0:
            continue
        for key, val in [("speed",   float(steps.mean())),
                          ("x_range", float(xs.max() - xs.min())),
                          ("y_range", float(ys.max() - ys.min())),
                          ("dist",    float(steps.sum()))]:
            role_stats.setdefault(role, {}).setdefault(key, []).append(val)

    if not role_pts:
        logger.warning("Keine Tracks für Trajektorien-Analyse")
        return

    named   = sorted({r for r in role_map.values()
                       if r not in {"Noise / Unklar"}}, key=str)
    ordered = [r for r in named if r in role_pts]
    total   = sum(track_count.values())
    logger.info("Trajektorien: %d Tracks in %d Rollen", total, len(ordered))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor("#111111")
    ax_court = fig.add_axes([0.02, 0.05, 0.52, 0.88])
    ax_stats  = fig.add_axes([0.60, 0.08, 0.38, 0.82])

    _draw_full_court(ax_court, title="Positions-Dichte + Laufwege nach Rolle")

    legend_patches = []
    for role in ordered[:6]:
        chex = _TEAM_COLORS.get(role, "#aaaaaa")
        pts  = role_pts.get(role)
        if pts is None or len(pts[0]) < 20:
            continue

        # Layer 1: smooth density background
        cmap = _team_cmap(chex)
        hist, xc, yc = _density_grid(
            np.array(pts[0]), np.array(pts[1]),
            x_range=(0.0, 1.0), bins=45, sigma=3.0)
        if hist.max() > 1e-9:
            XX, YY = np.meshgrid(xc, yc)
            levels = np.linspace(hist.max() * 0.08, hist.max(), 9)
            ax_court.contourf(XX, YY, hist, levels=levels, cmap=cmap, zorder=3)

        # Layer 2: individual track lines (max 15, semi-transparent)
        tracks_for_role = role_tracks_list.get(role, [])
        step = max(1, len(tracks_for_role) // 15)
        for t in tracks_for_role[::step][:15]:
            ax_court.plot(t["xs"], t["ys"], color=chex,
                         alpha=0.28, lw=1.0, zorder=5)
            # Small dot at track start
            ax_court.scatter(t["xs"][0], t["ys"][0],
                            c=chex, s=10, alpha=0.45, zorder=6)

        legend_patches.append(
            mpatches.Patch(color=chex,
                           label=f"{role}  ({track_count.get(role, 0)} Tracks)"))

    ax_court.legend(handles=legend_patches, fontsize=9, loc="lower right",
                    facecolor="#2a2a2a", labelcolor="white", framealpha=0.9)

    # ── Horizontal stats bars ─────────────────────────────────────────────────
    ax_stats.set_facecolor("#1a1a2a")
    for sp in ax_stats.spines.values():
        sp.set_edgecolor("#555555")

    stat_keys   = ["speed", "x_range", "y_range", "dist"]
    stat_labels = ["Ø Geschw.", "x-Bereich", "y-Bereich", "Gesamtstrecke"]
    y_pos = np.arange(len(stat_keys))
    bar_h = 0.7 / max(len(ordered), 1)

    for i, role in enumerate(ordered[:6]):
        chex  = _TEAM_COLORS.get(role, "#aaaaaa")
        stats = role_stats.get(role, {})
        vals  = [float(np.mean(stats.get(k, [0.0]))) for k in stat_keys]
        off   = (i - len(ordered) / 2 + 0.5) * bar_h
        ax_stats.barh(y_pos + off, vals, bar_h * 0.88,
                      color=chex, alpha=0.88, label=role,
                      edgecolor="#ffffff33", linewidth=0.5)

    ax_stats.set_yticks(y_pos)
    ax_stats.set_yticklabels(stat_labels, color="white", fontsize=11)
    ax_stats.tick_params(axis="x", colors="#aaaaaa")
    ax_stats.set_xlabel("Mittelwert (normierte Koordinaten)",
                        color="#aaaaaa", fontsize=10)
    ax_stats.legend(fontsize=9, facecolor="#2a2a2a", labelcolor="white",
                    framealpha=0.9, loc="lower right")
    ax_stats.set_title("Bewegungs-Statistiken nach Rolle",
                       fontsize=13, fontweight="bold", color="white", pad=10)
    ax_stats.grid(axis="x", color="#444455", alpha=0.6, linestyle="--")

    fig.suptitle(f"Laufweg-Analyse  —  {total} Tracks",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    path = output_dir / "tactical_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Trajektorien-Plot: %s", path)
