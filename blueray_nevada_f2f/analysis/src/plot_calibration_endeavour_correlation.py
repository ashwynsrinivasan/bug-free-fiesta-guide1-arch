#!/usr/bin/env python3
"""
Endeavour calibration: laser vs VOA final-state correlation from ``P*_log.csv``.

Reads the **last row** of each log (final settled state) under ``logs_Endeavour_*``.

Writes under ``analysis/results/calibration/co-relation/``:

1. ``endeavour_final_ld_vs_voa_per_channel.png`` — 16 subplots (one channel each),
   scatter: final laser current (mA) vs final VOA current (mA); title shows 2×2 Pearson
   correlation matrix for that channel across calibration runs.

2. ``endeavour_final_ld_voa_by_channel_violin_box.png`` — two subplots: channel vs final
   laser current, and channel vs final VOA current. Styling matches
   ``tp2p5_distribution_vs_freq_error`` (grey violin behind gen1 white box, μ̃/σ annotations).

Samples with final laser **< 50 mA** (or 0) are dropped from the **VOA** distribution and from
any paired view. **Laser** scatter and top distribution use laser in **100–200 mA** (and still ≥ 50 mA).
Axes: laser **100–200 mA**, VOA **0–10 mA**.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_tp2p4_onet_sftp import (
    DIST_FIGSIZE,
    _GEN1_BOXPLOT_KW,
    _force_white_box_faces,
    _grey_violin_behind_box,
    _raise_boxplot_zorder,
)

MIN_LASER_MA = 50.0
LASER_DISPLAY_LO_HI = (100.0, 200.0)
YLIM_LASER = (100.0, 200.0)
YLIM_VOA = (0.0, 10.0)
YTICKS_LASER = np.arange(100, 201, 20)
YTICKS_VOA = np.arange(0, 11, 2)


def _channel_labels() -> list[str]:
    return [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)]


def iter_endeavour_log_files(cal_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in cal_root.rglob("P*_log.csv"):
        if "logs_Endeavour" not in p.as_posix():
            continue
        if "_log.csv" not in p.name:
            continue
        out.append(p)
    return sorted(out)


def final_state_row(path: Path) -> pd.Series | None:
    try:
        df = pd.read_csv(path)
        if df.empty or "timestamp" not in df.columns:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return None
        return df.iloc[-1]
    except Exception:
        return None


def collect_final_states(cal_root: Path) -> pd.DataFrame:
    """One row per log file: columns log_path, tile-ish, current_LD_0..15, current_VOA_0..15."""
    rows: list[dict] = []
    for path in iter_endeavour_log_files(cal_root):
        last = final_state_row(path)
        if last is None:
            continue
        d: dict = {"log_path": str(path)}
        try:
            for i in range(16):
                d[f"current_LD_{i}"] = float(pd.to_numeric(last[f"current_LD_{i}"], errors="coerce"))
                d[f"current_VOA_{i}"] = float(pd.to_numeric(last[f"current_VOA_{i}"], errors="coerce"))
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(d)
    return pd.DataFrame(rows)


def _per_channel_filtered_arrays(
    df: pd.DataFrame,
    *,
    laser_in_range: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per channel: paired (laser, voa) where laser >= MIN_LASER_MA; optional laser window [lo, hi]."""
    laser_out: list[np.ndarray] = []
    voa_out: list[np.ndarray] = []
    lo, hi = laser_in_range if laser_in_range else (None, None)
    for i in range(16):
        ld = df[f"current_LD_{i}"].to_numpy(dtype=float)
        voa = df[f"current_VOA_{i}"].to_numpy(dtype=float)
        ok = np.isfinite(ld) & np.isfinite(voa) & (ld >= MIN_LASER_MA)
        if lo is not None and hi is not None:
            ok = ok & (ld >= lo) & (ld <= hi)
        laser_out.append(ld[ok])
        voa_out.append(voa[ok])
    return laser_out, voa_out


def _pearson_corr_matrix_line(x: np.ndarray, y: np.ndarray) -> str:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return "corr = n/a"
    if np.std(x) == 0 or np.std(y) == 0:
        return "corr = n/a (zero variance)"
    c = np.corrcoef(x, y)
    return f"[[{c[0, 0]:.3f}, {c[0, 1]:.3f}], [{c[1, 0]:.3f}, {c[1, 1]:.3f}]]"


def plot_ld_vs_voa_per_channel(df: pd.DataFrame, output_path: Path) -> None:
    labels = _channel_labels()
    laser_arrs, voa_arrs = _per_channel_filtered_arrays(
        df, laser_in_range=LASER_DISPLAY_LO_HI
    )
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(14, 14), layout="constrained")
    axes_flat = axes.ravel()
    rng = np.random.default_rng(0)
    for i in range(16):
        ax = axes_flat[i]
        x = laser_arrs[i]
        y = voa_arrs[i]
        corr_line = _pearson_corr_matrix_line(x, y)
        if x.size:
            jitter_x = rng.normal(0, 0.35, size=len(x))
            jitter_y = rng.normal(0, 0.06, size=len(y))
            ax.scatter(x + jitter_x, y + jitter_y, s=14, alpha=0.35, c="#1565c0", edgecolors="none")
        ax.set_title(f"{labels[i]}\n{corr_line}", fontsize=8)
        ax.set_xlabel("Laser I (mA)", fontsize=7)
        ax.set_ylabel("VOA I (mA)", fontsize=7)
        ax.set_xlim(YLIM_LASER)
        ax.set_ylim(YLIM_VOA)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Endeavour calibration: final state — laser vs VOA (per channel); "
        f"laser ≥ {MIN_LASER_MA:.0f} mA and {LASER_DISPLAY_LO_HI[0]:.0f}–{LASER_DISPLAY_LO_HI[1]:.0f} mA; "
        f"VOA {YLIM_VOA[0]:.0f}–{YLIM_VOA[1]:.0f} mA",
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _tp2p5_style_distribution_panel(
    ax,
    per_channel_values: list[np.ndarray],
    *,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    ylabel: str,
    annotation_unit: str,
) -> None:
    """Same layout as ``plot_tp2p4_freq_error_distribution`` / tp2p5_distribution_vs_freq_error."""
    y_ann = ylo + (yhi - ylo) * 0.055
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for i, vals in enumerate(per_channel_values):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        _grey_violin_behind_box(ax, vals, position=float(i), width=0.62)
        box_data.append(vals)
        box_positions.append(i)
    if not box_data:
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(yticks)
        return
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        vert=True,
        widths=0.55,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    u = annotation_unit
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        med = int(round(float(np.median(vals))))
        std = int(round(float(np.std(vals))))
        fmt = f"μ̃={med}{u}\nσ={std}{u}"
        ax.text(
            pos,
            y_ann,
            fmt,
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=6,
            clip_on=False,
        )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels(
        [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 15.5)


def plot_violin_box_by_channel(df: pd.DataFrame, output_path: Path) -> None:
    laser_arrs, _ = _per_channel_filtered_arrays(df, laser_in_range=LASER_DISPLAY_LO_HI)
    _, voa_arrs = _per_channel_filtered_arrays(df, laser_in_range=None)
    sns.set_style("whitegrid")
    w, h = DIST_FIGSIZE
    fig, axes = plt.subplots(2, 1, figsize=(w, h * 2), layout="constrained")

    _tp2p5_style_distribution_panel(
        axes[0],
        laser_arrs,
        ylo=YLIM_LASER[0],
        yhi=YLIM_LASER[1],
        yticks=YTICKS_LASER,
        ylabel="Final laser current (mA)",
        annotation_unit="mA",
    )
    _tp2p5_style_distribution_panel(
        axes[1],
        voa_arrs,
        ylo=YLIM_VOA[0],
        yhi=YLIM_VOA[1],
        yticks=YTICKS_VOA,
        ylabel="Final VOA current (mA)",
        annotation_unit="mA",
    )

    fig.suptitle(
        f"Endeavour calibration: final state by channel (tp2p5_distribution_vs_freq_error style); "
        f"laser panel: {LASER_DISPLAY_LO_HI[0]:.0f}–{LASER_DISPLAY_LO_HI[1]:.0f} mA (and ≥{MIN_LASER_MA:.0f} mA); "
        f"VOA panel: paired VOA when laser ≥ {MIN_LASER_MA:.0f} mA",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Endeavour final-state LD vs VOA correlation plots.")
    parser.add_argument(
        "--cal-root",
        type=Path,
        default=None,
        help="clm_calibration root (default: monorepo data/clm_calibration)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: analysis/results/calibration/co-relation)",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    cal_root = (args.cal_root or (repo_root / "data" / "clm_calibration")).resolve()
    if not cal_root.is_dir():
        print(f"Missing calibration root {cal_root}", file=sys.stderr)
        return 1

    out_dir = (
        args.out_dir
        or (here.parent.parent / "results" / "calibration" / "co-relation")
    ).resolve()

    print(f"Scanning {cal_root} …")
    df = collect_final_states(cal_root)
    if df.empty or len(df) < 2:
        print("Not enough Endeavour logs with valid final rows.", file=sys.stderr)
        return 1
    print(
        f"Using {len(df)} calibration logs (final row each); "
        f"VOA stats: laser ≥ {MIN_LASER_MA:.0f} mA; "
        f"laser scatter/top panel: {LASER_DISPLAY_LO_HI[0]:.0f}–{LASER_DISPLAY_LO_HI[1]:.0f} mA."
    )

    p1 = out_dir / "endeavour_final_ld_vs_voa_per_channel.png"
    p2 = out_dir / "endeavour_final_ld_voa_by_channel_violin_box.png"
    plot_ld_vs_voa_per_channel(df, p1)
    print(f"Saved: {p1}")
    plot_violin_box_by_channel(df, p2)
    print(f"Saved: {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
