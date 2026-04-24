#!/usr/bin/env python3
"""
Endeavour calibration logs: five-panel time series per ``P*_log.csv`` under ``logs_Endeavour_*``.

Panels (x = elapsed time from first row, minutes):

1. Laser drive current (mA) — ``current_LD_0`` … ``current_LD_15``
2. VOA current (mA) — ``current_VOA_0`` … ``current_VOA_15``
3. Optical power in fiber (mW) — ``power_PIC_i / 1e3``
4. Frequency error (GHz) — measured optical frequency vs ``wavelength_grid.yaml``
   target: ``(frequency_i / 1000 - f_target_thz) * 1000`` (same scale as grid ``frequency_thz``).
5. PIC temperature (°C) — ``temp_pic`` (single trace, module-level).

Writes PNGs under ``analysis/results/calibration/tiles/endeavour/``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analyze_tp2p4_onet_sftp import default_config_dir


def _channel_legend_labels() -> list[str]:
    return [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)]


def _target_frequency_thz(wl_grid: dict, channel_index: int) -> float:
    bank = channel_index // 8
    ch = channel_index % 8
    return float(wl_grid["banks"][f"bank{bank}"]["grids"][ch + 1]["frequency_thz"])


def _tile_sn_from_calibration_path(p: Path) -> str | None:
    """Tile SN from ``Y##########_date`` or ``.../Y##########/...`` parents."""
    m = re.match(r"^(Y\d{10})(?:_\d{4}-\d{2}-\d{2})?$", p.name)
    if m:
        return m.group(1)
    for parent in p.parents:
        m2 = re.match(r"^(Y\d{10})(?:_\d{4}-\d{2}-\d{2})?$", parent.name)
        if m2:
            return m2.group(1)
    m3 = re.search(r"(Y\d{10})", p.as_posix())
    return m3.group(1) if m3 else None


def _freq_error_ghz_series(freq_col: pd.Series, target_thz: float) -> np.ndarray:
    """Grid targets ``frequency_thz``; log column equals ``round(f_thz * 1000)`` (see grid 230.35 ↔ 230350)."""
    v = pd.to_numeric(freq_col, errors="coerce").to_numpy(dtype=float)
    return (v / 1000.0 - target_thz) * 1000.0


def load_endeavour_log_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column in {path}")
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    t0 = df["timestamp"].min()
    df["t_min"] = (df["timestamp"] - t0) / 60.0
    return df


def plot_five_panels(df: pd.DataFrame, wl_grid: dict, title: str, output_path: Path) -> None:
    labels = _channel_legend_labels()
    colors = plt.cm.tab20(np.linspace(0, 1, 16))

    fig, axes = plt.subplots(5, 1, figsize=(12, 17.5), sharex=True, layout="constrained")
    t = df["t_min"].values

    for i in range(16):
        c = colors[i]
        ld = df[f"current_LD_{i}"]
        voa = df[f"current_VOA_{i}"]
        pic = pd.to_numeric(df[f"power_PIC_{i}"], errors="coerce") / 1e3
        tgt = _target_frequency_thz(wl_grid, i)
        fe = _freq_error_ghz_series(df[f"frequency_{i}"], tgt)

        axes[0].plot(t, ld, color=c, linewidth=0.8, alpha=0.85, label=labels[i])
        axes[1].plot(t, voa, color=c, linewidth=0.8, alpha=0.85)
        axes[2].plot(t, pic, color=c, linewidth=0.8, alpha=0.85)
        axes[3].plot(t, fe, color=c, linewidth=0.8, alpha=0.85)

    axes[0].set_ylabel("Laser current (mA)", fontsize=10, fontweight="bold")
    axes[0].set_title(title, fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=6, loc="upper right", framealpha=0.9)

    axes[1].set_ylabel("VOA current (mA)", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Optical power in fiber (mW)", fontsize=10, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Frequency error (GHz)", fontsize=10, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    if "temp_pic" in df.columns:
        tpic = pd.to_numeric(df["temp_pic"], errors="coerce")
        axes[4].plot(t, tpic, color="#bf360c", linewidth=1.2, label="T_PIC")
        axes[4].legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        axes[4].text(0.5, 0.5, "temp_pic column missing", ha="center", va="center", transform=axes[4].transAxes)
    axes[4].set_ylabel("T_PIC (°C)", fontsize=10, fontweight="bold")
    axes[4].set_xlabel("Time (min from start of log)", fontsize=10, fontweight="bold")
    axes[4].grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def iter_endeavour_log_files(cal_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in cal_root.rglob("P*_log.csv"):
        if "logs_Endeavour" not in p.as_posix():
            continue
        if "_log.csv" not in p.name:
            continue
        out.append(p)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Endeavour calibration P*_log time series (5 panels).")
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
        help="Output directory (default: analysis/results/calibration/tiles/endeavour)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N log files (debug).")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[5]
    cal_root = (args.cal_root or (repo / "data" / "clm_calibration")).resolve()
    if not cal_root.is_dir():
        print(f"Missing calibration root {cal_root}", file=sys.stderr)
        return 1

    out_dir = (
        args.out_dir
        or (Path(__file__).resolve().parent.parent / "results" / "calibration" / "tiles" / "endeavour")
    ).resolve()

    cfg = default_config_dir()
    grid_path = cfg / "wavelength_grid.yaml"
    if not grid_path.is_file():
        print(f"Missing {grid_path}", file=sys.stderr)
        return 1
    wl_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))

    logs = iter_endeavour_log_files(cal_root)
    if args.limit is not None:
        logs = logs[: args.limit]

    ok, bad = 0, 0
    for path in logs:
        sn = _tile_sn_from_calibration_path(path)
        rel = path.relative_to(cal_root)
        safe_rel = re.sub(r"[^\w.\-]+", "_", rel.with_suffix("").as_posix().replace("/", "__"))
        out_name = f"{sn}__{safe_rel}.png" if sn else f"unknown__{safe_rel}.png"
        outp = out_dir / out_name
        try:
            df = load_endeavour_log_csv(path)
            if len(df) < 2:
                bad += 1
                continue
            title = f"{sn or 'unknown'} | {rel.as_posix()}"
            plot_five_panels(df, wl_grid, title, outp)
            ok += 1
        except Exception as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            bad += 1

    print(f"Wrote {ok} figures under {out_dir} ({bad} skipped).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(None))
