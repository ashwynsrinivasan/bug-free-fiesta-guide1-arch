#!/usr/bin/env python3
"""
Enablence EC101 datasheet CSVs (LD + PD) → bank-channel box + violin plots (TP2-4 distribution style).

Reads ``data/enablence_data_sftp/ec101/*Datasheet_LD*.csv`` and matching ``*Datasheet_PD*.csv``,
merges on ``Chip ID``, ``Wavelengths``, and lane label (``Laser`` / ``PD``).

Metrics (per row, after merge):
  * ``LD IL (dB)``, ``PD IL (dB)``
  * ``LD_Freq_Error_GHz`` / ``PD_Freq_Error_GHz``: passband-aware error — mean optical frequency at
    ``λ ± (1dB PBW)/2`` vs grid target frequency for the lane (uses center λ and passband width).
  * ``LD_Freq_Error_linear_GHz`` / ``PD_Freq_Error_linear_GHz``: small-signal linear formula from
    center λ only (same spirit as TP2-4 / ``_freq_error_nm_ghz_from_row``).
  * ``LD_minus_PD_Freq_Error_GHz`` = LD − PD (passband-aware errors).

Also writes ``ec101_merged_long.csv`` under ``analysis/results/ec101/``.
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
import yaml

from analyze_tp2p4_onet_sftp import (
    COMBINED_BANKS_FIGSIZE,
    DIST_FIGSIZE,
    FREQ_ERROR_Y_TICKS,
    _GEN1_BOXPLOT_KW,
    _combined_banks_strip_xaxis,
    _force_white_box_faces,
    _grey_violin_behind_box,
    _plot_combined_banks_distribution,
    _raise_boxplot_zorder,
    _save_single_figure,
)

_C = 299792458.0 * 1e9  # same convention as ``_freq_error_nm_ghz_from_row``


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[5]


def default_ec101_data_dir() -> Path:
    return (_repo_root_from_here() / "data" / "enablence_data_sftp" / "ec101").resolve()


def default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "ec101"


def default_wl_grid_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "wavelength_grid.yaml"


def load_wl_grid(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_LANE_RE = re.compile(r"^(\d+)([ABab])$")


def laser_to_bank_channel(lane: str) -> tuple[int, int] | None:
    """``1A``..``8A`` → bank 0 ch 0..7; ``1B``..``8B`` → bank 1."""
    m = _LANE_RE.match(str(lane).strip())
    if not m:
        return None
    n = int(m.group(1))
    letter = m.group(2).upper()
    if n < 1 or n > 8:
        return None
    bank = 0 if letter == "A" else 1
    return bank, n - 1


def _wl_nm_to_freq_thz(wl_nm: float) -> float:
    return (299792458.0 / (float(wl_nm) * 1e-9)) / 1e12


def freq_error_ghz_linear_from_center_wl(measured_wl_nm: float, bank: int, channel: int, wl_grid: dict) -> float:
    bank_key = f"bank{int(bank)}"
    grid_num = int(channel) + 1
    target_wl = float(wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"])
    wl_error_nm = float(measured_wl_nm) - target_wl
    freq_error_hz = -(_C / (target_wl**2)) * wl_error_nm
    return float(freq_error_hz / 1e9)


def freq_error_ghz_from_wl_and_passband(
    measured_wl_nm: float, passband_nm: float, bank: int, channel: int, wl_grid: dict
) -> float:
    """Mean THz at λ ± PBW/2 vs target grid THz → error in GHz (uses both center λ and 1dB width)."""
    lam_c = float(measured_wl_nm)
    d = float(passband_nm) / 2.0
    lam_lo = lam_c - d
    lam_hi = lam_c + d
    if lam_lo <= 0 or lam_hi <= 0:
        return float("nan")
    f_lo = _wl_nm_to_freq_thz(lam_lo)
    f_hi = _wl_nm_to_freq_thz(lam_hi)
    fc = (f_lo + f_hi) / 2.0
    bank_key = f"bank{int(bank)}"
    grid_num = int(channel) + 1
    tgt_wl = float(wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"])
    ft = _wl_nm_to_freq_thz(tgt_wl)
    return float((fc - ft) * 1000.0)


def paired_pd_path(ld_csv: Path) -> Path | None:
    if "Datasheet_LD" not in ld_csv.name:
        return None
    pd_name = ld_csv.name.replace("Datasheet_LD", "Datasheet_PD")
    p = ld_csv.parent / pd_name
    return p if p.is_file() else None


def load_merged_ec101(data_dir: Path) -> pd.DataFrame:
    ld_files = sorted(data_dir.glob("*Datasheet_LD*.csv"))
    parts: list[pd.DataFrame] = []
    for lp in ld_files:
        pp = paired_pd_path(lp)
        if pp is None:
            print(f"EC101: skip LD without PD: {lp.name}", file=sys.stderr)
            continue
        ld = pd.read_csv(lp)
        pd_df = pd.read_csv(pp)
        tag = lp.stem.replace("Testing_Report_Lightmatter_Datasheet_LD_", "")
        pd_side = pd_df.rename(columns={"PD": "Laser"})[
            [
                "Chip ID",
                "Wavelengths",
                "Laser",
                "PD IL (dB)",
                "PD Wavelength (nm)",
                "PD 1dB Passband Width (nm)",
            ]
        ]
        m = ld.merge(pd_side, on=["Chip ID", "Wavelengths", "Laser"], how="inner")
        m["Report_Batch"] = tag
        if m.empty:
            print(f"EC101: empty merge for {lp.name} + {pp.name}", file=sys.stderr)
            continue
        parts.append(m)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def add_bank_channel_and_metrics(df: pd.DataFrame, wl_grid: dict) -> pd.DataFrame:
    out = df.copy()
    bc = out["Laser"].map(laser_to_bank_channel)
    out["Bank"] = bc.map(lambda x: x[0] if x else np.nan)
    out["Channel"] = bc.map(lambda x: x[1] if x else np.nan)
    out = out.dropna(subset=["Bank", "Channel"])
    out["Bank"] = out["Bank"].astype(int)
    out["Channel"] = out["Channel"].astype(int)

    out["LD_Freq_Error_linear_GHz"] = out.apply(
        lambda r: freq_error_ghz_linear_from_center_wl(
            r["LD Wavelength (nm)"], r["Bank"], r["Channel"], wl_grid
        ),
        axis=1,
    )
    out["PD_Freq_Error_linear_GHz"] = out.apply(
        lambda r: freq_error_ghz_linear_from_center_wl(
            r["PD Wavelength (nm)"], r["Bank"], r["Channel"], wl_grid
        ),
        axis=1,
    )
    out["LD_Freq_Error_GHz"] = out.apply(
        lambda r: freq_error_ghz_from_wl_and_passband(
            r["LD Wavelength (nm)"],
            r["LD 1dB Passband Width (nm)"],
            r["Bank"],
            r["Channel"],
            wl_grid,
        ),
        axis=1,
    )
    out["PD_Freq_Error_GHz"] = out.apply(
        lambda r: freq_error_ghz_from_wl_and_passband(
            r["PD Wavelength (nm)"],
            r["PD 1dB Passband Width (nm)"],
            r["Bank"],
            r["Channel"],
            wl_grid,
        ),
        axis=1,
    )
    out["LD_minus_PD_Freq_Error_GHz"] = out["LD_Freq_Error_GHz"] - out["PD_Freq_Error_GHz"]
    return out


def _bank_channel_positions(df: pd.DataFrame, y_col: str) -> tuple[list[np.ndarray], list[int]]:
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in (0, 1):
        sub_b = df[df["Bank"] == bank]
        for channel in range(8):
            sub = sub_b[sub_b["Channel"] == channel]
            if sub.empty:
                continue
            v = sub[y_col].dropna().astype(float).values
            if v.size == 0:
                continue
            box_data.append(v)
            box_positions.append(bank * 8 + channel)
    return box_data, box_positions


def _y_limits_from_values(vals: np.ndarray, *, pad_frac: float = 0.12) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _nice_yticks(ylo: float, yhi: float, *, max_ticks: int = 11) -> np.ndarray:
    span = yhi - ylo
    if span <= 0:
        return np.array([ylo, yhi])
    raw = np.linspace(ylo, yhi, max_ticks)
    return raw


def plot_bank_channel_distribution(
    df: pd.DataFrame,
    y_col: str,
    output_path: Path,
    *,
    ylabel: str,
    ylo: float | None,
    yhi: float | None,
    yticks: np.ndarray | None,
    annotation_decimals: int,
    annotation_unit: str,
    title: str | None = None,
) -> None:
    box_data, box_positions = _bank_channel_positions(df, y_col)
    if not box_data:
        print(f"EC101: no data for {output_path.name}", file=sys.stderr)
        return
    pooled = np.concatenate(box_data)
    if ylo is None or yhi is None:
        ylo_a, yhi_a = _y_limits_from_values(pooled)
        ylo = ylo_a if ylo is None else ylo
        yhi = yhi_a if yhi is None else yhi
    if yticks is None:
        yticks = _nice_yticks(ylo, yhi)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    y_ann = ylo + (yhi - ylo) * 0.055
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        _grey_violin_behind_box(ax, vals, position=float(pos), width=0.62)
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        vert=True,
        widths=0.55,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    fmt = f"{{:.{annotation_decimals}f}}"
    u = annotation_unit
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        std = float(np.std(vals))
        ax.text(
            pos,
            y_ann,
            f"μ̃={fmt.format(med)}{u}\nσ={fmt.format(std)}{u}",
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=6,
            clip_on=False,
        )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 15.5)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    _save_single_figure(fig, output_path)


def plot_combined_banks_float(
    df: pd.DataFrame,
    y_col: str,
    output_path: Path,
    *,
    ylabel: str,
    ylo: float | None,
    yhi: float | None,
    yticks: np.ndarray | None,
    annotation_decimals: int,
    annotation_unit: str,
    empty_msg: str,
) -> None:
    vals = df[y_col].dropna().astype(float).values
    if vals.size == 0:
        print(empty_msg, file=sys.stderr)
        return
    if ylo is None or yhi is None:
        ylo_a, yhi_a = _y_limits_from_values(vals)
        ylo = ylo_a if ylo is None else ylo
        yhi = yhi_a if yhi is None else yhi
    if yticks is None:
        yticks = _nice_yticks(ylo, yhi)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=COMBINED_BANKS_FIGSIZE, layout="constrained")
    pos = 0.0
    _grey_violin_behind_box(ax, vals, position=pos)
    bp = ax.boxplot(
        [vals],
        positions=[pos],
        vert=True,
        widths=0.35,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    med = float(np.median(vals))
    std = float(np.std(vals))
    fmt = f"{{:.{annotation_decimals}f}}"
    u = annotation_unit
    ax.set_title(f"μ̃={fmt.format(med)}{u}, σ={fmt.format(std)}{u}", fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 0.5)
    _combined_banks_strip_xaxis(ax)
    _save_single_figure(fig, output_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EC101 Enablence LD/PD → bank-channel distributions")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--wl-grid", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    data_dir = (args.data_dir or default_ec101_data_dir()).resolve()
    wl_path = (args.wl_grid or default_wl_grid_path()).resolve()
    out_dir = (args.out_dir or default_results_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"EC101: data dir missing: {data_dir}", file=sys.stderr)
        return 1
    if not wl_path.is_file():
        print(f"EC101: wavelength grid missing: {wl_path}", file=sys.stderr)
        return 1

    wl_grid = load_wl_grid(wl_path)
    raw = load_merged_ec101(data_dir)
    if raw.empty:
        print("EC101: no merged rows", file=sys.stderr)
        return 1
    df = add_bank_channel_and_metrics(raw, wl_grid)
    csv_path = out_dir / "ec101_merged_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"EC101: wrote {csv_path} ({len(df)} rows)")

    fe_specs = [
        (
            "LD_Freq_Error_GHz",
            "ec101_distribution_ld_freq_error_ghz.png",
            "ec101_distribution_ld_freq_error_ghz_combined_banks.png",
            "LD frequency error (GHz)",
            -50.0,
            50.0,
            FREQ_ERROR_Y_TICKS,
            0,
            "GHz",
            "LD: mean f(λ ± PBW/2) vs grid (LD Wavelength + LD 1dB PBW)",
        ),
        (
            "PD_Freq_Error_GHz",
            "ec101_distribution_pd_freq_error_ghz.png",
            "ec101_distribution_pd_freq_error_ghz_combined_banks.png",
            "PD frequency error (GHz)",
            -50.0,
            50.0,
            FREQ_ERROR_Y_TICKS,
            0,
            "GHz",
            "PD: mean f(λ ± PBW/2) vs grid (PD Wavelength + PD 1dB PBW)",
        ),
    ]
    fe_specs_ld_pd_diff = (
        "LD_minus_PD_Freq_Error_GHz",
        "ec101_distribution_ld_minus_pd_freq_error_ghz.png",
        "ec101_distribution_ld_minus_pd_freq_error_ghz_combined_banks.png",
        "LD − PD frequency error (GHz)",
        "LD freq error minus PD freq error (same channel)",
    )

    il_specs = [
        ("LD IL (dB)", "ec101_distribution_ld_il_db.png", "ec101_distribution_ld_il_db_combined_banks.png", "LD IL (dB)"),
        ("PD IL (dB)", "ec101_distribution_pd_il_db.png", "ec101_distribution_pd_il_db_combined_banks.png", "PD IL (dB)"),
    ]

    for col, fn, fnc, ylab, ylo, yhi, yticks, dec, unit, ttl in fe_specs:
        plot_bank_channel_distribution(
            df,
            col,
            out_dir / fn,
            ylabel=ylab,
            ylo=ylo,
            yhi=yhi,
            yticks=yticks,
            annotation_decimals=dec,
            annotation_unit=unit,
            title=ttl,
        )
        _plot_combined_banks_distribution(
            out_dir / fnc,
            df[col].dropna().astype(float).values,
            ylo=ylo,
            yhi=yhi,
            yticks=yticks,
            ylabel=ylab,
            use_mean_for_annotation=False,
            annotation_unit=unit,
            empty_msg=f"EC101: empty {col} combined",
            annotation_center_decimals=None,
            annotation_std_decimals=0,
        )

    col, fn, fnc, ylab, ttl = fe_specs_ld_pd_diff
    ld_pd_vals = df[col].dropna().astype(float).values
    ylo_d, yhi_d = _y_limits_from_values(ld_pd_vals, pad_frac=0.12) if ld_pd_vals.size else (0.0, 1.0)
    plot_bank_channel_distribution(
        df,
        col,
        out_dir / fn,
        ylabel=ylab,
        ylo=ylo_d,
        yhi=yhi_d,
        yticks=_nice_yticks(ylo_d, yhi_d),
        annotation_decimals=2,
        annotation_unit=" GHz",
        title=ttl,
    )
    plot_combined_banks_float(
        df,
        col,
        out_dir / fnc,
        ylabel=ylab,
        ylo=ylo_d,
        yhi=yhi_d,
        yticks=_nice_yticks(ylo_d, yhi_d),
        annotation_decimals=2,
        annotation_unit=" GHz",
        empty_msg=f"EC101: empty {col} combined",
    )

    for col, fn, fnc, ylab in il_specs:
        plot_bank_channel_distribution(
            df,
            col,
            out_dir / fn,
            ylabel=ylab,
            ylo=None,
            yhi=None,
            yticks=None,
            annotation_decimals=2,
            annotation_unit=" dB",
        )
        vals = df[col].dropna().astype(float).values
        ylo_i, yhi_i = _y_limits_from_values(vals)
        plot_combined_banks_float(
            df,
            col,
            out_dir / fnc,
            ylabel=ylab,
            ylo=ylo_i,
            yhi=yhi_i,
            yticks=_nice_yticks(ylo_i, yhi_i),
            annotation_decimals=2,
            annotation_unit=" dB",
            empty_msg=f"EC101: empty {col} combined",
        )

    print(f"EC101: figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
