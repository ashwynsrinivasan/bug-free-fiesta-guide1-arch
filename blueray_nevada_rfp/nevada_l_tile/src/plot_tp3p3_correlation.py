#!/usr/bin/env python3
"""
TP3-3: laser current vs per-channel optical power and vs frequency error (single test point tree).

- Loads ``*TP3-3*Scan*.csv`` under ``TP3-3`` (includes PreBI/PostBI-style names).
- **Channel power**: bank total ``Power(mW)`` × normalized OSA linear power from
  ``OSAl_Power(dBm)`` (10^(dBm/10)) within each (file, tile, bank, Time) group.
- **Frequency error**: ``OSA_Wave(nm)`` vs ``wavelength_grid.yaml`` targets via
  ``_freq_error_nm_ghz_from_row`` (same as TP2-4 analysis).
- Rows: ``T_MUX(C)`` in ~50 °C (49.9–50.1), ``Set Laser(mA)`` > 0.
- Drops tiles whose laser-on rows are **only** 150 mA (no sweep).
- Optional ``--skip-filters``: skip ``filter.yaml`` cascade (recommended when TP2-6
  is absent, since the cascade is TP2-based).

Outputs under ``analysis/results/correlation/``:

  * ``tp3p3_current_vs_channel_power_combined.png``
  * ``tp3p3_current_vs_channel_power_by_channel.png``
  * ``tp3p3_current_vs_freq_error_combined.png``
  * ``tp3p3_current_vs_freq_error_by_channel.png``

Figure titles include the 2×2 Pearson correlation matrix for (current, y).
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
    default_config_dir,
    default_data_root,
    get_valid_tiles_onet,
    load_filters,
    tile_sn_from_csv_path,
)
from analyze_tp2p4_onet_sftp import _freq_error_nm_ghz_from_row

_TILE_SN = re.compile(r"(Y\d{10})")


def _tile_sn_from_tp33_path(csv_path: Path) -> str | None:
    sn = tile_sn_from_csv_path(csv_path)
    if sn is not None:
        return sn
    m = _TILE_SN.search(csv_path.name)
    return m.group(1) if m else None


# Per-channel mW after OSA split; bank total ~100–130 mW → channels often < 30 mW
TP3P3_CHANNEL_POWER_Y_TICKS = np.arange(0, 31, 5)
# TP3-3 freq error has long tails vs TP2-4-style ±50 GHz window
TP3P3_FREQ_ERROR_YLIM = (-250.0, 250.0)
TP3P3_FREQ_ERROR_Y_TICKS = np.arange(-250, 251, 50)


def load_tp3p3_scan(
    tp3_path: Path,
    wl_grid: dict,
    valid_tiles: set[str] | None,
) -> pd.DataFrame:
    """Laser-on rows @ T_MUX ~50C with Channel_Power_mW and Frequency_Error_GHz."""
    parts: list[pd.DataFrame] = []
    csv_files = sorted(tp3_path.glob("*TP3-3*Scan*.csv"))
    for csv_file in csv_files:
        tile = _tile_sn_from_tp33_path(csv_file)
        if tile is None:
            print(f"TP3-3 skip (no tile SN): {csv_file.name}", file=sys.stderr)
            continue
        if valid_tiles is not None and tile not in valid_tiles:
            continue
        try:
            df = pd.read_csv(csv_file)
            df["Tile_SN"] = tile
            df["_src"] = csv_file.name
            df["T_MUX(C)"] = pd.to_numeric(df["T_MUX(C)"], errors="coerce")
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue
            df["Set Laser(mA)"] = pd.to_numeric(df["Set Laser(mA)"], errors="coerce")
            df["Power(mW)"] = pd.to_numeric(df["Power(mW)"], errors="coerce")
            df["OSA_Wave(nm)"] = pd.to_numeric(df["OSA_Wave(nm)"], errors="coerce")
            df["Bank"] = pd.to_numeric(df["Bank"], errors="coerce").astype(int)
            df["Channel"] = pd.to_numeric(df["Channel"], errors="coerce").astype(int)
            df = df[df["Set Laser(mA)"] > 0].dropna(
                subset=["Set Laser(mA)", "Power(mW)", "OSA_Wave(nm)", "Bank", "Channel"]
            )
            if df.empty:
                continue
            parts.append(df)
        except Exception as e:
            print(f"TP3-3 skip {csv_file.name}: {e}", file=sys.stderr)
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    return _add_channel_power_and_freq_error(raw, wl_grid)


def _add_channel_power_and_freq_error(df: pd.DataFrame, wl_grid: dict) -> pd.DataFrame:
    out = df.copy()
    if "Time" not in out.columns:
        out["Time"] = out.groupby(["_src", "Tile_SN", "Bank"]).cumcount().astype(str)
    dbm = pd.to_numeric(out["OSAl_Power(dBm)"], errors="coerce")
    lin = np.power(10.0, dbm / 10.0)
    lin = np.where(np.isfinite(lin) & (lin > 0), lin, np.nan)
    out["_osa_lin"] = lin
    gcols = ["_src", "Tile_SN", "Bank", "Time"]
    sums = out.groupby(gcols, observed=True)["_osa_lin"].transform("sum")
    n_ch = out.groupby(gcols, observed=True)["Channel"].transform("count")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = out["_osa_lin"] / sums
    use_uniform = (~np.isfinite(sums)) | (sums <= 0) | (~np.isfinite(frac))
    frac_uniform = 1.0 / n_ch.replace(0, np.nan)
    frac = np.where(use_uniform, frac_uniform, frac)
    p_bank = pd.to_numeric(out["Power(mW)"], errors="coerce")
    out["Channel_Power_mW"] = p_bank * frac
    out.drop(columns=["_osa_lin"], inplace=True)

    out["Frequency_Error_GHz"] = out.apply(
        lambda r: _freq_error_nm_ghz_from_row(r, wl_grid)["Frequency_Error_GHz"], axis=1
    )
    out = out.dropna(subset=["Channel_Power_mW", "Frequency_Error_GHz"])
    return out


def drop_tiles_all_channels_only_150ma(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    bad: set[str] = set()
    for tile, g in df.groupby("Tile_SN"):
        u = np.unique(np.round(g["Set Laser(mA)"].values.astype(float)))
        if u.size == 1 and float(u[0]) == 150.0:
            bad.add(tile)
    if not bad:
        return df
    print(f"Excluding {len(bad)} tiles with only 150 mA laser-on data (no sweep).")
    return df[~df["Tile_SN"].isin(bad)].copy()


def _channel_labels() -> list[str]:
    return [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)]


def _channel_order() -> list[tuple[int, int]]:
    return [(0, ch) for ch in range(8)] + [(1, ch) for ch in range(8)]


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


def _scatter_current_vs_y(
    ax,
    df: pd.DataFrame,
    *,
    y_col: str,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    ylabel: str,
    subtitle: str,
) -> None:
    x = df["Set Laser(mA)"].values.astype(float)
    y = df[y_col].values.astype(float)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.35, size=len(x))
    ax.scatter(x + jitter, y, s=6, alpha=0.22, c="#1565c0", edgecolors="none", rasterized=True)
    ax.set_xlabel("Set Laser (mA)", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    xm, xM = float(np.nanmin(x)), float(np.nanmax(x))
    pad = max(2.0, (xM - xm) * 0.03)
    ax.set_xlim(xm - pad, xM + pad)
    ax.grid(True, alpha=0.3)
    ax.set_title(subtitle, fontsize=8, loc="left")


def plot_current_vs_metric_combined(
    df: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    output_path: Path,
    title_tag: str,
) -> None:
    if df.empty:
        print(f"No data for combined plot {output_path.name}")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    x = df["Set Laser(mA)"].values.astype(float)
    y = df[y_col].values.astype(float)
    corr_line = _pearson_corr_matrix_line(x, y)
    full_title = f"{title_tag}\ncorr (I_mA, y): {corr_line}"
    _scatter_current_vs_y(
        ax, df, y_col=y_col, ylo=ylo, yhi=yhi, yticks=yticks, ylabel=ylabel, subtitle=""
    )
    ax.set_title("")
    fig.suptitle(full_title, fontsize=10, y=1.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_current_vs_metric_by_channel(
    df: pd.DataFrame,
    *,
    y_col: str,
    ylabel: str,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    output_path: Path,
    title_tag: str,
) -> None:
    labels = _channel_labels()
    order = _channel_order()
    sns.set_style("whitegrid")
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.15 * n), layout="constrained", sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (bank, ch), lab in zip(axes, order, labels):
        sub = df[(df["Bank"] == bank) & (df["Channel"] == ch)]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(lab, fontsize=9, fontweight="bold")
            ax.set_ylim(ylo, yhi)
            ax.set_title(f"{lab}: corr n/a", fontsize=8, loc="left")
            continue
        x = sub["Set Laser(mA)"].values.astype(float)
        y = sub[y_col].values.astype(float)
        cl = _pearson_corr_matrix_line(x, y)
        _scatter_current_vs_y(
            ax,
            sub,
            y_col=y_col,
            ylo=ylo,
            yhi=yhi,
            yticks=yticks,
            ylabel=f"{lab}\n{ylabel}",
            subtitle="",
        )
        ax.set_title(f"{lab}  corr (I,y): {cl}", fontsize=7, loc="left")
    axes[-1].set_xlabel("Set Laser (mA)", fontsize=10, fontweight="bold")
    fig.suptitle(title_tag, fontsize=10, y=1.002)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TP3-3 current vs OSA-split power / freq error correlation plots.")
    parser.add_argument("--data-root", type=Path, default=None, help="ONET SFTP root (default: monorepo data/clm_data_onet_sftp)")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output directory (default: blueray_nevada_rfp/nevada_l_tile/results/correlation)",
    )
    parser.add_argument("--skip-filters", action="store_true", help="Skip filter.yaml tile cascade (TP2-based).")
    args = parser.parse_args(argv)

    data_root = (args.data_root or default_data_root()).resolve()
    tp3 = data_root / "TP3-3"
    if not tp3.is_dir():
        print(f"No TP3-3 at {tp3}", file=sys.stderr)
        return 1

    config_dir = default_config_dir()
    grid_path = config_dir / "wavelength_grid.yaml"
    if not grid_path.is_file():
        print(f"Missing {grid_path}", file=sys.stderr)
        return 1
    wl_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))

    valid_tiles: set[str] | None = None
    if not args.skip_filters:
        filters = load_filters(config_dir)
        print("Applying filter.yaml cascade for tile allow-list …")
        valid_tiles = get_valid_tiles_onet(data_root, filters, wl_grid)
        if not valid_tiles:
            print(
                "No tiles passed filters (cascade uses TP2 data). Use --skip-filters for TP3-3-only trees.",
                file=sys.stderr,
            )
            return 1
        print(f"  {len(valid_tiles)} tiles after filters.")

    df = load_tp3p3_scan(tp3, wl_grid, valid_tiles)
    df = drop_tiles_all_channels_only_150ma(df)
    if df.empty:
        print("No TP3-3 rows after filters / T_MUX / laser-on.", file=sys.stderr)
        return 1
    print(
        f"TP3-3: {len(df)} rows, {df['Tile_SN'].nunique()} tiles, "
        f"currents (mA) unique: {sorted(df['Set Laser(mA)'].round().astype(int).unique())}"
    )

    results = (
        args.results or (Path(__file__).resolve().parent.parent / "results" / "correlation")
    ).resolve()
    results.mkdir(parents=True, exist_ok=True)
    tag = "ONET SFTP (tiles passing filter.yaml)" if valid_tiles else "ONET SFTP (--skip-filters)"
    ptag = f"{tag}; TP3-3; T_MUX~50C; channel mW = bank Power × OSA linear share"

    plot_current_vs_metric_combined(
        df,
        y_col="Channel_Power_mW",
        ylabel="Channel power (mW), OSA split",
        ylo=0.0,
        yhi=30.0,
        yticks=TP3P3_CHANNEL_POWER_Y_TICKS,
        output_path=results / "tp3p3_current_vs_channel_power_combined.png",
        title_tag=f"{ptag}; excluded 150-only sweep tiles",
    )
    plot_current_vs_metric_by_channel(
        df,
        y_col="Channel_Power_mW",
        ylabel="mW",
        ylo=0.0,
        yhi=30.0,
        yticks=TP3P3_CHANNEL_POWER_Y_TICKS,
        output_path=results / "tp3p3_current_vs_channel_power_by_channel.png",
        title_tag=f"{ptag}; by bank-channel",
    )
    ylo_f, yhi_f = TP3P3_FREQ_ERROR_YLIM
    plot_current_vs_metric_combined(
        df,
        y_col="Frequency_Error_GHz",
        ylabel="Frequency error (GHz)",
        ylo=ylo_f,
        yhi=yhi_f,
        yticks=TP3P3_FREQ_ERROR_Y_TICKS,
        output_path=results / "tp3p3_current_vs_freq_error_combined.png",
        title_tag=f"{ptag}; OSA λ vs grid; y-axis ±250 GHz (long tails)",
    )
    plot_current_vs_metric_by_channel(
        df,
        y_col="Frequency_Error_GHz",
        ylabel="GHz",
        ylo=ylo_f,
        yhi=yhi_f,
        yticks=TP3P3_FREQ_ERROR_Y_TICKS,
        output_path=results / "tp3p3_current_vs_freq_error_by_channel.png",
        title_tag=f"{ptag}; freq error vs current",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
