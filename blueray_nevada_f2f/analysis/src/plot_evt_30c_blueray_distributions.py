#!/usr/bin/env python3
"""
EVT OFC 30 °C sheet → Blueray-style violin + gen1 white box distributions.

Reads ``ips_clm_gen1/ips_clm_evt_ofc/temperature_aggressors/ofc_data.xlsx`` (tab 30C)
with the same parsing as ``module_analysis.temperature_aggressors_2.analyze_30C_*``.

Output: ``blueray_nevada_f2f/analysis/results/evt_*.png`` (same directory as other Blueray EVT figures).

Tiles are included only if **minimum** last-cycle MPD channel power is **≥ 12 mW** (see ``evt_plot_filters``).

X-axis: **Bank channel** (A-Ch1–8 then B-Ch1–8, same order as ``plot_tp2p4_freq_error_distribution``),
pooling all module tiles. Styling matches TP2p4 grey violin + gen1 box + μ̃/σ annotations.

Also writes **combined_banks** panels (single pooled column, no x labels), matching
``*_distribution_vs_freq_error_combined_banks.png`` / TP2-6 combined style.
``evt_module_distribution_vs_optical_power_dBm_30C_combined_banks.png`` uses the same pooled **mW**
data as the mW combined figure, with y-axis **0–20 mW** (``dBm`` in the basename is legacy).
"""
from __future__ import annotations

import ast
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_tp2p4_onet_sftp import _plot_combined_banks_distribution
from evt_30c_common import build_30c_power_frame, default_ofc_excel_path
from evt_plot_filters import EVT_MIN_OPTICAL_POWER_MW, filter_30c_power_and_freq_by_min_tile_power
from mfg_ofc_plotter import (
    SET_A_BANK,
    SET_B_BANK,
    _bank_channel_xticklabels_16,
    _ofc_violin_gen1_box_per_bank,
)

N_DWDM_CHANNELS = 8
N_BANK_CHANNEL_COLS = N_DWDM_CHANNELS * 2  # Set B (bank 0) columns 0–7, Set A (bank 1) 8–15

# Match Blueray distribution figures (e.g. analyze_tp2p4_onet_sftp.DIST_FIGSIZE)
FIGSIZE = (10, 5)
FREQ_ERROR_Y_TICKS = np.arange(-50, 51, 10)
EVT_30C_PNG_PREFIX = "evt_"


def default_excel_path() -> Path:
    return default_ofc_excel_path()


def default_output_dir() -> Path:
    """``blueray_nevada_f2f/analysis/results`` (with ``evt_*`` filenames)."""
    return (Path(__file__).resolve().parent.parent / "results").resolve()


def build_30c_freq_error_frame(df_30c: pd.DataFrame) -> pd.DataFrame:
    """Reference = min cycle; measured = max cycle; GHz error per channel (matches module_analysis)."""
    first_cycle = df_30c["cycle_number"].min()
    last_cycle = df_30c["cycle_number"].max()
    df_reference = df_30c[df_30c["cycle_number"] == first_cycle]
    df_last = df_30c[df_30c["cycle_number"] == last_cycle]

    ref_wavelengths: dict[tuple, list] = {}
    for _, row in df_reference.iterrows():
        tile_id = row["tile_id"]
        bank_type = row["bank_type"]
        try:
            wl_ref = ast.literal_eval(row["wavelength_nm"])
            wl_ref = [w * 1e-9 for w in wl_ref]
            ref_wavelengths[(tile_id, bank_type)] = wl_ref
        except (SyntaxError, TypeError, ValueError):
            continue

    c_speed_light = 299792458.0
    rows: list[dict] = []

    for _, row in df_last.iterrows():
        tile_id = row["tile_id"]
        bank_type = row["bank_type"]
        try:
            wavelengths_raw = ast.literal_eval(row["wavelength_nm"])
            wavelengths_nm = [w * 1e-9 for w in wavelengths_raw]
        except (SyntaxError, TypeError, ValueError):
            continue
        ref_wl = ref_wavelengths.get((tile_id, bank_type))
        if ref_wl is None:
            continue
        for channel_idx, wl_nm in enumerate(wavelengths_nm):
            if channel_idx >= len(ref_wl):
                break
            ref_wl_nm = ref_wl[channel_idx]
            if wl_nm <= 0 or ref_wl_nm <= 0:
                continue
            measured_freq_thz = c_speed_light / (wl_nm * 1e-9) / 1e12
            ref_freq_thz = c_speed_light / (ref_wl_nm * 1e-9) / 1e12
            freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000.0
            bank_csv = SET_A_BANK if bank_type == "BANK_A" else SET_B_BANK
            rows.append(
                {
                    "tile_id": int(tile_id),
                    "bank_csv": bank_csv,
                    "freq_error_ghz": float(freq_error_ghz),
                    "channel": channel_idx,
                }
            )
    return pd.DataFrame(rows)


def _collect_bank_channel_violin_series(df: pd.DataFrame, value_col: str) -> tuple[list, list, list[int]]:
    """Pool all modules: one violin per (bank_csv, channel), x = bank*8 + channel (bank 0 first)."""
    positions: list[int] = []
    values: list[np.ndarray] = []
    banks: list[int] = []
    for bank in (SET_B_BANK, SET_A_BANK):
        for ch in range(N_DWDM_CHANNELS):
            vals = df.loc[
                (df["bank_csv"] == bank) & (df["channel"] == ch), value_col
            ].dropna().values.astype(float)
            if vals.size == 0:
                continue
            positions.append(bank * N_DWDM_CHANNELS + ch)
            values.append(vals)
            banks.append(bank)
    return positions, values, banks


def plot_power_mw(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        print("30C power: empty frame, skip")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    pos, vals, banks = _collect_bank_channel_violin_series(df, "pic_mpd_value_mw")
    if not vals:
        plt.close(fig)
        print("30C power: no bank×channel values, skip")
        return
    yticks = np.arange(0, 21, 5)
    _ofc_violin_gen1_box_per_bank(
        ax,
        pos,
        vals,
        banks,
        ylo=0.0,
        yhi=20.0,
        yticks=yticks,
        annotation_unit="mW",
    )
    ax.set_xticks(np.arange(N_BANK_CHANNEL_COLS))
    ax.set_xticklabels(_bank_channel_xticklabels_16(), fontsize=9, rotation=45, ha="right")
    ax.set_xlim(-0.5, N_BANK_CHANNEL_COLS - 0.5)
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Power in fiber (mW)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 20.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_power_mw_combined_banks(df: pd.DataFrame, out: Path) -> None:
    vals = df["pic_mpd_value_mw"].dropna().values.astype(float) if not df.empty else np.array([])
    yticks = np.arange(0, 21, 5)
    _plot_combined_banks_distribution(
        out,
        vals,
        ylo=0.0,
        yhi=20.0,
        yticks=yticks,
        ylabel="Power in fiber (mW)",
        use_mean_for_annotation=False,
        annotation_unit="mW",
        empty_msg="30C power (combined banks): no data, skip",
    )


def plot_power_dbm(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        print("30C dBm: empty frame, skip")
        return
    d = df.copy()
    d["power_dbm"] = 10.0 * np.log10(np.maximum(d["pic_mpd_value_mw"].values, 1e-12))
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    pos, vals, banks = _collect_bank_channel_violin_series(d, "power_dbm")
    if not vals:
        plt.close(fig)
        print("30C dBm: no bank×channel values, skip")
        return
    yticks = np.arange(0, 16, 2)
    _ofc_violin_gen1_box_per_bank(
        ax,
        pos,
        vals,
        banks,
        ylo=0.0,
        yhi=15.0,
        yticks=yticks,
        annotation_unit="dBm",
    )
    ax.set_xticks(np.arange(N_BANK_CHANNEL_COLS))
    ax.set_xticklabels(_bank_channel_xticklabels_16(), fontsize=9, rotation=45, ha="right")
    ax.set_xlim(-0.5, N_BANK_CHANNEL_COLS - 0.5)
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Power in fiber (dBm)", fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_power_dbm_combined_banks(df: pd.DataFrame, out: Path) -> None:
    """Pooled **mW** (same as optical-power combined), y-axis 0–20 mW — matches Blueray mW convention."""
    if df.empty:
        print("30C dBm combined (mW axis): empty frame, skip")
        return
    vals = df["pic_mpd_value_mw"].dropna().values.astype(float)
    if vals.size == 0:
        print("30C dBm combined (mW axis): no data, skip")
        return
    yticks = np.arange(0, 21, 5)
    _plot_combined_banks_distribution(
        out,
        vals,
        ylo=0.0,
        yhi=20.0,
        yticks=yticks,
        ylabel="Power in fiber (mW)",
        use_mean_for_annotation=False,
        annotation_unit="mW",
        empty_msg="30C dBm combined (mW axis): no data, skip",
    )


def plot_freq_error(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        print("30C freq error: empty frame, skip")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    pos, vals, banks = _collect_bank_channel_violin_series(df, "freq_error_ghz")
    if not vals:
        plt.close(fig)
        print("30C freq error: no bank×channel values, skip")
        return
    _ofc_violin_gen1_box_per_bank(
        ax,
        pos,
        vals,
        banks,
        ylo=-50.0,
        yhi=50.0,
        yticks=FREQ_ERROR_Y_TICKS,
        annotation_unit="GHz",
    )
    ax.set_xticks(np.arange(N_BANK_CHANNEL_COLS))
    ax.set_xticklabels(_bank_channel_xticklabels_16(), fontsize=9, rotation=45, ha="right")
    ax.set_xlim(-0.5, N_BANK_CHANNEL_COLS - 0.5)
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=12, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_freq_error_combined_banks(df: pd.DataFrame, out: Path) -> None:
    vals = df["freq_error_ghz"].dropna().values.astype(float) if not df.empty else np.array([])
    _plot_combined_banks_distribution(
        out,
        vals,
        ylo=-50.0,
        yhi=50.0,
        yticks=FREQ_ERROR_Y_TICKS,
        ylabel="Frequency Error (GHz)",
        use_mean_for_annotation=False,
        annotation_unit="GHz",
        empty_msg="30C freq error (combined banks): no data, skip",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EVT 30C OFC → Blueray violin distribution PNGs")
    p.add_argument("--excel", type=Path, default=None, help="ofc_data.xlsx path")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG outputs")
    args = p.parse_args(argv)

    excel = (args.excel or default_excel_path()).resolve()
    out_dir = (args.output_dir or default_output_dir()).resolve()

    if not excel.is_file():
        print(f"Excel not found: {excel}", file=sys.stderr)
        return 1

    df_30c = pd.read_excel(excel, sheet_name="30C")
    df_power = build_30c_power_frame(df_30c)
    df_freq = build_30c_freq_error_frame(df_30c)
    df_power, df_freq = filter_30c_power_and_freq_by_min_tile_power(
        df_power, df_freq, min_mw=EVT_MIN_OPTICAL_POWER_MW
    )
    if df_power.empty:
        print(
            f"No 30C rows left after optical power ≥ {EVT_MIN_OPTICAL_POWER_MW} mW filter; exiting.",
            file=sys.stderr,
        )
        return 1

    p = EVT_30C_PNG_PREFIX
    plot_power_mw(df_power, out_dir / f"{p}module_distribution_vs_optical_power_30C.png")
    plot_power_mw_combined_banks(
        df_power,
        out_dir / f"{p}module_distribution_vs_optical_power_30C_combined_banks.png",
    )
    plot_power_dbm(df_power, out_dir / f"{p}module_distribution_vs_optical_power_dBm_30C.png")
    plot_power_dbm_combined_banks(
        df_power,
        out_dir / f"{p}module_distribution_vs_optical_power_dBm_30C_combined_banks.png",
    )
    plot_freq_error(df_freq, out_dir / f"{p}module_distribution_vs_freqerror_30C.png")
    plot_freq_error_combined_banks(
        df_freq,
        out_dir / f"{p}module_distribution_vs_freqerror_30C_combined_banks.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
