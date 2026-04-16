#!/usr/bin/env python3
"""
EVT TP2-4 / TP2-5 / TP2-6 from ips_clm_gen1 ``clm_mfg_data_v1`` / ``v2`` → Blueray ``analysis/results``.

Uses the same figures as full ONET / gen1 analysis:

- ``plot_tp2p4_freq_error_distribution`` → ``evt_tp2p4_distribution_vs_freq_error.png``,
  plus ``plot_freq_error_distribution_combined_banks`` → ``evt_tp2p4_distribution_vs_freq_error_combined_banks.png``;
  ``evt_tp2p5_distribution_vs_freq_error.png`` and ``evt_tp2p5_distribution_vs_freq_error_combined_banks.png`` when TP2-5 rows exist.
- ``plot_tp2p6_power_*`` → ``evt_tp2p6_power_tile_vs_power.png``,
  ``evt_tp2p6_power_distribution_vs_power.png``,
  ``evt_tp2p6_power_distribution_vs_power_combined_banks.png`` (when TP2-6 rows exist).

Tile list: ``analysis/config/evt_tp_scan_tiles.yaml`` (key ``tile_sn``, **slot order** 1..N unless
``clm_mfg_data/analysis_src/tile_module_slot.yaml`` maps ``Tile_SN`` → slot). **Only** ``Tile_SN`` whose
``tile_id`` passes the **30C OFC** min-channel MPD gate (≥ 12 mW; see ``evt_plot_filters``) are included
in MFG figures — TP2-6 / TP2-4 powers are **not** used for this filter.

Also writes OSA vs fiber reconcile tables (CSV still in mW) and scaled channel-power distribution figures:

- ``evt_tp2p4_osa_power_bank_summary.csv``
- ``evt_tp2p4_osa_power_per_channel.csv``
- ``evt_tp2p4_osa_scaled_power_distribution_vs_power.png`` (per-channel y-axis **mW**, 0–25)
- ``evt_tp2p4_osa_scaled_power_distribution_vs_power_combined_banks.png`` (pooled **mW**, y-axis 0–25)

TP2-6 EVT power PNGs use y-axis **Power in fiber (mW)**.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from analyze_tp2p4_onet_sftp import (
    plot_freq_error_distribution_combined_banks,
    plot_tp2p4_freq_error_distribution,
    plot_tp2p4_osa_scaled_power_distribution,
    plot_tp2p4_osa_scaled_power_distribution_combined_banks,
    plot_tp2p6_power_distribution,
    plot_tp2p6_power_distribution_combined_banks,
    plot_tp2p6_power_tiles,
)
from evt_30c_common import (
    build_30c_power_frame,
    default_clm_mfg_data_base,
    default_ofc_excel_path,
    evt_tile_sn_from_30c_passing_ids,
)
from evt_plot_filters import EVT_MIN_OPTICAL_POWER_MW, tile_ids_passing_min_channel_mpd
from tp2p4_osa_power_reconcile import collect_evt_tp2p4_osa_data, write_evt_tp2p4_osa_csvs


def default_mfg_base() -> Path:
    return default_clm_mfg_data_base()


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


def default_results_dir() -> Path:
    """``blueray_nevada_f2f/analysis/results`` (same tree as ``tp2p4_*.png``)."""
    return Path(__file__).resolve().parent.parent / "results"


def load_evt_tile_sn(config_dir: Path) -> list[str]:
    p = config_dir / "evt_tp_scan_tiles.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"EVT tile list not found: {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = doc.get("tile_sn") or []
    return [str(x).strip() for x in raw if x is not None and str(x).strip()]


# MFG v1 often uses Y + 8 digits (e.g. Y25170083); ONET exports may use Y + 10 digits.
_TILE_SN = re.compile(r"^Y\d{8,10}$")


def _tile_sn_from_tp2p4_filename(csv_path: Path) -> str | None:
    for part in csv_path.stem.split("-"):
        if _TILE_SN.match(part):
            return part
    return None


def load_mfg_tp2p4_scan(tp4_dir: Path, wl_grid: dict, allow: set[str]) -> pd.DataFrame:
    """Match ``tpanalysis._load_tp2p4_data`` (filename Tile_SN, T_MUX ~50C)."""
    all_data: list[pd.DataFrame] = []
    c = 299792458 * 1e9
    for csv_file in sorted(glob.glob(str(tp4_dir / "*TP2-4 Scan.csv"))):
        p = Path(csv_file)
        try:
            tile_sn = _tile_sn_from_tp2p4_filename(p)
            if tile_sn is None or tile_sn not in allow:
                continue
            df = pd.read_csv(csv_file)
            df["Tile_SN"] = tile_sn
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()

            def calc_freq_error(row):
                bank = row["Bank"]
                channel = row["Channel"]
                measured_wl = row["OSA_Wave(nm)"]
                bank_key = f"bank{bank}"
                grid_num = channel + 1
                target_wl = wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"]
                wl_error_nm = measured_wl - target_wl
                freq_error_hz = -(c / (target_wl**2)) * wl_error_nm
                freq_error_ghz = freq_error_hz / 1e9
                return pd.Series(
                    {"Wavelength_Error_nm": wl_error_nm, "Frequency_Error_GHz": freq_error_ghz}
                )

            df[["Wavelength_Error_nm", "Frequency_Error_GHz"]] = df.apply(calc_freq_error, axis=1)
            df = df[
                [
                    "Tile_SN",
                    "Bank",
                    "Channel",
                    "T_MUX(C)",
                    "OSA_Wave(nm)",
                    "Wavelength_Error_nm",
                    "Frequency_Error_GHz",
                ]
            ].copy()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_mfg_tp2p5_scan(tp5_dir: Path, wl_grid: dict, allow: set[str]) -> pd.DataFrame:
    """Match ``tpanalysis._load_tp2p5_data`` (CSV ``Tile_SN`` column)."""
    all_data: list[pd.DataFrame] = []
    c = 299792458 * 1e9
    for csv_file in sorted(glob.glob(str(tp5_dir / "*TP2-5 Scan.csv"))):
        try:
            df = pd.read_csv(csv_file)
            df = df[df["Tile_SN"].isin(allow)].copy()
            if df.empty:
                continue
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue

            def calc_freq_error(row):
                bank = row["Bank"]
                channel = row["Channel"]
                measured_wl = row["OSA_Wave(nm)"]
                bank_key = f"bank{bank}"
                grid_num = channel + 1
                target_wl = wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"]
                wl_error = measured_wl - target_wl
                freq_error_hz = -(c / (target_wl**2)) * wl_error
                return freq_error_hz / 1e9

            df["Frequency_Error_GHz"] = df.apply(calc_freq_error, axis=1)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_mfg_tp2p6_scan(tp6_dir: Path, allow: set[str]) -> pd.DataFrame:
    """Match ``load_tp2p6_onet`` / gen1 ``_load_tp2p6_data``: laser-on rows; Tile_SN from filename stem."""
    all_data: list[pd.DataFrame] = []
    for csv_file in sorted(glob.glob(str(tp6_dir / "*TP2-6 Test.csv"))):
        p = Path(csv_file)
        try:
            tile_sn = _tile_sn_from_tp2p4_filename(p)
            if tile_sn is None or tile_sn not in allow:
                continue
            df = pd.read_csv(csv_file)
            df = df[df["Set Laser(mA)"] > 0].copy()
            if df.empty:
                continue
            df["Tile_SN"] = tile_sn
            df = df[["Tile_SN", "Bank", "Channel", "Power(mW)"]].copy()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EVT TP2-4/5/6 from MFG data → Blueray distribution PNGs")
    p.add_argument("--mfg-base", type=Path, default=None, help="clm_mfg_data directory")
    p.add_argument(
        "--ofc-excel",
        type=Path,
        default=None,
        help="ofc_data.xlsx (30C tab); default under ips_clm_gen1/.../temperature_aggressors — EVT ≥12 mW gate",
    )
    p.add_argument("--config-dir", type=Path, default=None, help="Blueray config (evt yaml + wavelength_grid)")
    p.add_argument("--results", type=Path, default=None, help="Output dir (default: analysis/results)")
    p.add_argument(
        "--skip-osa-csv",
        action="store_true",
        help="Do not write evt_tp2p4_osa_power_*.csv under results",
    )
    args = p.parse_args(argv)

    mfg = (args.mfg_base or default_mfg_base()).resolve()
    config_dir = (args.config_dir or default_config_dir()).resolve()
    results = (args.results or default_results_dir()).resolve()
    results.mkdir(parents=True, exist_ok=True)

    allow_list = load_evt_tile_sn(config_dir)
    if not allow_list:
        print("evt_tp_scan_tiles.yaml has empty tile_sn", file=sys.stderr)
        return 1

    excel = (args.ofc_excel or default_ofc_excel_path()).resolve()
    if not excel.is_file():
        print(f"30C OFC Excel not found (required for EVT tile gate): {excel}", file=sys.stderr)
        return 1
    df_30c = pd.read_excel(excel, sheet_name="30C")
    df_power = build_30c_power_frame(df_30c)
    passing_ids = tile_ids_passing_min_channel_mpd(df_power, min_mw=EVT_MIN_OPTICAL_POWER_MW)
    allow = evt_tile_sn_from_30c_passing_ids(passing_ids, allow_list, mfg)
    if not allow:
        print(
            "No EVT Tile_SN left after 30C OFC min-channel MPD gate "
            f"(≥ {EVT_MIN_OPTICAL_POWER_MW} mW) and yaml/slot mapping; exiting.",
            file=sys.stderr,
        )
        return 1
    print(
        f"MFG EVT: 30C OFC ≥{EVT_MIN_OPTICAL_POWER_MW} mW → {len(passing_ids)} tile_id(s) → "
        f"{len(allow)} Tile_SN"
    )

    osa_summary, osa_ch = collect_evt_tp2p4_osa_data(mfg_base=mfg, evt_tile_sn=allow)
    if not args.skip_osa_csv:
        write_evt_tp2p4_osa_csvs(osa_summary, osa_ch, results)

    osa_plot = osa_ch.dropna(subset=["OSA_Power_mW_scaled"]).copy()
    if not osa_plot.empty:
        osa_plot["Version"] = osa_plot["mfg_version"]
        plot_tp2p4_osa_scaled_power_distribution(
            osa_plot,
            results / "evt_tp2p4_osa_scaled_power_distribution_vs_power.png",
        )
        plot_tp2p4_osa_scaled_power_distribution_combined_banks(
            osa_plot,
            results / "evt_tp2p4_osa_scaled_power_distribution_vs_power_combined_banks.png",
        )
    else:
        print("No EVT TP2-4 OSA scaled rows; skip evt_tp2p4_osa_scaled_power_distribution_*.png")

    grid_path = config_dir / "wavelength_grid.yaml"
    wl_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))

    frames: list[pd.DataFrame] = []
    frames5: list[pd.DataFrame] = []
    frames6: list[pd.DataFrame] = []
    for version, sub in [("v1", mfg / "clm_mfg_data_v1"), ("v2", mfg / "clm_mfg_data_v2")]:
        tp4 = sub / "TP2-4"
        tp5 = sub / "TP2-5"
        tp6 = sub / "TP2-6"
        if tp4.is_dir():
            d4 = load_mfg_tp2p4_scan(tp4, wl_grid, allow)
            if not d4.empty:
                d4 = d4.copy()
                d4["Version"] = version
                frames.append(d4)
                print(f"TP2-4 {version}: {len(d4)} rows, {d4['Tile_SN'].nunique()} tiles")
        if tp5.is_dir():
            d5 = load_mfg_tp2p5_scan(tp5, wl_grid, allow)
            if not d5.empty:
                d5 = d5.copy()
                d5["Version"] = version
                frames5.append(d5)
                print(f"TP2-5 {version}: {len(d5)} rows, {d5['Tile_SN'].nunique()} tiles")
        if tp6.is_dir():
            d6 = load_mfg_tp2p6_scan(tp6, allow)
            if not d6.empty:
                d6 = d6.copy()
                d6["Version"] = version
                frames6.append(d6)
                print(f"TP2-6 {version}: {len(d6)} rows, {d6['Tile_SN'].nunique()} tiles")

    if frames:
        df4 = pd.concat(frames, ignore_index=True)
        out4 = results / "evt_tp2p4_distribution_vs_freq_error.png"
        plot_tp2p4_freq_error_distribution(df4, out4)
        plot_freq_error_distribution_combined_banks(
            df4,
            results / "evt_tp2p4_distribution_vs_freq_error_combined_banks.png",
        )
    else:
        print(
            "No EVT TP2-4 scan rows loaded; skip evt_tp2p4_distribution_vs_freq_error*.png",
        )

    if frames5:
        df5 = pd.concat(frames5, ignore_index=True)
        out5 = results / "evt_tp2p5_distribution_vs_freq_error.png"
        plot_tp2p4_freq_error_distribution(df5, out5)
        plot_freq_error_distribution_combined_banks(
            df5,
            results / "evt_tp2p5_distribution_vs_freq_error_combined_banks.png",
        )
    else:
        print(
            "No EVT TP2-5 scan rows loaded; skip evt_tp2p5_distribution_vs_freq_error*.png",
        )

    if frames6:
        df6 = pd.concat(frames6, ignore_index=True)
        plot_tp2p6_power_tiles(df6, results / "evt_tp2p6_power_tile_vs_power.png")
        plot_tp2p6_power_distribution(df6, results / "evt_tp2p6_power_distribution_vs_power.png")
        plot_tp2p6_power_distribution_combined_banks(
            df6, results / "evt_tp2p6_power_distribution_vs_power_combined_banks.png"
        )
    else:
        print("No EVT TP2-6 test rows loaded; skip evt_tp2p6_power_*.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
