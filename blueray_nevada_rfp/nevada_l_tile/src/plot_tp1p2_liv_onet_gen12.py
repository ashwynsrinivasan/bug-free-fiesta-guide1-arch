#!/usr/bin/env python3
"""
Reproduce ips_clm_gen1-style ``tp1p2_liv_summary.png`` (median LIV + representative curve + insets)
using ONET SFTP data:

  * **Gen1 (LM):** ``data/clm_data_onet_sftp/oPO Data/3pcs OPO Module V1/LM/TP1-2`` … *LIV.csv
  * **Gen2 (ONET root):** ``data/clm_data_onet_sftp/TP1-2`` … *LIV.csv

By default applies the same ``filter.yaml`` cascade as ``analyze_tp2p4_onet_sftp`` (TP2-6 → TP2-5 → TP2-4)
**separately** on each data root, then loads only LIV rows for tiles that pass that root’s cascade.
Use ``--skip-filters`` to include every *LIV.csv row that loads.

Writes:
  ``blueray_nevada_rfp/nevada_l_tile/results/LIV/tp1p2_liv_summary_gen1_lm_gen2_onet.png``
  ``blueray_nevada_rfp/nevada_l_tile/results/LIV/tp1p2_liv_gen1_lm_gen2_onet.png`` (simple overlay, no insets)

Optical power y-axis is fixed to **0–40 mW** (main panel and insets).
Inset violins at 120 / 135 / 150 / 165 mA use **1.5×IQR** outlier filtering on
``Power(mW)`` (``tpanalysis._filter_tp1p2_inset_powers``); main L–I curve is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from analyze_tp2p4_onet_sftp import (
    default_config_dir,
    default_data_root,
    default_gen1_lm_data_root,
    get_valid_tiles_onet,
    load_filters,
)
from tp2p4_osa_power_reconcile import tile_sn_from_tp2p4_path


def _ips_clm_mfg_data() -> Path:
    """…/ips_clm_gen1/clm_mfg_data (for reusing tpanalysis plotters)."""
    here = Path(__file__).resolve()
    arch = here.parents[3]
    return (arch / "ips_clm_gen1" / "clm_mfg_data").resolve()


def _import_tpanalysis():
    analysis_src = _ips_clm_mfg_data() / "analysis_src"
    if not (analysis_src / "analyze_test_points.py").is_file():
        raise FileNotFoundError(f"analyze_test_points.py not found under {analysis_src}")
    sys.path.insert(0, str(analysis_src))
    from analyze_test_points import tpanalysis  # noqa: WPS433

    return tpanalysis


def load_tp1p2_liv_csvs(tp_path: Path, valid_tiles: set[str] | None, cohort: str) -> pd.DataFrame:
    """Load ``*LIV.csv`` under ``tp_path``; optional tile allow-list; tag ``Version`` = *cohort*."""
    all_data: list[pd.DataFrame] = []
    if not tp_path.is_dir():
        print(f"  Warning: TP1-2 path missing: {tp_path}")
        return pd.DataFrame()

    for csv_file in sorted(tp_path.glob("*LIV.csv")):
        tile_sn = tile_sn_from_tp2p4_path(csv_file)
        if tile_sn is None:
            continue
        if valid_tiles is not None and tile_sn not in valid_tiles:
            continue
        try:
            df = pd.read_csv(csv_file)
            df["Tile_SN"] = tile_sn
            df["Version"] = cohort
            df["Set Laser(mA)"] = pd.to_numeric(df["Set Laser(mA)"], errors="coerce")
            df["Power(mW)"] = pd.to_numeric(df["Power(mW)"], errors="coerce")
            df["Bank"] = pd.to_numeric(df["Bank"], errors="coerce")
            df["Channel"] = pd.to_numeric(df["Channel"], errors="coerce")
            df = df.dropna(subset=["Set Laser(mA)", "Power(mW)", "Bank", "Channel"])
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"  Warning: skip {csv_file.name}: {e}")

    if not all_data:
        return pd.DataFrame()
    out = pd.concat(all_data, ignore_index=True)
    n_tiles = out["Tile_SN"].nunique()
    print(f"  LIV {cohort}: {len(out)} rows, {n_tiles} tiles from {tp_path.name}/")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TP1-2 LIV summary: ONET gen2 + LM gen1 (ONET SFTP).")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output root (default: blueray_nevada_rfp/nevada_l_tile/results)",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Do not apply filter.yaml; load all *LIV.csv rows that parse.",
    )
    args = parser.parse_args(argv)

    onet_root = default_data_root().resolve()
    lm_root = default_gen1_lm_data_root().resolve()
    config_dir = default_config_dir()
    grid_path = (config_dir / "wavelength_grid.yaml").resolve()
    if not grid_path.is_file():
        print(f"Missing wavelength grid: {grid_path}", file=sys.stderr)
        return 1

    wl_grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    filters = None
    valid_onet: set[str] | None = None
    valid_lm: set[str] | None = None

    if args.skip_filters:
        print("Skipping filter.yaml; using all LIV files that load.")
        valid_onet = None
        valid_lm = None
    else:
        filters = load_filters(config_dir)
        print(f"Applying filter.yaml per data root (ONET root vs LM) …")
        valid_onet = get_valid_tiles_onet(onet_root, filters, wl_grid)
        valid_lm = get_valid_tiles_onet(lm_root, filters, wl_grid)
        print(f"  ONET SFTP (gen2): {len(valid_onet)} tiles pass")
        print(f"  LM (gen1): {len(valid_lm)} tiles pass")

    tp1_onet = onet_root / "TP1-2"
    tp1_lm = lm_root / "TP1-2"

    df_onet = load_tp1p2_liv_csvs(tp1_onet, valid_onet, "gen2_onet")
    df_lm = load_tp1p2_liv_csvs(tp1_lm, valid_lm, "gen1_lm")
    parts = [d for d in (df_onet, df_lm) if not d.empty]
    if not parts:
        print("No LIV data loaded; nothing to plot.")
        return 1

    combined = pd.concat(parts, ignore_index=True)
    n_tot = combined["Tile_SN"].nunique()
    n_g2 = int(combined.loc[combined["Version"] == "gen2_onet", "Tile_SN"].nunique()) if not df_onet.empty else 0
    n_g1 = int(combined.loc[combined["Version"] == "gen1_lm", "Tile_SN"].nunique()) if not df_lm.empty else 0
    print(
        f"Combined LIV: {len(combined)} rows, {n_tot} distinct tiles "
        f"(gen2 ONET: {n_g2}; gen1 LM: {n_g1})."
    )

    results = (args.results or (Path(__file__).resolve().parent.parent / "results")).resolve()
    liv_dir = results / "LIV"
    liv_dir.mkdir(parents=True, exist_ok=True)

    TPanalysis = _import_tpanalysis()
    tp = TPanalysis(_ips_clm_mfg_data())

    summary_path = liv_dir / "tp1p2_liv_summary_gen1_lm_gen2_onet.png"
    simple_path = liv_dir / "tp1p2_liv_gen1_lm_gen2_onet.png"
    ylim_mw = (0.0, 40.0)
    tp._plot_tp1p2_liv_overlay(combined, summary_path, power_ylim=ylim_mw)
    tp._plot_tp1p2_liv_simple(combined, simple_path, power_ylim=ylim_mw)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
