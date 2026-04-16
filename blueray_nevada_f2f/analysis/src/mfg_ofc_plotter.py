#!/usr/bin/env python3
"""
MFG OFC module distribution plots (TP2-5 frequency error, TP2-6 optical power, TP2-6 total power).

Why 16 vs 32 “fibers”
----------------------
- TP2-5 / TP2-6 CSVs only have ``Bank`` ∈ {0,1} and ``Channel`` ∈ {0..7}. That is **2 bank outputs × 8
  DWDM lanes = 16 series per tile** — not a bug in the data.
- A full **DWDM laser module** in EVT uses **fibers 1–32** = **16 tile slots × 2 banks** (Set A / Set B),
  same formula as ``ips_clm_evt_ofc/.../module_analysis.py``:
  ``fiber_num = (slot - 1) * 2 + 1`` for Set A (bank 1), ``+ 2`` for Set B (bank 0), ``slot`` = 1..16.
- TP2-6 files often have **32 rows per tile** because they include **16 laser-on + 16 laser-off** rows
  (one per bank×channel), not 32 output fibers.

To plot the **module fiber 1–32** axis for MFG tiles, add ``clm_mfg_data/analysis_src/tile_module_slot.yaml``
mapping each ``Tile_SN`` to module slot ``1..16`` (see ``tile_module_slot.example.yaml``). Without coverage,
plots fall back to **16 DWDM lanes** (F01–F16) so every CSV row maps to one x position.

Output: blueray_nevada_f2f/analysis/results/ofc/module_distribution_vs_*.png

Styling matches ``analyze_tp2p4_onet_sftp`` distribution panels: grey violin, white gen1 box,
red median, μ̃/σ annotations, ``Bank channel`` + A-Ch1–8 / B-Ch1–8 (16-lane mode), figure (10, 5).
Frequency error y-axis: −50…50 GHz (10 GHz ticks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Set A = bank 1, Set B = bank 0 — matches EVT / TP2p4 CSV ``Bank`` column
SET_A_BANK = 1
SET_B_BANK = 0

N_DWDM_LANES = 16  # 2 banks × 8 channels in CSV
N_MODULE_FIBERS = 32  # EVT: 16 slots × 2 banks
N_DWDM_CHANNELS = 8

# Match TP2p4 / Blueray distribution panels (e.g. ``plot_tp2p4_freq_error_distribution``)
DIST_FIGSIZE = (10, 5)


def _force_white_box_faces(bp: dict) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)


def _raise_boxplot_zorder(bp: dict, z: float = 4.0) -> None:
    for key in ("boxes", "medians", "whiskers", "caps", "fliers"):
        if key not in bp:
            continue
        for artist in bp[key]:
            artist.set_zorder(z)


def _grey_violin_at(ax, values: np.ndarray, *, position: float, width: float = 0.62) -> None:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    vp = ax.violinplot(
        [vals],
        positions=[position],
        vert=True,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#9e9e9e")
        body.set_edgecolor("#616161")
        body.set_alpha(0.48)
        body.set_zorder(0.4)
    for key in ("cbars", "cmins", "cmaxes"):
        if key not in vp:
            continue
        for ln in vp[key]:
            ln.set_visible(False)


def default_mfg_base() -> Path:
    here = Path(__file__).resolve()
    guide1 = here.parents[3]
    return (guide1 / "ips_clm_gen1" / "clm_mfg_data").resolve()


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "ofc"


def load_tile_module_slot_map(mfg_base: Path) -> dict[str, int]:
    """Tile_SN -> EVT module slot 1..16. File optional; empty dict if missing."""
    p = mfg_base / "analysis_src" / "tile_module_slot.yaml"
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not raw or not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        try:
            slot = int(v)
        except (TypeError, ValueError):
            continue
        out[key] = slot
    return out


def module_fiber_number(slot_1_to_16: int, bank: int) -> int:
    """Match EVT module_analysis: Set A (bank 1) → odd fiber, Set B (bank 0) → even fiber."""
    s = int(slot_1_to_16)
    b = int(bank)
    if b == SET_A_BANK:
        return (s - 1) * 2 + 1
    return (s - 1) * 2 + 2


def bank_from_module_fiber(fiber_1_to_32: int) -> int:
    return SET_A_BANK if (int(fiber_1_to_32) % 2 == 1) else SET_B_BANK


def slot_map_covers_frame(df: pd.DataFrame, slot_map: dict[str, int]) -> bool:
    if not slot_map or df.empty or "Tile_SN" not in df.columns:
        return False
    for sn in df["Tile_SN"].astype(str).unique():
        if sn in slot_map:
            return True
    return False


def add_module_fiber_column(df: pd.DataFrame, slot_map: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    sn = out["Tile_SN"].astype(str)
    slot = sn.map(slot_map)
    bank = out["Bank"].astype(int)
    mf = pd.Series(np.nan, index=out.index, dtype=float)
    slot_num = pd.to_numeric(slot, errors="coerce")
    ok = slot_num.notna() & (slot_num >= 1) & (slot_num <= 16)
    if ok.any():
        slot_i = slot_num[ok].astype(int)
        b = bank[ok]
        mf.loc[ok] = (slot_i - 1) * 2 + np.where(b.values == SET_A_BANK, 1, 2)
    out["ModuleFiber"] = mf
    return out


def _bank_channel_xticklabels_16() -> list[str]:
    """TP2p4 convention: bank 0 → A-Ch*, bank 1 → B-Ch*."""
    return [f"A-Ch{i}" for i in range(1, N_DWDM_CHANNELS + 1)] + [
        f"B-Ch{i}" for i in range(1, N_DWDM_CHANNELS + 1)
    ]


def _ofc_violin_gen1_box_per_bank(
    ax,
    box_data_positions: list,
    box_data_values: list,
    box_banks: list[int],
    *,
    ylo: float,
    yhi: float,
    yticks: np.ndarray | None = None,
    annotation_unit: str | None = None,
    use_median_in_annotation: bool = True,
    annotation_fontsize: int = 7,
    annotation_pad_frac: float = 0.055,
) -> None:
    """Grey violin + gen1 white box + red median; optional μ̃/σ text (``analyze_tp2p4`` distribution style)."""
    for pos, vals in zip(box_data_positions, box_data_values):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        _grey_violin_at(ax, vals, position=float(pos))

    bp = ax.boxplot(
        box_data_values,
        positions=box_data_positions,
        vert=True,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="red", linewidth=2.5),
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)

    if annotation_unit:
        y_ann = ylo + (yhi - ylo) * annotation_pad_frac
        u = annotation_unit
        for pos, vals in zip(box_data_positions, box_data_values):
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0:
                continue
            if use_median_in_annotation:
                c_stat = int(round(float(np.median(vals))))
                sym = "μ̃"
            else:
                c_stat = int(round(float(np.mean(vals))))
                sym = "μ"
            std = int(round(float(np.std(vals))))
            ax.text(
                pos,
                y_ann,
                f"{sym}={c_stat}{u}\nσ={std}{u}",
                fontsize=annotation_fontsize,
                ha="center",
                va="bottom",
                zorder=6,
                clip_on=False,
            )

    ax.set_ylim(ylo, yhi)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=0.3, axis="y")


def _collect_dwdm_lane_series(df: pd.DataFrame, value_column: str) -> tuple[list, list, list[int]]:
    """Bank 0 all channels, then bank 1 (same x order as ``plot_tp2p4_freq_error_distribution``)."""
    box_data_positions: list = []
    box_data_values: list = []
    box_banks: list[int] = []
    for bank in (SET_B_BANK, SET_A_BANK):
        df_bank = df[df["Bank"] == bank]
        for channel in range(N_DWDM_CHANNELS):
            d = df_bank[df_bank["Channel"] == channel]
            if d.empty:
                continue
            pos = bank * N_DWDM_CHANNELS + channel
            box_data_positions.append(pos)
            box_data_values.append(d[value_column].values.astype(float))
            box_banks.append(bank)
    return box_data_positions, box_data_values, box_banks


def _collect_module_fiber_32_series(df: pd.DataFrame, value_column: str, slot_map: dict[str, int]) -> tuple[list, list, list[int]]:
    """One violin per module fiber 1..32; pools all 8 DWDM lanes on that bank output."""
    enriched = add_module_fiber_column(df, slot_map)
    enriched = enriched.dropna(subset=["ModuleFiber"]).copy()
    enriched["ModuleFiber"] = enriched["ModuleFiber"].astype(int)
    n_drop = len(df) - len(enriched)
    if n_drop > 0:
        print(f"  MFG OFC: dropped {n_drop} rows with no/invalid tile_module_slot.yaml entry")

    box_data_positions: list = []
    box_data_values: list = []
    box_banks: list[int] = []
    for f in range(1, N_MODULE_FIBERS + 1):
        vals = enriched.loc[enriched["ModuleFiber"] == f, value_column].values.astype(float)
        if vals.size == 0:
            continue
        box_data_positions.append(f - 1)
        box_data_values.append(vals)
        box_banks.append(bank_from_module_fiber(f))
    return box_data_positions, box_data_values, box_banks


def plot_module_distribution_vs_freqerror(
    df: pd.DataFrame, output_path: Path, slot_map: dict[str, int]
) -> None:
    if df.empty:
        print("No data available for frequency error plot")
        return

    use_module = slot_map_covers_frame(df, slot_map)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")

    if use_module:
        box_data_positions, box_data_values, box_banks = _collect_module_fiber_32_series(
            df, "Frequency_Error_GHz", slot_map
        )
        n_x = N_MODULE_FIBERS
        xlabel = "Module fiber 1–32 (EVT slot map); 8 DWDM lanes pooled per fiber"
    else:
        if not slot_map:
            print(
                "MFG OFC: tile_module_slot.yaml missing or empty → 16 DWDM lanes on x-axis "
                "(CSV Bank×Channel). Add Tile_SN→slot 1–16 to plot fibers 1–32 like EVT."
            )
        else:
            print("MFG OFC: tile_module_slot.yaml has no Tile_SN overlap with this frame → 16 DWDM lanes.")
        box_data_positions, box_data_values, box_banks = _collect_dwdm_lane_series(df, "Frequency_Error_GHz")
        n_x = N_DWDM_LANES
        xlabel = "Bank channel"

    if not box_data_values:
        print("No data available for frequency error plot")
        plt.close(fig)
        return

    yticks = np.arange(-50, 51, 10)
    _ofc_violin_gen1_box_per_bank(
        ax,
        box_data_positions,
        box_data_values,
        box_banks,
        ylo=-50.0,
        yhi=50.0,
        yticks=yticks,
        annotation_unit="GHz",
    )

    ax.set_xticks(np.arange(n_x))
    if use_module:
        ax.set_xticklabels([str(i + 1) for i in range(n_x)], fontsize=7, rotation=45, ha="right")
    else:
        ax.set_xticklabels(_bank_channel_xticklabels_16(), fontsize=9, rotation=45, ha="right")
    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=12, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_module_distribution_vs_optical_power(
    df: pd.DataFrame, output_path: Path, slot_map: dict[str, int]
) -> None:
    if df.empty:
        print("No data available for power plot")
        return

    use_module = slot_map_covers_frame(df, slot_map)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")

    if use_module:
        box_data_positions, box_data_values, box_banks = _collect_module_fiber_32_series(df, "Power(mW)", slot_map)
        n_x = N_MODULE_FIBERS
        xlabel = "Module fiber 1–32 (EVT slot map); 8 DWDM lanes pooled per fiber"
    else:
        if not slot_map:
            print(
                "MFG OFC: tile_module_slot.yaml missing or empty → 16 DWDM lanes on x-axis. "
                "Add Tile_SN→slot 1–16 for fibers 1–32."
            )
        else:
            print("MFG OFC: slot map has no Tile_SN overlap with power frame → 16 DWDM lanes.")
        box_data_positions, box_data_values, box_banks = _collect_dwdm_lane_series(df, "Power(mW)")
        n_x = N_DWDM_LANES
        xlabel = "Bank channel"

    if not box_data_values:
        print("No data available for power plot")
        plt.close(fig)
        return

    yticks = np.arange(0, 21, 5)
    _ofc_violin_gen1_box_per_bank(
        ax,
        box_data_positions,
        box_data_values,
        box_banks,
        ylo=0.0,
        yhi=20.0,
        yticks=yticks,
        annotation_unit="mW",
    )

    ax.set_xticks(np.arange(n_x))
    if use_module:
        ax.set_xticklabels([str(i + 1) for i in range(n_x)], fontsize=7, rotation=45, ha="right")
    else:
        ax.set_xticklabels(_bank_channel_xticklabels_16(), fontsize=9, rotation=45, ha="right")
    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Power in fiber (mW)", fontsize=12, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _total_power_per_bank_output_from_tp2p6(df_power: pd.DataFrame) -> pd.DataFrame:
    if df_power.empty:
        return pd.DataFrame(columns=["Tile_SN", "Bank", "Total_Power_mW"])
    g = df_power.groupby(["Tile_SN", "Bank", "Channel"], as_index=False)["Power(mW)"].mean()
    t = g.groupby(["Tile_SN", "Bank"], as_index=False)["Power(mW)"].sum()
    return t.rename(columns={"Power(mW)": "Total_Power_mW"})


def plot_module_distribution_vs_total_power(
    df_power: pd.DataFrame, output_path: Path, slot_map: dict[str, int]
) -> None:
    df_tot = _total_power_per_bank_output_from_tp2p6(df_power)
    if df_tot.empty:
        print("No data available for total power (per bank output) plot")
        return

    df_tot = df_tot[
        (df_tot["Total_Power_mW"] >= 60.0) & (df_tot["Total_Power_mW"] <= 160.0)
    ].copy()
    if df_tot.empty:
        print("No data available for total power plot after 60–160 mW filter")
        return

    use_module = slot_map_covers_frame(df_tot, slot_map)
    sns.set_style("whitegrid")

    if use_module:
        df_tot = add_module_fiber_column(df_tot, slot_map)
        df_tot = df_tot.dropna(subset=["ModuleFiber"]).copy()
        df_tot["ModuleFiber"] = df_tot["ModuleFiber"].astype(int)
        fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
        box_data_positions: list = []
        box_data_values: list = []
        box_banks: list[int] = []
        for f in range(1, N_MODULE_FIBERS + 1):
            vals = df_tot.loc[df_tot["ModuleFiber"] == f, "Total_Power_mW"].values.astype(float)
            if vals.size == 0:
                continue
            box_data_positions.append(f - 1)
            box_data_values.append(vals)
            box_banks.append(bank_from_module_fiber(f))
        n_x = N_MODULE_FIBERS
    else:
        if not slot_map:
            print("MFG OFC: tile_module_slot.yaml missing → total power plot uses Set A / Set B only (2 violins).")
        fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
        box_data_positions = []
        box_data_values = []
        box_banks = []
        for bank in [SET_A_BANK, SET_B_BANK]:
            df_b = df_tot[df_tot["Bank"] == bank]
            if df_b.empty:
                continue
            vals = df_b["Total_Power_mW"].values.astype(float)
            pos = 0 if bank == SET_A_BANK else 1
            box_data_positions.append(pos)
            box_data_values.append(vals)
            box_banks.append(bank)
        n_x = 2

    if not box_data_values:
        print("No data available for total power plot")
        plt.close(fig)
        return

    yticks = np.arange(0, 201, 50)
    _ofc_violin_gen1_box_per_bank(
        ax,
        box_data_positions,
        box_data_values,
        box_banks,
        ylo=0.0,
        yhi=200.0,
        yticks=yticks,
        annotation_unit="mW",
    )

    if use_module:
        ax.set_xticks(np.arange(n_x))
        ax.set_xticklabels([str(i + 1) for i in range(n_x)], fontsize=7, rotation=45, ha="right")
        ax.set_xlim(-0.5, n_x - 0.5)
        x_title = "Module fiber 1–32 — total power summed over 8 DWDM lanes on that bank output (TP2-6)"
    else:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Set A", "Set B"], fontsize=11)
        ax.set_xlim(-0.5, 1.5)
        x_title = "Total power per bank output — sum over 8 DWDM channels (TP2-6)"

    ax.set_ylabel("Total power in fiber (mW)", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_title, fontsize=12, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_mfg_ofc_plots(
    df_freq: pd.DataFrame,
    df_power: pd.DataFrame,
    output_dir: Path | None = None,
    mfg_base: Path | None = None,
) -> Path:
    mfg = (mfg_base or default_mfg_base()).resolve()
    slot_map = load_tile_module_slot_map(mfg)

    out = (output_dir or default_output_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)

    plot_module_distribution_vs_freqerror(df_freq, out / "module_distribution_vs_freqerror.png", slot_map)
    plot_module_distribution_vs_optical_power(df_power, out / "module_distribution_vs_optical_power.png", slot_map)
    plot_module_distribution_vs_total_power(df_power, out / "module_distribution_vs_total_power.png", slot_map)
    return out


def ofc_plotter(mfg_base_path: Path | None = None, output_dir: Path | None = None) -> Path:
    mfg = (mfg_base_path or default_mfg_base()).resolve()
    mfg_src = mfg / "analysis_src"
    if not mfg_src.is_dir():
        raise FileNotFoundError(f"MFG analysis_src not found: {mfg_src}")

    if str(mfg_src) not in sys.path:
        sys.path.insert(0, str(mfg_src))

    from analyze_test_points import tpanalysis

    analyzer = tpanalysis(base_path=mfg)
    df_freq, df_power = analyzer.load_ofc_filtered_data()
    out = run_mfg_ofc_plots(df_freq, df_power, output_dir=output_dir, mfg_base=mfg)
    print("\nMFG OFC module distribution plots completed!")
    print(f"Plots saved to: {out}")
    return out


if __name__ == "__main__":
    ofc_plotter()
