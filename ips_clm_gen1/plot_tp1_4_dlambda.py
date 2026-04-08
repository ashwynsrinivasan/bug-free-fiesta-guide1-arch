#!/usr/bin/env python3
"""Violin + box plots of dlambda (nm) and dfreq (GHz) by Bank and Channel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

_C_NM_PER_S = 299_792_458e9  # speed of light in nm/s


def _make_plot(
    datasets: list[np.ndarray],
    groups: list[dict],
    banks: list,
    channels: list,
    overall_mean: float,
    ylim: float,
    ylabel: str,
    xlabel: str,
    title: str | None,
    bank_labels: dict,
    show_mean_table: bool,
) -> plt.Figure:
    """Draw a single violin+boxplot figure and return it."""
    violin_color = (0.45 * 0.85, 0.65 * 0.85, 0.95 * 0.85)
    violin_rgba = (*violin_color, 0.55)
    violin_edge = (*violin_color[:3], 0.9)

    xpos = [g["pos"] for g in groups]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)

    # ── Violin plots ─────────────────────────────────────────────────────────
    parts = ax.violinplot(
        datasets,
        positions=xpos,
        widths=0.65,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(violin_rgba)
        pc.set_edgecolor(violin_edge)
        pc.set_linewidth(0.8)
        pc.set_zorder(1)

    # ── Box plots overlaid ───────────────────────────────────────────────────
    ax.boxplot(
        datasets,
        positions=xpos,
        widths=0.22,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
        boxprops=dict(facecolor="white", color="black", zorder=3),
        medianprops=dict(color="black", linewidth=1.2, zorder=4),
        whiskerprops=dict(color="black", linewidth=0.9),
        capprops=dict(color="black", linewidth=0.9),
        flierprops=dict(
            marker="o",
            markersize=1.8,
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            alpha=0.7,
            zorder=2,
        ),
    )

    # ── Axis limits & labels ─────────────────────────────────────────────────
    ax.set_ylim(-ylim, ylim)
    ax.set_xlim(xpos[0] - 0.8, xpos[-1] + 0.8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=12)

    # ── Two-level x-axis ─────────────────────────────────────────────────────
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(g["channel"]) for g in groups], fontsize=9)
    ax.tick_params(axis="x", which="major", length=4)

    blended = blended_transform_factory(ax.transData, ax.transAxes)
    for bank in banks:
        bank_groups = [g for g in groups if g["bank"] == bank]
        mid_pos = (bank_groups[0]["pos"] + bank_groups[-1]["pos"]) / 2.0
        ax.text(
            mid_pos, -0.09,
            bank_labels.get(bank, str(bank)),
            transform=blended,
            ha="center", va="top", fontsize=11,
        )
        if bank != banks[-1]:
            last_idx = groups.index(bank_groups[-1])
            divider_x = (bank_groups[-1]["pos"] + groups[last_idx + 1]["pos"]) / 2.0
            ax.axvline(x=divider_x, color="0.65", linewidth=0.8, linestyle="-", zorder=0)

    ax.set_xlabel(xlabel, labelpad=18)

    # ── Mean-table annotation (top-left, optional) ───────────────────────────
    if show_mean_table:
        means = {
            (g["bank"], g["channel"]): float(np.mean(d))
            for g, d in zip(groups, datasets)
        }
        col_width = max(len(f"Mean({c}): {means[(banks[0], c)]:.5g}") for c in channels) + 3
        lines = [f"{'Bank':>{col_width}}"]
        lines.append("   ".join(f"{'  ' + bank_labels.get(b, str(b)):>{col_width}}" for b in banks))
        for ch in channels:
            row = "   ".join(
                f"{'Mean(' + str(ch) + '): ' + f'{means[(b, ch)]:.5g}':>{col_width}}"
                for b in banks
            )
            lines.append(row)
        ax.text(
            0.01, 0.99, "\n".join(lines),
            transform=ax.transAxes,
            fontsize=6.5,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7", alpha=0.92),
            zorder=10,
        )

    # ── Overall mean (top-right) ──────────────────────────────────────────────
    ax.text(
        0.99, 0.99,
        f"Mean: {overall_mean:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        zorder=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    label = ylabel
    violin_patch = mpatches.Patch(
        facecolor=(*violin_color, 0.85), edgecolor=(*violin_color, 1.0),
        linewidth=0.8, label=label,
    )
    box_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", linewidth=0.9, label=label,
    )
    ax.legend(
        handles=[violin_patch, box_patch],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.82),
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    return fig


def main() -> None:
    xlsx = Path(__file__).resolve().parent / "TP1-4.xlsx"
    out_dir = Path(__file__).resolve().parent

    df = pd.read_excel(xlsx)
    need = {"Bank", "Channel", "dlambda", "PeakWave(nm)"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # ── Frequency conversion: df (GHz) = -c/λ² · dλ ─────────────────────────
    df["dfreq_GHz"] = (
        -(_C_NM_PER_S / df["PeakWave(nm)"] ** 2) * df["dlambda"] / 1e9
    )

    banks = sorted(df["Bank"].unique())
    channels = sorted(df["Channel"].unique())
    n_ch = len(channels)
    bank_gap = 1.5

    def _build_groups(plot_df: pd.DataFrame, metric: str) -> tuple[list, list, list]:
        groups: list[dict] = []
        for i, bank in enumerate(banks):
            for j, ch in enumerate(channels):
                pos = i * (n_ch + bank_gap) + j
                groups.append({"bank": bank, "channel": ch, "pos": pos})
        datasets = [
            plot_df.loc[
                (plot_df["Bank"] == g["bank"]) & (plot_df["Channel"] == g["channel"]),
                metric,
            ].values
            for g in groups
        ]
        return groups, datasets

    # ── Plot 1: dlambda (nm) ─────────────────────────────────────────────────
    ylim_nm = 0.3
    df_nm = df[(df["dlambda"] >= -ylim_nm) & (df["dlambda"] <= ylim_nm)].copy()
    groups_nm, datasets_nm = _build_groups(df_nm, "dlambda")
    overall_nm = float(np.mean(np.concatenate(datasets_nm)))

    fig1 = _make_plot(
        datasets=datasets_nm,
        groups=groups_nm,
        banks=banks,
        channels=channels,
        overall_mean=overall_nm,
        ylim=ylim_nm,
        ylabel="dlambda",
        xlabel="Bank / Channel",
        title="TP1-4 all channels on at T_op free space\ndlambda vs. Bank & Channel",
        bank_labels={b: str(b) for b in banks},
        show_mean_table=True,
    )
    out1 = out_dir / "TP1-4_dlambda_bank_channel.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Wrote {out1}")

    # ── Plot 2: dfreq (GHz) ──────────────────────────────────────────────────
    ylim_ghz = 50.0
    df_ghz = df[(df["dfreq_GHz"] >= -ylim_ghz) & (df["dfreq_GHz"] <= ylim_ghz)].copy()
    groups_ghz, datasets_ghz = _build_groups(df_ghz, "dfreq_GHz")
    overall_ghz = float(np.mean(np.concatenate(datasets_ghz)))

    fig2 = _make_plot(
        datasets=datasets_ghz,
        groups=groups_ghz,
        banks=banks,
        channels=channels,
        overall_mean=overall_ghz,
        ylim=ylim_ghz,
        ylabel="dfreq (GHz)",
        xlabel="Bank / Channel",
        title=None,
        bank_labels={banks[0]: "A", banks[1]: "B"},
        show_mean_table=False,
    )
    out2 = out_dir / "TP1-4_dfreq_GHz_bank_channel.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
