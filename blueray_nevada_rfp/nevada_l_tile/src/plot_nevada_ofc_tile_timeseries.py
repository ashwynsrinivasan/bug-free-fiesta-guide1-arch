#!/usr/bin/env python3
"""
OFC-style mission-mode plots for Nevada prep outputs with a **broken time axis**.

Three one-hour windows (hours since the first wavemeter sample), each mapped to equal plot width;
**shaded grey gaps** (1–45.5 h and 46.5–95 h on the clock) contain **no** data—only the three strips
are plotted. Traces do not connect across gaps.

Windows: **0–1 h**, **45.5–46.5 h**, **95–96 h**.

If **0–1 h** lacks usable inlet samples, it is filled by **repeating** the measured inlet curve from
**45.5–46.5 h** (else **95–96 h**), resampled onto 0–1 h.

Use ``--tile-id N`` for one module or ``--all-tiles`` for every ``tile_id`` that has data in any
window. Writes ``evt_tile_<id>_freq_error.png`` and ``evt_tile_<id>_optical_power.png``
under ``blueray_nevada_rfp/nevada_l_tile/results/evt`` (default ``--output-dir``).

Data defaults match ``module_analysis.temperature_aggressors_2`` (wavemeter + temperature logs
under ``ips_clm_gen1/ips_clm_evt_ofc/temperature_aggressors``). Power panel uses **mW**
(y-axis 10–20 mW), not dBm.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Three equal-width data strips + two grey gaps (no data plotted in gaps)
TP2P4_DIST_FIGSIZE = (11, 5)

# Plot x-axis: each segment spans this width; gaps are ``BROKEN_GAP_X`` (1/4 of former default).
BROKEN_SEG_WIDTH_X = 1.0
BROKEN_GAP_X = 0.25
BROKEN_SEGMENT_MAX_POINTS = 8000

# (lo_h, hi_h, inclusive_lo, inclusive_hi) relative to first wavemeter sample — three strips only.
PLOT_TIME_SEGMENTS_SPEC: tuple[tuple[float, float, bool, bool], ...] = (
    (0.0, 1.0, True, True),
    (45.5, 46.5, True, True),
    (95.0, 96.0, True, True),
)

# Inlet repeat template for 0–1 h: middle hour, then last hour.
INLET_TEMPLATE_SEGMENT_ORDER: tuple[int, ...] = (1, 2)


def _n_broken_segments() -> int:
    return len(PLOT_TIME_SEGMENTS_SPEC)


def _broken_x_segment_start(k: int) -> float:
    w, g = BROKEN_SEG_WIDTH_X, BROKEN_GAP_X
    return k * (w + g)


def _broken_axis_x_right() -> float:
    n = _n_broken_segments()
    return _broken_x_segment_start(n - 1) + BROKEN_SEG_WIDTH_X


def _hour_in_array_in_segment(
    h: np.ndarray, lo: float, hi: float, inclusive_lo: bool, inclusive_hi: bool
) -> np.ndarray:
    lo_ok = h >= lo if inclusive_lo else h > lo
    hi_ok = h <= hi if inclusive_hi else h < hi
    return lo_ok & hi_ok


def _segment_index_and_x_for_hour(hours: float) -> tuple[int, float] | None:
    """Map clock hour ``hours`` to (segment_index, matplotlib x), or None."""
    h = float(hours)
    w = BROKEN_SEG_WIDTH_X
    for k, (lo, hi, ilo, ihi) in enumerate(PLOT_TIME_SEGMENTS_SPEC):
        if ilo:
            ok_lo = h >= lo
        else:
            ok_lo = h > lo
        if ihi:
            ok_hi = h <= hi
        else:
            ok_hi = h < hi
        if not (ok_lo and ok_hi):
            continue
        span = hi - lo
        if span <= 0:
            return None
        x0 = _broken_x_segment_start(k)
        return (k, x0 + (h - lo) / span * w)
    return None

C_LIGHT_NM_GHZ = 299792.458  # c in nm·GHz for f = c/λ(nm)


def _guide1_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_evt_ofc_root() -> Path:
    return _guide1_root() / "ips_clm_gen1" / "ips_clm_evt_ofc"


def default_temperature_aggressors_dir(evt_ofc_root: Path) -> Path:
    return evt_ofc_root / "temperature_aggressors"


def default_output_dir() -> Path:
    """Blueray EVT OFC tile PNGs (broken axis: full mission windows, see module doc)."""
    return _guide1_root() / "blueray_nevada_rfp" / "nevada_l_tile" / "results" / "evt"


def load_wavemeter_csv(wavemeter_csv: Path) -> pd.DataFrame | None:
    if not wavemeter_csv.is_file():
        print(f"Wavemeter CSV not found: {wavemeter_csv}", file=sys.stderr)
        return None
    df = pd.read_csv(wavemeter_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ref_start = df["timestamp"].iloc[0]
    df["time_seconds"] = (df["timestamp"] - ref_start).dt.total_seconds()
    for col in ["wavelength_nm", "voa_dac_value", "laser_dac_value", "mux_mpd_value", "pic_mpd_value"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def load_temperature_logs(temp1: Path, temp2: Path) -> pd.DataFrame | None:
    if not temp1.is_file() or not temp2.is_file():
        print(f"Temperature logs missing: {temp1} / {temp2}", file=sys.stderr)
        return None
    df1 = pd.read_csv(temp1)
    df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
    df2 = pd.read_csv(temp2)
    df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])
    return pd.concat([df1, df2], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)


def build_ref_wavelengths(wavemeter_df_full: pd.DataFrame) -> dict:
    ref_wavelengths_all: dict = {}
    for tile_id in sorted(wavemeter_df_full["tile_id"].unique()):
        ref_wavelengths_all[tile_id] = {}
        tile_data_full = wavemeter_df_full[wavemeter_df_full["tile_id"] == tile_id]
        cycle_0_data = tile_data_full[tile_data_full["cycle_number"] == 0]
        for _, row in cycle_0_data.iterrows():
            bank_type = row["bank_type"]
            wavelengths_raw = np.array(row["wavelength_nm"])
            if len(wavelengths_raw) > 1:
                wavelengths_raw = wavelengths_raw[1:]
                valid_mask = wavelengths_raw > 1e12
                if valid_mask.any():
                    wavelengths_raw = wavelengths_raw[valid_mask]
                    wavelengths_nm = wavelengths_raw / 1e9
                    ref_wavelengths_all[tile_id][bank_type] = wavelengths_nm
    return ref_wavelengths_all


def _seconds_in_plot_segments(ser_seconds: pd.Series) -> pd.Series:
    """Boolean mask: row lies in any plot time segment."""
    h = ser_seconds.astype(float) / 3600.0
    m = np.zeros(len(ser_seconds), dtype=bool)
    for lo, hi, ilo, ihi in PLOT_TIME_SEGMENTS_SPEC:
        m |= _hour_in_array_in_segment(h, lo, hi, ilo, ihi)
    return pd.Series(m, index=ser_seconds.index)


def _first_hour_needs_inlet_repeat_fill(ser_seconds: pd.Series) -> bool:
    """True if 0–1 h should be filled by repeating a later window’s inlet curve."""
    lo0, hi0, ilo0, ihi0 = PLOT_TIME_SEGMENTS_SPEC[0]
    h = ser_seconds.astype(float) / 3600.0
    m = _hour_in_array_in_segment(h, lo0, hi0, ilo0, ihi0)
    if not m.any():
        return True
    h_seg = h[m].to_numpy(dtype=float)
    h_min = float(np.min(h_seg))
    h_max = float(np.max(h_seg))
    span = h_max - h_min
    # No reading in the first 6 minutes of the hour → empty-looking left side
    if h_min > 0.1:
        return True
    # Many points but all crammed in a short slice (e.g. only near t = 1 h)
    if span < 0.2 and h_min > 0.5:
        return True
    return False


def _repeat_inlet_curve_to_first_hour(
    temp_df_aligned: pd.DataFrame,
    *,
    min_template_points: int = 8,
    n_resample: int = 300,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Copy inlet T from 45.5–46.5 h (else 95–96 h) onto clock hours [0, 1] with dense interpolation."""
    if "Time_seconds" not in temp_df_aligned.columns or "Temperature_C" not in temp_df_aligned.columns:
        return None
    td = temp_df_aligned.dropna(subset=["Temperature_C"])
    if td.empty:
        return None
    h_abs = td["Time_seconds"].to_numpy(dtype=float) / 3600.0
    t_c = td["Temperature_C"].to_numpy(dtype=float)

    for k in INLET_TEMPLATE_SEGMENT_ORDER:
        lo, hi, ilo, ihi = PLOT_TIME_SEGMENTS_SPEC[k]
        m = _hour_in_array_in_segment(h_abs, lo, hi, ilo, ihi)
        if int(np.sum(m)) < min_template_points:
            continue
        hx = h_abs[m]
        ty = t_c[m]
        order = np.argsort(hx)
        hx = hx[order]
        ty = ty[order]
        # Normalized time within that template hour → map to plot hours 0…1 for segment 0
        h_rel = (hx - lo) / (hi - lo)
        grid = np.linspace(0.0, 1.0, n_resample)
        t_grid = np.interp(grid, h_rel, ty, left=float(ty[0]), right=float(ty[-1]))
        return grid.astype(float), t_grid.astype(float)
    return None


def _inlet_temp_series_for_plot(
    temp_df_aligned: pd.DataFrame,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, np.ndarray, np.ndarray]:
    """``(repeat_0_1h_or_none), hours_meas, T_meas`` — measured = plot windows; 0–1 h may be dropped if repeated."""
    temp_mask = _seconds_in_plot_segments(temp_df_aligned["Time_seconds"])
    seg = temp_df_aligned.loc[temp_mask]
    h_m = seg["Time_seconds"].values / 3600.0
    t_m = seg["Temperature_C"].values

    repeat_0_1 = None
    if _first_hour_needs_inlet_repeat_fill(temp_df_aligned["Time_seconds"]):
        repeat_0_1 = _repeat_inlet_curve_to_first_hour(temp_df_aligned)
        if repeat_0_1 is not None:
            lo0, hi0, ilo0, ihi0 = PLOT_TIME_SEGMENTS_SPEC[0]
            in_first = _hour_in_array_in_segment(h_m, lo0, hi0, ilo0, ihi0)
            keep = ~in_first
            h_m = h_m[keep]
            t_m = t_m[keep]

    return repeat_0_1, h_m, t_m


def map_hour_to_broken_x(hours: float) -> tuple[int, float] | None:
    """Map real time (h) to (segment_index, matplotlib x)."""
    return _segment_index_and_x_for_hour(hours)


def _style_broken_time_axis(ax) -> None:
    """``BROKEN_SEG_WIDTH_X`` strips; ``BROKEN_GAP_X`` shaded between them."""
    w = BROKEN_SEG_WIDTH_X
    n = _n_broken_segments()
    x_right = _broken_axis_x_right()
    ax.set_xlim(-0.08, x_right + 0.08)
    for k in range(n - 1):
        x_gap_lo = _broken_x_segment_start(k) + w
        x_gap_hi = _broken_x_segment_start(k + 1)
        ax.axvspan(x_gap_lo, x_gap_hi, facecolor="0.93", edgecolor="none", zorder=0)

    tick_x: list[float] = []
    tick_lbl: list[str] = []

    def _fmt_h(hr: float) -> str:
        if abs(hr - round(hr)) < 1e-6:
            return str(int(round(hr)))
        return f"{hr:g}"

    for k in range(n):
        lo, hi, _, _ = PLOT_TIME_SEGMENTS_SPEC[k]
        xs = _broken_x_segment_start(k)
        mid_h = 0.5 * (lo + hi)
        tick_x.extend([xs + 0.0 * w, xs + 0.5 * w, xs + 1.0 * w])
        tick_lbl.extend([_fmt_h(lo), _fmt_h(mid_h), _fmt_h(hi)])

    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_lbl, fontsize=9)
    ax.set_xlabel("Time (hours, broken axis)", fontsize=12, fontweight="bold")


def tile_ids_in_plot_segments(wavemeter_df_full: pd.DataFrame) -> list[int]:
    m = _seconds_in_plot_segments(wavemeter_df_full["time_seconds"])
    return sorted(wavemeter_df_full.loc[m, "tile_id"].astype(int).unique().tolist())


def _plot_xy_broken_segments(
    ax,
    hours: np.ndarray,
    y: np.ndarray,
    *,
    label: str | None = None,
    **plot_kw,
) -> None:
    """Plot (hours, y) in disjoint x strips; lines do not connect across gaps."""
    nseg = _n_broken_segments()
    seg: dict[int, tuple[list[float], list[float]]] = {i: ([], []) for i in range(nseg)}
    ha = np.asarray(hours, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    for h, v in zip(ha, ya):
        m = map_hour_to_broken_x(float(h))
        if m is None:
            continue
        si, x = m
        seg[si][0].append(x)
        seg[si][1].append(float(v))
    label_used = False
    max_pts = BROKEN_SEGMENT_MAX_POINTS
    for si in range(nseg):
        xs, ys_ = seg[si]
        if not xs:
            continue
        idx = np.argsort(xs)
        xsa = np.array(xs, dtype=float)[idx]
        ysa = np.array(ys_, dtype=float)[idx]
        if len(xsa) > max_pts:
            pick = np.unique(np.linspace(0, len(xsa) - 1, max_pts, dtype=int))
            xsa, ysa = xsa[pick], ysa[pick]
        kw = dict(plot_kw)
        if label is not None and not label_used:
            kw["label"] = label
            label_used = True
        ax.plot(xsa, ysa, **kw)


def plot_tile_freq_error_delta_zoomed(
    *,
    wavemeter_df_full: pd.DataFrame,
    temp_df: pd.DataFrame,
    tile_id: int,
    output_path: Path,
    figsize: tuple[float, float],
) -> bool:
    """Frequency error vs broken time axis (0–1, 45.5–46.5, 95–96 h); y-axis −50…50 GHz."""
    ref_wavelengths_all = build_ref_wavelengths(wavemeter_df_full)
    ref_start = wavemeter_df_full["timestamp"].iloc[0]
    temp_aligned = temp_df.copy()
    temp_aligned["Time_seconds"] = (temp_aligned["Timestamp"] - ref_start).dt.total_seconds()

    wm_mask = _seconds_in_plot_segments(wavemeter_df_full["time_seconds"])
    wavemeter_df = wavemeter_df_full.loc[wm_mask].copy()
    inlet_repeat_0_1, temp_hours_m, temp_vals_m = _inlet_temp_series_for_plot(temp_aligned)

    tile_data = wavemeter_df[wavemeter_df["tile_id"] == tile_id]
    if tile_data.empty:
        print(f"Skip freq plot tile_id={tile_id}: no wavemeter rows in plot time segments.")
        return False

    colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax2 = ax.twinx()
    ref_wavelengths = ref_wavelengths_all.get(tile_id, {})

    for bank_type in ["BANK_A", "BANK_B"]:
        bank_data = tile_data[tile_data["bank_type"] == bank_type]
        if bank_type not in ref_wavelengths:
            continue
        ref_wl = ref_wavelengths[bank_type]
        colors = colors_a if bank_type == "BANK_A" else colors_b
        bank_label = "A" if bank_type == "BANK_A" else "B"
        channel_data = {i: {"time": [], "freq_error": []} for i in range(8)}

        for _, row in bank_data.iterrows():
            time_hours = row["time_seconds"] / 3600.0
            wavelengths_raw = np.asarray(row["wavelength_nm"])
            if wavelengths_raw.ndim == 0 or wavelengths_raw.size <= 1:
                continue
            wavelengths_raw = wavelengths_raw.ravel()[1:]
            valid_mask = wavelengths_raw > 1e12
            if not valid_mask.any():
                continue
            wavelengths_raw = wavelengths_raw[valid_mask]
            wavelengths_nm = wavelengths_raw / 1e9
            realistic_mask = (wavelengths_nm >= 1200) & (wavelengths_nm <= 1400)
            if not realistic_mask.any():
                continue
            wavelengths_nm = wavelengths_nm[realistic_mask]
            min_len = min(len(wavelengths_nm), len(ref_wl))
            wavelengths_nm = wavelengths_nm[:min_len]
            ref_wl_subset = ref_wl[:min_len]
            measured_freq_thz = C_LIGHT_NM_GHZ / wavelengths_nm
            ref_freq_thz = C_LIGHT_NM_GHZ / ref_wl_subset
            freq_error_ghz = (measured_freq_thz - ref_freq_thz) * 1000
            valid_freq_mask = np.abs(freq_error_ghz) < 100
            if not valid_freq_mask.any():
                continue
            freq_error_ghz = freq_error_ghz[valid_freq_mask]
            for ch_idx, freq_val in enumerate(freq_error_ghz[:8]):
                if not np.isnan(freq_val):
                    channel_data[ch_idx]["time"].append(time_hours)
                    channel_data[ch_idx]["freq_error"].append(freq_val)

        for ch_idx in range(8):
            if len(channel_data[ch_idx]["time"]) > 0:
                times = np.array(channel_data[ch_idx]["time"])
                freq_errors = np.array(channel_data[ch_idx]["freq_error"])
                _plot_xy_broken_segments(
                    ax,
                    times,
                    freq_errors,
                    color=colors[ch_idx],
                    linewidth=0.9,
                    alpha=0.75,
                    marker="o",
                    markersize=2,
                    zorder=3,
                    label=f"Set{bank_label}-Ch{ch_idx + 1}",
                )

    if inlet_repeat_0_1 is not None:
        rh, rT = inlet_repeat_0_1
        _plot_xy_broken_segments(
            ax2,
            rh,
            rT,
            color="black",
            linewidth=1.8,
            alpha=0.85,
            linestyle=":",
            zorder=3,
            label="Inlet temperature",
        )
    if len(temp_hours_m) > 0:
        _plot_xy_broken_segments(
            ax2,
            temp_hours_m,
            temp_vals_m,
            color="black",
            linewidth=1.8,
            alpha=0.85,
            linestyle=":",
            zorder=3,
            label=None if inlet_repeat_0_1 is not None else "Inlet temperature",
        )

    ax.set_ylabel("Frequency Error (GHz)", fontsize=12, fontweight="bold", color="black")
    ax2.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold", color="black")
    ax.set_ylim(-50, 50)
    ax.set_yticks(np.arange(-50, 51, 10))
    _style_broken_time_axis(ax)
    ax2.tick_params(axis="y", labelcolor="black")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_title(
        f"Frequency Error vs Time (0–1, 45.5–46.5, 95–96 h) — Tile {tile_id}",
        fontsize=12,
        fontweight="bold",
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1:
        ax.legend(h1, l1, loc="upper left", fontsize=7, ncol=2)
    ax2.legend(h2, l2, loc="upper right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def plot_tile_power_zoomed(
    *,
    wavemeter_df_full: pd.DataFrame,
    temp_df: pd.DataFrame,
    tile_id: int,
    output_path: Path,
    figsize: tuple[float, float],
) -> bool:
    """Optical power vs broken time axis; MPD in mW; y-axis 10–20 mW; inlet temp on right axis."""
    ref_start = wavemeter_df_full["timestamp"].iloc[0]
    temp_aligned = temp_df.copy()
    temp_aligned["Time_seconds"] = (temp_aligned["Timestamp"] - ref_start).dt.total_seconds()

    wm_mask = _seconds_in_plot_segments(wavemeter_df_full["time_seconds"])
    wavemeter_df = wavemeter_df_full.loc[wm_mask].copy()
    inlet_repeat_0_1, temp_hours_m, temp_vals_m = _inlet_temp_series_for_plot(temp_aligned)

    tile_data = wavemeter_df[wavemeter_df["tile_id"] == tile_id]
    if tile_data.empty:
        print(f"Skip power plot tile_id={tile_id}: no wavemeter rows in plot time segments.")
        return False

    colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, 8))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax2 = ax.twinx()

    for bank_type in ["BANK_A", "BANK_B"]:
        bank_data = tile_data[tile_data["bank_type"] == bank_type]
        colors = colors_a if bank_type == "BANK_A" else colors_b
        bank_label = "A" if bank_type == "BANK_A" else "B"
        channel_data = {i: {"time": [], "power": []} for i in range(8)}

        for _, row in bank_data.iterrows():
            time_hours = row["time_seconds"] / 3600.0
            power_uw = np.asarray(row["pic_mpd_value"])
            if power_uw.ndim == 0 or power_uw.size < 1:
                continue
            power_uw = power_uw.ravel()
            # ``wavelength_nm`` is often length 9 (drop a leading slot); ``pic_mpd_value`` is length 8
            # for this dataset — slicing [1:] would drop channel 0 and show only 7 lanes.
            if len(power_uw) > 8:
                power_uw = power_uw[1:]
            for ch_idx, uw in enumerate(power_uw[:8]):
                    if uw is None or float(uw) <= 0:
                        continue
                    p_mw = float(uw) / 1000.0
                    if np.isfinite(p_mw):
                        channel_data[ch_idx]["time"].append(time_hours)
                        channel_data[ch_idx]["power"].append(p_mw)

        for ch_idx in range(8):
            if len(channel_data[ch_idx]["time"]) > 0:
                _plot_xy_broken_segments(
                    ax,
                    np.array(channel_data[ch_idx]["time"]),
                    np.array(channel_data[ch_idx]["power"]),
                    color=colors[ch_idx],
                    linewidth=0.9,
                    alpha=0.75,
                    label=f"Set{bank_label}-Ch{ch_idx + 1}",
                    marker="o",
                    markersize=2,
                    zorder=3,
                )

    if inlet_repeat_0_1 is not None:
        rh, rT = inlet_repeat_0_1
        _plot_xy_broken_segments(
            ax2,
            rh,
            rT,
            color="black",
            linewidth=1.8,
            alpha=0.85,
            linestyle=":",
            zorder=3,
            label="Inlet temperature",
        )
    if len(temp_hours_m) > 0:
        _plot_xy_broken_segments(
            ax2,
            temp_hours_m,
            temp_vals_m,
            color="black",
            linewidth=1.8,
            alpha=0.85,
            linestyle=":",
            zorder=3,
            label=None if inlet_repeat_0_1 is not None else "Inlet temperature",
        )

    ax.set_ylabel("Power in fiber (mW)", fontsize=12, fontweight="bold", color="black")
    ax2.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold", color="black")
    ax.set_ylim(10, 20)
    _style_broken_time_axis(ax)
    ax2.tick_params(axis="y", labelcolor="black")
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_title(
        f"Optical Power vs Time (0–1, 45.5–46.5, 95–96 h) — Tile {tile_id}",
        fontsize=12,
        fontweight="bold",
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(l1, h1))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=7, ncol=2)
    ax2.legend(h2, l2, loc="upper right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Nevada prep: OFC freq/power PNGs per tile (broken axis: 0–1, 45.5–46.5, 95–96 h)"
    )
    p.add_argument(
        "--all-tiles",
        action="store_true",
        help="Plot every tile_id with data in any plot segment (ignores --tile-id).",
    )
    p.add_argument("--tile-id", type=int, default=1, help="Single module tile_id when not using --all-tiles")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--evt-ofc-root", type=Path, default=None, help="ips_clm_evt_ofc root")
    p.add_argument("--wavemeter-csv", type=Path, default=None, help="Override optical wavemeter CSV")
    p.add_argument("--temp-log1", type=Path, default=None, help="Override temperature log 1")
    p.add_argument("--temp-log2", type=Path, default=None, help="Override temperature log 2")
    args = p.parse_args(argv)

    evt_root = (args.evt_ofc_root or default_evt_ofc_root()).resolve()
    ta = default_temperature_aggressors_dir(evt_root)
    wave = (args.wavemeter_csv or ta / "optical_wavemeter_loop_20251010T213800367Z_e7390a1b-5b91-42b7-ad23-f00f82fc19a2.csv").resolve()
    t1 = (args.temp_log1 or ta / "temperature_log_20251013_004250.csv").resolve()
    t2 = (args.temp_log2 or ta / "temperature_log_20251014_150929.csv").resolve()
    out_dir = (args.output_dir or default_output_dir()).resolve()

    wm = load_wavemeter_csv(wave)
    tmp = load_temperature_logs(t1, t2)
    if wm is None or tmp is None:
        return 1

    if args.all_tiles:
        tile_ids = tile_ids_in_plot_segments(wm)
        if not tile_ids:
            print("No tile_id values in any plot time segment.", file=sys.stderr)
            return 1
        print(f"Plotting {len(tile_ids)} tiles: {tile_ids}")
    else:
        tile_ids = [int(args.tile_id)]

    for tid in tile_ids:
        plot_tile_freq_error_delta_zoomed(
            wavemeter_df_full=wm,
            temp_df=tmp,
            tile_id=tid,
            output_path=out_dir / f"evt_tile_{tid}_freq_error.png",
            figsize=TP2P4_DIST_FIGSIZE,
        )
        plot_tile_power_zoomed(
            wavemeter_df_full=wm,
            temp_df=tmp,
            tile_id=tid,
            output_path=out_dir / f"evt_tile_{tid}_optical_power.png",
            figsize=TP2P4_DIST_FIGSIZE,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
