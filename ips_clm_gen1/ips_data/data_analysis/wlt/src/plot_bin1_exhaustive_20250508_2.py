"""
Exhaustive plots for bin=1 dies from ``wlt_data_summary_20250508_2.csv``.

Writes PNGs under ``wlt/results/bin1_exhaustive/`` (summary + heatmaps + scatters
in Plotly/Kaleido; per-column histograms in Matplotlib for speed).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_DATA_ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_OUT_DIR = _RESULTS_DIR / "bin1_exhaustive"
_CSV_NAME = "wlt_data_summary_20250508_2.csv"

_SUMMARY_COLS = [
    "minimum_facet_power_dbm",
    "backside_temperature_target_c",
    "worst_smsr_db_hz",
    "worst_rin_db_hz",
    "worst_relative_frequency_error_ghz",
    "total_wpe_estimate_percent",
]

_RE_FREQ = re.compile(r"^FREQ_(\d+)_(\d+)_(36|50)$")
_RE_WPE = re.compile(r"^WPE_(\d+)_(\d+)_(36|50)$")
_RE_IPD2 = re.compile(r"^IPD2_(\d+)_(\d+)_(36|50)$")
_RE_VLZR = re.compile(r"^VLZR_(\d+)_(\d+)_(36|50)$")
_RE_SMSR = re.compile(r"^SMSR_(\d+)_(\d+)_36$")
_RE_RIN = re.compile(r"^RIN_MAX_DB_6\.5GHZ_(\d+)_(\d+)_36$")
_RE_EACH = re.compile(r"^EACH_DF_DIV_145~165_(\d+)_(36|50)$")


def _csv_path() -> Path:
    return _DATA_ANALYSIS_DIR / _CSV_NAME


def _ensure_dirs() -> None:
    for sub in (
        _OUT_DIR,
        _OUT_DIR / "summary_histograms",
        _OUT_DIR / "pairwise_scatter",
        _OUT_DIR / "gelpak_maps",
        _OUT_DIR / "heatmaps_mean",
        _OUT_DIR / "heatmaps_std",
        _OUT_DIR / "box_by_gelpak",
        _OUT_DIR / "line_mean_vs_channel",
        _OUT_DIR / "histograms_all_numeric",
    ):
        sub.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str, max_len: int = 160) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return s[:max_len]


def _save_plotly(fig, relative_path: str) -> None:
    path = _OUT_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=1000, height=600, scale=1)


def _add_diode_bias(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diode_bias_v"] = (
        10 ** (out["minimum_facet_power_dbm"] / 10)
        / out["total_wpe_estimate_percent"]
        / 1000
        * 100
        / 145e-3
    )
    return out


def load_bin1() -> pd.DataFrame:
    path = _csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df = df[df["bin"] == 1].copy()
    if df.empty:
        raise ValueError("No rows with bin==1")
    return _add_diode_bias(df)


def plot_summary_histograms(df: pd.DataFrame) -> None:
    cols = _SUMMARY_COLS + ["diode_bias_v"]
    titles = {
        "minimum_facet_power_dbm": "Minimum facet power (dBm)",
        "backside_temperature_target_c": "Backside temperature target (°C)",
        "worst_smsr_db_hz": "Worst SMSR (dB)",
        "worst_rin_db_hz": "Worst RIN (dBc/Hz)",
        "worst_relative_frequency_error_ghz": "Worst relative frequency error (GHz)",
        "total_wpe_estimate_percent": "Total WPE estimate (%)",
        "diode_bias_v": "Diode bias (V)",
    }
    for c in cols:
        fig = px.histogram(
            df,
            x=c,
            nbins=80,
            title=f"Bin 1 — {titles.get(c, c)} (n={len(df)})",
        )
        fig.update_layout(template="plotly_white")
        _save_plotly(fig, f"summary_histograms/hist_{_safe_filename(c)}.png")


def plot_pairwise_scatters(df: pd.DataFrame) -> None:
    cols = _SUMMARY_COLS + ["diode_bias_v"]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            cc = df[a].corr(df[b])
            fig = px.scatter(
                df,
                x=a,
                y=b,
                color="gelpaknumber",
                hover_data=["mmid", "gelpakx", "gelpaky", "batch"],
                title=f"Bin 1 — {a} vs {b} (r={cc:.3f})",
            )
            fig.update_layout(template="plotly_white", legend_title_text="Gelpak")
            fname = f"pairwise_scatter/scatter_{_safe_filename(a)}__vs__{_safe_filename(b)}.png"
            _save_plotly(fig, fname)


def plot_gelpak_maps(df: pd.DataFrame) -> None:
    for c in _SUMMARY_COLS:
        fig = px.scatter(
            df,
            x="gelpakx",
            y="gelpaky",
            color=c,
            facet_col="gelpaknumber",
            facet_col_wrap=4,
            hover_data=["mmid", "bin"],
            title=f"Bin 1 — wafer map by gelpak ({c})",
            height=1400,
            width=1600,
        )
        fig.update_layout(template="plotly_white")
        _save_plotly(fig, f"gelpak_maps/map_{_safe_filename(c)}.png")


def _matrix_agg(df: pd.DataFrame, pattern: re.Pattern[str], agg: str) -> dict[str, pd.DataFrame]:
    """agg is 'mean' or 'std'."""
    colmap: dict[tuple[int, int, str], str] = {}
    for col in df.columns:
        m = pattern.match(col)
        if not m:
            continue
        g = m.groups()
        if pattern is _RE_EACH:
            ch_s, t_s = g
            colmap[(0, int(ch_s), t_s)] = col
        elif pattern in (_RE_SMSR, _RE_RIN):
            wl_s, ch_s = g
            colmap[(int(wl_s), int(ch_s), "36_only")] = col
        else:
            wl_s, ch_s, t_s = g
            colmap[(int(wl_s), int(ch_s), t_s)] = col

    out: dict[str, pd.DataFrame] = {}
    for t_label in sorted({k[2] for k in colmap}):
        keys = [(wl, ch) for (wl, ch, t) in colmap if t == t_label]
        if not keys:
            continue
        chs = sorted({ch for _, ch in keys})
        wls = sorted({wl for wl, _ in keys})
        mat = pd.DataFrame(index=wls, columns=chs, dtype=float)
        for wl in wls:
            for ch in chs:
                key = (wl, ch, t_label)
                if key not in colmap:
                    continue
                s = df[colmap[key]]
                mat.loc[wl, ch] = s.mean() if agg == "mean" else s.std()
        out[t_label] = mat
    return out


def plot_heatmaps(df: pd.DataFrame) -> None:
    specs: list[tuple[str, re.Pattern[str], str]] = [
        ("FREQ", _RE_FREQ, "Frequency"),
        ("WPE", _RE_WPE, "WPE (%)"),
        ("IPD2", _RE_IPD2, "IPD2"),
        ("VLZR", _RE_VLZR, "VLZR"),
        ("SMSR", _RE_SMSR, "SMSR (dB)"),
        ("RIN", _RE_RIN, "RIN (dBc/Hz)"),
        ("EACH_DF", _RE_EACH, "EACH DF div"),
    ]
    for short, pat, label in specs:
        for agg, subdir in (("mean", "heatmaps_mean"), ("std", "heatmaps_std")):
            mats = _matrix_agg(df, pat, agg)
            for t_key, mat in mats.items():
                if mat.empty:
                    continue
                ytitle = "Channel (EACH)" if short == "EACH_DF" else "λ (nm)"
                ttl = f"Bin 1 — {label} {agg} ({t_key}°C) — {short}"
                if t_key == "36_only":
                    ttl = f"Bin 1 — {label} {agg} (36°C) — {short}"
                fig = go.Figure(
                    data=go.Heatmap(
                        z=mat.values.astype(float),
                        x=[str(c) for c in mat.columns],
                        y=[str(r) for r in mat.index],
                        colorscale="Plasma" if agg == "mean" else "Cividis",
                    )
                )
                fig.update_layout(
                    title=ttl,
                    template="plotly_white",
                    xaxis_title="Channel",
                    yaxis_title=ytitle,
                )
                fname = f"{subdir}/heatmap_{short}_{agg}_{_safe_filename(t_key)}.png"
                _save_plotly(fig, fname)


def plot_box_by_gelpak(df: pd.DataFrame) -> None:
    for c in _SUMMARY_COLS:
        fig = px.box(
            df,
            x="gelpaknumber",
            y=c,
            points="all",
            title=f"Bin 1 — {c} by gelpak (n={len(df)})",
        )
        fig.update_layout(template="plotly_white")
        _save_plotly(fig, f"box_by_gelpak/box_{_safe_filename(c)}.png")


def plot_line_mean_vs_channel(df: pd.DataFrame) -> None:
    def lines_for_family(prefix: str, pattern: re.Pattern[str], temps: list[str]) -> None:
        for temp in temps:
            series: dict[int, tuple[list[int], list[float]]] = {}
            for col in df.columns:
                m = pattern.match(col)
                if not m:
                    continue
                g = m.groups()
                if len(g) == 3:
                    wl, ch, t = int(g[0]), int(g[1]), g[2]
                    if t != temp:
                        continue
                else:
                    continue
                if wl not in series:
                    series[wl] = ([], [])
                chv = ch
                series[wl][0].append(chv)
                series[wl][1].append(df[col].mean())
            if not series:
                continue
            fig = go.Figure()
            for wl in sorted(series):
                xs, ys = series[wl]
                order = np.argsort(xs)
                fig.add_trace(
                    go.Scatter(
                        x=np.array(xs)[order],
                        y=np.array(ys)[order],
                        mode="lines+markers",
                        name=f"{wl} nm",
                    )
                )
            fig.update_layout(
                title=f"Bin 1 — mean {prefix} vs channel ({temp}°C)",
                template="plotly_white",
                xaxis_title="Channel",
                yaxis_title=prefix,
            )
            _save_plotly(
                fig,
                f"line_mean_vs_channel/line_{prefix}_{temp}C.png",
            )

    lines_for_family("FREQ", _RE_FREQ, ["36", "50"])
    lines_for_family("WPE", _RE_WPE, ["36", "50"])
    lines_for_family("IPD2", _RE_IPD2, ["36", "50"])
    lines_for_family("VLZR", _RE_VLZR, ["36", "50"])

    for prefix, pattern in (("SMSR", _RE_SMSR), ("RIN", _RE_RIN)):
        series: dict[int, tuple[list[int], list[float]]] = {}
        for col in df.columns:
            m = pattern.match(col)
            if not m:
                continue
            wl, ch = int(m.group(1)), int(m.group(2))
            if wl not in series:
                series[wl] = ([], [])
            series[wl][0].append(ch)
            series[wl][1].append(df[col].mean())
        fig = go.Figure()
        for wl in sorted(series):
            xs, ys = series[wl]
            order = np.argsort(xs)
            fig.add_trace(
                go.Scatter(
                    x=np.array(xs)[order],
                    y=np.array(ys)[order],
                    mode="lines+markers",
                    name=f"{wl} nm",
                )
            )
        fig.update_layout(
            title=f"Bin 1 — mean {prefix} vs channel (36°C)",
            template="plotly_white",
            xaxis_title="Channel",
            yaxis_title=prefix,
        )
        _save_plotly(fig, f"line_mean_vs_channel/line_{prefix}_36C_only.png")


def plot_all_numeric_histograms(df: pd.DataFrame) -> None:
    skip = {"mmid"}
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    out = _OUT_DIR / "histograms_all_numeric"
    for col in num:
        if col in skip:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(s.to_numpy(), bins=min(80, max(10, int(np.sqrt(len(s))))))
        plt.title(f"Bin 1 — {col} (n={len(s)})")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / f"{_safe_filename(col)}.png", dpi=120)
        plt.close()


def main() -> None:
    _ensure_dirs()
    df = load_bin1()
    print(f"Loaded bin==1: {len(df)} rows from {_csv_path().name}")

    plot_summary_histograms(df)
    plot_pairwise_scatters(df)
    plot_gelpak_maps(df)
    plot_heatmaps(df)
    plot_box_by_gelpak(df)
    plot_line_mean_vs_channel(df)
    plot_all_numeric_histograms(df)

    print(f"Wrote figures under: {_OUT_DIR}")


if __name__ == "__main__":
    main()
