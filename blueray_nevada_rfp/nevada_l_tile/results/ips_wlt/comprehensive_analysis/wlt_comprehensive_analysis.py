"""
WLT Data Comprehensive Analysis
Data: wlt_data_summary_20250508_2.csv
Compares two operating temperatures (36C and 50C) across all metrics,
current levels, and channels.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../../../..",  # repo root
    "data/ips_wlt_data/wlt_data_summary_20250508_2.csv",
)
DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../" * 7, "data/ips_wlt_data/wlt_data_summary_20250508_2.csv")
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
COLORS = {36: "#2196F3", 50: "#F44336"}   # blue = 36C, red = 50C
CURRENTS = [125, 145, 165]
TEMPS = [36, 50]
CHANNELS = list(range(1, 17))
ALPHA = 0.35

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    # Try repo-relative path first, then a direct path
    candidates = [
        DATA_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../../../../../data/ips_wlt_data/wlt_data_summary_20250508_2.csv"),
        os.path.expanduser(
            "~/gitsrc/friendly-system-lmi/data/ips_wlt_data/wlt_data_summary_20250508_2.csv"
        ),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            print(f"Loading data from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find CSV. Tried:\n" + "\n".join(candidates))


df = load_data()
print(f"Loaded {len(df)} rows × {len(df.columns)} columns")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_cols(metric, current, temp):
    """Return sorted list of 16 channel columns for a given metric/current/temp."""
    prefix = f"{metric}_{current}_"
    suffix = f"_{temp}"
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)],
        key=lambda c: int(c.split("_")[-2]),
    )


def get_cols_df_div(temp):
    """EACH_DF_DIV has a special naming: EACH_DF_DIV_145~165_CH_TEMP."""
    prefix = "EACH_DF_DIV_145~165_"
    suffix = f"_{temp}"
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)],
        key=lambda c: int(c.split("_")[-2]),
    )


def col_vals(cols):
    """Flatten all values across the given columns into one array (drop NaN)."""
    return df[cols].values.flatten()


def channel_means(cols):
    """Return per-channel mean across all devices."""
    return df[cols].mean().values


def channel_medians(cols):
    return df[cols].median().values


def channel_stds(cols):
    return df[cols].std().values


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ─── 1. WPE — Wall Plug Efficiency ───────────────────────────────────────────
print("\n[1] WPE — Wall Plug Efficiency")
metric = "WPE"
unit = "%"

# 1a. Distribution histograms: one subplot per current, overlay 36C and 50C
fig = plt.figure(figsize=(15, 5))
fig.suptitle("WPE Distribution — 36°C vs 50°C", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        vals = col_vals(get_cols(metric, cur, t))
        vals = vals[~np.isnan(vals)]
        plt.hist(vals, bins=60, alpha=ALPHA + 0.1, color=COLORS[t],
                 label=f"{t}°C (μ={vals.mean():.2f})", density=True)
        # KDE overlay
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde = stats.gaussian_kde(vals)
        plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=1.8)
    plt.xlabel(f"WPE ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.legend()
fig.tight_layout()
save(fig, "01a_wpe_distribution_by_current.png")

# 1b. Channel-by-channel mean ± std — one subplot per current
fig = plt.figure(figsize=(15, 5))
fig.suptitle("WPE — Per-Channel Mean ± Std (36°C vs 50°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        cols = get_cols(metric, cur, t)
        means = channel_means(cols)
        stds = channel_stds(cols)
        plt.errorbar(CHANNELS, means, yerr=stds, color=COLORS[t], marker="o",
                     markersize=4, linewidth=1.5, capsize=3, label=f"{t}°C")
    plt.xlabel("Channel")
    plt.ylabel(f"WPE ({unit})" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
    plt.legend()
fig.tight_layout()
save(fig, "01b_wpe_channel_mean_by_current.png")

# 1c. Temperature sensitivity: ΔWPE = WPE_36 - WPE_50, per channel per current
fig = plt.figure(figsize=(15, 5))
fig.suptitle("WPE Temperature Sensitivity: Δ(36°C − 50°C) per Channel", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    delta_means = channel_means(cols36) - channel_means(cols50)
    delta_stds = np.sqrt(channel_stds(cols36) ** 2 + channel_stds(cols50) ** 2)
    plt.bar(CHANNELS, delta_means, yerr=delta_stds, color="#4CAF50", alpha=0.7,
            capsize=3, width=0.6)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Channel")
    plt.ylabel("ΔWPE (pp)" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
fig.tight_layout()
save(fig, "01c_wpe_temperature_delta.png")

# 1d. Boxplot per current level, grouped by temperature
fig = plt.figure(figsize=(14, 6))
fig.suptitle("WPE Box Plot — All Channels, All Currents (36°C vs 50°C)", fontweight="bold")
ax = plt.gca()
positions, data_36, data_50 = [], [], []
xtick_pos, xtick_lbl = [], []
offset = 0
for cur in CURRENTS:
    vals36 = col_vals(get_cols(metric, cur, 36))
    vals50 = col_vals(get_cols(metric, cur, 50))
    data_36.append(vals36[~np.isnan(vals36)])
    data_50.append(vals50[~np.isnan(vals50)])
    xtick_pos.append(offset + 1.25)
    xtick_lbl.append(f"{cur} mA")
    offset += 3

for j, (v36, v50, base) in enumerate(zip(data_36, data_50, range(0, 9, 3))):
    bp36 = plt.boxplot(v36, positions=[base + 0.75], widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color="white", linewidth=2))
    bp50 = plt.boxplot(v50, positions=[base + 1.75], widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(color="white", linewidth=2))
    for patch in bp36["boxes"]:
        patch.set_facecolor(COLORS[36])
        patch.set_alpha(0.75)
    for patch in bp50["boxes"]:
        patch.set_facecolor(COLORS[50])
        patch.set_alpha(0.75)

plt.xticks(xtick_pos, xtick_lbl)
plt.ylabel(f"WPE ({unit})")
legend_elems = [Line2D([0], [0], color=COLORS[36], lw=8, alpha=0.75, label="36°C"),
                Line2D([0], [0], color=COLORS[50], lw=8, alpha=0.75, label="50°C")]
plt.legend(handles=legend_elems)
fig.tight_layout()
save(fig, "01d_wpe_boxplot_comparison.png")

# 1e. WPE scatter: 36C vs 50C (per device, averaged over channels)
fig = plt.figure(figsize=(12, 4))
fig.suptitle("WPE Scatter: 36°C vs 50°C (per device, per current)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    mean36 = df[cols36].mean(axis=1)
    mean50 = df[cols50].mean(axis=1)
    plt.scatter(mean36, mean50, alpha=0.3, s=6, color="#9C27B0")
    lims = [min(mean36.min(), mean50.min()), max(mean36.max(), mean50.max())]
    plt.plot(lims, lims, "k--", linewidth=0.8, label="y=x")
    r, p = stats.pearsonr(mean36.dropna(), mean50.dropna())
    plt.xlabel("WPE @ 36°C (%)")
    plt.ylabel("WPE @ 50°C (%)" if i == 0 else "")
    plt.title(f"{cur} mA  (r={r:.3f})")
    plt.legend(fontsize=8)
fig.tight_layout()
save(fig, "01e_wpe_scatter_36_vs_50.png")


# ─── 2. FREQ — Lasing Frequency ───────────────────────────────────────────────
print("\n[2] FREQ — Lasing Frequency")
metric = "FREQ"
unit = "GHz"

fig = plt.figure(figsize=(15, 5))
fig.suptitle("FREQ Distribution — 36°C vs 50°C", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        vals = col_vals(get_cols(metric, cur, t))
        vals = vals[~np.isnan(vals)]
        plt.hist(vals, bins=60, alpha=ALPHA + 0.1, color=COLORS[t],
                 label=f"{t}°C (μ={vals.mean():.2f})", density=True)
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde = stats.gaussian_kde(vals)
        plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=1.8)
    plt.xlabel(f"Frequency ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.legend()
fig.tight_layout()
save(fig, "02a_freq_distribution_by_current.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("FREQ — Per-Channel Mean ± Std (36°C vs 50°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        cols = get_cols(metric, cur, t)
        means = channel_means(cols)
        stds = channel_stds(cols)
        plt.errorbar(CHANNELS, means, yerr=stds, color=COLORS[t], marker="o",
                     markersize=4, linewidth=1.5, capsize=3, label=f"{t}°C")
    plt.xlabel("Channel")
    plt.ylabel(f"Frequency ({unit})" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
    plt.legend()
fig.tight_layout()
save(fig, "02b_freq_channel_mean_by_current.png")

# Frequency vs channel: all 3 currents on same plot for each temp
fig = plt.figure(figsize=(12, 5))
fig.suptitle("FREQ — Channel Profile: All Currents (36°C vs 50°C)", fontweight="bold")
cur_colors = {125: "#1565C0", 145: "#388E3C", 165: "#E65100"}
cur_ls = {36: "-", 50: "--"}
for i, t in enumerate(TEMPS):
    plt.subplot(1, 2, i + 1)
    for cur in CURRENTS:
        cols = get_cols(metric, cur, t)
        means = channel_means(cols)
        plt.plot(CHANNELS, means, marker="o", markersize=4, color=cur_colors[cur],
                 linewidth=1.5, label=f"{cur} mA")
    plt.xlabel("Channel")
    plt.ylabel(f"Frequency ({unit})" if i == 0 else "")
    plt.title(f"{t}°C")
    plt.xticks(CHANNELS)
    plt.legend()
fig.tight_layout()
save(fig, "02c_freq_channel_all_currents.png")

# ΔFREQ = FREQ_50 - FREQ_36 (thermal shift)
fig = plt.figure(figsize=(15, 5))
fig.suptitle("FREQ Thermal Shift: Δ(50°C − 36°C) per Channel", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    delta_means = channel_means(cols50) - channel_means(cols36)
    delta_stds = np.sqrt(channel_stds(cols36) ** 2 + channel_stds(cols50) ** 2)
    plt.bar(CHANNELS, delta_means, yerr=delta_stds, color="#FF9800", alpha=0.7,
            capsize=3, width=0.6)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Channel")
    plt.ylabel("ΔFreq (GHz)" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
fig.tight_layout()
save(fig, "02d_freq_thermal_shift.png")


# ─── 3. VLZR — Laser Voltage ──────────────────────────────────────────────────
print("\n[3] VLZR — Laser Voltage")
metric = "VLZR"
unit = "V"

fig = plt.figure(figsize=(15, 5))
fig.suptitle("VLZR Distribution — 36°C vs 50°C", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        vals = col_vals(get_cols(metric, cur, t))
        vals = vals[~np.isnan(vals)]
        plt.hist(vals, bins=60, alpha=ALPHA + 0.1, color=COLORS[t],
                 label=f"{t}°C (μ={vals.mean():.3f})", density=True)
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde = stats.gaussian_kde(vals)
        plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=1.8)
    plt.xlabel(f"Voltage ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.legend()
fig.tight_layout()
save(fig, "03a_vlzr_distribution_by_current.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("VLZR — Per-Channel Mean ± Std (36°C vs 50°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        cols = get_cols(metric, cur, t)
        means = channel_means(cols)
        stds = channel_stds(cols)
        plt.errorbar(CHANNELS, means, yerr=stds, color=COLORS[t], marker="o",
                     markersize=4, linewidth=1.5, capsize=3, label=f"{t}°C")
    plt.xlabel("Channel")
    plt.ylabel(f"Voltage ({unit})" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
    plt.legend()
fig.tight_layout()
save(fig, "03b_vlzr_channel_mean_by_current.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("VLZR Temperature Sensitivity: Δ(36°C − 50°C) per Channel", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    delta_means = channel_means(cols36) - channel_means(cols50)
    delta_stds = np.sqrt(channel_stds(cols36) ** 2 + channel_stds(cols50) ** 2)
    plt.bar(CHANNELS, delta_means, yerr=delta_stds, color="#00BCD4", alpha=0.7,
            capsize=3, width=0.6)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Channel")
    plt.ylabel("ΔV (V)" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
fig.tight_layout()
save(fig, "03c_vlzr_temperature_delta.png")

fig = plt.figure(figsize=(12, 4))
fig.suptitle("VLZR Scatter: 36°C vs 50°C (per device, per current)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    mean36 = df[cols36].mean(axis=1)
    mean50 = df[cols50].mean(axis=1)
    plt.scatter(mean36, mean50, alpha=0.3, s=6, color="#607D8B")
    lims = [min(mean36.min(), mean50.min()), max(mean36.max(), mean50.max())]
    plt.plot(lims, lims, "k--", linewidth=0.8, label="y=x")
    r, _ = stats.pearsonr(mean36.dropna(), mean50.dropna())
    plt.xlabel("VLZR @ 36°C (V)")
    plt.ylabel("VLZR @ 50°C (V)" if i == 0 else "")
    plt.title(f"{cur} mA  (r={r:.3f})")
    plt.legend(fontsize=8)
fig.tight_layout()
save(fig, "03d_vlzr_scatter_36_vs_50.png")


# ─── 4. IPD2 — Monitor Photocurrent ──────────────────────────────────────────
print("\n[4] IPD2 — Monitor Photocurrent")
metric = "IPD2"
unit = "mA"

fig = plt.figure(figsize=(15, 5))
fig.suptitle("IPD2 Distribution — 36°C vs 50°C", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        vals = col_vals(get_cols(metric, cur, t))
        vals = vals[~np.isnan(vals)]
        plt.hist(vals, bins=60, alpha=ALPHA + 0.1, color=COLORS[t],
                 label=f"{t}°C (μ={vals.mean():.3f})", density=True)
        kde_x = np.linspace(vals.min(), vals.max(), 300)
        kde = stats.gaussian_kde(vals)
        plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=1.8)
    plt.xlabel(f"Photocurrent ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.legend()
fig.tight_layout()
save(fig, "04a_ipd2_distribution_by_current.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("IPD2 — Per-Channel Mean ± Std (36°C vs 50°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    for t in TEMPS:
        cols = get_cols(metric, cur, t)
        means = channel_means(cols)
        stds = channel_stds(cols)
        plt.errorbar(CHANNELS, means, yerr=stds, color=COLORS[t], marker="o",
                     markersize=4, linewidth=1.5, capsize=3, label=f"{t}°C")
    plt.xlabel("Channel")
    plt.ylabel(f"IPD2 ({unit})" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
    plt.legend()
fig.tight_layout()
save(fig, "04b_ipd2_channel_mean_by_current.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("IPD2 Temperature Sensitivity: Δ(36°C − 50°C) per Channel", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    delta_means = channel_means(cols36) - channel_means(cols50)
    delta_stds = np.sqrt(channel_stds(cols36) ** 2 + channel_stds(cols50) ** 2)
    plt.bar(CHANNELS, delta_means, yerr=delta_stds, color="#9C27B0", alpha=0.7,
            capsize=3, width=0.6)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Channel")
    plt.ylabel("ΔIPD2 (mA)" if i == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(CHANNELS)
fig.tight_layout()
save(fig, "04c_ipd2_temperature_delta.png")

fig = plt.figure(figsize=(12, 4))
fig.suptitle("IPD2 Scatter: 36°C vs 50°C (per device, per current)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols36 = get_cols(metric, cur, 36)
    cols50 = get_cols(metric, cur, 50)
    mean36 = df[cols36].mean(axis=1)
    mean50 = df[cols50].mean(axis=1)
    plt.scatter(mean36, mean50, alpha=0.3, s=6, color="#E91E63")
    lims = [min(mean36.min(), mean50.min()), max(mean36.max(), mean50.max())]
    plt.plot(lims, lims, "k--", linewidth=0.8, label="y=x")
    r, _ = stats.pearsonr(mean36.dropna(), mean50.dropna())
    plt.xlabel("IPD2 @ 36°C (mA)")
    plt.ylabel("IPD2 @ 50°C (mA)" if i == 0 else "")
    plt.title(f"{cur} mA  (r={r:.3f})")
    plt.legend(fontsize=8)
fig.tight_layout()
save(fig, "04d_ipd2_scatter_36_vs_50.png")


# ─── 5. EACH_DF_DIV — Differential Frequency Tuning (145~165 mA) ─────────────
print("\n[5] EACH_DF_DIV — Differential Frequency Division (145–165 mA)")
unit = "GHz"

fig = plt.figure(figsize=(10, 4))
fig.suptitle("EACH_DF_DIV Distribution — 36°C vs 50°C (145–165 mA)", fontweight="bold")
for i, t in enumerate(TEMPS):
    plt.subplot(1, 2, i + 1)
    cols = get_cols_df_div(t)
    vals = col_vals(cols)
    vals = vals[~np.isnan(vals)]
    plt.hist(vals, bins=60, alpha=0.6, color=COLORS[t],
             label=f"{t}°C (μ={vals.mean():.4f})", density=True)
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    kde = stats.gaussian_kde(vals)
    plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=2)
    plt.xlabel(f"DF Div ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{t}°C")
    plt.legend()
fig.tight_layout()
save(fig, "05a_each_df_div_distribution.png")

fig = plt.figure(figsize=(10, 4))
fig.suptitle("EACH_DF_DIV — Per-Channel Mean ± Std (36°C vs 50°C)", fontweight="bold")
plt.subplot(1, 1, 1)
for t in TEMPS:
    cols = get_cols_df_div(t)
    means = channel_means(cols)
    stds = channel_stds(cols)
    plt.errorbar(CHANNELS, means, yerr=stds, color=COLORS[t], marker="o",
                 markersize=5, linewidth=2, capsize=4, label=f"{t}°C")
plt.xlabel("Channel")
plt.ylabel(f"DF Division ({unit})")
plt.title("EACH_DF_DIV per Channel (145–165 mA)")
plt.xticks(CHANNELS)
plt.legend()
fig.tight_layout()
save(fig, "05b_each_df_div_channel_mean.png")

fig = plt.figure(figsize=(8, 4))
fig.suptitle("EACH_DF_DIV: Δ(50°C − 36°C) per Channel", fontweight="bold")
cols36 = get_cols_df_div(36)
cols50 = get_cols_df_div(50)
delta_means = channel_means(cols50) - channel_means(cols36)
delta_stds = np.sqrt(channel_stds(cols36) ** 2 + channel_stds(cols50) ** 2)
plt.bar(CHANNELS, delta_means, yerr=delta_stds, color="#FF5722", alpha=0.7,
        capsize=3, width=0.6)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Channel")
plt.ylabel("Δ DF Div (GHz)")
plt.xticks(CHANNELS)
fig.tight_layout()
save(fig, "05c_each_df_div_delta.png")

fig = plt.figure(figsize=(6, 5))
fig.suptitle("EACH_DF_DIV Scatter: 36°C vs 50°C", fontweight="bold")
mean36 = df[get_cols_df_div(36)].mean(axis=1)
mean50 = df[get_cols_df_div(50)].mean(axis=1)
plt.scatter(mean36, mean50, alpha=0.3, s=6, color="#FF5722")
lims = [min(mean36.min(), mean50.min()), max(mean36.max(), mean50.max())]
plt.plot(lims, lims, "k--", linewidth=0.8, label="y=x")
r, _ = stats.pearsonr(mean36.dropna(), mean50.dropna())
plt.xlabel("DF Div @ 36°C (GHz)")
plt.ylabel("DF Div @ 50°C (GHz)")
plt.title(f"r = {r:.3f}")
plt.legend()
fig.tight_layout()
save(fig, "05d_each_df_div_scatter_36_vs_50.png")


# ─── 6. SMSR — Side Mode Suppression Ratio (36°C only) ──────────────────────
print("\n[6] SMSR — Side Mode Suppression Ratio (36°C only)")
unit = "dB"

fig = plt.figure(figsize=(15, 5))
fig.suptitle("SMSR Distribution — All Current Levels (36°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols = get_cols("SMSR", cur, 36)
    vals = col_vals(cols)
    vals = vals[~np.isnan(vals)]
    plt.hist(vals, bins=50, alpha=0.7, color=COLORS[36], density=True,
             label=f"μ={vals.mean():.2f}, σ={vals.std():.2f}")
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    kde = stats.gaussian_kde(vals)
    plt.plot(kde_x, kde(kde_x), color="darkblue", linewidth=2)
    plt.xlabel(f"SMSR ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA @ 36°C")
    plt.legend()
fig.tight_layout()
save(fig, "06a_smsr_distribution_36c.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("SMSR — Per-Channel Mean ± Std @ 36°C (All Currents)", fontweight="bold")
cur_colors_smsr = {125: "#1565C0", 145: "#2E7D32", 165: "#BF360C"}
plt.subplot(1, 1, 1)
for cur in CURRENTS:
    cols = get_cols("SMSR", cur, 36)
    means = channel_means(cols)
    stds = channel_stds(cols)
    plt.errorbar(CHANNELS, means, yerr=stds, color=cur_colors_smsr[cur], marker="o",
                 markersize=4, linewidth=1.5, capsize=3, label=f"{cur} mA")
plt.xlabel("Channel")
plt.ylabel(f"SMSR ({unit})")
plt.title("SMSR per Channel @ 36°C")
plt.xticks(CHANNELS)
plt.legend()
fig.tight_layout()
save(fig, "06b_smsr_channel_mean_36c.png")

# SMSR violin per current
fig = plt.figure(figsize=(8, 5))
fig.suptitle("SMSR — Distribution vs Current Level @ 36°C", fontweight="bold")
ax = plt.gca()
vdata = [col_vals(get_cols("SMSR", cur, 36)) for cur in CURRENTS]
vdata = [v[~np.isnan(v)] for v in vdata]
parts = plt.violinplot(vdata, positions=[1, 2, 3], showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor(COLORS[36])
    pc.set_alpha(0.6)
plt.xticks([1, 2, 3], [f"{c} mA" for c in CURRENTS])
plt.ylabel(f"SMSR ({unit})")
fig.tight_layout()
save(fig, "06c_smsr_violin_by_current_36c.png")


# ─── 7. RIN — Relative Intensity Noise (36°C only) ───────────────────────────
print("\n[7] RIN — Relative Intensity Noise (36°C only)")
unit = "dB/Hz"
rin_metric = "RIN_MAX_DB_6.5GHZ"

def get_rin_cols(current):
    prefix = f"{rin_metric}_{current}_"
    suffix = "_36"
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)],
        key=lambda c: int(c.split("_")[-2]),
    )

fig = plt.figure(figsize=(15, 5))
fig.suptitle("RIN Distribution — All Current Levels (36°C)", fontweight="bold")
for i, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, i + 1)
    cols = get_rin_cols(cur)
    vals = col_vals(cols)
    vals = vals[~np.isnan(vals)]
    plt.hist(vals, bins=50, alpha=0.7, color=COLORS[36], density=True,
             label=f"μ={vals.mean():.2f}, σ={vals.std():.2f}")
    kde_x = np.linspace(vals.min(), vals.max(), 300)
    kde = stats.gaussian_kde(vals)
    plt.plot(kde_x, kde(kde_x), color="darkblue", linewidth=2)
    plt.xlabel(f"RIN ({unit})")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"{cur} mA @ 36°C")
    plt.legend()
fig.tight_layout()
save(fig, "07a_rin_distribution_36c.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("RIN — Per-Channel Mean ± Std @ 36°C (All Currents)", fontweight="bold")
plt.subplot(1, 1, 1)
for cur in CURRENTS:
    cols = get_rin_cols(cur)
    means = channel_means(cols)
    stds = channel_stds(cols)
    plt.errorbar(CHANNELS, means, yerr=stds, color=cur_colors_smsr[cur], marker="o",
                 markersize=4, linewidth=1.5, capsize=3, label=f"{cur} mA")
plt.xlabel("Channel")
plt.ylabel(f"RIN ({unit})")
plt.title("RIN per Channel @ 36°C")
plt.xticks(CHANNELS)
plt.legend()
fig.tight_layout()
save(fig, "07b_rin_channel_mean_36c.png")

fig = plt.figure(figsize=(8, 5))
fig.suptitle("RIN — Distribution vs Current Level @ 36°C", fontweight="bold")
vdata = [col_vals(get_rin_cols(cur)) for cur in CURRENTS]
vdata = [v[~np.isnan(v)] for v in vdata]
parts = plt.violinplot(vdata, positions=[1, 2, 3], showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor(COLORS[36])
    pc.set_alpha(0.6)
plt.xticks([1, 2, 3], [f"{c} mA" for c in CURRENTS])
plt.ylabel(f"RIN ({unit})")
fig.tight_layout()
save(fig, "07c_rin_violin_by_current_36c.png")


# ─── 8. Cross-Metric Summary: Mean Values vs Current Level ───────────────────
print("\n[8] Cross-Metric Summary")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Summary: Mean Metric vs Drive Current (36°C vs 50°C)", fontweight="bold",
             fontsize=13)

metrics_info = [
    ("WPE", "%",   [(cur, t) for cur in CURRENTS for t in TEMPS]),
    ("VLZR", "V",  [(cur, t) for cur in CURRENTS for t in TEMPS]),
    ("FREQ", "GHz",[(cur, t) for cur in CURRENTS for t in TEMPS]),
    ("IPD2", "mA", [(cur, t) for cur in CURRENTS for t in TEMPS]),
]

for idx, (met, u, combos) in enumerate(metrics_info):
    plt.subplot(2, 2, idx + 1)
    for t in TEMPS:
        means_per_cur = []
        stds_per_cur = []
        for cur in CURRENTS:
            vals = col_vals(get_cols(met, cur, t))
            vals = vals[~np.isnan(vals)]
            means_per_cur.append(vals.mean())
            stds_per_cur.append(vals.std())
        plt.errorbar(CURRENTS, means_per_cur, yerr=stds_per_cur,
                     color=COLORS[t], marker="o", markersize=6,
                     linewidth=2, capsize=4, label=f"{t}°C")
    plt.xlabel("Drive Current (mA)")
    plt.ylabel(f"{met} ({u})")
    plt.title(met)
    plt.xticks(CURRENTS)
    plt.legend()

fig.tight_layout()
save(fig, "08_summary_mean_vs_current.png")


# ─── 9. Per-Bin Analysis: WPE and FREQ distribution by bin ──────────────────
print("\n[9] Per-Bin Analysis")
bins = sorted(df["bin"].unique())

fig = plt.figure(figsize=(15, 5))
fig.suptitle("WPE Distribution by Bin — 36°C vs 50°C (145 mA)", fontweight="bold")
for i, b in enumerate(bins):
    plt.subplot(1, len(bins), i + 1)
    sub = df[df["bin"] == b]
    for t in TEMPS:
        cols = get_cols("WPE", 145, t)
        vals = sub[cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        plt.hist(vals, bins=40, alpha=0.5, color=COLORS[t],
                 label=f"{t}°C (n={len(vals)})", density=True)
        kde_x = np.linspace(vals.min(), vals.max(), 200)
        kde = stats.gaussian_kde(vals)
        plt.plot(kde_x, kde(kde_x), color=COLORS[t], linewidth=1.8)
    plt.xlabel("WPE (%)")
    plt.ylabel("Density" if i == 0 else "")
    plt.title(f"Bin {b}")
    plt.legend(fontsize=8)
fig.tight_layout()
save(fig, "09a_wpe_by_bin_145mA.png")

fig = plt.figure(figsize=(15, 5))
fig.suptitle("WPE Mean by Bin and Current — 36°C vs 50°C", fontweight="bold")
x = np.arange(len(bins))
width = 0.35
for ti, t in enumerate(TEMPS):
    for ci, cur in enumerate(CURRENTS):
        bin_means = []
        bin_stds = []
        for b in bins:
            sub = df[df["bin"] == b]
            cols = get_cols("WPE", cur, t)
            vals = sub[cols].values.flatten()
            vals = vals[~np.isnan(vals)]
            bin_means.append(vals.mean() if len(vals) > 0 else np.nan)
            bin_stds.append(vals.std() if len(vals) > 0 else np.nan)

fig = plt.figure(figsize=(14, 5))
fig.suptitle("WPE Mean per Bin per Current (36°C vs 50°C)", fontweight="bold")
for ci, cur in enumerate(CURRENTS):
    plt.subplot(1, 3, ci + 1)
    x = np.arange(len(bins))
    w = 0.35
    for ti, t in enumerate(TEMPS):
        bin_means = []
        bin_stds = []
        for b in bins:
            sub = df[df["bin"] == b]
            cols = get_cols("WPE", cur, t)
            vals = sub[cols].values.flatten()
            vals = vals[~np.isnan(vals)]
            bin_means.append(vals.mean() if len(vals) > 0 else np.nan)
            bin_stds.append(vals.std() if len(vals) > 0 else np.nan)
        plt.bar(x + ti * w, bin_means, width=w, color=COLORS[t], alpha=0.75,
                label=f"{t}°C", yerr=bin_stds, capsize=3)
    plt.xlabel("Bin")
    plt.ylabel("WPE (%)" if ci == 0 else "")
    plt.title(f"{cur} mA")
    plt.xticks(x + w / 2, [f"Bin {b}" for b in bins])
    plt.legend()
fig.tight_layout()
save(fig, "09b_wpe_mean_per_bin_per_current.png")


# ─── 10. Summary Statistics Table ────────────────────────────────────────────
print("\n[10] Summary Statistics Table")

rows = []
for met, u in [("WPE", "%"), ("FREQ", "GHz"), ("VLZR", "V"), ("IPD2", "mA")]:
    for cur in CURRENTS:
        for t in TEMPS:
            vals = col_vals(get_cols(met, cur, t))
            vals = vals[~np.isnan(vals)]
            rows.append({
                "Metric": met,
                "Unit": u,
                "Current (mA)": cur,
                "Temp (°C)": t,
                "N": len(vals),
                "Mean": round(vals.mean(), 4),
                "Median": round(np.median(vals), 4),
                "Std": round(vals.std(), 4),
                "P5": round(np.percentile(vals, 5), 4),
                "P95": round(np.percentile(vals, 95), 4),
                "Min": round(vals.min(), 4),
                "Max": round(vals.max(), 4),
            })

summary_df = pd.DataFrame(rows)
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_statistics.csv")
summary_df.to_csv(csv_path, index=False)
print(f"  Saved: summary_statistics.csv")

# Plot summary table as a figure
fig = plt.figure(figsize=(18, len(rows) * 0.35 + 1.5))
fig.suptitle("Summary Statistics Table", fontweight="bold")
ax = plt.gca()
ax.axis("off")
col_labels = list(summary_df.columns)
tbl = plt.table(
    cellText=summary_df.values.tolist(),
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.auto_set_column_width(list(range(len(col_labels))))
fig.tight_layout()
save(fig, "10_summary_statistics_table.png")


# ─── 11. Temperature Coefficient Summary ─────────────────────────────────────
print("\n[11] Temperature Coefficient Summary")

fig = plt.figure(figsize=(14, 6))
fig.suptitle("Temperature Coefficients: Δ per °C (per metric, per current)", fontweight="bold")

metrics_tc = [("WPE", "%"), ("VLZR", "V"), ("FREQ", "GHz"), ("IPD2", "mA")]
delta_T = 50 - 36

for idx, (met, u) in enumerate(metrics_tc):
    plt.subplot(1, 4, idx + 1)
    tc_vals = []
    for cur in CURRENTS:
        vals36 = col_vals(get_cols(met, cur, 36))
        vals50 = col_vals(get_cols(met, cur, 50))
        mean36 = np.nanmean(vals36)
        mean50 = np.nanmean(vals50)
        tc = (mean50 - mean36) / delta_T
        tc_vals.append(tc)
    colors_bar = ["#1565C0", "#388E3C", "#E65100"]
    bars = plt.bar([f"{c}mA" for c in CURRENTS], tc_vals,
                   color=colors_bar, alpha=0.75, width=0.5)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Current")
    plt.ylabel(f"TC ({u}/°C)" if idx == 0 else "")
    plt.title(f"{met}\n(Δ/°C)")
    for bar, val in zip(bars, tc_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=7)

fig.tight_layout()
save(fig, "11_temperature_coefficients.png")


# ─── 12. Channel Uniformity: CV (Std/Mean) per metric per temperature ─────────
print("\n[12] Channel Uniformity (CV)")

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Channel Uniformity — Coefficient of Variation (Std/Mean) per Channel",
             fontweight="bold")
metrics_cv = [("WPE", "%"), ("FREQ", "GHz"), ("VLZR", "V"), ("IPD2", "mA")]

for idx, (met, u) in enumerate(metrics_cv):
    plt.subplot(2, 2, idx + 1)
    for cur in CURRENTS:
        for t in TEMPS:
            cols = get_cols(met, cur, t)
            cv_vals = []
            for col in cols:
                col_data = df[col].dropna()
                if len(col_data) > 1 and col_data.mean() != 0:
                    cv_vals.append(100 * col_data.std() / abs(col_data.mean()))
                else:
                    cv_vals.append(np.nan)
            ls = "-" if t == 36 else "--"
            color = cur_colors[cur]
            plt.plot(CHANNELS, cv_vals, linestyle=ls, color=color, marker="o",
                     markersize=3, linewidth=1.2, alpha=0.8,
                     label=f"{cur}mA/{t}°C")
    plt.xlabel("Channel")
    plt.ylabel("CV (%)" if idx % 2 == 0 else "")
    plt.title(f"{met}")
    plt.xticks(CHANNELS)
    plt.legend(fontsize=7, ncol=2)

fig.tight_layout()
save(fig, "12_channel_uniformity_cv.png")


print(f"\n{'='*60}")
print(f"Analysis complete! {len(os.listdir(OUT_DIR))} files saved to:")
print(f"  {OUT_DIR}")
print("="*60)
