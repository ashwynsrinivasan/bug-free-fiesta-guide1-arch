"""
ES0 tested dies — WLT analysis.

Script form of ``wlt_data_analysis.ipynb`` in the parent ``data_analysis`` folder.
Reads CSV summaries from ``data_analysis/``. Running this module (``python wlt_data_analysis.py``)
always writes every figure to ``wlt/results/`` as PNG (no extra flags).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

_DATA_ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _csv(name: str) -> Path:
    return _DATA_ANALYSIS_DIR / name


def _save_fig(fig, filename: str) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(_RESULTS_DIR / filename))


def load_and_prepare_wlt_data() -> pd.DataFrame:
    wlt_data_20250225 = pd.read_csv(_csv("wlt_data_summary_20250225.csv"))
    wlt_data_20250225["Date"] = "2025-02-25"
    wlt_data_20250225["TBI"] = False
    wlt_data_20250228 = pd.read_csv(_csv("wlt_data_summary_20250228.csv"))
    wlt_data_20250228["Date"] = "2025-02-28"
    wlt_data_20250228["TBI"] = False
    wlt_data_20250403 = pd.read_csv(_csv("wlt_data_summary_20250403.csv"))
    wlt_data_20250403["Date"] = "2025-04-03"
    wlt_data_20250403["TBI"] = True

    wlt_data_20250507_1 = pd.read_csv(_csv("wlt_data_summary_20250507_1.csv"))
    wlt_data_20250507_1["Date"] = "2025-05-07"
    wlt_data_20250507_1["TBI"] = False

    wlt_data_20250507_2 = pd.read_csv(_csv("wlt_data_summary_20250507_2.csv"))
    wlt_data_20250507_2["Date"] = "2025-05-07"
    wlt_data_20250507_2["TBI"] = False

    wlt_data_20250508 = pd.read_csv(_csv("wlt_data_summary_20250508.csv"))
    wlt_data_20250508["Date"] = "2025-05-08"
    wlt_data_20250508["TBI"] = True

    wlt_data = pd.concat(
        [
            wlt_data_20250225,
            wlt_data_20250228,
            wlt_data_20250403,
            wlt_data_20250507_1,
            wlt_data_20250507_2,
            wlt_data_20250508,
        ],
        ignore_index=True,
    )
    wlt_data["Diode Bias (V)"] = (
        10 ** (wlt_data["Minimum Facet Power (dBm)"] / 10)
        / wlt_data["Total WPE Estimate (%)"]
        / 1000
        * 100
        / 145e-3
    )
    return wlt_data


def assign_phase1(wlt_data: pd.DataFrame) -> None:
    # Product Samples
    power_trim_product = 13.5
    WPE_trim_product = 11
    freq_trim_product = 25
    # smsr_trim_product = 50
    # rin_trim_product = -155

    # Test Samples
    power_trim_testing = 12.5
    WPE_trim_testing = 10

    # Rests are Mechanical Samples
    wlt_data["Phase1"] = "Mechanical Sample"
    wlt_data.loc[
        (wlt_data["Minimum Facet Power (dBm)"] >= power_trim_testing)
        & (wlt_data["Total WPE Estimate (%)"] >= WPE_trim_testing),
        "Phase1",
    ] = "Test Sample"
    wlt_data.loc[
        (wlt_data["Minimum Facet Power (dBm)"] > power_trim_product)
        & (wlt_data["Worst Relative Frequency Error (GHz)"] < freq_trim_product)
        & (wlt_data["Total WPE Estimate (%)"] > WPE_trim_product),
        "Phase1",
    ] = "Product Sample"


def plot_histograms_all_dates(wlt_data: pd.DataFrame) -> None:
    fig1 = px.histogram(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig1.update_layout(
        xaxis_range=[8, 16],
        title=dict(
            text=(
                f"Median={wlt_data['Minimum Facet Power (dBm)'].median():0.2f}, "
                f"Std={wlt_data['Minimum Facet Power (dBm)'].std():0.2f}, min spec = 14 dBm"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig1, "histogram_min_facet_power.png")

    fig2 = px.histogram(
        wlt_data,
        x="Worst Relative Frequency Error (GHz)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig2.update_layout(
        xaxis_range=[0, 50],
        title=dict(
            text=(
                f"Median={wlt_data['Worst Relative Frequency Error (GHz)'].median():0.2f}, "
                f"Std={wlt_data['Worst Relative Frequency Error (GHz)'].std():0.2f}, "
                "max spec = 36 GHz"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig2, "histogram_worst_relative_frequency_error.png")

    fig3 = px.histogram(
        wlt_data,
        x="Total WPE Estimate (%)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig3.update_layout(
        xaxis_range=[5, 20],
        title=dict(
            text=(
                f"Median={wlt_data['Total WPE Estimate (%)'].median():0.2f}, "
                f"Std={wlt_data['Total WPE Estimate (%)'].std():0.2f}, typical spec = 14%"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig3, "histogram_total_wpe_estimate.png")

    fig4 = px.histogram(
        wlt_data,
        x="Backside Temperature (C)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig4.update_layout(
        xaxis_range=[30, 60],
        title=dict(
            text=(
                f"Median={wlt_data['Backside Temperature (C)'].median():0.1f}, "
                f"Std={wlt_data['Backside Temperature (C)'].std():0.1f}, "
                "typical spec = 35 to 50C"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig4, "histogram_backside_temperature.png")

    fig5 = px.histogram(
        wlt_data,
        x="Worst RIN (dBc/Hz)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig5.update_layout(
        xaxis_range=[-160, -140],
        title=dict(
            text=(
                f"Median={wlt_data['Worst RIN (dBc/Hz)'].median():0.1f}, "
                f"Std={wlt_data['Worst RIN (dBc/Hz)'].std():0.1f}, max spec = -145 dBc/Hz"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig5, "histogram_worst_rin.png")

    fig6 = px.histogram(
        wlt_data,
        x="Worst SMSR (dB)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig6.update_layout(
        xaxis_range=[40, 60],
        title=dict(
            text=(
                f"Median={wlt_data['Worst SMSR (dB)'].median():0.1f}, "
                f"Std={wlt_data['Worst SMSR (dB)'].std():0.1f}, min spec = 40 dB"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig6, "histogram_worst_smsr.png")

    fig7 = px.histogram(
        wlt_data,
        x="Diode Bias (V)",
        color="Date",
        nbins=100,
        title="WLT Distribution",
        width=800,
        height=400,
    )
    fig7.update_layout(
        xaxis_range=[0, 2],
        title=dict(
            text=(
                f"Median={wlt_data['Diode Bias (V)'].median():0.1f}, "
                f"Std={wlt_data['Diode Bias (V)'].std():0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig7, "histogram_max_bias.png")


def plot_scatters_all_dates(wlt_data: pd.DataFrame) -> None:
    fig1 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Total WPE Estimate (%)",
        color="Date",
        width=800,
        height=400,
    )
    fig1.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Total WPE Estimate (%)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig1, "scatter_min_facet_power_vs_total_wpe_estimate.png")

    fig2 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Diode Bias (V)",
        color="Date",
        width=800,
        height=400,
    )
    fig2.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Diode Bias (V)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig2, "scatter_min_facet_power_vs_bias_voltage.png")

    fig3 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Worst Relative Frequency Error (GHz)",
        color="Date",
        width=800,
        height=400,
    )
    fig3.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Worst Relative Frequency Error (GHz)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig3, "scatter_min_facet_power_vs_worst_relative_frequency_error.png")

    fig4 = px.scatter(
        wlt_data,
        x="Worst RIN (dBc/Hz)",
        y="Worst Relative Frequency Error (GHz)",
        color="Date",
        width=800,
        height=400,
    )
    fig4.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Worst RIN (dBc/Hz)'].corr(wlt_data['Worst Relative Frequency Error (GHz)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig4, "scatter_worst_rin_vs_worst_relative_frequency_error.png")

    fig5 = px.scatter(
        wlt_data,
        x="Worst RIN (dBc/Hz)",
        y="Worst SMSR (dB)",
        color="Date",
        width=800,
        height=400,
    )
    fig5.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Worst RIN (dBc/Hz)'].corr(wlt_data['Worst SMSR (dB)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig5, "scatter_worst_rin_vs_worst_smsr.png")


def plot_scatters_phase1(wlt_data: pd.DataFrame) -> None:
    fig1 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Total WPE Estimate (%)",
        color="Phase1",
        width=800,
        height=400,
    )
    fig1.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Total WPE Estimate (%)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig1, "scatter_min_facet_power_vs_total_wpe_estimate_phase1_with_bins.png")

    fig2 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Diode Bias (V)",
        color="Phase1",
        width=800,
        height=400,
    )
    fig2.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Diode Bias (V)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig2, "scatter_min_facet_power_vs_bias_voltage_phase1_with_bins.png")

    fig3 = px.scatter(
        wlt_data,
        x="Minimum Facet Power (dBm)",
        y="Worst Relative Frequency Error (GHz)",
        color="Phase1",
        width=800,
        height=400,
    )
    fig3.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Minimum Facet Power (dBm)'].corr(wlt_data['Worst Relative Frequency Error (GHz)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig3, "scatter_min_facet_power_vs_worst_relative_frequency_error_phase1_with_bins.png")

    fig4 = px.scatter(
        wlt_data,
        x="Worst RIN (dBc/Hz)",
        y="Worst Relative Frequency Error (GHz)",
        color="Phase1",
        width=800,
        height=400,
    )
    fig4.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Worst RIN (dBc/Hz)'].corr(wlt_data['Worst Relative Frequency Error (GHz)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig4, "scatter_worst_rin_vs_worst_relative_frequency_error_phase1_with_bins.png")

    fig5 = px.scatter(
        wlt_data,
        x="Worst RIN (dBc/Hz)",
        y="Worst SMSR (dB)",
        color="Phase1",
        width=800,
        height=400,
    )
    fig5.update_layout(
        title=dict(
            text=(
                "Cross-correlation="
                f"{wlt_data['Worst RIN (dBc/Hz)'].corr(wlt_data['Worst SMSR (dB)']):0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig5, "scatter_worst_rin_vs_worst_smsr_phase1_with_bins.png")


def plot_histogram_phase1(wlt_data: pd.DataFrame) -> None:
    fig1 = px.histogram(wlt_data, x="Phase1", color="Date", width=800, height=400)
    _save_fig(fig1, "histogram_phase1_with_bins.png")


def plot_histograms_product_samples(product_samples: pd.DataFrame) -> None:
    fig1 = px.histogram(
        product_samples,
        x="Minimum Facet Power (dBm)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig1.update_layout(
        xaxis_range=[13, 16],
        title=dict(
            text=(
                f"Median={product_samples['Minimum Facet Power (dBm)'].median():0.2f}, "
                f"Std={product_samples['Minimum Facet Power (dBm)'].std():0.2f}, min spec = 14 dBm"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig1, "histogram_min_facet_power_product_samples.png")

    fig2 = px.histogram(
        product_samples,
        x="Worst Relative Frequency Error (GHz)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig2.update_layout(
        xaxis_range=[0, 50],
        title=dict(
            text=(
                f"Median={product_samples['Worst Relative Frequency Error (GHz)'].median():0.2f}, "
                f"Std={product_samples['Worst Relative Frequency Error (GHz)'].std():0.2f}, "
                "max spec = 36 GHz"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig2, "histogram_worst_relative_frequency_error_product_samples.png")

    fig3 = px.histogram(
        product_samples,
        x="Total WPE Estimate (%)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig3.update_layout(
        xaxis_range=[10, 20],
        title=dict(
            text=(
                f"Median={product_samples['Total WPE Estimate (%)'].median():0.2f}, "
                f"Std={product_samples['Total WPE Estimate (%)'].std():0.2f}, typical spec = 14%"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig3, "histogram_total_wpe_estimate_product_samples.png")

    fig4 = px.histogram(
        product_samples,
        x="Backside Temperature (C)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig4.update_layout(
        xaxis_range=[30, 60],
        title=dict(
            text=(
                f"Median={product_samples['Backside Temperature (C)'].median():0.1f}, "
                f"Std={product_samples['Backside Temperature (C)'].std():0.1f}, "
                "typical spec = 35 to 50C"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig4, "histogram_backside_temperature_product_samples.png")

    fig5 = px.histogram(
        product_samples,
        x="Worst RIN (dBc/Hz)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig5.update_layout(
        xaxis_range=[-160, -140],
        title=dict(
            text=(
                f"Median={product_samples['Worst RIN (dBc/Hz)'].median():0.1f}, "
                f"Std={product_samples['Worst RIN (dBc/Hz)'].std():0.1f}, max spec = -145 dBc/Hz"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig5, "histogram_worst_rin_product_samples.png")

    fig6 = px.histogram(
        product_samples,
        x="Worst SMSR (dB)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig6.update_layout(
        xaxis_range=[40, 60],
        title=dict(
            text=(
                f"Median={product_samples['Worst SMSR (dB)'].median():0.1f}, "
                f"Std={product_samples['Worst SMSR (dB)'].std():0.1f}, min spec = 40 dB"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig6, "histogram_worst_smsr_product_samples.png")

    fig7 = px.histogram(
        product_samples,
        x="Diode Bias (V)",
        color="Date",
        nbins=51,
        title="Product Samples",
        width=800,
        height=400,
    )
    fig7.update_layout(
        xaxis_range=[1, 1.6],
        title=dict(
            text=(
                f"Median={product_samples['Diode Bias (V)'].median():0.1f}, "
                f"Std={product_samples['Diode Bias (V)'].std():0.1f}"
            ),
            font=dict(size=25),
        ),
    )
    _save_fig(fig7, "histogram_max_bias_product_samples.png")


def main() -> None:
    """Load data, print summaries, and write all figures under ``wlt/results/``."""
    wlt_data = load_and_prepare_wlt_data()
    print(wlt_data.describe())

    plot_histograms_all_dates(wlt_data)
    plot_scatters_all_dates(wlt_data)

    print(wlt_data.columns)

    assign_phase1(wlt_data)
    print(wlt_data["Phase1"].unique())

    plot_scatters_phase1(wlt_data)
    plot_histogram_phase1(wlt_data)

    product_samples = wlt_data[wlt_data["Phase1"] == "Product Sample"]
    print(product_samples.describe())

    plot_histograms_product_samples(product_samples)

    print(f"Figures written under: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
