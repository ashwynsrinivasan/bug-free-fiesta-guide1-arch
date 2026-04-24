#!/usr/bin/env python3
"""
Load OPO tile lens coupling *Result_PF.csv and plot Coupling Loss(dB) vs channel (A0–A7, B0–B7).

Data:
  - On-disk CSVs under clm_data_onet_sftp (folders).
  - Zip/rar at the data root whose name contains "OPO tile Lens coupling original data" are
    extracted to results/opo_lens_coupling/data/<archive_stem>/ and those CSVs are loaded too.
  - If the same Tile_SN×Channel exists on-disk and in an extracted copy, the on-disk row wins.

When the same Tile_SN + Channel appears in both Others/65Pcs_* and Burn in */65Pcs_*,
one row is kept (Others path preferred).

Tiles excluded from plots/stats: Y2532000054; Y2544000406, Y2544000456, Y2609000890 (corrupt Coupling Loss in source files).

Outputs: TP2-style figures under analysis/results/ and analysis/results/opo_lens_coupling/;
extracted archives under results/opo_lens_coupling/data/;
opo_lens_coupling_corrupt_tile_report.csv (tiles with Coupling Loss outside report band; all ch rows).
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def default_data_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    return (repo_root / "data" / "clm_data_onet_sftp").resolve()


def default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "opo_lens_coupling"


def default_analysis_results_dir() -> Path:
    """Parent `results/` next to tp2p4_*.png."""
    return Path(__file__).resolve().parent.parent / "results"


# Match analyze_tp2p4_onet_sftp distribution figure sizes and gen1 box styling
DIST_FIGSIZE = (10, 5)
COMBINED_BANKS_FIGSIZE = (3, 4)
_GEN1_BOXPLOT_KW = dict(
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
    whiskerprops=dict(color="black", linewidth=2),
    capprops=dict(color="black", linewidth=2),
    medianprops=dict(color="red", linewidth=2.5),
)

# X-axis order (matches CSV Channel labels)
CHANNEL_LABELS = [f"A{i}" for i in range(8)] + [f"B{i}" for i in range(8)]

# Filename: Y##########_Result_PF.csv
_TILE_CSV = re.compile(r"^(Y\d{10})_Result_PF\.csv$", re.IGNORECASE)

# Dropped from plots/stats (all 16 channel rows per file)
# Y2532000054: positive Coupling Loss vs convention; Y2544000406 / Y2544000456 / Y2609000890: bogus Coupling Loss column in source CSVs
EXCLUDED_TILE_SN: frozenset[str] = frozenset(
    {
        "Y2532000054",
        "Y2544000406",
        "Y2544000456",
        "Y2609000890",
    }
)

# Zip/rar at data root matching this substring are extracted into results/.../data/
OPO_ARCHIVE_NAME_SUBSTR = "OPO tile Lens coupling original data"

# Report-only: flag rows with non-physical Coupling Loss(dB) (e.g. WL mixed into column).
# Does **not** filter plots or pooled stats.
CORRUPT_COUPLING_LOSS_HIGH_DB = 20.0
CORRUPT_COUPLING_LOSS_LOW_DB = -12.0


def _find_unar() -> str | None:
    brew_unar = Path("/opt/homebrew/bin/unar")
    if brew_unar.is_file():
        return str(brew_unar)
    return shutil.which("unar")


def _loss_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).strip().lower().replace(" ", "")
        if "couplingloss" in cl or (cl.startswith("coupling") and "loss" in cl):
            return c
    return None


def _read_one_result_pf(text: str, *, batch: str, rel_name: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"  skip parse {rel_name}: {e}", file=sys.stderr)
        return None
    if df.empty or "Channel" not in df.columns:
        return None
    loss_col = _loss_column(df)
    if not loss_col:
        print(f"  skip no Coupling Loss column: {rel_name}", file=sys.stderr)
        return None
    out = df[["Channel", loss_col]].copy()
    out.rename(columns={loss_col: "Coupling_Loss_dB"}, inplace=True)
    m = out["Channel"].astype(str).str.strip().str.upper().str.replace("CH", "", regex=False)
    out["Channel"] = m
    out["Coupling_Loss_dB"] = pd.to_numeric(out["Coupling_Loss_dB"], errors="coerce")
    out = out.dropna(subset=["Coupling_Loss_dB", "Channel"])
    if out.empty:
        return None
    stem = Path(rel_name).name
    mo = _TILE_CSV.match(stem)
    tile_sn = mo.group(1).upper() if mo else ""
    out["Tile_SN"] = tile_sn
    out["Batch"] = batch
    out["Source_File"] = rel_name
    return out


def _load_csv_file(path: Path, batch: str) -> pd.DataFrame | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  skip read {path}: {e}", file=sys.stderr)
        return None
    return _read_one_result_pf(text, batch=batch, rel_name=str(path))


def _iter_result_pf_csvs_under(tree_root: Path) -> list[Path]:
    """Y##########_Result_PF.csv under tree_root (recursive)."""
    found: list[Path] = []
    for csv_path in tree_root.rglob("*_Result_PF.csv"):
        if not csv_path.is_file():
            continue
        if "__MACOSX" in csv_path.parts:
            continue
        if not _TILE_CSV.match(csv_path.name):
            continue
        found.append(csv_path)
    return sorted(found, key=lambda p: str(p).lower())


def _dest_has_result_pf(dest: Path) -> bool:
    return any(dest.rglob("*_Result_PF.csv"))


def extract_opo_archives(
    data_root: Path,
    extract_root: Path,
    *,
    unar_cmd: str | None,
    force: bool,
) -> None:
    """Extract matching zip/rar from data_root into extract_root/<archive_stem>/."""
    extract_root.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []
    for p in sorted(data_root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if OPO_ARCHIVE_NAME_SUBSTR not in p.name:
            continue
        suf = p.suffix.lower()
        if suf not in (".zip", ".rar"):
            continue
        archives.append(p)

    if not archives:
        print(f"No OPO zip/rar at {data_root} (name must contain {OPO_ARCHIVE_NAME_SUBSTR!r}).")
        return

    print(f"Extracting {len(archives)} archive(s) → {extract_root} …")
    for arc in archives:
        # Same stem can exist as both .rar and .zip (e.g. 20260107); keep separate dirs.
        ext_tag = arc.suffix.lower().lstrip(".") or "arc"
        dest = extract_root / f"{arc.stem}_{ext_tag}"
        if dest.exists() and _dest_has_result_pf(dest) and not force:
            print(f"  skip (already has CSVs): {arc.name}")
            continue
        if force and dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if arc.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(arc, "r") as zf:
                    zf.extractall(dest)
                print(f"  zip → {dest.name}/")
            except zipfile.BadZipFile as e:
                print(f"  bad zip {arc.name}: {e}", file=sys.stderr)
        else:
            if not unar_cmd:
                print(f"  skip rar (install unar): {arc.name}", file=sys.stderr)
                continue
            r = subprocess.run(
                [unar_cmd, "-f", "-o", str(dest), str(arc)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode != 0:
                print(f"  unar failed {arc.name}: {r.stderr or r.stdout}", file=sys.stderr)
            else:
                print(f"  rar → {dest.name}/")


def _load_tree_grouped(
    anchor: Path,
    batch_label_for: Callable[[Path], str],
) -> pd.DataFrame:
    paths = _iter_result_pf_csvs_under(anchor)
    if not paths:
        return pd.DataFrame()

    by_parent: dict[str, list[Path]] = {}
    for p in paths:
        by_parent.setdefault(str(p.parent.resolve()), []).append(p)

    frames: list[pd.DataFrame] = []
    seen_files: set[str] = set()
    for parent, plist in sorted(by_parent.items(), key=lambda x: x[0]):
        label = batch_label_for(plist[0])
        for csv_path in plist:
            key = str(csv_path.resolve())
            if key in seen_files:
                continue
            seen_files.add(key)
            part = _load_csv_file(csv_path, batch=label)
            if part is not None:
                frames.append(part)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["Channel"].isin(CHANNEL_LABELS)].copy()


def _merge_disk_over_extract(df_disk: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """Prefer on-disk clm_data row when Tile_SN×Channel exists in both."""
    if df_ext.empty:
        return df_disk
    if df_disk.empty:
        return df_ext
    d0 = df_disk.copy()
    d0["_merge_prio"] = 0
    d1 = df_ext.copy()
    d1["_merge_prio"] = 1
    df = pd.concat([d0, d1], ignore_index=True)
    df = df.sort_values(["Tile_SN", "Channel", "_merge_prio", "Source_File"])
    before = len(df)
    df = df.drop_duplicates(subset=["Tile_SN", "Channel"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(
            f"Deduped {dropped} tile×channel rows (on-disk clm_data preferred over extracted archive)."
        )
    return df.drop(columns=["_merge_prio"])


def _dedupe_tile_channel_prefer_others(df: pd.DataFrame) -> pd.DataFrame:
    """If the same tile+channel appears in Others/65Pcs_* and Burn-in */65Pcs_*, keep Others."""
    if df.empty:
        return df
    norm = df["Source_File"].astype(str).str.replace("\\", "/")

    def _prio(src: str) -> int:
        if "Others/65Pcs_OPO" in src or "/Others/65Pcs_OPO" in src:
            return 0
        if "Burn in data/" in src and "65Pcs_OPO" in src:
            return 1
        return 2

    df = df.copy()
    df["_prio"] = norm.map(_prio)
    df = df.sort_values(["Tile_SN", "Channel", "_prio", "Source_File"])
    before = len(df)
    df = df.drop_duplicates(subset=["Tile_SN", "Channel"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Deduped {dropped} duplicate tile×channel rows (preferred Others/65Pcs over Burn-in copy).")
    return df.drop(columns=["_prio"])


def load_all_opo_lens_frames(data_root: Path, extract_root: Path | None) -> pd.DataFrame:
    data_root = data_root.resolve()
    print(f"Loading on-disk *_Result_PF.csv under {data_root} …")
    df_disk = _load_tree_grouped(
        data_root,
        lambda p: f"dir:{p.parent.resolve().relative_to(data_root).as_posix()}",
    )
    df_disk = _dedupe_tile_channel_prefer_others(df_disk)
    if not df_disk.empty:
        for name, g in sorted(df_disk.groupby("Batch"), key=lambda x: str(x[0])):
            print(f"  {name}: +{len(g)} channel-rows")

    df_ext = pd.DataFrame()
    if extract_root is not None:
        er = extract_root.resolve()
        if er.is_dir():
            print(f"Loading extracted *_Result_PF.csv under {er} …")
            df_ext = _load_tree_grouped(
                er,
                lambda p: f"extracted:{p.parent.resolve().relative_to(er).as_posix()}",
            )
            if not df_ext.empty:
                for name, g in sorted(df_ext.groupby("Batch"), key=lambda x: str(x[0])):
                    print(f"  {name}: +{len(g)} channel-rows")

    df = _merge_disk_over_extract(df_disk, df_ext)
    if df.empty:
        return df
    if EXCLUDED_TILE_SN:
        excl = {s.upper() for s in EXCLUDED_TILE_SN}
        before = len(df)
        df = df[~df["Tile_SN"].isin(excl)].copy()
        dropped = before - len(df)
        if dropped:
            print(f"Excluded tile(s) {sorted(excl)}: dropped {dropped} channel-rows.")
    df["Channel"] = pd.Categorical(df["Channel"], categories=CHANNEL_LABELS, ordered=True)
    return df


def corrupt_coupling_loss_mask(series: pd.Series) -> pd.Series:
    """True where Coupling_Loss_dB is outside the physical band used for reporting only."""
    v = pd.to_numeric(series, errors="coerce")
    return (v > CORRUPT_COUPLING_LOSS_HIGH_DB) | (v < CORRUPT_COUPLING_LOSS_LOW_DB)


def build_corrupt_tile_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiles that have at least one corrupt Coupling Loss(dB) row; returns **all** channel rows
    for those tiles (same columns as df), sorted for readability.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    bad = corrupt_coupling_loss_mask(df["Coupling_Loss_dB"])
    corrupt_tiles = df.loc[bad, "Tile_SN"].dropna().unique()
    if len(corrupt_tiles) == 0:
        return pd.DataFrame(columns=df.columns)
    out = df[df["Tile_SN"].isin(corrupt_tiles)].copy()
    out["Coupling_Loss_dB"] = pd.to_numeric(out["Coupling_Loss_dB"], errors="coerce")
    ch_str = out["Channel"].astype(str)
    out["_ch_order"] = ch_str.map({c: i for i, c in enumerate(CHANNEL_LABELS)})
    out = out.sort_values(["Tile_SN", "Batch", "_ch_order"]).drop(columns=["_ch_order"])
    return out


def write_corrupt_tile_report(df: pd.DataFrame, output_path: Path) -> int:
    """Write CSV of all rows for tiles with any corrupt loss value. Returns row count written."""
    report = build_corrupt_tile_report(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report.empty:
        # Still write header-only + note
        pd.DataFrame(
            columns=["Tile_SN", "Channel", "Coupling_Loss_dB", "Batch", "Source_File"]
        ).to_csv(output_path, index=False)
        print(f"Saved (no corrupt tiles): {output_path}")
        return 0
    report.to_csv(output_path, index=False)
    n_tiles = report["Tile_SN"].nunique()
    print(
        f"Saved corrupt-tile report: {output_path} "
        f"({n_tiles} tile(s), {len(report)} channel-rows; "
        f"flag if loss > {CORRUPT_COUPLING_LOSS_HIGH_DB} or < {CORRUPT_COUPLING_LOSS_LOW_DB} dB)"
    )
    return len(report)


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


def _combined_banks_strip_xaxis(ax) -> None:
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)


def _grey_violin_behind_box(ax, values: np.ndarray, *, position: float = 0.0, width: float = 0.62) -> None:
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


def _coupling_ylim_ticks(vals: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Pad min/max; tick step 0.5 dB if span ≤ 6 else 1 dB."""
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    pad = 0.5
    ylo = float(np.floor((vmin - pad) * 2) / 2)
    yhi = float(np.ceil((vmax + pad) * 2) / 2)
    span = yhi - ylo
    step = 0.5 if span <= 6.0 else 1.0
    yticks = np.arange(ylo, yhi + step * 0.25, step)
    return ylo, yhi, yticks


def _vertical_gen1_boxplot_violin_coupling(
    ax,
    box_data: list[np.ndarray],
    box_positions: list[int],
    *,
    ylo: float,
    yhi: float,
    box_width: float = 0.55,
    annotation_pad_frac: float = 0.055,
    annotation_fontsize: int = 6,
) -> None:
    """Same layout as TP2-4 freq distribution; annotations in dB (one decimal)."""
    if not box_data:
        ax.set_ylim(ylo, yhi)
        return
    y_ann = ylo + (yhi - ylo) * annotation_pad_frac
    u = "dB"
    violin_width = 0.62
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        _grey_violin_behind_box(ax, vals, position=float(pos), width=violin_width)
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        vert=True,
        widths=box_width,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        std = float(np.std(vals))
        fmt = f"μ̃={med:.1f}{u}\nσ={std:.1f}{u}"
        ax.text(
            pos,
            y_ann,
            fmt,
            fontsize=annotation_fontsize,
            ha="center",
            va="bottom",
            zorder=6,
            clip_on=False,
        )


def _save_figure(fig, output_path: Path, *, dpi: int = 1200) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_opo_lens_distribution_vs_loss(df: pd.DataFrame, output_path: Path) -> None:
    """Per-channel gen1 + grey violin (TP2-4 distribution style)."""
    if df.empty:
        print("No data for OPO lens distribution vs loss plot")
        return
    vals_all = df["Coupling_Loss_dB"].values.astype(float)
    ylo, yhi, yticks = _coupling_ylim_ticks(vals_all)

    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for pos, ch in enumerate(CHANNEL_LABELS):
        sub = df[df["Channel"] == ch]["Coupling_Loss_dB"].values.astype(float)
        if sub.size == 0:
            continue
        box_data.append(sub)
        box_positions.append(pos)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_violin_coupling(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.55,
        annotation_fontsize=6,
    )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coupling Loss (dB)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 15.5)
    _save_figure(fig, output_path)


def plot_opo_lens_distribution_vs_loss_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """All channels pooled; 3×4 combined_banks panel; stripped x-axis (TP2 style)."""
    if df.empty:
        print("No data for OPO lens combined distribution plot")
        return
    vals = df["Coupling_Loss_dB"].values.astype(float)
    ylo, yhi, yticks = _coupling_ylim_ticks(vals)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=COMBINED_BANKS_FIGSIZE, layout="constrained")
    pos = 0.0
    _grey_violin_behind_box(ax, vals, position=pos)
    bp = ax.boxplot(
        [vals],
        positions=[pos],
        vert=True,
        widths=0.35,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    med = float(np.median(vals))
    std = float(np.std(vals))
    u = "dB"
    ax.set_title(
        f"μ̃={med:.2f}{u}, σ={std:.2f}{u}",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel("Coupling Loss (dB)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 0.5)
    _combined_banks_strip_xaxis(ax)
    _save_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OPO lens coupling loss vs channel (TP2-style gen1 + combined).")
    parser.add_argument("--data-root", type=Path, default=None, help="clm_data_onet_sftp root")
    parser.add_argument("--results", type=Path, default=None, help="Output subdirectory (opo_lens_coupling)")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract zip/rar (reuse existing results/.../data/).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archives even if destination folders already contain CSVs.",
    )
    args = parser.parse_args()

    data_root = (args.data_root or default_data_root()).resolve()
    results_sub = (args.results or default_results_dir()).resolve()
    results_root = default_analysis_results_dir().resolve()
    extract_root = results_sub / "data"

    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    unar_cmd = _find_unar()
    if not args.skip_extract:
        extract_opo_archives(
            data_root,
            extract_root,
            unar_cmd=unar_cmd,
            force=args.force_extract,
        )
    else:
        print(f"Skipping archive extraction; loading from {extract_root} if present.")

    df = load_all_opo_lens_frames(data_root, extract_root)
    if df.empty:
        raise SystemExit("No Coupling Loss rows loaded; check paths and CSV schema.")

    n_tiles = df["Tile_SN"].nunique()
    n_rows = len(df)
    print(f"Pooled: {n_rows} channel measurements, {n_tiles} tiles, folder-groups: {df['Batch'].nunique()}")

    dist_name = "opo_lens_coupling_distribution_vs_loss.png"
    combined_name = "opo_lens_coupling_distribution_vs_loss_combined_banks.png"

    for base in (results_root, results_sub):
        plot_opo_lens_distribution_vs_loss(df, base / dist_name)
        plot_opo_lens_distribution_vs_loss_combined_banks(df, base / combined_name)

    summ = (
        df.groupby("Channel", observed=True)["Coupling_Loss_dB"]
        .agg(count="count", median="median", mean="mean", std="std")
        .reindex(CHANNEL_LABELS)
    )
    csv_path = results_sub / "opo_lens_coupling_loss_by_channel_summary.csv"
    summ.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    corrupt_path = results_sub / "opo_lens_coupling_corrupt_tile_report.csv"
    write_corrupt_tile_report(df, corrupt_path)


if __name__ == "__main__":
    main()
