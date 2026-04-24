#!/usr/bin/env python3
"""
Regulators / aggressors mission-mode plots (``module_analysis.regulators_aggressors``).

**Default run (no flags)**  
Runs the gen1 mirror **and**, when ``data/clm_evt_oct2026`` (or ``--oct2026-root``) has
discoverable runs, the Oct2026 batch. Then runs ``plot_nevada_ofc_tile_timeseries.py --all-tiles``
into ``analysis/results/evt`` and, if present, ``nevada_prep/analysis/results``. Use
``--no-oct2026`` for gen1 only; ``--no-nevada-ofc`` to skip Nevada tile PNGs; ``--oct2026`` for
Oct2026 only (no gen1 mirror); Nevada tile PNGs still run after Oct2026 unless ``--no-nevada-ofc``.

**Gen1 mirror**  
Reads ``ips_clm_evt_ofc/regulators_aggressors/*.xlsx`` →
``blueray_nevada_rfp/nevada_l_tile/results/evt/regulators_aggressors/``.

**Oct 2026 batch** (``--oct2026-root``)  
- For each immediate subdirectory of ``data/clm_evt_oct2026`` that contains
  ``regulators_aggressors`` or ``regulator_aggressors``, runs the full analysis with
  that folder as ``data_path``; outputs go to
  ``.../evt/regulators_aggressors_oct2026/<subdir_name>/``.
- If no such subfolders exist, looks first for the combined EVT workbook
  ``Different VR for 30C- 45C.xlsx`` (and a few spelling variants) in the oct2026
  root and analyzes that as **one** run. If that file is missing, falls back to separate
  ``Different VR for {30..45}C.xlsx`` files (one run per temperature found). **PowerOpt**
  uses shared ``ips.clm.power.optimization.xlsx`` in the same root when present. Flat
  runs go under ``.../regulators_aggressors_oct2026/explicit_vr_30_45C/<slug>/``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _arch_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_evt_ofc_root() -> Path:
    return _arch_root() / "ips_clm_gen1" / "ips_clm_evt_ofc"


def default_results_dir() -> Path:
    return _arch_root() / "blueray_nevada_rfp" / "nevada_l_tile" / "results" / "evt" / "regulators_aggressors"


def default_evt_plot_dir() -> Path:
    """Blueray ``evt_tile_*`` PNGs (Nevada OFC broken-axis plots)."""
    return _arch_root() / "blueray_nevada_rfp" / "nevada_l_tile" / "results" / "evt"


def nevada_prep_evt_plot_dir() -> Path:
    """Optional second sink for the same tile PNGs (Nevada prep tree)."""
    return _arch_root() / "blueray_nevada_rfp" / "nevada_prep" / "analysis" / "results"


def run_nevada_ofc_tile_timeseries(output_dir: Path) -> int:
    """Run ``plot_nevada_ofc_tile_timeseries.py --all-tiles`` into ``output_dir``."""
    script = _arch_root() / "blueray_nevada_rfp" / "nevada_l_tile" / "src" / "plot_nevada_ofc_tile_timeseries.py"
    if not script.is_file():
        print(f"Skipping Nevada OFC plots (missing script): {script}", file=sys.stderr)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"Nevada OFC tile timeseries → {output_dir}")
    print("=" * 80)
    proc = subprocess.run(
        [sys.executable, str(script), "--all-tiles", "--output-dir", str(output_dir)],
        cwd=str(_arch_root()),
    )
    return int(proc.returncode)


def run_nevada_ofc_all_default_outputs() -> int:
    """Write tile PNGs to Blueray ``evt/`` and mirror to ``nevada_prep/.../results`` if ``nevada_prep`` exists."""
    primary = default_evt_plot_dir()
    rc = run_nevada_ofc_tile_timeseries(primary)
    if rc != 0:
        return rc
    nevada_prep_root = _arch_root() / "blueray_nevada_rfp" / "nevada_prep"
    if nevada_prep_root.is_dir():
        rc2 = run_nevada_ofc_tile_timeseries(nevada_prep_evt_plot_dir())
        if rc2 != 0:
            return rc2
    return 0


def default_oct2026_root() -> Path:
    # _arch_root = .../src/bug-free-fiesta-guide1-arch → repo root is parent.parent
    return _arch_root().parent.parent / "data" / "clm_evt_oct2026"


# Parent folder under regulators_aggressors_oct2026 for flat VR layouts (combined or per-°C)
OCT2026_EXPLICIT_VR_30_45_DIR = "explicit_vr_30_45C"

# Single workbook spanning 30–45 °C (try in order; first existing file wins)
OCT2026_COMBINED_VR_30_45_NAMES = (
    "Different VR for 30C- 45C.xlsx",
    "Different VR for 30C-45C.xlsx",
    "Different VR for 30C - 45C.xlsx",
)


def _import_regulators_class():
    src = default_evt_ofc_root() / "analysis_src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from module_analysis import regulators_aggressors

    return regulators_aggressors


_regulators_aggressors = None


def _ra_cls():
    global _regulators_aggressors
    if _regulators_aggressors is None:
        _regulators_aggressors = _import_regulators_class()
    return _regulators_aggressors


class BluerayRegulatorsAggressors(_ra_cls()):
    """Gen1 analysis with Blueray results under ``analysis/results/evt``."""

    def __init__(self, evt_ofc_root: Path, results_dir: Path) -> None:
        data_path = Path(evt_ofc_root) / "regulators_aggressors"
        super().__init__(
            evt_ofc_root,
            data_path=data_path,
            results_path=results_dir,
            quiet_header=True,
        )
        print("=" * 80)
        print("Regulators aggressors (Blueray EVT results)")
        print("=" * 80)
        print(f"Data path: {self.data_path}")
        print(f"Results path: {self.results_path}\n")


def _find_reg_folder(parent: Path) -> Path | None:
    for name in ("regulators_aggressors", "regulator_aggressors"):
        p = parent / name
        if p.is_dir():
            return p
    return None


def iter_oct2026_runs(
    oct_root: Path,
) -> tuple[list[tuple[str, Path, str, str]], str | None]:
    """
    Return (runs, flat_results_parent_name).

    Each run is (slug, data_path, evt_workbook, poweropt_workbook). Workbook names are
    relative to ``data_path`` (oct_root for flat VR layout).

    ``flat_results_parent_name`` is None for per-subdirectory layout; for combined or
    per-degree ``Different VR …`` workbooks it is ``OCT2026_EXPLICIT_VR_30_45_DIR``.
    """
    runs: list[tuple[str, Path, str, str]] = []
    oct_root = oct_root.resolve()
    if not oct_root.is_dir():
        return runs, None

    subdirs = [p for p in oct_root.iterdir() if p.is_dir() and p.name not in (".git",)]
    reg_subdirs = [(p.name, rf) for p in subdirs if (rf := _find_reg_folder(p)) is not None]

    if reg_subdirs:
        for slug, reg_path in reg_subdirs:
            safe = re.sub(r"[^\w.\-]+", "_", slug)
            runs.append((safe, reg_path, "ips.clm.evt.xlsx", "ips.clm.power.optimization.xlsx"))
        return runs, None

    # Flat layout: optional shared power optimization workbook
    power_shared = oct_root / "ips.clm.power.optimization.xlsx"
    power_name = "ips.clm.power.optimization.xlsx" if power_shared.is_file() else ""

    def _append_flat_run(workbook_path: Path) -> None:
        slug = re.sub(r"[^\w.\-]+", "_", workbook_path.stem.replace(" ", "_"))
        if power_name:
            runs.append((slug, oct_root, workbook_path.name, power_name))
        else:
            runs.append((slug, oct_root, workbook_path.name, workbook_path.name))

    # Prefer one combined workbook: "Different VR for 30C- 45C.xlsx", etc.
    for name in OCT2026_COMBINED_VR_30_45_NAMES:
        f = oct_root / name
        if f.is_file():
            _append_flat_run(f)
            return runs, OCT2026_EXPLICIT_VR_30_45_DIR

    # Fallback: separate "Different VR for 30C.xlsx" … "45C.xlsx"
    for t in range(30, 46):
        f = oct_root / f"Different VR for {t}C.xlsx"
        if f.is_file():
            _append_flat_run(f)

    return runs, OCT2026_EXPLICIT_VR_30_45_DIR if runs else None


def run_oct2026(oct_root: Path, results_parent: Path) -> int:
    runs, flat_parent = iter_oct2026_runs(oct_root)
    if not runs:
        print(
            f"No oct2026 runs under {oct_root} (no */regulator*_aggressors/, nor "
            f"'Different VR for 30C- 45C.xlsx', nor per-degree Different VR …C.xlsx).",
            file=sys.stderr,
        )
        return 1

    base_out = results_parent / flat_parent if flat_parent else results_parent
    RA = _ra_cls()
    layout = f"explicit VR 30-45C → {flat_parent}/" if flat_parent else "per-subfolder regulators_aggressors/"
    print(f"Oct2026 batch: {len(runs)} run(s) from {oct_root} ({layout})\n")
    for slug, data_path, evt_wb, pow_wb in runs:
        out = base_out / slug
        out.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"Run: {slug}")
        print(f"  data_path: {data_path}")
        print(f"  evt workbook: {evt_wb}")
        print(f"  poweropt workbook: {pow_wb}")
        print(f"  results: {out}")
        print("=" * 80)
        RA(
            oct_root,
            data_path=data_path,
            results_path=out,
            evt_workbook=evt_wb,
            poweropt_workbook=pow_wb,
            quiet_header=True,
        ).analyze_all()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regulators aggressors → blueray_nevada_rfp/nevada_l_tile/results/evt")
    p.add_argument(
        "--oct2026",
        action="store_true",
        help="Run only the Oct2026 batch (skip gen1). Default run already includes Oct2026 when data exists.",
    )
    p.add_argument(
        "--no-oct2026",
        action="store_true",
        help="Run only the gen1 mirror; skip Oct2026 even if data is present.",
    )
    p.add_argument(
        "--no-nevada-ofc",
        action="store_true",
        help="Skip Nevada OFC tile timeseries PNGs (plot_nevada_ofc_tile_timeseries --all-tiles).",
    )
    p.add_argument(
        "--oct2026-root",
        type=Path,
        default=None,
        help="Oct2026 data root (default: <repo>/data/clm_evt_oct2026); used by default run and --oct2026",
    )
    p.add_argument("--evt-ofc-root", type=Path, default=None, help="ips_clm_evt_ofc root (default run)")
    p.add_argument("--results-dir", type=Path, default=None, help="Output folder for default gen1 mirror run")
    args = p.parse_args(argv)

    if args.oct2026 and args.no_oct2026:
        print("Choose at most one of --oct2026 and --no-oct2026.", file=sys.stderr)
        return 2

    if args.oct2026:
        oct_root = (args.oct2026_root or default_oct2026_root()).resolve()
        results_parent = _arch_root() / "blueray_nevada_rfp" / "nevada_l_tile" / "results" / "evt" / "regulators_aggressors_oct2026"
        rc = run_oct2026(oct_root, results_parent)
        if rc != 0:
            return rc
    else:
        evt_root = (args.evt_ofc_root or default_evt_ofc_root()).resolve()
        out = (args.results_dir or default_results_dir()).resolve()

        data = evt_root / "regulators_aggressors"
        if not data.is_dir():
            print(f"Missing data directory: {data}", file=sys.stderr)
            return 1

        BluerayRegulatorsAggressors(evt_root, out).analyze_all()

        if not args.no_oct2026:
            oct_root = (args.oct2026_root or default_oct2026_root()).resolve()
            runs, _ = iter_oct2026_runs(oct_root)
            if runs:
                results_parent = (
                    _arch_root()
                    / "blueray_nevada_rfp"
                    / "nevada_l_tile"
                    / "results"
                    / "evt"
                    / "regulators_aggressors_oct2026"
                )
                rc = run_oct2026(oct_root, results_parent)
                if rc != 0:
                    return rc

    if not args.no_nevada_ofc:
        rcn = run_nevada_ofc_all_default_outputs()
        if rcn != 0:
            return rcn
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
