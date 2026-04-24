#!/usr/bin/env python3
"""
TP2-4 Scan: reconcile per-channel OSA power (``OSAl_Power(dBm)``) with bank ``Power(mW)``.

The CSV repeats one **bank-level** ``Power(mW)`` value on each of the eight channel rows. That column is
treated here as **total power in the fiber** for that bank at the mux temperature of the row.

Cross-check (MFG v1 EVT-style tiles): summing linear power from ``OSAl_Power(dBm)`` over the eight DWDM
channels typically lands near **~0.3–0.8×** ``Power(mW)`` depending on tile/bank — it does **not**
match ``Power(mW)`` or ``2×Power(mW)`` without scaling.

When you need channel powers that **add up to a defined bank total** while keeping the **same relative
shape** as the OSA measurements, use a single scale factor per bank (per snapshot):

    P_i_scaled = P_i_linear × (target_total / sum_j P_j_linear)

with ``P_i_linear = 10^(OSAl_Power(dBm)/10)`` mW. Default ``target_total = 2 * Power(mW)`` (override
if your calibration uses a different target).

CLI: summarize CSVs or print one file in long form with ``OSA_Power_mW_raw`` and ``OSA_Power_mW_scaled``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

OSA_DBM_COL = "OSAl_Power(dBm)"
FIBER_POWER_COL = "Power(mW)"
TMUX_COL = "T_MUX(C)"


def osal_dbm_to_mw(dbm: pd.Series) -> pd.Series:
    x = pd.to_numeric(dbm, errors="coerce")
    return 10.0 ** (x / 10.0)


def tile_sn_from_tp2p4_path(csv_path: Path) -> str | None:
    for part in csv_path.stem.split("-"):
        if re.fullmatch(r"Y\d{8,10}", part):
            return part
    return None


def _snapshot_keys(df: pd.DataFrame) -> list[str]:
    keys = ["Bank"]
    if "Time" in df.columns:
        keys.append("Time")
    return keys


def reconcile_tp2p4_dataframe(
    df: pd.DataFrame,
    *,
    target_multiplier: float = 2.0,
    tmux_lo: float = 49.9,
    tmux_hi: float = 50.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return ``(enriched_rows, bank_summary)``.

    - ``enriched_rows``: original rows (after T_MUX filter) with ``OSA_Power_mW_raw``,
      ``OSA_Power_mW_scaled``, ``Fiber_Power_mW_bank``, ``Target_total_mW_bank``, ``OSA_scale_to_target``.
    - ``bank_summary``: one row per bank snapshot with sums and scale factor.

    Requires eight channels (0..7) per snapshot; others are skipped with a warning.
    """
    if OSA_DBM_COL not in df.columns or FIBER_POWER_COL not in df.columns:
        raise ValueError(f"Need columns {OSA_DBM_COL!r} and {FIBER_POWER_COL!r}")

    work = df.copy()
    if TMUX_COL in work.columns:
        work = work[(work[TMUX_COL] >= tmux_lo) & (work[TMUX_COL] <= tmux_hi)].copy()
    work["_osa_mw"] = osal_dbm_to_mw(work[OSA_DBM_COL])

    snap_keys = _snapshot_keys(work)
    enriched_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for group_key, g in work.groupby(snap_keys, sort=True):
        g = g.copy()
        if isinstance(group_key, tuple):
            bank = group_key[0]
        else:
            bank = group_key
        if len(g) != 8 or set(g["Channel"].tolist()) != set(range(8)):
            print(
                f"skip snapshot Bank={bank!r} keys={group_key!r}: "
                f"expected 8 rows ch 0-7, got {len(g)} rows",
                file=sys.stderr,
            )
            continue

        fiber = float(pd.to_numeric(g[FIBER_POWER_COL], errors="coerce").mean())
        sum_osa = float(g["_osa_mw"].sum())
        target = float(target_multiplier) * fiber
        if sum_osa <= 0 or not (fiber > 0):
            scale = float("nan")
            g["OSA_Power_mW_scaled"] = float("nan")
        else:
            scale = target / sum_osa
            g["OSA_Power_mW_scaled"] = g["_osa_mw"] * scale

        g["OSA_Power_mW_raw"] = g["_osa_mw"]
        g["Fiber_Power_mW_bank"] = fiber
        g["Target_total_mW_bank"] = target
        g["OSA_scale_to_target"] = scale
        enriched_parts.append(g.drop(columns=["_osa_mw"]))

        summary_rows.append(
            {
                "Bank": bank,
                "Fiber_Power_mW": fiber,
                "Target_total_mW": target,
                "Sum_OSA_Power_mW_raw": sum_osa,
                "sum_OSA_div_fiber": sum_osa / fiber if fiber else float("nan"),
                "OSA_scale_to_target": scale,
            }
        )

    if not enriched_parts:
        return pd.DataFrame(), pd.DataFrame(summary_rows)

    out = pd.concat(enriched_parts, ignore_index=True)
    summ = pd.DataFrame(summary_rows)
    return out, summ


def collect_evt_tp2p4_osa_data(
    *,
    mfg_base: Path,
    evt_tile_sn: set[str],
    target_multiplier: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load EVT TP2-4 Scan rows with OSA linear + scaled channel power (T_MUX ~50C).

    Returns ``(bank_summary_df, per_channel_df)`` — empty frames if nothing loads.
    """
    summary_rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []

    for version, sub in [("v1", mfg_base / "clm_mfg_data_v1"), ("v2", mfg_base / "clm_mfg_data_v2")]:
        tp4 = sub / "TP2-4"
        if not tp4.is_dir():
            continue
        for csv_file in sorted(tp4.glob("*TP2-4 Scan.csv")):
            tile = tile_sn_from_tp2p4_path(csv_file)
            if tile is None or tile not in evt_tile_sn:
                continue
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"OSA EVT collect: skip {csv_file.name}: {e}", file=sys.stderr)
                continue
            df = df.copy()
            df["Tile_SN"] = tile
            enriched, summ = reconcile_tp2p4_dataframe(df, target_multiplier=target_multiplier)
            if summ.empty:
                continue
            for _, r in summ.iterrows():
                summary_rows.append(
                    {
                        "mfg_version": version,
                        "source_file": csv_file.name,
                        "tile_sn": tile,
                        **r.to_dict(),
                    }
                )
            if not enriched.empty:
                e2 = enriched.copy()
                e2["mfg_version"] = version
                e2["source_file"] = csv_file.name
                long_parts.append(e2)

    sdf = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    ldf = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    return sdf, ldf


def write_evt_tp2p4_osa_csvs(
    summary_df: pd.DataFrame,
    per_channel_df: pd.DataFrame,
    results_dir: Path,
) -> tuple[Path | None, Path | None]:
    """Write ``evt_tp2p4_osa_power_*.csv``; returns paths or None if skipped."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path: Path | None = None
    long_path: Path | None = None

    if not summary_df.empty:
        summary_path = results_dir / "evt_tp2p4_osa_power_bank_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path} ({len(summary_df)} rows)")
    else:
        print("No EVT TP2-4 OSA bank summaries; skip evt_tp2p4_osa_power_bank_summary.csv")

    if not per_channel_df.empty:
        long_path = results_dir / "evt_tp2p4_osa_power_per_channel.csv"
        per_channel_df.to_csv(long_path, index=False)
        print(f"Saved: {long_path} ({len(per_channel_df)} rows)")
    else:
        print("No EVT TP2-4 OSA per-channel rows; skip evt_tp2p4_osa_power_per_channel.csv")

    return summary_path, long_path


def export_evt_tp2p4_osa_power_csvs(
    *,
    mfg_base: Path,
    results_dir: Path,
    evt_tile_sn: set[str],
    target_multiplier: float = 2.0,
) -> tuple[Path | None, Path | None]:
    """
    Scan ``clm_mfg_data_v1`` / ``v2`` TP2-4 for tiles in ``evt_tile_sn``; write Blueray results:

    - ``evt_tp2p4_osa_power_bank_summary.csv`` — one row per bank snapshot
    - ``evt_tp2p4_osa_power_per_channel.csv`` — per-channel raw/scaled OSA mW and fiber targets

    Returns ``(summary_path_or_none, per_channel_path_or_none)``.
    """
    sdf, ldf = collect_evt_tp2p4_osa_data(
        mfg_base=mfg_base,
        evt_tile_sn=evt_tile_sn,
        target_multiplier=target_multiplier,
    )
    return write_evt_tp2p4_osa_csvs(sdf, ldf, results_dir)


def summarize_directory(tp4_dir: Path, *, target_multiplier: float) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_file in sorted(tp4_dir.glob("*TP2-4 Scan.csv")):
        tile = tile_sn_from_tp2p4_path(csv_file)
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"{csv_file.name}: read error {e}", file=sys.stderr)
            continue
        if "Tile_SN" not in df.columns and tile:
            df = df.copy()
            df["Tile_SN"] = tile
        _, summ = reconcile_tp2p4_dataframe(df, target_multiplier=target_multiplier)
        if summ.empty:
            continue
        for _, r in summ.iterrows():
            rows.append({"file": csv_file.name, "tile_sn": tile, **r.to_dict()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TP2-4 OSA power vs fiber Power(mW) cross-check and scaling")
    p.add_argument(
        "path",
        type=Path,
        help="TP2-4 Scan.csv file or directory of *TP2-4 Scan.csv",
    )
    p.add_argument(
        "--target-multiplier",
        type=float,
        default=2.0,
        help="Target bank total = this × Power(mW) (default: 2)",
    )
    p.add_argument(
        "--long",
        action="store_true",
        help="For a single CSV: print per-row scaled columns (CSV to stdout)",
    )
    args = p.parse_args(argv)

    path = args.path.resolve()
    if path.is_dir():
        df = summarize_directory(path, target_multiplier=args.target_multiplier)
        if df.empty:
            print("No bank summaries produced.", file=sys.stderr)
            return 1
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", 30)
        print(df.to_string(index=False))
        print(
            f"\nAcross {len(df)} bank-rows: sum_OSA_div_fiber "
            f"min={df['sum_OSA_div_fiber'].min():.4g} max={df['sum_OSA_div_fiber'].max():.4g} "
            f"median={df['sum_OSA_div_fiber'].median():.4g}",
            file=sys.stderr,
        )
        return 0

    if not path.is_file():
        print(f"Not a file or directory: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    tile = tile_sn_from_tp2p4_path(path)
    if tile and "Tile_SN" not in df.columns:
        df["Tile_SN"] = tile

    enriched, summ = reconcile_tp2p4_dataframe(df, target_multiplier=args.target_multiplier)
    if args.long:
        if enriched.empty:
            print("No rows after reconciliation.", file=sys.stderr)
            return 1
        cols = [
            c
            for c in enriched.columns
            if c
            not in (
                OSA_DBM_COL,
            )
        ]
        enriched[cols].to_csv(sys.stdout, index=False)
        return 0

    print("=== Bank snapshot summary ===")
    print(summ.to_string(index=False))
    if not enriched.empty:
        chk = enriched.groupby(["Bank", "OSA_scale_to_target"], sort=False)["OSA_Power_mW_scaled"].sum()
        print("\nCheck: sum(scaled OSA mW) per bank snapshot (should match Target_total):")
        for (bank, scale), total in chk.items():
            tgt = enriched.loc[
                (enriched["Bank"] == bank) & (enriched["OSA_scale_to_target"] == scale),
                "Target_total_mW_bank",
            ].iloc[0]
            print(f"  Bank {bank}: sum={total:.6f} mW, target={tgt:.6f} mW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
