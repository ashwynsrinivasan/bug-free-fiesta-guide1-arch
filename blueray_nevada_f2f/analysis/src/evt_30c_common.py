"""
Shared 30 °C OFC sheet parsing and EVT module-slot → ``Tile_SN`` resolution.

``tile_id`` in ``ofc_data.xlsx`` (tab ``30C``) is the EVT module slot index (typically 1..16).
``evt_tp_scan_tiles.yaml`` lists ``tile_sn`` in **slot order** (first row = slot 1) unless
``clm_mfg_data/analysis_src/tile_module_slot.yaml`` maps ``Tile_SN`` → slot explicitly.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import yaml

# Set A = bank 1, Set B = bank 0 — matches TP2p4 CSV ``Bank`` column
SET_A_BANK = 1
SET_B_BANK = 0


def default_ofc_excel_path() -> Path:
    here = Path(__file__).resolve()
    guide1 = here.parents[3]
    return (
        guide1 / "ips_clm_gen1" / "ips_clm_evt_ofc" / "temperature_aggressors" / "ofc_data.xlsx"
    ).resolve()


def default_clm_mfg_data_base() -> Path:
    here = Path(__file__).resolve()
    guide1 = here.parents[3]
    return (guide1 / "ips_clm_gen1" / "clm_mfg_data").resolve()


def build_30c_power_frame(df_30c: pd.DataFrame) -> pd.DataFrame:
    """Last cycle only; per-channel mW with EVT tile/bank corrections (matches module_analysis)."""
    last_cycle = df_30c["cycle_number"].max()
    df_last = df_30c[df_30c["cycle_number"] == last_cycle]
    rows: list[dict] = []

    for _, row in df_last.iterrows():
        tile_id = row["tile_id"]
        bank_type = row["bank_type"]
        try:
            pic_values_uw = ast.literal_eval(row["pic_mpd_value"])
        except (SyntaxError, TypeError, ValueError):
            continue
        for channel_idx, value_uw in enumerate(pic_values_uw):
            if tile_id == 9 and bank_type == "BANK_A":
                value_uw = value_uw * 1.1
            value_mw = value_uw / 1000.0
            if tile_id == 7 and bank_type == "BANK_A" and value_mw < 10:
                value_mw = value_mw + 0.5
            bank_csv = SET_A_BANK if bank_type == "BANK_A" else SET_B_BANK
            rows.append(
                {
                    "tile_id": int(tile_id),
                    "bank_csv": bank_csv,
                    "pic_mpd_value_mw": float(value_mw),
                    "channel": channel_idx,
                }
            )
    return pd.DataFrame(rows)


def load_mfg_tile_sn_to_slot_map(mfg_base: Path) -> dict[str, int]:
    """``Tile_SN`` → EVT module slot 1..16. Optional file; empty dict if missing."""
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


def evt_tile_sn_from_30c_passing_ids(
    passing_tile_ids: set[int],
    ordered_evt_tile_sn: list[str],
    mfg_base: Path,
) -> set[str]:
    """
    Map 30C ``tile_id`` values that passed the MPD gate to ``Tile_SN``.

    Prefer ``tile_module_slot.yaml`` (invert SN→slot). Otherwise slot *n* uses the *n*th entry
    in ``ordered_evt_tile_sn`` (1-based slot index). Only serials present in that list are returned.
    """
    yaml_set = set(ordered_evt_tile_sn)
    sn_to_slot = load_mfg_tile_sn_to_slot_map(mfg_base)
    slot_to_sn: dict[int, str] = {}
    for sn, slot in sn_to_slot.items():
        if slot not in slot_to_sn:
            slot_to_sn[slot] = sn

    out: set[str] = set()
    n_yaml = len(ordered_evt_tile_sn)
    for tid in passing_tile_ids:
        tid_i = int(tid)
        sn = slot_to_sn.get(tid_i)
        if sn is None and n_yaml and 1 <= tid_i <= n_yaml:
            sn = ordered_evt_tile_sn[tid_i - 1]
        if sn is not None and sn in yaml_set:
            out.add(sn)
    return out
