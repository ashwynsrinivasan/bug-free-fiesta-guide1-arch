"""
EVT plot helpers.

The **≥ EVT_MIN_OPTICAL_POWER_MW** gate is applied only to **30C OFC** MPD data
(``pic_mpd_value_mw`` per channel). MFG EVT and ONET ``--evt-tp-scan`` use the set of
``Tile_SN`` derived from the ``tile_id`` values that pass that 30C gate (see ``evt_30c_common``).
"""
from __future__ import annotations

import pandas as pd

EVT_MIN_OPTICAL_POWER_MW = 12.0


def tile_ids_passing_min_channel_mpd(
    df_power: pd.DataFrame,
    *,
    min_mw: float = EVT_MIN_OPTICAL_POWER_MW,
    power_col: str = "pic_mpd_value_mw",
    tile_col: str = "tile_id",
) -> set[int]:
    """``tile_id`` values whose minimum ``power_col`` across all rows is ≥ ``min_mw``."""
    if df_power.empty or tile_col not in df_power.columns or power_col not in df_power.columns:
        return set()
    g = df_power.groupby(tile_col, sort=False)[power_col].min()
    return {int(i) for i in g[g >= float(min_mw)].index.tolist()}


def filter_30c_power_and_freq_by_min_tile_power(
    df_power: pd.DataFrame,
    df_freq: pd.DataFrame,
    *,
    min_mw: float = EVT_MIN_OPTICAL_POWER_MW,
    power_col: str = "pic_mpd_value_mw",
    tile_col: str = "tile_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop tiles whose minimum ``power_col`` in ``df_power`` is strictly below ``min_mw``."""
    if df_power.empty or tile_col not in df_power.columns:
        return df_power, df_freq
    keep = tile_ids_passing_min_channel_mpd(
        df_power, min_mw=min_mw, power_col=power_col, tile_col=tile_col
    )
    n0 = df_power[tile_col].nunique()
    print(f"30C EVT optical power ≥ {min_mw} mW: {n0} → {len(keep)} tiles (by min channel MPD)")
    df_p = df_power[df_power[tile_col].isin(keep)].copy()
    if df_freq.empty or tile_col not in df_freq.columns:
        return df_p, df_freq
    df_f = df_freq[df_freq[tile_col].isin(keep)].copy()
    return df_p, df_f
