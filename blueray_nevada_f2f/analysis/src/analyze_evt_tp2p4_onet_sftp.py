#!/usr/bin/env python3
"""
EVT-focused TP2-4 / TP2-5 Scan + TP2-5 total power (same pipeline as ``analyze_tp2p4_onet_sftp``).

Uses ``config/evt_tp_scan_tiles.yaml`` and the **30C OFC** min-channel MPD ≥12 mW gate (same as MFG EVT),
and writes
``evt_tp2p4_*``, ``evt_tp2p5_*``, ``evt_tp2p5_totalpower_*``, ``evt_tp2p6_power_*``.
Equivalent to: ``python analyze_tp2p4_onet_sftp.py --evt-tp-scan [args...]``
"""
from __future__ import annotations

import sys

from analyze_tp2p4_onet_sftp import main as _main


if __name__ == "__main__":
    _main(["--evt-tp-scan", *sys.argv[1:]])
