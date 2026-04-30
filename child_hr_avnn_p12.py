#!/usr/bin/env python3
"""
child_hr_avnn_p12.py

Generate HR & AVNN plots (Apple vs Oura, 5-minute windows) specifically for
participant P012, even if P012 is not listed in
graphs/child_day_overlays/child_day_summary.csv.

This script:
  - Loads raw Apple and Oura HR for pid=12
  - Infers all local calendar days where both have data
  - For each such day, calls the existing helper in child_hr_avnn_plots.py
    to create:
        graphs/child_hr_avnn/P012/P012_YYYY-MM-DD_hr_avnn.png
        graphs/child_hr_avnn/P012/P012_YYYY-MM-DD_hr_avnn.csv
"""

import os
from datetime import date

import numpy as np

from child_hr_avnn_plots import (
    OUT_BASE,
    LOCAL_TZ,
    _read_apple_raw_hr,
    _read_oura_raw_hr,
    _run_for_child_day,
)


def _local_dates_with_data(idx, tz: str) -> set[date]:
    """Return set of local dates present in a datetime index."""
    if idx.tz is None:
        # Treat as UTC then convert to local
        local_dates = idx.tz_localize("UTC").tz_convert(tz).date
    else:
        local_dates = idx.tz_convert(tz).date
    # np.unique on dates returns ndarray of date objects
    return set(np.unique(local_dates.astype("datetime64[D]")).astype("datetime64[D]").astype(object))


def main() -> None:
    pid = 12
    print(f"Running HR & AVNN plots for P{pid:03d}")

    ap_raw = _read_apple_raw_hr(pid)
    ou_raw = _read_oura_raw_hr(pid)
    if ap_raw.empty or ou_raw.empty:
        print(f"[WARN] Missing Apple or Oura HR for P{pid:03d}. Nothing to do.")
        return

    # Infer all local dates where both have any data
    ap_dates = _local_dates_with_data(ap_raw.index, LOCAL_TZ)
    ou_dates = _local_dates_with_data(ou_raw.index, LOCAL_TZ)
    common_dates = sorted(ap_dates.intersection(ou_dates))
    if not common_dates:
        print(f"[WARN] No overlapping local dates for P{pid:03d}. Nothing to do.")
        return

    out_dir = os.path.join(OUT_BASE, f"P{pid:03d}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"P{pid:03d}: {len(common_dates)} overlapping day(s) -> {out_dir}")

    for d in common_dates:
        day_str = d.isoformat()
        _run_for_child_day(pid, day_str, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()

