#!/usr/bin/env python3
"""
Exploratory HR: for each participant, plot one figure for their labeled day with the
most HR samples.

Each PNG has two panels (same look as before): BPM vs time-of-day and ΔBPM vs time-of-day
(ΔBPM = BPM − that day’s median), points colored by class, thin gray line in time order.

Inputs: Oura labeled HR CSVs (*OrHrLabeled*.csv) under OuraRing/.
Outputs: graphs/hr_baseline_exploration/*.png only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

FILE_RE = re.compile(r"^(P\d+)OrHrLabeled(\d{4}-\d{2}-\d{2})\.csv$", re.I)
DEFAULT_GLOB = "**/*OrHrLabeled*.csv"

CLASS_NORMALIZATION_MAP = {
    "ela/history": "ELA/History",
    "history": "ELA/History",
    "histry": "ELA/History",
    "friday funday": "Friday Funday",
    "funday friday": "Friday Funday",
}

ROLLING_WINDOW_POINTS = 7


def _time_of_day_seconds(df: pd.DataFrame) -> pd.Series:
    if "Time_In_PST" in df.columns:
        t = pd.to_datetime(df["Time_In_PST"], format="%H:%M:%S", errors="coerce")
        return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(float)
    if "Time_In_ISO" in df.columns:
        t = pd.to_datetime(df["Time_In_ISO"], errors="coerce")
        return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(float)
    if "time" in df.columns:
        return pd.to_numeric(df["time"], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _normalize_class_name(value: str) -> str:
    raw = str(value).strip()
    key = raw.casefold()
    return CLASS_NORMALIZATION_MAP.get(key, raw)


def _read_labeled_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"class", "bpm"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    out = pd.DataFrame(
        {
            "class": df["class"].astype(str).map(_normalize_class_name),
            "bpm": pd.to_numeric(df["bpm"], errors="coerce"),
        }
    )
    out["tod_seconds"] = _time_of_day_seconds(df)
    out = out.dropna(subset=["tod_seconds", "bpm"])
    return out


def _format_hhmmss(seconds_val: float) -> str:
    if not np.isfinite(seconds_val):
        return ""
    total = int(round(seconds_val))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _set_time_ticks(ax: plt.Axes, x: np.ndarray) -> None:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi <= lo:
        ax.set_xticks([lo])
        ax.set_xticklabels([_format_hhmmss(lo)], rotation=18, ha="right", fontsize=7)
        return
    ticks = np.linspace(lo, hi, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_hhmmss(t) for t in ticks], rotation=18, ha="right", fontsize=7)


def _add_day_metrics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("tod_seconds").copy()
    baseline = float(np.nanmedian(g["bpm"].to_numpy()))
    g["baseline_median_bpm"] = baseline
    g["delta_bpm"] = g["bpm"] - baseline
    g["bpm_sd_roll"] = (
        g["bpm"].rolling(window=ROLLING_WINDOW_POINTS, center=True, min_periods=3).std()
    )
    return g


def _draw_class_colored_series(
    ax: plt.Axes,
    gg: pd.DataFrame,
    ycol: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    cmap = plt.get_cmap("tab20")
    classes = sorted(gg["class"].astype(str).unique())
    class_to_i = {cls: i % 20 for i, cls in enumerate(classes)}
    ax.plot(
        gg["tod_seconds"],
        gg[ycol],
        color="0.82",
        linewidth=0.7,
        zorder=1,
        alpha=0.95,
    )
    for cls in classes:
        m = gg["class"].astype(str) == cls
        if not m.any():
            continue
        color = cmap(class_to_i[cls])
        ax.scatter(
            gg.loc[m, "tod_seconds"],
            gg.loc[m, ycol],
            s=16,
            alpha=0.88,
            color=color,
            edgecolors="0.25",
            linewidths=0.25,
            label=cls,
            zorder=2,
        )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("Time of day", fontsize=8)
    _set_time_ticks(ax, gg["tod_seconds"].to_numpy(dtype=float))
    ax.grid(True, alpha=0.28)
    if show_legend:
        ax.legend(fontsize=5.5, loc="upper left", framealpha=0.92, ncol=1)


def _participant_figure(participant: str, date_str: str, g: pd.DataFrame, out_path: Path) -> None:
    gg = g.sort_values("tod_seconds")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    _draw_class_colored_series(axs[0], gg, "bpm", "BPM", show_legend=True)
    axs[0].set_title("BPM vs time-of-day", fontsize=9)
    _draw_class_colored_series(axs[1], gg, "delta_bpm", "ΔBPM (vs day median)", show_legend=False)
    axs[1].set_title("ΔBPM vs time-of-day", fontsize=9)
    fig.suptitle(
        f"{participant} — day with most samples: {date_str} (n={len(gg)})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-root",
        default=str(ROOT / "OuraRing"),
        help="Root directory for labeled Oura CSVs.",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        default=DEFAULT_GLOB,
        help="Glob under --input-root.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "graphs" / "hr_baseline_exploration"),
        help="Output directory for PNGs only.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_root.glob(args.glob_pattern))
    if not csvs:
        print(f"No files under {input_root} matching {args.glob_pattern}", file=sys.stderr)
        sys.exit(1)

    # participant -> list of (n_points, date_str, dataframe)
    by_p: dict[str, list[tuple[int, str, pd.DataFrame]]] = {}
    skipped = 0

    for path in csvs:
        m = FILE_RE.match(path.name)
        if not m:
            skipped += 1
            continue
        participant = m.group(1).upper()
        date_str = m.group(2)
        try:
            raw = _read_labeled_csv(path)
        except Exception as exc:
            print(f"Skip {path}: {exc}")
            skipped += 1
            continue
        if raw.empty:
            skipped += 1
            continue
        g = _add_day_metrics(raw)
        n = len(g)
        by_p.setdefault(participant, []).append((n, date_str, g))

    if not by_p:
        print("No usable labeled files.")
        sys.exit(1)

    for participant, days in sorted(by_p.items()):
        # Max points; tie-break: later calendar date (string sort works for ISO dates)
        best_n, best_date, best_g = max(days, key=lambda t: (t[0], t[1]))
        out_path = out_dir / f"{participant}_baseline_maxpoints_day.png"
        _participant_figure(participant, best_date, best_g, out_path)
        print(f"Wrote {out_path} ({participant} {best_date}, n={best_n})")

    print(f"Skipped non-matching or empty files: {skipped}")
    print("Done.")


if __name__ == "__main__":
    main()
