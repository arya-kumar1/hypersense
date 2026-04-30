#!/usr/bin/env python3
"""
Create Spearman-correlation outputs for all participant labeled Oura HR files.

Per participant-day CSV:
- bpm vs time-of-day (TOTAL)
- bpm vs class order (TOTAL, class encoded by within-file median time ordering)
- bpm vs time-of-day for each class
- one participant plot with two panels (time and class correlation views)

Overall across all participant-day files:
- bpm vs time-of-day (TOTAL)
- bpm vs class order (TOTAL, class encoded by global median time ordering)
- bpm vs time-of-day for each class (global)
- overall plots
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

try:
    from scipy.stats import spearmanr
except ImportError:
    print("This script requires scipy (pip install scipy).", file=sys.stderr)
    sys.exit(1)


FILE_RE = re.compile(r"^(P\d+)OrHrLabeled(\d{4}-\d{2}-\d{2})\.csv$", re.I)
DEFAULT_GLOB = "**/*OrHrLabeled*.csv"

CLASS_NORMALIZATION_MAP = {
    "ela/history": "ELA/History",
    "history": "ELA/History",
    "histry": "ELA/History",
    "friday funday": "Friday Funday",
    "funday friday": "Friday Funday",
}

BPM_HIGHLIGHT_MIN = 60
BPM_HIGHLIGHT_MAX = 120


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-root",
        default=str(root / "OuraRing"),
        help="Root directory to recursively search for labeled Oura HR CSV files.",
    )
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        default=DEFAULT_GLOB,
        help="Glob pattern used under --input-root.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(root / "graphs" / "spearman_time_class_total"),
        help="Output directory for plots and summary CSVs.",
    )
    return ap.parse_args()


def _time_of_day_seconds(df: pd.DataFrame) -> pd.Series:
    if "Time_In_PST" in df.columns:
        t = pd.to_datetime(df["Time_In_PST"], format="%H:%M:%S", errors="coerce")
        return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(float)
    if "Time_In_ISO" in df.columns:
        t = pd.to_datetime(df["Time_In_ISO"], errors="coerce")
        return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(float)
    if "time" in df.columns:
        # Fallback: use epoch ordering only if no clock-time columns exist.
        return pd.to_numeric(df["time"], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _spearman_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "rho": np.nan, "p": np.nan}
    rho, p = spearmanr(x[mask], y[mask])
    return {"n": n, "rho": float(rho), "p": float(p)}


def _class_rank_map(df: pd.DataFrame) -> dict[str, int]:
    ordering = (
        df.groupby("class", as_index=False)["tod_seconds"]
        .median()
        .sort_values("tod_seconds")
        .reset_index(drop=True)
    )
    return {cls: i + 1 for i, cls in enumerate(ordering["class"].tolist())}


def _normalize_class_name(value: str) -> str:
    raw = str(value).strip()
    key = raw.casefold()
    return CLASS_NORMALIZATION_MAP.get(key, raw)


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
        ax.set_xticklabels([_format_hhmmss(lo)], rotation=20, ha="right")
        return
    ticks = np.linspace(lo, hi, 4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_format_hhmmss(t) for t in ticks], rotation=20, ha="right")


def _annotate_stats(ax: plt.Axes, stats: dict) -> None:
    if stats["n"] >= 3:
        txt = f"rho={stats['rho']:.3f}\np={stats['p']:.3g}\nn={stats['n']}"
    else:
        txt = f"n={stats['n']} (need >=3)"
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor="0.8"),
    )


def _add_bpm_highlight_band(ax: plt.Axes) -> None:
    """Highlight the typical 60-120 bpm range on scatter panels."""
    ax.axhspan(
        BPM_HIGHLIGHT_MIN,
        BPM_HIGHLIGHT_MAX,
        facecolor="gold",
        alpha=0.12,
        zorder=0,
    )
    ax.axhline(BPM_HIGHLIGHT_MIN, color="goldenrod", linestyle="--", linewidth=0.9, alpha=0.85)
    ax.axhline(BPM_HIGHLIGHT_MAX, color="goldenrod", linestyle="--", linewidth=0.9, alpha=0.85)


def _scatter_time_panel(ax: plt.Axes, g: pd.DataFrame, title: str) -> dict:
    x = g["tod_seconds"].to_numpy(dtype=float)
    y = g["bpm"].to_numpy(dtype=float)
    stats = _spearman_stats(x, y)

    _add_bpm_highlight_band(ax)
    ax.scatter(x, y, alpha=0.8, s=28, edgecolors="0.3", linewidths=0.35)
    ax.set_title(title)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("bpm")
    _set_time_ticks(ax, x)
    ax.grid(True, alpha=0.3)
    _annotate_stats(ax, stats)
    return stats


def _scatter_class_panel(
    ax: plt.Axes,
    g: pd.DataFrame,
    class_rank_map: dict[str, int],
    title: str,
) -> dict:
    gg = g.copy()
    gg["class_rank"] = gg["class"].map(class_rank_map).astype(float)
    x = gg["class_rank"].to_numpy(dtype=float)
    y = gg["bpm"].to_numpy(dtype=float)
    stats = _spearman_stats(x, y)

    # jitter to make repeated class ranks visible
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, size=len(gg))
    _add_bpm_highlight_band(ax)
    ax.scatter(x + jitter, y, alpha=0.75, s=26, edgecolors="0.3", linewidths=0.35)
    ax.set_title(title)
    ax.set_xlabel("Class (ordered by median time-of-day)")
    ax.set_ylabel("bpm")
    inv = {v: k for k, v in class_rank_map.items()}
    ticks = sorted(inv.keys())
    ax.set_xticks(ticks)
    ax.set_xticklabels([inv[t] for t in ticks], rotation=25, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    _annotate_stats(ax, stats)
    return stats


def _read_participant_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"class", "bpm"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "class": df["class"].astype(str).map(_normalize_class_name),
            "bpm": pd.to_numeric(df["bpm"], errors="coerce"),
        }
    )
    out["tod_seconds"] = _time_of_day_seconds(df)
    out = out.dropna(subset=["tod_seconds", "bpm"])
    return out


def process_one_csv(path: Path, out_plot_dir: Path) -> tuple[list[dict], dict | None, pd.DataFrame | None]:
    m = FILE_RE.match(path.name)
    if not m:
        return [], None
    participant = m.group(1).upper()
    date_str = m.group(2)

    df = _read_participant_file(path)
    if df.empty:
        return [], None, None

    class_rank_map = _class_rank_map(df)
    df["class_rank"] = df["class"].map(class_rank_map).astype(float)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))

    records: list[dict] = []
    st_time_total = _scatter_time_panel(axs[0], df, "TOTAL: bpm vs time-of-day")
    st_class_total = _scatter_class_panel(axs[1], df, class_rank_map, "TOTAL: bpm vs class")

    records.append(
        {
            "participant": participant,
            "date": date_str,
            "analysis": "TOTAL_BPM_VS_TIME",
            "class": "TOTAL",
            "n": st_time_total["n"],
            "spearman_rho": st_time_total["rho"],
            "spearman_p": st_time_total["p"],
            "source_csv": str(path),
        }
    )
    total_rec = records[-1].copy()
    records.append(
        {
            "participant": participant,
            "date": date_str,
            "analysis": "TOTAL_BPM_VS_CLASS",
            "class": "TOTAL",
            "n": st_class_total["n"],
            "spearman_rho": st_class_total["rho"],
            "spearman_p": st_class_total["p"],
            "source_csv": str(path),
        }
    )

    for cls, g in df.groupby("class"):
        st = _spearman_stats(
            g["tod_seconds"].to_numpy(dtype=float),
            g["bpm"].to_numpy(dtype=float),
        )
        records.append(
            {
                "participant": participant,
                "date": date_str,
                "analysis": "CLASS_BPM_VS_TIME",
                "class": cls,
                "n": st["n"],
                "spearman_rho": st["rho"],
                "spearman_p": st["p"],
                "source_csv": str(path),
            }
        )

    fig.suptitle(f"{participant} {date_str} - Spearman correlations", fontsize=12, y=1.02)
    fig.tight_layout()
    out_plot = out_plot_dir / f"{participant}_{date_str}_spearman_time_and_class_total.png"
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_out = df.copy()
    df_out["participant"] = participant
    df_out["date"] = date_str
    return records, total_rec, df_out


def _overall_plots_and_records(all_df: pd.DataFrame, out_plot_dir: Path) -> list[dict]:
    records: list[dict] = []

    st_time_total = _spearman_stats(
        all_df["tod_seconds"].to_numpy(dtype=float),
        all_df["bpm"].to_numpy(dtype=float),
    )
    records.append(
        {
            "participant": "ALL",
            "date": "ALL",
            "analysis": "OVERALL_TOTAL_BPM_VS_TIME",
            "class": "TOTAL",
            "n": st_time_total["n"],
            "spearman_rho": st_time_total["rho"],
            "spearman_p": st_time_total["p"],
            "source_csv": "ALL_FILES",
        }
    )

    global_class_rank = _class_rank_map(all_df)
    all_df = all_df.copy()
    all_df["class_rank"] = all_df["class"].map(global_class_rank).astype(float)
    st_class_total = _spearman_stats(
        all_df["class_rank"].to_numpy(dtype=float),
        all_df["bpm"].to_numpy(dtype=float),
    )
    records.append(
        {
            "participant": "ALL",
            "date": "ALL",
            "analysis": "OVERALL_TOTAL_BPM_VS_CLASS",
            "class": "TOTAL",
            "n": st_class_total["n"],
            "spearman_rho": st_class_total["rho"],
            "spearman_p": st_class_total["p"],
            "source_csv": "ALL_FILES",
        }
    )

    for cls, g in all_df.groupby("class"):
        st = _spearman_stats(g["tod_seconds"].to_numpy(dtype=float), g["bpm"].to_numpy(dtype=float))
        records.append(
            {
                "participant": "ALL",
                "date": "ALL",
                "analysis": "OVERALL_CLASS_BPM_VS_TIME",
                "class": cls,
                "n": st["n"],
                "spearman_rho": st["rho"],
                "spearman_p": st["p"],
                "source_csv": "ALL_FILES",
            }
        )

    # Overall time plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.3))
    _scatter_time_panel(ax, all_df, "ALL participants: bpm vs time-of-day")
    fig.tight_layout()
    fig.savefig(out_plot_dir / "OVERALL_spearman_bpm_vs_time_of_day.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overall class plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    _scatter_class_panel(ax, all_df, global_class_rank, "ALL participants: bpm vs class")
    fig.tight_layout()
    fig.savefig(out_plot_dir / "OVERALL_spearman_bpm_vs_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return records


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_plot_dir = out_dir / "plots"
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_root.glob(args.glob_pattern))
    if not csvs:
        print(f"No files found under {input_root} with glob {args.glob_pattern}")
        sys.exit(1)

    all_records: list[dict] = []
    totals: list[dict] = []
    long_rows: list[pd.DataFrame] = []
    skipped = 0

    for path in csvs:
        try:
            recs, total, df_long = process_one_csv(path, out_plot_dir)
        except Exception as exc:
            skipped += 1
            print(f"Skipping {path}: {exc}")
            continue
        if recs:
            all_records.extend(recs)
        else:
            skipped += 1
        if total is not None:
            totals.append(total)
        if df_long is not None:
            long_rows.append(df_long)

    if not all_records:
        print("No usable participant files were processed.")
        sys.exit(1)

    all_long_df = pd.concat(long_rows, ignore_index=True)
    all_records.extend(_overall_plots_and_records(all_long_df, out_plot_dir))

    summary_df = pd.DataFrame(all_records).sort_values(["participant", "date", "analysis", "class"])
    totals_df = pd.DataFrame(totals).sort_values(["participant", "date"])

    summary_csv = out_dir / "spearman_bpm_vs_time_and_class_summary.csv"
    totals_csv = out_dir / "spearman_participant_total_bpm_vs_time.csv"
    summary_df.to_csv(summary_csv, index=False)
    totals_df.to_csv(totals_csv, index=False)

    print(f"Processed files: {len(csvs) - skipped} / {len(csvs)}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {totals_csv}")
    print(f"Plots dir: {out_plot_dir}")


if __name__ == "__main__":
    main()
