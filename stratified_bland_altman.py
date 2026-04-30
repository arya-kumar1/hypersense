#!/usr/bin/env python3
"""
stratified_bland_altman.py

Create stratified Bland–Altman analyses for Apple Watch vs Oura Ring HR:
  - By participant (small multiples + per-participant bias/LoA table)
  - By time-of-day bin (08:30–15:00, 30-min display bins)
  - By class label (using class intervals extracted from HealthApp labeled exports)
  - By HR intensity (bands over the Bland–Altman mean HR; quantiles or fixed thresholds)

Outputs are written under ./graphs/bland_altman_stratified by default.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Defaults ----------------
DEFAULT_HEALTHAPP_ROOT = "./HealthApp"
DEFAULT_OURA_DIR = "./OuraRing"
DEFAULT_OUT_DIR = "./graphs/bland_altman_stratified"

DEFAULT_BIN_SECONDS = 300
DEFAULT_LOCAL_TZ = "US/Pacific"
DEFAULT_HR_MIN, DEFAULT_HR_MAX = 40, 180

# Match hr_by_class_and_time_heatmaps.py time windows
DEFAULT_TIME_BINS: list[tuple[str, str]] = [
    ("08:30", "09:00"),
    ("09:00", "09:30"),
    ("09:30", "10:00"),
    ("10:00", "10:30"),
    ("10:30", "11:00"),
    ("11:00", "11:30"),
    ("11:30", "12:00"),
    ("12:00", "12:30"),
    ("12:30", "13:00"),
    ("13:00", "13:30"),
    ("13:30", "14:00"),
    ("14:00", "14:30"),
    ("14:30", "15:00"),
]


# ---------------- Utilities ----------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df


def _tz_to_utc(series: pd.Series, *, local_tz: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(local_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def _ensure_dtindex_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _pid_from_pcode(pcode: str) -> int:
    m = re.search(r"(\d+)", str(pcode))
    return int(m.group(1)) if m else -1


def _pcode(pid: int) -> str:
    return f"P{pid:03d}"


def _participants(healthapp_root: str, oura_dir: str) -> list[int]:
    health_pids, oura_pids = [], []
    for d in glob.glob(os.path.join(healthapp_root, "HealthAppP*")):
        m = re.match(r".*HealthAppP0*([0-9]+)$", d)
        if m:
            health_pids.append(int(m.group(1)))
    for d in glob.glob(os.path.join(oura_dir, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m:
            oura_pids.append(int(m.group(1)))
    return sorted(set(health_pids).union(oura_pids))


def _find_healthapp_base(healthapp_root: str, pid: int) -> str | None:
    for name in (f"HealthAppP{pid}", f"HealthAppP{pid:02d}", f"HealthAppP{pid:03d}"):
        base = os.path.join(healthapp_root, name)
        if os.path.isdir(base):
            return base
    return None


def _find_apple_hr_files(healthapp_root: str, pid: int) -> list[str]:
    base = _find_healthapp_base(healthapp_root, pid)
    if not base:
        return []
    pats = [
        os.path.join(base, "Labeled", "Record", "**", "HeartRate.csv"),
        os.path.join(base, "Labeled", "Record", "**", "*HeartRate*.csv"),
        os.path.join(base, "Record", "Record", "**", "HeartRate.csv"),
        os.path.join(base, "Record", "Record", "**", "*HeartRate*.csv"),
        os.path.join(base, "Record", "**", "HeartRate.csv"),
        os.path.join(base, "Record", "**", "*HeartRate*.csv"),
    ]
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(list(dict.fromkeys(out)))


def _read_apple_binned(
    healthapp_root: str,
    pid: int,
    *,
    bin_seconds: int,
    hr_min: float,
    hr_max: float,
    local_tz: str,
) -> pd.DataFrame:
    paths = _find_apple_hr_files(healthapp_root, pid)
    if not paths:
        return pd.DataFrame(columns=["apple_bpm"])

    def _pick_time_col(df: pd.DataFrame) -> str | None:
        for c in ("CreationDate", "EndDate", "StartDate", "Time", "Date", "Time_In_PST"):
            if c in df.columns:
                return c
        best, bestn = None, 0
        for c in df.columns:
            try:
                n = pd.to_datetime(df[c], errors="coerce").notna().sum()
                if n > bestn:
                    best, bestn = c, n
            except Exception:
                pass
        return best

    series: list[pd.Series] = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            if "Type" in df.columns:
                df = df[df["Type"].astype(str).str.contains("HeartRate", case=False, na=False)]
            if "Unit" in df.columns:
                df = df[df["Unit"].astype(str).str.lower().str.contains("count/min")]
            if "Value" not in df.columns:
                continue
            tcol = _pick_time_col(df)
            if not tcol:
                continue
            ts = _tz_to_utc(df[tcol], local_tz=local_tz)
            val = pd.to_numeric(df["Value"], errors="coerce")
            mask = (val >= hr_min) & (val <= hr_max)
            s = pd.Series(val.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Apple read fail {p}: {e}")

    if not series:
        return pd.DataFrame(columns=["apple_bpm"])
    s_all = pd.concat(series).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("apple_bpm")


def _read_oura_binned(
    oura_dir: str,
    pid: int,
    *,
    bin_seconds: int,
    hr_min: float,
    hr_max: float,
) -> pd.DataFrame:
    folder = None
    for cand in (os.path.join(oura_dir, f"P{pid}OuraRing"), os.path.join(oura_dir, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand
            break
    if not folder:
        return pd.DataFrame(columns=["oura_bpm"])

    series: list[pd.Series] = []
    for p in sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv"))):
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                for c in df.columns:
                    if isinstance(c, str) and ("T" in c and ("Z" in c or "+" in c)):
                        tcol = c
                        break
            if not tcol:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            bpm = pd.to_numeric(df.get("bpm"), errors="coerce")
            mask = (bpm >= hr_min) & (bpm <= hr_max)
            s = pd.Series(bpm.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Oura read fail {p}: {e}")

    if not series:
        return pd.DataFrame(columns=["oura_bpm"])
    s_all = pd.concat(series).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("oura_bpm")


def load_all_pairs(
    *,
    healthapp_root: str,
    oura_dir: str,
    bin_seconds: int,
    hr_min: float,
    hr_max: float,
    local_tz: str,
) -> pd.DataFrame:
    """
    Return a DataFrame of inner-joined binned points across all participants:
      index: UTC bin timestamp
      cols: participant (P###), apple_bpm, oura_bpm, mean_ba, diff_ba
    """
    all_pairs: list[pd.DataFrame] = []
    for pid in _participants(healthapp_root, oura_dir):
        ap = _read_apple_binned(
            healthapp_root,
            pid,
            bin_seconds=bin_seconds,
            hr_min=hr_min,
            hr_max=hr_max,
            local_tz=local_tz,
        )
        ou = _read_oura_binned(oura_dir, pid, bin_seconds=bin_seconds, hr_min=hr_min, hr_max=hr_max)
        ap = _ensure_dtindex_utc(ap)
        ou = _ensure_dtindex_utc(ou)
        if ap.empty or ou.empty:
            continue
        mapped = ap.join(ou, how="inner").dropna()
        if mapped.empty:
            continue
        mapped["participant"] = _pcode(pid)
        all_pairs.append(mapped)

    if not all_pairs:
        return pd.DataFrame(columns=["participant", "apple_bpm", "oura_bpm", "mean_ba", "diff_ba"])

    pairs = pd.concat(all_pairs).sort_index()
    apple = pairs["apple_bpm"].astype(float)
    oura = pairs["oura_bpm"].astype(float)
    pairs["mean_ba"] = (apple + oura) / 2.0
    pairs["diff_ba"] = apple - oura
    return pairs


# ---------------- Stratification: time-of-day ----------------
def assign_time_bin(ts_utc_index: pd.DatetimeIndex, *, local_tz: str, time_bins: list[tuple[str, str]]) -> pd.Series:
    """
    Label each UTC timestamp with a time-of-day bin like '08:30-09:00' in local time.
    Returns '' for timestamps outside configured bins.
    """
    dt_local = ts_utc_index.tz_convert(local_tz)
    cur_min = dt_local.hour * 60 + dt_local.minute

    starts = []
    ends = []
    labels = []
    for s, e in time_bins:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        starts.append(sh * 60 + sm)
        ends.append(eh * 60 + em)
        labels.append(f"{s}-{e}")

    out = np.array([""] * len(cur_min), dtype=object)
    cur_min = np.asarray(cur_min, dtype=int)
    for start, end, lab in zip(starts, ends, labels):
        mask = (cur_min >= start) & (cur_min < end)
        out[mask] = lab
    return pd.Series(out, index=ts_utc_index, name="time_bin")


# ---------------- Stratification: class label ----------------
def canon_class_label(x: str) -> str:
    s = str(x).strip()
    key = s.lower()
    mapping = {
        "history": "History",
        "ela": "ELA",
        "ela/history": "ELA",
        "ela & history": "ELA",
    }
    return mapping.get(key, s.title())


def extract_class_intervals_for_pid(
    *,
    healthapp_root: str,
    pid: int,
    local_tz: str,
) -> pd.DataFrame:
    """
    Scan HealthApp labeled exports for (class_label, start_utc, end_utc).
    """
    base = _find_healthapp_base(healthapp_root, pid)
    if not base:
        return pd.DataFrame(columns=["class_label", "start_utc", "end_utc"])

    rec_root = Path(base) / "Labeled" / "Record"
    if not rec_root.exists():
        return pd.DataFrame(columns=["class_label", "start_utc", "end_utc"])

    all_rows: list[pd.DataFrame] = []
    for day_dir in sorted([d for d in rec_root.iterdir() if d.is_dir()]):
        for f in day_dir.glob("*.csv"):
            df = None
            try:
                df = _safe_read_csv(str(f))
            except Exception:
                df = None
            if df is None or df.empty:
                continue

            df.columns = [str(c).strip() for c in df.columns]

            class_col = None
            for cand in ("class", "Class", "label", "Label", "subject", "period", "activity"):
                if cand in df.columns:
                    class_col = cand
                    break

            start_col = None
            for cand in ("StartDate", "startDate", "Start", "start_time", "start"):
                if cand in df.columns:
                    start_col = cand
                    break

            end_col = None
            for cand in ("EndDate", "endDate", "End", "end_time", "end"):
                if cand in df.columns:
                    end_col = cand
                    break

            if not (class_col and start_col):
                continue

            sub = df[[class_col, start_col] + ([end_col] if end_col else [])].copy()
            sub.rename(columns={class_col: "class_label"}, inplace=True)
            sub["class_label"] = sub["class_label"].astype(str).apply(canon_class_label)
            sub["start_utc"] = _tz_to_utc(sub[start_col], local_tz=local_tz)
            if end_col:
                sub["end_utc"] = _tz_to_utc(sub[end_col], local_tz=local_tz)
            else:
                sub["end_utc"] = sub["start_utc"] + pd.Timedelta(minutes=5)
            sub = sub.dropna(subset=["class_label", "start_utc", "end_utc"])
            sub = sub[sub["end_utc"] > sub["start_utc"]]
            if not sub.empty:
                all_rows.append(sub[["class_label", "start_utc", "end_utc"]])

    if not all_rows:
        return pd.DataFrame(columns=["class_label", "start_utc", "end_utc"])

    out = pd.concat(all_rows, ignore_index=True).sort_values("start_utc")
    return out


def assign_class_label_to_timestamps(
    ts_utc_index: pd.DatetimeIndex, intervals: pd.DataFrame
) -> pd.Series:
    """
    Assign each timestamp to the (single) class interval that contains it.
    If no interval contains a timestamp, the label is ''.

    Implementation: for each timestamp, pick the last interval start <= ts,
    then check ts < end.
    """
    if intervals.empty:
        return pd.Series([""] * len(ts_utc_index), index=ts_utc_index, name="class_label")

    ints = intervals.dropna(subset=["start_utc", "end_utc", "class_label"]).copy()
    ints = ints.sort_values("start_utc")
    starts = pd.to_datetime(ints["start_utc"], utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
    ends = pd.to_datetime(ints["end_utc"], utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
    labels = ints["class_label"].astype(str).to_numpy()

    ts = ts_utc_index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(starts, ts, side="right") - 1
    out = np.array([""] * len(ts), dtype=object)
    valid = (pos >= 0) & (pos < len(starts))
    pv = pos[valid]
    tv = ts[valid]
    within = tv < ends[pv]
    out_idx = np.flatnonzero(valid)[within]
    out[out_idx] = labels[pv[within]]
    return pd.Series(out, index=ts_utc_index, name="class_label")


# ---------------- Bland–Altman stats + plotting ----------------
@dataclass(frozen=True)
class BAStats:
    n: int
    bias: float
    sd_diff: float
    loa_lower: float
    loa_upper: float


def bland_altman_stats(diff: np.ndarray) -> BAStats:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = int(diff.size)
    if n == 0:
        return BAStats(n=0, bias=np.nan, sd_diff=np.nan, loa_lower=np.nan, loa_upper=np.nan)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n >= 2 else float("nan")
    loa_u = bias + 1.96 * sd if np.isfinite(sd) else float("nan")
    loa_l = bias - 1.96 * sd if np.isfinite(sd) else float("nan")
    return BAStats(n=n, bias=bias, sd_diff=sd, loa_lower=loa_l, loa_upper=loa_u)


def _downsample_xy(x: np.ndarray, y: np.ndarray, *, max_points: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def plot_ba_panel(
    ax: plt.Axes,
    mean_ba: np.ndarray,
    diff_ba: np.ndarray,
    *,
    title: str,
    max_points: int = 5000,
) -> BAStats:
    mean_ba = np.asarray(mean_ba, dtype=float)
    diff_ba = np.asarray(diff_ba, dtype=float)
    mask = np.isfinite(mean_ba) & np.isfinite(diff_ba)
    x = mean_ba[mask]
    y = diff_ba[mask]
    st = bland_altman_stats(y)

    x_plot, y_plot = _downsample_xy(x, y, max_points=max_points)
    ax.scatter(x_plot, y_plot, s=8, alpha=0.55, c="steelblue", edgecolors="none")

    if st.n > 0 and np.isfinite(st.bias):
        ax.axhline(st.bias, color="black", linewidth=1.6, linestyle="-")
    if st.n > 1 and np.isfinite(st.loa_upper):
        ax.axhline(st.loa_upper, color="0.35", linewidth=1.0, linestyle="--")
        ax.axhline(st.loa_lower, color="0.35", linewidth=1.0, linestyle="--")

    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    ann = f"n={st.n}"
    if st.n > 0 and np.isfinite(st.bias):
        ann += f"\nbias={st.bias:.2f}"
    if st.n > 1 and np.isfinite(st.sd_diff):
        ann += f"\nLoA=[{st.loa_lower:.1f},{st.loa_upper:.1f}]"
    ax.text(
        0.02,
        0.98,
        ann,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    return st


def _small_multiples_grid(n_panels: int, *, ncols: int = 3) -> tuple[plt.Figure, np.ndarray]:
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.4), squeeze=False)
    return fig, axs


def save_stratified_ba(
    df: pd.DataFrame,
    *,
    group_col: str,
    out_dir: str,
    stem: str,
    title: str,
    ncols: int = 3,
    max_panels: int = 24,
    max_points_per_panel: int = 5000,
) -> pd.DataFrame:
    """
    Create BA small multiples and a stats CSV for a grouping column.
    Returns the stats DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)
    work = df.copy()
    work[group_col] = work[group_col].astype(str)
    work = work[work[group_col] != ""]
    if work.empty:
        print(f"[WARN] No rows for {stem} (group_col={group_col}).")
        return pd.DataFrame(columns=[group_col, "n_pairs", "bias_bpm", "sd_diff_bpm", "loa_lower_bpm", "loa_upper_bpm"])

    groups = [(k, g) for k, g in work.groupby(group_col)]
    groups.sort(key=lambda kv: (-len(kv[1]), kv[0]))  # biggest first
    if len(groups) > max_panels:
        print(f"[INFO] {stem}: plotting top {max_panels} groups by n (of {len(groups)}).")
        groups_plot = groups[:max_panels]
    else:
        groups_plot = groups

    stats_rows: list[dict] = []
    for k, g in groups:
        st = bland_altman_stats(g["diff_ba"].to_numpy(dtype=float))
        stats_rows.append(
            {
                group_col: k,
                "n_pairs": st.n,
                "bias_bpm": st.bias,
                "sd_diff_bpm": st.sd_diff,
                "loa_lower_bpm": st.loa_lower,
                "loa_upper_bpm": st.loa_upper,
            }
        )
    stats_df = pd.DataFrame(stats_rows).sort_values(["n_pairs", group_col], ascending=[False, True])
    stats_path = os.path.join(out_dir, f"{stem}_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[INFO] Wrote {stats_path}")

    # Plot
    fig, axs = _small_multiples_grid(len(groups_plot), ncols=ncols)
    axs_flat = axs.flat

    for i, (k, g) in enumerate(groups_plot):
        ax = axs_flat[i]
        plot_ba_panel(
            ax,
            g["mean_ba"].to_numpy(dtype=float),
            g["diff_ba"].to_numpy(dtype=float),
            title=str(k),
            max_points=max_points_per_panel,
        )
        if i % ncols == 0:
            ax.set_ylabel("Apple − Oura (bpm)", fontsize=9)
        ax.set_xlabel("Mean HR (bpm)", fontsize=9)

    # Hide unused axes
    for j in range(len(groups_plot), axs.size):
        axs_flat[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{stem}_small_multiples.png")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote {out_png}")
    return stats_df


# ---------------- Stratification: HR intensity ----------------
def add_intensity_band(
    df: pd.DataFrame,
    *,
    mode: str,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    fixed_thresholds: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Add 'intensity_band' based on mean_ba.
      - mode='quantile': split by mean_ba quantiles into 3 bands
      - mode='fixed': split by thresholds (t1, t2) into <t1, [t1,t2), >=t2
    """
    out = df.copy()
    m = pd.to_numeric(out["mean_ba"], errors="coerce")
    if mode == "fixed":
        if not fixed_thresholds:
            raise ValueError("fixed_thresholds required when mode='fixed'")
        t1, t2 = fixed_thresholds
        bands = np.where(
            m < t1,
            f"<{t1:g}",
            np.where(m < t2, f"{t1:g}-{t2:g}", f">={t2:g}"),
        )
        out["intensity_band"] = bands
        return out

    if mode == "quantile":
        q1, q2 = quantiles
        t1 = float(np.nanquantile(m.to_numpy(dtype=float), q1))
        t2 = float(np.nanquantile(m.to_numpy(dtype=float), q2))
        bands = np.where(
            m < t1,
            f"low (<{t1:.1f})",
            np.where(m < t2, f"mid ({t1:.1f}-{t2:.1f})", f"high (≥{t2:.1f})"),
        )
        out["intensity_band"] = bands
        return out

    raise ValueError("mode must be 'quantile' or 'fixed'")


# ---------------- Main ----------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--healthapp-root", default=DEFAULT_HEALTHAPP_ROOT)
    ap.add_argument("--oura-dir", default=DEFAULT_OURA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--bin-seconds", type=int, default=DEFAULT_BIN_SECONDS)
    ap.add_argument("--local-tz", default=DEFAULT_LOCAL_TZ)
    ap.add_argument("--hr-min", type=float, default=float(DEFAULT_HR_MIN))
    ap.add_argument("--hr-max", type=float, default=float(DEFAULT_HR_MAX))
    ap.add_argument("--max-panels", type=int, default=24, help="Max panels per stratification plot (top-n by n_pairs).")
    ap.add_argument("--ncols", type=int, default=3, help="Number of columns in small multiples.")
    ap.add_argument("--max-points-per-panel", type=int, default=5000)
    ap.add_argument(
        "--intensity-mode",
        choices=["quantile", "fixed"],
        default="quantile",
        help="How to define HR intensity bands over the BA mean HR.",
    )
    ap.add_argument(
        "--fixed-thresholds",
        default="90,110",
        help="Comma-separated thresholds for --intensity-mode fixed (e.g., '90,110').",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pairs = load_all_pairs(
        healthapp_root=args.healthapp_root,
        oura_dir=args.oura_dir,
        bin_seconds=args.bin_seconds,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        local_tz=args.local_tz,
    )
    if pairs.empty:
        raise SystemExit("[ERROR] No overlapping Apple/Oura bins found.")

    pairs = _ensure_dtindex_utc(pairs)
    pairs["timestamp_utc"] = pairs.index
    print(f"[INFO] Total paired bins: {len(pairs):,}")

    # 1) By participant
    save_stratified_ba(
        pairs,
        group_col="participant",
        out_dir=out_dir,
        stem="by_participant",
        title="Bland–Altman (Apple − Oura) by participant",
        ncols=args.ncols,
        max_panels=args.max_panels,
        max_points_per_panel=args.max_points_per_panel,
    )

    # 2) By time-of-day bin
    pairs_time = pairs.copy()
    pairs_time["time_bin"] = assign_time_bin(
        pairs_time.index, local_tz=args.local_tz, time_bins=DEFAULT_TIME_BINS
    ).values
    save_stratified_ba(
        pairs_time,
        group_col="time_bin",
        out_dir=out_dir,
        stem="by_time_of_day",
        title="Bland–Altman (Apple − Oura) by time of day (local)",
        ncols=args.ncols,
        max_panels=args.max_panels,
        max_points_per_panel=args.max_points_per_panel,
    )

    # 3) By class label
    class_rows: list[pd.DataFrame] = []
    for p, g in pairs.groupby("participant"):
        pid = _pid_from_pcode(p)
        if pid < 0:
            continue
        intervals = extract_class_intervals_for_pid(
            healthapp_root=args.healthapp_root,
            pid=pid,
            local_tz=args.local_tz,
        )
        if intervals.empty:
            continue
        labels = assign_class_label_to_timestamps(g.index, intervals)
        gg = g.copy()
        gg["class_label"] = labels.values
        class_rows.append(gg)

    if class_rows:
        pairs_class = pd.concat(class_rows).sort_index()
    else:
        pairs_class = pairs.copy()
        pairs_class["class_label"] = ""

    save_stratified_ba(
        pairs_class,
        group_col="class_label",
        out_dir=out_dir,
        stem="by_class",
        title="Bland–Altman (Apple − Oura) by class label",
        ncols=args.ncols,
        max_panels=args.max_panels,
        max_points_per_panel=args.max_points_per_panel,
    )

    # 4) By HR intensity (bands over BA mean)
    if args.intensity_mode == "fixed":
        try:
            a, b = [float(x.strip()) for x in args.fixed_thresholds.split(",")]
            thr = (a, b)
        except Exception as e:
            raise SystemExit(f"[ERROR] Could not parse --fixed-thresholds: {e}")
        pairs_int = add_intensity_band(pairs, mode="fixed", fixed_thresholds=thr)
        title = f"Bland–Altman (Apple − Oura) by HR intensity bands (<{thr[0]:g}, {thr[0]:g}-{thr[1]:g}, ≥{thr[1]:g})"
    else:
        pairs_int = add_intensity_band(pairs, mode="quantile")
        title = "Bland–Altman (Apple − Oura) by HR intensity bands (mean-HR quantiles)"

    save_stratified_ba(
        pairs_int,
        group_col="intensity_band",
        out_dir=out_dir,
        stem="by_intensity",
        title=title,
        ncols=args.ncols,
        max_panels=args.max_panels,
        max_points_per_panel=args.max_points_per_panel,
    )

    print(f"[DONE] Wrote stratified outputs under: {out_dir}")


if __name__ == "__main__":
    main()

