#!/usr/bin/env python3
"""
child_hr_avnn_plots.py

Create HR and AVNN-only comparison plots (Apple vs Oura) for each participant
on each day, using 5-minute windowed metrics. One subfolder per child under
graphs/child_hr_avnn/ (e.g. graphs/child_hr_avnn/P001/, graphs/child_hr_avnn/P002/).

Metrics plotted per 5-minute window:
  - HR (mean bpm)
  - AVNN (mean RR in ms; RR = 60000 / bpm)

Participant-days are taken from graphs/child_day_overlays/child_day_summary.csv.

Outputs:
  graphs/child_hr_avnn/P###/P###_YYYY-MM-DD_hr_avnn.png
  graphs/child_hr_avnn/P###/P###_YYYY-MM-DD_hr_avnn.csv
"""

import os
import re
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
HEALTHAPP_ROOT = "./HealthApp"
OURA_DIR       = "./OuraRing"

SUMMARY_CSV    = "./graphs/child_day_overlays/child_day_summary.csv"
OUT_BASE       = "./graphs/child_hr_avnn"  # subfolder per child: OUT_BASE/P001/, P002/, ...

LOCAL_TZ       = "US/Pacific"
BIN_SECONDS    = 300  # 5-minute windows

HR_MIN, HR_MAX = 40, 180

MAX_POINTS_LINE = 2500  # downsample line markers for dense windows (plotting only)
# ----------------------------------------

# --------- time helpers ---------
def _tz_to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(LOCAL_TZ, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
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


def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df


# --------- participant/file discovery ---------
def _healthapp_base_for_pid(pid: int) -> str | None:
    for name in (f"HealthAppP{pid}", f"HealthAppP{pid:02d}", f"HealthAppP{pid:03d}"):
        base = os.path.join(HEALTHAPP_ROOT, name)
        if os.path.isdir(base):
            return base
    return None


def _find_apple_hr_files(pid: int) -> list[str]:
    base = _healthapp_base_for_pid(pid)
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


def _read_apple_raw_hr(pid: int) -> pd.DataFrame:
    """Return Apple raw HR samples with UTC datetime index and column 'bpm'."""
    paths = _find_apple_hr_files(pid)
    if not paths:
        return pd.DataFrame(columns=["bpm"])

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

    series = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            if "Type" in df.columns:
                df = df[df["Type"].astype(str).str.contains("HeartRate", case=False, na=False)]
            if "Unit" in df.columns:
                df = df[df["Unit"].astype(str).str.lower().str.contains("count/min")]

            hr_col = "Value" if "Value" in df.columns else ("value" if "value" in df.columns else None)
            if not hr_col:
                continue
            tcol = _pick_time_col(df)
            if not tcol:
                continue

            ts = _tz_to_utc(df[tcol])
            bpm = pd.to_numeric(df[hr_col], errors="coerce")
            mask = (bpm >= HR_MIN) & (bpm <= HR_MAX)
            s = pd.Series(bpm.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Apple read fail {p}: {e}")

    if not series:
        return pd.DataFrame(columns=["bpm"])

    s_all = pd.concat(series).sort_index()
    out = s_all.to_frame("bpm")
    out = _ensure_dtindex_utc(out)
    return out


def _read_oura_raw_hr(pid: int) -> pd.DataFrame:
    """Return Oura raw HR samples with UTC datetime index and column 'bpm'."""
    folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"), os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand
            break
    if not folder:
        return pd.DataFrame(columns=["bpm"])

    series = []
    for p in sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv"))):
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            bpm = pd.to_numeric(df.get("bpm"), errors="coerce")
            mask = (bpm >= HR_MIN) & (bpm <= HR_MAX)
            s = pd.Series(bpm.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Oura read fail {p}: {e}")

    if not series:
        return pd.DataFrame(columns=["bpm"])

    s_all = pd.concat(series).sort_index()
    out = s_all.to_frame("bpm")
    out = _ensure_dtindex_utc(out)
    return out


# --------- windowed metrics (HR and AVNN only) ---------
@dataclass(frozen=True)
class WindowMetricsHRAVNN:
    hr_bpm: float
    avnn_ms: float
    n_samples: int


def _compute_window_metrics_hr_avnn(df: pd.DataFrame) -> WindowMetricsHRAVNN:
    """Compute only HR and AVNN for a single 5-minute window."""
    bpm = pd.to_numeric(df["bpm"], errors="coerce").dropna()
    n = int(bpm.shape[0])
    if n == 0:
        return WindowMetricsHRAVNN(hr_bpm=np.nan, avnn_ms=np.nan, n_samples=0)

    hr_bpm = float(bpm.mean())
    rr_ms = 60000.0 / bpm.to_numpy(dtype=float)
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    avnn_ms = float(np.mean(rr_ms)) if rr_ms.size else np.nan
    return WindowMetricsHRAVNN(hr_bpm=hr_bpm, avnn_ms=avnn_ms, n_samples=n)


def _metrics_by_window_hr_avnn(raw: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with index=timestamp_utc, columns hr_bpm, avnn_ms."""
    if raw.empty:
        return pd.DataFrame()
    raw = _ensure_dtindex_utc(raw)

    rows = []
    for ts, g in raw.groupby(pd.Grouper(freq=f"{BIN_SECONDS}s")):
        m = _compute_window_metrics_hr_avnn(g)
        rows.append({
            "timestamp_utc": ts,
            "hr_bpm": m.hr_bpm,
            "avnn_ms": m.avnn_ms,
            "n_samples": m.n_samples,
        })
    out = pd.DataFrame(rows).set_index("timestamp_utc").sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def _restrict_to_local_day(raw: pd.DataFrame, day_local: str) -> pd.DataFrame:
    """Keep only samples whose LOCAL_TZ date matches day_local."""
    if raw.empty:
        return raw
    idx_local_date = raw.index.tz_convert(LOCAL_TZ).date
    mask = (idx_local_date.astype(str) == str(day_local))
    return raw.loc[mask]


def _get_all_participant_days() -> list[tuple[int, str]]:
    """Return list of (participant_id, date_local) from summary CSV."""
    if not os.path.isfile(SUMMARY_CSV):
        raise FileNotFoundError(f"Summary not found: {SUMMARY_CSV}. Run oneperkid.py first.")
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["participant"].notna()].copy()
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce")
    df = df[df["participant"].notna()]  # drop "average" or non-numeric rows
    df = df[df["date_local"].notna()]
    df = df.astype({"participant": int})
    out = []
    for _, row in df.iterrows():
        out.append((int(row["participant"]), str(row["date_local"]).strip()))
    return out


def _run_for_child_day(pid: int, day_local: str, out_dir: str) -> None:
    """Generate HR and AVNN plot and CSV for one participant on one day."""
    print(f"  P{pid:03d} {day_local}")

    ap_raw = _read_apple_raw_hr(pid)
    ou_raw = _read_oura_raw_hr(pid)
    if ap_raw.empty or ou_raw.empty:
        print(f"    [WARN] Missing Apple or Oura HR for P{pid:03d}. Skipping.")
        return

    ap_day = _restrict_to_local_day(ap_raw, day_local)
    ou_day = _restrict_to_local_day(ou_raw, day_local)
    if ap_day.empty or ou_day.empty:
        print(f"    [WARN] No data for P{pid:03d} on {day_local}. Skipping.")
        return

    ap_m = _metrics_by_window_hr_avnn(ap_day).add_prefix("apple_")
    ou_m = _metrics_by_window_hr_avnn(ou_day).add_prefix("oura_")

    joined = ap_m.join(ou_m, how="inner")
    if joined.empty:
        print(f"    [WARN] No overlapping 5-min windows for P{pid:03d} on {day_local}. Skipping.")
        return

    # Keep only HR and AVNN columns for CSV
    csv_cols = ["apple_hr_bpm", "oura_hr_bpm", "apple_avnn_ms", "oura_avnn_ms"]
    joined_csv = joined[[c for c in csv_cols if c in joined.columns]]
    out_csv = os.path.join(out_dir, f"P{pid:03d}_{day_local}_hr_avnn.csv")
    joined_csv.to_csv(out_csv, index_label="timestamp_utc")

    # Plot in local time (HR and AVNN only)
    plot_df = joined.copy()
    plot_df.index = plot_df.index.tz_convert(LOCAL_TZ)
    if plot_df.shape[0] > MAX_POINTS_LINE:
        plot_df = plot_df.iloc[np.linspace(0, plot_df.shape[0] - 1, MAX_POINTS_LINE).astype(int)]

    metrics = [
        ("hr_bpm", "HR (bpm)"),
        ("avnn_ms", "AVNN (ms)"),
    ]
    n_panels = len(metrics)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        ax.plot(plot_df.index, plot_df[f"apple_{key}"], color="forestgreen", linewidth=1.5, label="Apple")
        ax.plot(plot_df.index, plot_df[f"oura_{key}"], color="crimson", linewidth=1.5, label="Oura")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    fig.suptitle(
        f"P{pid:03d} {day_local} — HR & AVNN (5-min windows, Apple vs Oura)",
        fontsize=14,
        fontweight="bold",
    )
    axes[-1].set_xlabel(f"Local Time ({LOCAL_TZ})")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_png = os.path.join(out_dir, f"P{pid:03d}_{day_local}_hr_avnn.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"    -> {out_png}")


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    participant_days = _get_all_participant_days()
    if not participant_days:
        print("[WARN] No participant-days found in summary. Exiting.")
        return

    # Group by participant so we create one subfolder per child
    by_pid: dict[int, list[str]] = {}
    for pid, day in participant_days:
        by_pid.setdefault(pid, []).append(day)

    for pid in sorted(by_pid.keys()):
        out_dir = os.path.join(OUT_BASE, f"P{pid:03d}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"P{pid:03d} ({len(by_pid[pid])} days) -> {out_dir}")
        for day_local in sorted(by_pid[pid]):
            _run_for_child_day(pid, day_local, out_dir)

    print(f"\nDone. Output base: {os.path.abspath(OUT_BASE)}")


if __name__ == "__main__":
    main()
