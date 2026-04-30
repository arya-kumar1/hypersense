#!/usr/bin/env python3
"""
best_child_multimetric_plots.py

Create per-5-minute window comparison plots (Apple vs Oura) for the child/day
with the best overlap in `graphs/child_day_overlays/child_day_summary.csv`.

This mirrors the style of multi-panel plots often used to compare ECG vs PPG
metrics, but here we derive "HRV-style" metrics from the *raw heart-rate samples*
within each 5-minute window by converting bpm -> RR (ms):
    RR_ms = 60000 / bpm

Per 5-minute window we compute (when enough samples exist):
  - HR (mean bpm)
  - AVNN (mean RR, ms)
  - SDNN (std RR, ms)
  - RMSSD (ms) from successive RR diffs
  - pNN50 (%) from successive RR diffs > 50 ms

Optionally (if SciPy is available), we also estimate LF and HF band power from
the interpolated RR tachogram in each window and report LF/HF.

Outputs:
  graphs/best_child_multimetric/P###_YYYY-MM-DD_multimetric.png
  graphs/best_child_multimetric/P###_YYYY-MM-DD_multimetric.csv
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

try:
    from scipy.signal import welch
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------- CONFIG ----------------
HEALTHAPP_ROOT = "./HealthApp"
OURA_DIR       = "./OuraRing"

SUMMARY_CSV    = "./graphs/child_day_overlays/child_day_summary.csv"
OUT_DIR        = "./graphs/best_child_multimetric"

LOCAL_TZ       = "US/Pacific"
BIN_SECONDS    = 300  # 5-minute windows

HR_MIN, HR_MAX = 40, 180

# Frequency bands (Hz) for HRV PSD integration (if SCIPY_OK)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

MAX_POINTS_LINE = 2500  # downsample line markers for dense windows (plotting only)
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


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
def _participants() -> list[int]:
    health_pids, oura_pids = [], []
    for d in glob.glob(os.path.join(HEALTHAPP_ROOT, "HealthAppP*")):
        m = re.match(r".*HealthAppP0*([0-9]+)$", d)
        if m:
            health_pids.append(int(m.group(1)))
    for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m:
            oura_pids.append(int(m.group(1)))
    return sorted(set(health_pids).union(oura_pids))


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
    """
    Return Apple raw HR samples with UTC datetime index and column 'bpm'.
    """
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
    """
    Return Oura raw HR samples with UTC datetime index and column 'bpm'.
    """
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


# --------- windowed metrics ---------
@dataclass(frozen=True)
class WindowMetrics:
    hr_bpm: float
    avnn_ms: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    lf_power: float
    hf_power: float
    lf_hf: float
    n_samples: int


def _bandpower_from_rr(rr_ms: np.ndarray, t_s: np.ndarray, band: tuple[float, float]) -> float:
    """
    Estimate band power for RR tachogram by interpolating RR to a uniform grid and
    integrating Welch PSD over the band. Returns NaN if insufficient data or SciPy missing.
    """
    if (not SCIPY_OK) or rr_ms.size < 8:
        return np.nan
    if t_s.size != rr_ms.size:
        return np.nan

    # Ensure strictly increasing time
    order = np.argsort(t_s)
    t_s = t_s[order]
    rr_ms = rr_ms[order]
    dt = np.diff(t_s)
    keep = np.isfinite(rr_ms) & np.isfinite(t_s)
    t_s = t_s[keep]
    rr_ms = rr_ms[keep]
    if t_s.size < 8:
        return np.nan
    # Remove duplicates
    _, uniq_idx = np.unique(t_s, return_index=True)
    t_s = t_s[uniq_idx]
    rr_ms = rr_ms[uniq_idx]
    if t_s.size < 8:
        return np.nan

    # Interpolate to 4 Hz (common for short-window HRV PSD)
    fs = 4.0
    t_grid = np.arange(t_s[0], t_s[-1], 1.0 / fs)
    if t_grid.size < 32:
        return np.nan
    rr_i = np.interp(t_grid, t_s, rr_ms)
    rr_i = rr_i - np.nanmean(rr_i)

    f, pxx = welch(rr_i, fs=fs, nperseg=min(256, rr_i.size))
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return np.nan
    return float(np.trapz(pxx[m], f[m]))


def _compute_window_metrics(df: pd.DataFrame) -> WindowMetrics:
    """
    df: rows in a single 5-minute window, index is UTC timestamps, column 'bpm'
    """
    bpm = pd.to_numeric(df["bpm"], errors="coerce").dropna()
    n = int(bpm.shape[0])
    if n == 0:
        return WindowMetrics(*(np.nan,) * 8, n_samples=0)  # type: ignore[arg-type]

    hr_bpm = float(bpm.mean())

    rr_ms = (60000.0 / bpm.to_numpy(dtype=float))
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if rr_ms.size < 2:
        return WindowMetrics(
            hr_bpm=hr_bpm,
            avnn_ms=float(np.nanmean(rr_ms)) if rr_ms.size else np.nan,
            sdnn_ms=np.nan,
            rmssd_ms=np.nan,
            pnn50_pct=np.nan,
            lf_power=np.nan,
            hf_power=np.nan,
            lf_hf=np.nan,
            n_samples=n,
        )

    avnn_ms = float(np.mean(rr_ms))
    sdnn_ms = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 3 else float(np.std(rr_ms, ddof=0))

    drr = np.diff(rr_ms)
    rmssd_ms = float(np.sqrt(np.mean(drr * drr))) if drr.size else np.nan
    pnn50_pct = float(100.0 * np.mean(np.abs(drr) > 50.0)) if drr.size else np.nan

    # PSD-derived metrics (optional)
    lf_power = hf_power = lf_hf = np.nan
    if SCIPY_OK:
        # Use time offsets in seconds within the window for interpolation
        t = (pd.to_datetime(bpm.index).view("int64") / 1e9).astype(float)
        t = t - t.min()
        rr_ms_t = (60000.0 / bpm.to_numpy(dtype=float))
        lf_power = _bandpower_from_rr(rr_ms_t, t, LF_BAND)
        hf_power = _bandpower_from_rr(rr_ms_t, t, HF_BAND)
        lf_hf = float(lf_power / hf_power) if (np.isfinite(lf_power) and np.isfinite(hf_power) and hf_power > 0) else np.nan

    return WindowMetrics(
        hr_bpm=hr_bpm,
        avnn_ms=avnn_ms,
        sdnn_ms=sdnn_ms,
        rmssd_ms=rmssd_ms,
        pnn50_pct=pnn50_pct,
        lf_power=lf_power,
        hf_power=hf_power,
        lf_hf=lf_hf,
        n_samples=n,
    )


def _metrics_by_window(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = _ensure_dtindex_utc(raw)

    # Group into fixed 5-minute windows
    rows = []
    for ts, g in raw.groupby(pd.Grouper(freq=f"{BIN_SECONDS}s")):
        m = _compute_window_metrics(g)
        rows.append({
            "timestamp_utc": ts,
            "hr_bpm": m.hr_bpm,
            "avnn_ms": m.avnn_ms,
            "sdnn_ms": m.sdnn_ms,
            "rmssd_ms": m.rmssd_ms,
            "pnn50_pct": m.pnn50_pct,
            "lf_power": m.lf_power,
            "hf_power": m.hf_power,
            "lf_hf": m.lf_hf,
            "n_samples": m.n_samples,
        })
    out = pd.DataFrame(rows).set_index("timestamp_utc").sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def _best_child_day() -> tuple[int, str]:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["participant"].notna()]
    df["n_pairs"] = pd.to_numeric(df["n_pairs"], errors="coerce")
    best = df.sort_values("n_pairs", ascending=False).iloc[0]
    pid = int(best["participant"])
    day = str(best["date_local"])
    return pid, day


def _restrict_to_local_day(raw: pd.DataFrame, day_local: str) -> pd.DataFrame:
    """
    Keep only samples whose LOCAL_TZ date matches `day_local`.
    """
    if raw.empty:
        return raw
    idx_local_date = raw.index.tz_convert(LOCAL_TZ).date
    mask = (idx_local_date.astype(str) == str(day_local))
    return raw.loc[mask]


def _run_for_child_day(pid: int, day_local: str) -> None:
    print(f"Processing P{pid:03d} on {day_local}")
    if not SCIPY_OK:
        print("[INFO] SciPy not found; LF/HF panels will be empty (NaN).")

    ap_raw = _read_apple_raw_hr(pid)
    ou_raw = _read_oura_raw_hr(pid)
    if ap_raw.empty or ou_raw.empty:
        print(f"[WARN] Missing Apple or Oura HR series for P{pid:03d}. Skipping.")
        return

    ap_day = _restrict_to_local_day(ap_raw, day_local)
    ou_day = _restrict_to_local_day(ou_raw, day_local)
    if ap_day.empty or ou_day.empty:
        print(f"[WARN] No data in that local day window for P{pid:03d} on {day_local}. Skipping.")
        return

    ap_m = _metrics_by_window(ap_day).add_prefix("apple_")
    ou_m = _metrics_by_window(ou_day).add_prefix("oura_")

    # Align to common window timestamps (inner join for plotting)
    joined = ap_m.join(ou_m, how="inner")
    if joined.empty:
        print(f"[WARN] No overlapping 5-minute windows after alignment for P{pid:03d} on {day_local}. Skipping.")
        return

    # Save per-window table
    out_csv = os.path.join(OUT_DIR, f"P{pid:03d}_{day_local}_multimetric.csv")
    joined.to_csv(out_csv, index_label="timestamp_utc")

    # Plot in local time
    plot_df = joined.copy()
    plot_df.index = plot_df.index.tz_convert(LOCAL_TZ)
    if plot_df.shape[0] > MAX_POINTS_LINE:
        plot_df = plot_df.iloc[np.linspace(0, plot_df.shape[0] - 1, MAX_POINTS_LINE).astype(int)]

    metrics = [
        ("hr_bpm", "HR (bpm)"),
        ("rmssd_ms", "RMSSD (ms)"),
        ("sdnn_ms", "SDNN (ms)"),
        ("avnn_ms", "AVNN (ms)"),
        ("pnn50_pct", "pNN50 (%)"),
        ("lf_power", "LF power (a.u.)"),
        ("hf_power", "HF power (a.u.)"),
        ("lf_hf", "LF/HF"),
    ]

    n_panels = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 9.5), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        ax.plot(plot_df.index, plot_df[f"apple_{key}"], color="forestgreen", linewidth=1.5, label="Apple")
        ax.plot(plot_df.index, plot_df[f"oura_{key}"], color="crimson", linewidth=1.5, label="Oura")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    # Turn off unused axes if any
    for j in range(n_panels, axes.size):
        axes[j].axis("off")

    # Shared labels/legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle(
        f"P{pid:03d} {day_local} — 5-min window metrics (Apple vs Oura) "
        + ("[LF/HF enabled]" if SCIPY_OK else "[LF/HF unavailable]"),
        fontsize=14,
        fontweight="bold",
    )
    axes[min(n_panels - 1, axes.size - 1)].set_xlabel(f"Local Time ({LOCAL_TZ})")
    fig.tight_layout(rect=[0, 0.02, 0.98, 0.94])

    out_png = os.path.join(OUT_DIR, f"P{pid:03d}_{day_local}_multimetric.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Saved CSV: {os.path.abspath(out_csv)}")
    print(f"Saved plot: {os.path.abspath(out_png)}")


def main():
    # Find best (participant, day) by n_pairs, then use that participant
    pid, _ = _best_child_day()
    print(f"Best participant (by max n_pairs on any day): P{pid:03d}")

    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["participant"].notna()]
    # Ensure participant column is numeric so filtering by pid works
    df["participant"] = pd.to_numeric(df["participant"], errors="coerce").astype("Int64")
    df["n_pairs"] = pd.to_numeric(df["n_pairs"], errors="coerce")
    df = df[df["participant"] == int(pid)]
    df = df[df["date_local"].notna()].sort_values("date_local")

    for _, row in df.iterrows():
        day_local = str(row["date_local"])
        print(f"--- P{pid:03d} {day_local} ---")
        _run_for_child_day(pid, day_local)


if __name__ == "__main__":
    main()

