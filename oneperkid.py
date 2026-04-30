# make_child_day_overlays_v2.py
# 1 plot = 1 child on 1 day (Apple vs Oura overlay), aiming ~40 and capping at 60.
# Improvements:
#  - Match HealthApp files named "HeartRate.csv" (not just "*_HeartRate.csv")
#  - Multi-bin strategy: try 5, 10, 15 min and keep the best-covered child-day
#  - Lower daily minimum pairs to include more valid days

import os, re, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional for daily r,p (not needed to plot)
try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------- CONFIG ----------------
HEALTHAPP_ROOT = "./HealthApp"          # HealthAppP1, HealthAppP07, ...
OURA_DIR       = "./OuraRing"           # P1OuraRing, P001OuraRing, ...

OUT_DIR        = "./graphs/child_day_overlays"
SUMMARY_CSV    = os.path.join(OUT_DIR, "child_day_summary.csv")

# We will try bins in this order and keep the best-covered per child-day
BIN_TRY_SECONDS = [300, 600, 900]       # 5, 10, 15 minutes

HR_MIN, HR_MAX = 40, 180                # plausible HR window
LOCAL_TZ       = "US/Pacific"           # "day" is local calendar day

MIN_POINTS_DAY = 8                      # minimum paired bins on a day (lower than before)
TARGET_GRAPHS  = 40                     # aim for ~30–40 plots
MAX_GRAPHS     = 60                     # hard cap

# Plot styling
BREAK_GAP_MULTIPLIER = 3                # break Apple line if gap > 3 bins
APPLE_SMOOTH_BINS    = 3                # small rolling median smoothing
CLIP_QUANTILES       = (0.01, 0.99)     # robust y-limits
SCATTER_SIZE         = 14
SCATTER_ALPHA        = 0.85

# Optional lag alignment (shift Oura by best integer number of bins)
ENABLE_LAG_ALIGNMENT = True
MAX_LAG_HOURS        = 1                # search ±1 hour
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --------- helpers ---------
def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

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

def _participants() -> list[int]:
    health_pids, oura_pids = [], []
    for d in glob.glob(os.path.join(HEALTHAPP_ROOT, "HealthAppP*")):
        m = re.match(r".*HealthAppP0*([0-9]+)$", d)
        if m: health_pids.append(int(m.group(1)))
    for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m: oura_pids.append(int(m.group(1)))
    return sorted(set(health_pids).union(oura_pids))

def _healthapp_base_for_pid(pid: int) -> str | None:
    for name in (f"HealthAppP{pid}", f"HealthAppP{pid:02d}", f"HealthAppP{pid:03d}"):
        base = os.path.join(HEALTHAPP_ROOT, name)
        if os.path.isdir(base):
            return base
    return None

def _find_apple_hr_files(pid: int) -> list[str]:
    """
    Include BOTH exact 'HeartRate.csv' and any '*HeartRate*.csv' under HealthApp paths.
    This avoids missing files named exactly 'HeartRate.csv'.
    """
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
    out = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    # de-dup
    return sorted(list(dict.fromkeys(out)))

def _read_apple_raw(pid: int) -> pd.Series:
    """
    Apple Health HR -> raw series (UTC index), no resampling. Returns pd.Series of HR.
    Accepts files with columns like:
      class,Time_In_PST,time,CreationDate,EndDate,StartDate,Type,Unit,Value,ID,...
    """
    paths = _find_apple_hr_files(pid)
    if not paths:
        return pd.Series(dtype=float)

    def _pick_time_col(df: pd.DataFrame) -> str | None:
        for c in ("CreationDate", "EndDate", "StartDate", "Time", "Date", "Time_In_PST"):
            if c in df.columns:
                return c
        # fallback: most-parsable
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
            # HR column
            hr_col = "Value" if "Value" in df.columns else ("value" if "value" in df.columns else None)
            if not hr_col:
                continue
            tcol = _pick_time_col(df)
            if not tcol:
                continue
            ts = _tz_to_utc(df[tcol])
            hr = pd.to_numeric(df[hr_col], errors="coerce")
            mask = (hr >= HR_MIN) & (hr <= HR_MAX)
            s = pd.Series(hr.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Apple read fail {p}: {e}")

    if not series:
        return pd.Series(dtype=float)

    return pd.concat(series).sort_index()

def _read_oura_raw(pid: int) -> pd.Series:
    """
    Oura HR -> raw series (UTC index), no resampling. Returns pd.Series of HR.
    """
    folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand; break
    if not folder:
        return pd.Series(dtype=float)

    series = []
    for p in sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv"))):
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                for c in df.columns:
                    if isinstance(c, str) and ("T" in c and ("Z" in c or "+" in c)):
                        tcol = c; break
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
        return pd.Series(dtype=float)

    return pd.concat(series).sort_index()

def _robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return s
    med = s.median()
    mad = np.median(np.abs(s - med))
    scale = mad if mad and not np.isnan(mad) else s.std(ddof=0) or 1.0
    return (s - med) / scale

def _best_lag_bins(ap: pd.Series, ou: pd.Series, bin_seconds: int) -> int:
    """Return lag (in bins) maximizing Pearson correlation (Oura shifted)."""
    if not ENABLE_LAG_ALIGNMENT:
        return 0
    max_lag_bins = max(1, int((MAX_LAG_HOURS * 3600) / bin_seconds))
    ap_z = _robust_z(ap)
    ou_z = _robust_z(ou)
    idx = ap_z.index.union(ou_z.index).sort_values()
    ap_z = ap_z.reindex(idx)
    ou_z = ou_z.reindex(idx)
    best_lag, best_corr = 0, np.nan
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        ou_shift = ou_z.shift(lag)
        df = pd.concat([ap_z, ou_shift], axis=1).dropna()
        if len(df) < 12:
            continue
        c = df.iloc[:,0].corr(df.iloc[:,1])
        if np.isnan(best_corr) or (c is not None and c > best_corr):
            best_lag, best_corr = lag, c
    return best_lag

def _resample(series: pd.Series, bin_seconds: int, name: str) -> pd.Series:
    if series.empty:
        return series
    s = series.copy()
    s.index = pd.to_datetime(s.index)  # ensure datetime index
    s = s.sort_index()
    return s.resample(f"{bin_seconds}s").mean().rename(name)

# --------- main ---------
def main():
    # Build best candidate per (participant, day) across multiple bin sizes
    best_by_day = {}  # key: (pid, day_str) -> dict with metrics + data for plotting

    pids = _participants()
    print("Participants:", pids)

    for pid in pids:
        ap_raw = _read_apple_raw(pid)
        ou_raw = _read_oura_raw(pid)
        if ap_raw.empty or ou_raw.empty:
            continue

        for bin_seconds in BIN_TRY_SECONDS:
            ap = _resample(ap_raw, bin_seconds, "apple_bpm")
            ou = _resample(ou_raw, bin_seconds, "oura_bpm")

            # mapped bins for all-time; we'll split by day next
            mapped = ap.to_frame().join(ou.to_frame(), how="inner").dropna()
            if mapped.empty:
                continue

            mapped["local_date"] = mapped.index.tz_convert(LOCAL_TZ).date

            for day, df in mapped.groupby("local_date"):
                n_pairs = int(len(df))
                if n_pairs < MIN_POINTS_DAY:
                    continue

                # Keep the best coverage per (pid, day)
                key = (pid, str(day))
                prev = best_by_day.get(key)
                if (prev is None) or (n_pairs > prev["n_pairs"]):
                    # Prepare day series for plotting
                    day_start = pd.Timestamp(f"{day} 00:00:00", tz=LOCAL_TZ).tz_convert("UTC")
                    day_end   = day_start + pd.Timedelta(days=1)
                    ap_day = ap[(ap.index >= day_start) & (ap.index < day_end)]
                    ou_day = ou[(ou.index >= day_start) & (ou.index < day_end)]

                    lag_bins = _best_lag_bins(ap_day, ou_day, bin_seconds)
                    ou_shift = ou_day.shift(lag_bins)

                    # daily correlation (optional)
                    try:
                        r = float(np.corrcoef(df["apple_bpm"], df["oura_bpm"])[0,1])
                    except Exception:
                        r = np.nan
                    p = np.nan
                    if SCIPY_OK:
                        try:
                            r_s, p_s = pearsonr(df["apple_bpm"], df["oura_bpm"])
                            r, p = float(r_s), float(p_s)
                        except Exception:
                            pass

                    best_by_day[key] = {
                        "participant": pid,
                        "day": str(day),
                        "n_pairs": n_pairs,
                        "bin_seconds": bin_seconds,
                        "r": r,
                        "p": p,
                        "lag_bins": int(lag_bins),
                        "ap_day": ap_day,
                        "ou_day_shift": ou_shift
                    }

    if not best_by_day:
        print("[INFO] No child-days with overlapping Apple & Oura data.")
        return

    # Rank by coverage and keep top N
    candidates = sorted(best_by_day.values(), key=lambda d: d["n_pairs"], reverse=True)
    keep_n = min(MAX_GRAPHS, max(TARGET_GRAPHS, 1))
    selected = candidates[:keep_n]

    # Plotting & summary
    rows = []
    for item in selected:
        pid        = item["participant"]
        day        = item["day"]
        n_pairs    = item["n_pairs"]
        r, p       = item["r"], item["p"]
        lag_bins   = item["lag_bins"]
        bin_sec    = item["bin_seconds"]
        ap_day     = item["ap_day"].copy()
        ou_shift   = item["ou_day_shift"].copy()

        gap_thresh = pd.Timedelta(seconds=bin_sec * BREAK_GAP_MULTIPLIER)

        # Convert to local time for axes
        ap_plot  = ap_day.copy()
        ap_plot.index = ap_plot.index.tz_convert(LOCAL_TZ)
        ou_plot  = ou_shift.copy()
        ou_plot.index = ou_plot.index.tz_convert(LOCAL_TZ)

        # Smooth Apple + break across long gaps
        ap_plot = ap_plot.rolling(APPLE_SMOOTH_BINS, center=True, min_periods=1).median()
        ap_plot[ap_plot.index.to_series().diff() > gap_thresh] = pd.NA

        # Robust y-limits
        y_all = pd.concat([ap_plot, ou_plot], axis=0)
        if not y_all.dropna().empty:
            y_lo, y_hi = y_all.quantile(CLIP_QUANTILES[0]), y_all.quantile(CLIP_QUANTILES[1])
        else:
            y_lo = y_hi = None

        # Make the figure
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(ap_plot.index, ap_plot, label=f"Apple (mean per {bin_sec//60} min)")
        ou_line = ou_plot.copy()
        ou_line[ou_line.index.to_series().diff() > gap_thresh] = np.nan  # don't connect long gaps
        ax.plot(
            ou_line.index, ou_line,
            color="tab:orange", linewidth=1.8,
            label=f"Oura (shift {lag_bins * (bin_sec//60)} min)"
        )
        title = f"P{pid} — {day}  (pairs={n_pairs}, r={r:.2f}"
        title += f"{'' if np.isnan(p) else f', p={p:.3g}'}; bin={bin_sec//60}m)"
        ax.set_title(title)
        ax.set_xlabel(f"Local Time ({LOCAL_TZ})")
        ax.set_ylabel("BPM (count/min)")
        ax.grid(True, alpha=0.25)
        if y_lo is not None and y_hi is not None:
            ax.set_ylim(y_lo, y_hi)
        ax.legend()
        fig.tight_layout()

        out_png = os.path.join(OUT_DIR, f"P{pid:03d}_{day}.png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)

        rows.append({
            "participant": pid,
            "date_local": day,
            "n_pairs": n_pairs,
            "pearson_r": r,
            "p_value": p,
            "bin_minutes": bin_sec // 60,
            "lag_bins": lag_bins,
            "lag_minutes": lag_bins * (bin_sec // 60),
            "plot_path": out_png,
        })

    # Write summary CSV
    pd.DataFrame(rows).sort_values(["participant","date_local"]).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {len(rows)} plots to {os.path.abspath(OUT_DIR)}")
    print("Summary:", os.path.abspath(SUMMARY_CSV))

if __name__ == "__main__":
    main()