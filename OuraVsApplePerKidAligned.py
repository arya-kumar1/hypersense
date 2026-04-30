"""
align_compare_hr_lagaligned.py
- Adaptive bin size per participant (from graphs/hr_sampling_summary_oura.csv, p95 rule)
- Compute best time lag (cross-correlation) to align Oura with Apple
- Save original + lag-aligned overlays and CSVs

Outputs:
  graphs/aligned_adaptive2/P###_aligned_{BIN}min.csv
  graphs/aligned_adaptive2/P###_aligned_{BIN}min_with_shift.csv
  graphs/comparison_adaptive2/P###_overlay_{BIN}min.png
  graphs/comparison_adaptive2/P###_overlay_aligned_{BIN}min.png
"""

import os, re, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytz

# ------------ CONFIG ------------
APPLE_DIR = "./heart-rate-applewatch-ouraring-main"
OURA_DIR  = "./OuraRing"
OURA_SUMMARY = "./graphs/hr_sampling_summary_oura.csv"

OUT_ALIGNED = "./graphs/aligned_adaptive3"
OUT_PLOTS   = "./graphs/comparison_adaptive3"

TIMEZONE = "US/Pacific"
APPLE_PREFERRED_TIME_COLS = ["startDate", "endDate"]

DEFAULT_BIN_SECONDS = 300  # if no p95 available
BREAK_GAP_MULTIPLIER = 3
APPLE_SMOOTH_BINS    = 3
CLIP_QUANTILES       = (0.01, 0.99)

# Lag search window (max absolute shift) — adjust if needed
MAX_LAG_HOURS = 2
# ---------------------------------

os.makedirs(OUT_ALIGNED, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)
local_tz = pytz.timezone(TIMEZONE)

def _safe_read_csv(path):
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def _pick_time_col(cols, preferred=None):
    preferred = preferred or []
    for c in preferred:
        if c in cols: return c
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["date","time","iso","timestamp"]):
            return c
    return None

def parse_apple_pid(path):
    b = os.path.basename(path)
    m = re.match(r"heart_rate(\d*)_filtered_.*\.csv$", b)
    if m:
        return int(m.group(1)) if m.group(1) else 1
    if b.startswith("heart_rate_filtered"): return 1
    return None

def choose_bin_seconds(p95_seconds):
    if p95_seconds is None or pd.isna(p95_seconds): return DEFAULT_BIN_SECONDS
    if p95_seconds <= 600:   return 300
    if p95_seconds <= 1200:  return 600
    return 900

def read_apple_resampled(path, bin_seconds):
    if not path:
        return pd.DataFrame(columns=["apple_bpm"])

    df = _safe_read_csv(path)

    # Prefer a timestamp column that already has a timezone offset
    tcol = None
    if "endDate" in df.columns:
        tcol = "endDate"
    elif "startDate" in df.columns:
        tcol = "startDate"
    else:
        tcol = _pick_time_col(df.columns, APPLE_PREFERRED_TIME_COLS)
    if not tcol:
        raise ValueError(f"Apple missing time column: {path}")

    # Parse WITHOUT forcing utc first (fix naive -> PST -> UTC)
    ts = pd.to_datetime(df[tcol], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("US/Pacific", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    # Apple HR column: "value" (count/min) or "bpm"
    hr_col = "value" if "value" in df.columns else ("bpm" if "bpm" in df.columns else None)
    hr = pd.to_numeric(df.get(hr_col), errors="coerce")

    # ── HR RANGE FILTER: keep only 40–180 bpm ───────────────────────────
    mask = (hr >= 40) & (hr <= 180)
    s = pd.Series(hr.where(mask).values, index=ts)  # out-of-range -> NaN
    # ────────────────────────────────────────────────────────────────────

    s = s.dropna().sort_index()
    return s.resample(f"{bin_seconds}s").mean().to_frame("apple_bpm")



def read_oura_resampled(folder, bin_seconds):
    if not folder: return pd.DataFrame(columns=["oura_bpm"])
    paths = sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv")))
    series_list = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else _pick_time_col(df.columns)
            if not tcol: 
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            hr = pd.to_numeric(df.get("bpm"), errors="coerce")

            # ── HR RANGE FILTER: keep only 40–180 bpm ────────────────────
            mask = (hr >= 40) & (hr <= 180)
            s = pd.Series(hr.where(mask).values, index=ts)  # out-of-range -> NaN
            # ─────────────────────────────────────────────────────────────

            s = s.dropna()
            series_list.append(s)
        except Exception as e:
            print(f"[WARN] Oura read fail {p}: {e}")
    if not series_list:
        return pd.DataFrame(columns=["oura_bpm"])
    s_all = pd.concat(series_list).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("oura_bpm")


def clip_series_quantiles(s, qlo=0.01, qhi=0.99):
    if s.dropna().empty: return s
    lo, hi = s.quantile(qlo), s.quantile(qhi)
    return s.clip(lower=lo, upper=hi)

def _robust_zscore(s: pd.Series) -> pd.Series:
    """Median-centered, MAD-scaled z-score. Falls back to std if MAD==0."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    med = s.median()
    mad = np.median(np.abs(s - med))  # true MAD
    scale = mad if mad and not np.isnan(mad) else s.std(ddof=0)
    if not scale or np.isnan(scale):
        scale = 1.0
    return (s - med) / scale

def best_lag_bins(ap: pd.Series, ou: pd.Series, max_lag_bins: int) -> tuple[int, float]:
    """
    Return (lag_bins, corr) maximizing Pearson correlation between
    Apple series and Oura shifted by 'lag' bins.
    Positive lag => shift Oura FORWARD (compare Apple(t) with Oura(t + lag)).
    """
    # z-score each series robustly (center & scale)
    ap_z = _robust_zscore(ap)
    ou_z = _robust_zscore(ou)

    # Put them on the same index (outer join), keep alignment by reindexing
    # We’ll dropna after shifting in each iteration.
    common_index = ap.index.union(ou.index).sort_values()
    ap_z = ap_z.reindex(common_index)
    ou_z = ou_z.reindex(common_index)

    best_lag, best_corr = 0, np.nan
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        ou_shift = ou_z.shift(lag)
        df = pd.concat([ap_z, ou_shift], axis=1, keys=["ap", "ou"]).dropna()
        if len(df) < 12:
            continue
        c = df["ap"].corr(df["ou"])
        if np.isnan(best_corr) or (c is not None and c > best_corr):
            best_corr, best_lag = c, lag
    return best_lag, best_corr

# Load Oura p95s
oura_p95_map = {}
if os.path.isfile(OURA_SUMMARY):
    try:
        sm = pd.read_csv(OURA_SUMMARY)
        for _, r in sm.iterrows():
            oura_p95_map[int(r["participant"])] = float(r["interval_p95_s"]) if pd.notna(r["interval_p95_s"]) else None
    except Exception as e:
        print(f"[WARN] Could not read Oura summary: {e} (using default bins).")

# Discover participants
apple_files = sorted(glob.glob(os.path.join(APPLE_DIR, "heart_rate*_filtered_*.csv")))
apple_map = {parse_apple_pid(f): f for f in apple_files if parse_apple_pid(f) is not None}

oura_ids = []
for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
    m = re.match(r".*P0*([0-9]+)OuraRing$", d)
    if m: oura_ids.append(int(m.group(1)))
oura_ids = sorted(set(oura_ids))

participants = sorted(set(apple_map.keys()).union(oura_ids))
print("Participants:", participants)

for pid in participants:
    p95 = oura_p95_map.get(pid, None)
    bin_seconds = choose_bin_seconds(p95)
    bin_minutes = bin_seconds // 60
    max_lag_bins = max(1, int((MAX_LAG_HOURS * 3600) / bin_seconds))

    apple_path = apple_map.get(pid)
    oura_folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand): oura_folder = cand; break

    ap_df = read_apple_resampled(apple_path, bin_seconds)
    ou_df = read_oura_resampled(oura_folder, bin_seconds)
    if ap_df.empty and ou_df.empty:
        print(f"[INFO] No data for P{pid}")
        continue

    # Smooth Apple slightly for correlation stability
    if "apple_bpm" in ap_df:
        ap_df["apple_bpm"] = ap_df["apple_bpm"].rolling(APPLE_SMOOTH_BINS, min_periods=1, center=True).median()

    # Outer join for saving; inner for lag finding
    aligned = ap_df.join(ou_df, how="outer")

    # Find best lag on overlap
    overlap = ap_df.join(ou_df, how="inner").dropna()
    if overlap.empty or overlap.shape[0] < 12:
        best_lag, best_corr = 0, np.nan
        ou_shifted = ou_df["oura_bpm"]
    else:
        best_lag, best_corr = best_lag_bins(ap_df["apple_bpm"], ou_df["oura_bpm"], max_lag_bins)
        ou_shifted = ou_df["oura_bpm"].shift(best_lag)

    # Save CSVs (UTC)
    csv_base = os.path.join(OUT_ALIGNED, f"P{pid:03d}_aligned_{bin_minutes}min")
    aligned.to_csv(csv_base + ".csv", index_label="timestamp_utc")
    # with shift column
    aligned_shift = aligned.copy()
    aligned_shift["oura_bpm_shifted"] = ou_shifted
    aligned_shift.to_csv(csv_base + "_with_shift.csv", index_label="timestamp_utc")
    print(f"P{pid}: bin={bin_minutes}m best_lag={best_lag} bins (~{best_lag*bin_minutes} min), corr={best_corr:.3f}  -> saved CSVs")

    # --- plotting setup ---
    # break Apple lines across long gaps
    gap_thresh = pd.Timedelta(seconds=bin_seconds * BREAK_GAP_MULTIPLIER)
    ap_plot = ap_df["apple_bpm"].copy()
    ap_plot.index = ap_plot.index.tz_convert(local_tz)
    gap_mask = ap_plot.index.to_series().diff() > gap_thresh
    ap_plot[gap_mask] = pd.NA

    # Prepare original and aligned Oura series for plotting
    ou_plot = ou_df["oura_bpm"].copy()
    ou_plot.index = ou_plot.index.tz_convert(local_tz)

    ou_plot_shift = ou_shifted.copy()
    if ou_plot_shift is not None and not isinstance(ou_plot_shift, pd.Series):
        ou_plot_shift = pd.Series(dtype=float)
    if isinstance(ou_plot_shift, pd.Series) and not ou_plot_shift.empty:
        ou_plot_shift.index = ou_df.index.tz_convert(local_tz)

    # y-limits (robust)
    y_all = pd.concat([ap_plot, ou_plot, ou_plot_shift], axis=0)
    y_all = clip_series_quantiles(y_all, *CLIP_QUANTILES)
    y_lo, y_hi = (y_all.min(), y_all.max()) if not y_all.dropna().empty else (None, None)

    # Overlay (original)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(ap_plot.index, ap_plot, label="Apple (mean per bin)")
    ax.scatter(ou_plot.index, ou_plot, s=12, color="tab:orange", alpha=0.9, edgecolors="white", linewidths=0.3, label="Oura (mean per bin)")
    ax.set_title(f"P{pid} — Original (bin={bin_minutes} min, {TIMEZONE})")
    ax.set_xlabel(f"Local Time ({TIMEZONE})"); ax.set_ylabel("BPM (count/min)")
    ax.grid(True, alpha=0.25); 
    if y_lo is not None and y_hi is not None: ax.set_ylim(y_lo, y_hi)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOTS, f"P{pid:03d}_overlay_{bin_minutes}min.png"), dpi=160); plt.close(fig)

    # Overlay (lag-aligned)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(ap_plot.index, ap_plot, label="Apple (mean per bin)")
    ax.scatter(ou_plot_shift.index, ou_plot_shift, s=12, color="tab:orange", alpha=0.9, edgecolors="white", linewidths=0.3,
               label=f"Oura (shifted by {best_lag*bin_minutes} min)")
    ax.set_title(f"P{pid} — Lag-aligned (bin={bin_minutes} min, lag={best_lag} bins, r={best_corr:.2f})")
    ax.set_xlabel(f"Local Time ({TIMEZONE})"); ax.set_ylabel("BPM (count/min)")
    ax.grid(True, alpha=0.25);
    if y_lo is not None and y_hi is not None: ax.set_ylim(y_lo, y_hi)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOTS, f"P{pid:03d}_overlay_aligned_{bin_minutes}min.png"), dpi=160); plt.close(fig)

print("Done.")
