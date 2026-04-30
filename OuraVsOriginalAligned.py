# align_compare_hr_healthapp_labeled.py
# Apple (HealthApp/Labeled) vs Oura HR with adaptive bins, lag alignment, robust TZ/index handling.

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytz

# ---------- CONFIG ----------
HEALTHAPP_ROOT = "./HealthApp"                     # contains HealthAppP1, HealthAppP7, ...
OURA_DIR       = "./OuraRing"
OURA_SUMMARY   = "./graphs/hr_sampling_summary_oura.csv"  # optional p95 for bin size

OUT_ALIGNED = "./graphs/aligned_healthapp"
OUT_PLOTS   = "./graphs/comparison_healthapp"

TIMEZONE = "US/Pacific"
DEFAULT_BIN_SECONDS = 300      # 5 min
HR_MIN, HR_MAX = 40, 180
BREAK_GAP_MULTIPLIER = 3
APPLE_SMOOTH_BINS    = 3
CLIP_Q = (0.01, 0.99)
MAX_LAG_HOURS = 2
# -----------------------------------------------

os.makedirs(OUT_ALIGNED, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)
local_tz = pytz.timezone(TIMEZONE)

# ---------- utils ----------
def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def ensure_dt_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("timestamp_utc","timestamp","time","date","Date","Time"):
            if cand in df.columns:
                idx = pd.to_datetime(df[cand], utc=True, errors="coerce")
                df = df.drop(columns=[cand])
                df.index = idx
                break
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def to_local_index(s: pd.Series, tz) -> pd.Series:
    out = s.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert(tz)
    return out

def choose_bin_seconds(p95_seconds: float | None) -> int:
    if p95_seconds is None or pd.isna(p95_seconds): return DEFAULT_BIN_SECONDS
    if p95_seconds <= 600: return 300
    if p95_seconds <= 1200: return 600
    return 900

def _robust_z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return s
    med = s.median()
    mad = np.median(np.abs(s - med))
    scale = mad if mad and not np.isnan(mad) else s.std(ddof=0) or 1.0
    return (s - med) / scale

def best_lag_bins(ap: pd.Series, ou: pd.Series, max_lag_bins: int) -> tuple[int, float]:
    ap_z = _robust_z(ap)
    ou_z = _robust_z(ou)
    idx = ap.index.union(ou.index).sort_values()
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
    return best_lag, best_corr

# ---------- file discovery ----------
def _healthapp_base_for_pid(pid: int) -> str | None:
    for name in (f"HealthAppP{pid}", f"HealthAppP{pid:02d}", f"HealthAppP{pid:03d}"):
        base = os.path.join(HEALTHAPP_ROOT, name)
        if os.path.isdir(base):
            return base
    return None

def find_apple_heartrate_files(pid: int) -> list[str]:
    """Look in .../Labeled/Record/** and .../Record/** for *_HeartRate.csv."""
    base = _healthapp_base_for_pid(pid)
    if base is None:
        return []
    patterns = [
        os.path.join(base, "Labeled", "Record", "**", "*_HeartRate.csv"),
        os.path.join(base, "Record", "**", "*_HeartRate.csv"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    # de-dup & sort
    return sorted(list(dict.fromkeys(found)))

# ---------- readers ----------
def read_apple_healthapp_resampled(pid: int, bin_seconds: int) -> pd.DataFrame:
    paths = find_apple_heartrate_files(pid)
    if not paths:
        print(f"[INFO] P{pid}: no Apple HeartRate files found under Labeled/Record or Record.")
        return pd.DataFrame(columns=["apple_bpm"])

    # Choose the best time column for these labeled files: prefer CreationDate, then EndDate, then StartDate.
    def _pick_time_col(df: pd.DataFrame) -> str | None:
        for c in ("CreationDate","EndDate","StartDate","Time","Date","Time_In_PST"):
            if c in df.columns:
                return c
        # fallback to the column that parses most values
        best, bestn = None, 0
        for c in df.columns:
            try:
                n = pd.to_datetime(df[c], errors="coerce").notna().sum()
                if n > bestn:
                    best, bestn = c, n
            except Exception:
                pass
        return best

    series_list = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            # Must be HeartRate rows
            if "Type" in df.columns:
                type_mask = df["Type"].astype(str).str.contains("HeartRate", case=False, na=False)
                df = df[type_mask]
            # Value + Unit filters
            if "Value" not in df.columns:
                print(f"[INFO] P{pid}: {os.path.basename(p)} has no Value column, skipping.")
                continue
            if "Unit" in df.columns:
                unit_mask = df["Unit"].astype(str).str.lower().str.contains("count/min")
                df = df[unit_mask]

            tcol = _pick_time_col(df)
            if not tcol:
                print(f"[INFO] P{pid}: {os.path.basename(p)} no usable time column; columns={list(df.columns)[:8]}...")
                continue

            # parse time -> UTC; handle strings like '2025-02-03 08:45:01 -0800'
            ts = pd.to_datetime(df[tcol], errors="coerce")
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize("US/Pacific", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")

            hr = pd.to_numeric(df["Value"], errors="coerce")
            mask_range = (hr >= HR_MIN) & (hr <= HR_MAX)
            s = pd.Series(hr.where(mask_range).values, index=ts).dropna()
            if not s.empty:
                series_list.append(s)
        except Exception as e:
            print(f"[WARN] Apple read fail {p}: {e}")

    if not series_list:
        print(f"[INFO] P{pid}: no valid Apple HR rows after filtering.")
        return pd.DataFrame(columns=["apple_bpm"])

    s_all = pd.concat(series_list).sort_index()
    print(f"[DEBUG] P{pid}: Apple points after filter = {len(s_all)} from {len(paths)} files.")
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("apple_bpm")

def read_oura_resampled(pid: int, bin_seconds: int) -> pd.DataFrame:
    folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand; break
    if not folder:
        return pd.DataFrame(columns=["oura_bpm"])

    paths = sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv")))
    series = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                # pick best time-like column if needed
                for c in df.columns:
                    if isinstance(c,str) and ("T" in c and ("Z" in c or "+" in c)):
                        tcol = c; break
            if not tcol: 
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            hr = pd.to_numeric(df.get("bpm"), errors="coerce")
            mask = (hr >= HR_MIN) & (hr <= HR_MAX)
            s = pd.Series(hr.where(mask).values, index=ts).dropna()
            series.append(s)
        except Exception as e:
            print(f"[WARN] Oura read fail {p}: {e}")
    if not series:
        return pd.DataFrame(columns=["oura_bpm"])
    s_all = pd.concat(series).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("oura_bpm")

# ---------- participants ----------
health_pids = []
for d in glob.glob(os.path.join(HEALTHAPP_ROOT, "HealthAppP*")):
    m = re.match(r".*HealthAppP0*([0-9]+)$", d)
    if m: health_pids.append(int(m.group(1)))
oura_pids = []
for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
    m = re.match(r".*P0*([0-9]+)OuraRing$", d)
    if m: oura_pids.append(int(m.group(1)))
participants = sorted(set(health_pids).union(oura_pids))
print("Participants:", participants)

# p95 bins (optional)
oura_p95_map = {}
if os.path.isfile(OURA_SUMMARY):
    sm = pd.read_csv(OURA_SUMMARY)
    for _, r in sm.iterrows():
        if pd.notna(r.get("participant")):
            oura_p95_map[int(r["participant"])] = float(r["interval_p95_s"]) if pd.notna(r.get("interval_p95_s")) else None

# ---------- main ----------
for pid in participants:
    bin_seconds = choose_bin_seconds(oura_p95_map.get(pid))
    bin_minutes = bin_seconds // 60
    max_lag_bins = max(1, int((MAX_LAG_HOURS * 3600) / bin_seconds))

    ap_df = read_apple_healthapp_resampled(pid, bin_seconds)
    ou_df = read_oura_resampled(pid, bin_seconds)

    ap_df = ensure_dt_utc_index(ap_df)
    ou_df = ensure_dt_utc_index(ou_df)

    if ap_df.empty and ou_df.empty:
        print(f"[INFO] No data for P{pid}")
        continue

    if "apple_bpm" in ap_df and not ap_df.empty:
        ap_df["apple_bpm"] = ap_df["apple_bpm"].rolling(APPLE_SMOOTH_BINS, center=True, min_periods=1).median()

    aligned = ap_df.join(ou_df, how="outer")

    # find best lag on overlap
    overlap = ap_df.join(ou_df, how="inner").dropna()
    if overlap.empty or overlap.shape[0] < 12:
        best_lag, best_corr = 0, np.nan
        ou_shifted = ou_df["oura_bpm"]
    else:
        best_lag, best_corr = best_lag_bins(ap_df["apple_bpm"], ou_df["oura_bpm"], max_lag_bins)
        ou_shifted = ou_df["oura_bpm"].shift(best_lag)

    # save CSVs (UTC)
    base = os.path.join(OUT_ALIGNED, f"P{pid:03d}_aligned_{bin_minutes}min")
    aligned.to_csv(base + ".csv", index_label="timestamp_utc")
    aligned_shift = aligned.copy()
    aligned_shift["oura_bpm_shifted"] = ou_shifted
    aligned_shift.to_csv(base + "_with_shift.csv", index_label="timestamp_utc")
    print(f"P{pid}: bin={bin_minutes}m best_lag={best_lag} (~{best_lag*bin_minutes} min), r={best_corr:.3f}")

    # plotting
    gap_thresh = pd.Timedelta(seconds=bin_seconds * BREAK_GAP_MULTIPLIER)

    ap_plot = to_local_index(ap_df["apple_bpm"], local_tz)
    if isinstance(ap_plot.index, pd.DatetimeIndex):
        ap_plot[ap_plot.index.to_series().diff() > gap_thresh] = pd.NA

    ou_plot = to_local_index(ou_df["oura_bpm"], local_tz)

    ou_plot_shift = ou_shifted.copy()
    if isinstance(ou_plot_shift, pd.Series) and not ou_plot_shift.empty:
        tmp = pd.DataFrame({"x": ou_plot_shift})
        tmp = ensure_dt_utc_index(tmp)
        ou_plot_shift = to_local_index(tmp["x"], local_tz)

    y_all = pd.concat([ap_plot, ou_plot, ou_plot_shift], axis=0)
    if not y_all.dropna().empty:
        y_lo, y_hi = y_all.quantile(CLIP_Q[0]), y_all.quantile(CLIP_Q[1])
    else:
        y_lo = y_hi = None

    # Original
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(ap_plot.index, ap_plot, label="Apple (mean per bin)")
    ax.scatter(ou_plot.index, ou_plot, s=12, color="tab:orange", alpha=0.9,
               edgecolors="white", linewidths=0.3, label="Oura (mean per bin)")
    ax.set_title(f"P{pid} — Original (bin={bin_minutes} min, {TIMEZONE})")
    ax.set_xlabel(f"Local Time ({TIMEZONE})"); ax.set_ylabel("BPM (count/min)")
    ax.grid(True, alpha=0.25)
    if y_lo is not None and y_hi is not None: ax.set_ylim(y_lo, y_hi)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOTS, f"P{pid:03d}_overlay_{bin_minutes}min.png"), dpi=160)
    plt.close(fig)

    # Lag-aligned
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(ap_plot.index, ap_plot, label="Apple (mean per bin)")
    ax.scatter(ou_plot_shift.index, ou_plot_shift, s=12, color="tab:orange", alpha=0.9,
               edgecolors="white", linewidths=0.3,
               label=f"Oura (shifted {best_lag*bin_minutes} min)")
    ax.set_title(f"P{pid} — Lag-aligned (bin={bin_minutes} min, lag={best_lag} bins, r={best_corr:.2f})")
    ax.set_xlabel(f"Local Time ({TIMEZONE})"); ax.set_ylabel("BPM (count/min)")
    ax.grid(True, alpha=0.25)
    if y_lo is not None and y_hi is not None: ax.set_ylim(y_lo, y_hi)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOTS, f"P{pid:03d}_overlay_aligned_{bin_minutes}min.png"), dpi=160)
    plt.close(fig)

print("Done.")
