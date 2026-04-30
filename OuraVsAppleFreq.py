"""
hr_sampling_frequency.py
Summarize how often Oura and Apple Watch record heart rate (sampling cadence).

Outputs (created if missing):
- graphs/hr_sampling_summary_oura.csv
- graphs/hr_sampling_summary_apple.csv

Assumptions:
- Apple: ./heart-rate-applewatch-ouraring-main/heart_rate{n}_filtered_*.csv
         Time column is usually "startDate" or "endDate". HR column "value" (count/min) or "bpm".
- Oura:  ./OuraRing/P{n}OuraRing/HeartRate/*.csv, one CSV per day, "Time_In_ISO" and "bpm".
"""

import os, re, glob, math
import pandas as pd

# ---------- CONFIG ----------
APPLE_DIR = "./heart-rate-applewatch-ouraring-main"
OURA_DIR  = "./OuraRing"
OUT_DIR   = "./graphs"
APPLE_PREFERRED_TIME_COLS = ["startDate", "endDate"]  # in this order
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Basic robust CSV reader (handles semicolon-delimited files)."""
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def _pick_time_col(cols, preferred=None):
    preferred = preferred or []
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in ["date", "time", "iso", "timestamp"]):
            return c
    return None

def parse_apple_pid(path: str):
    """Infer participant id from filename like heart_rate2_filtered_*.csv (blank => 1)."""
    b = os.path.basename(path)
    m = re.match(r"heart_rate(\d*)_filtered_.*\.csv$", b)
    if m:
        return int(m.group(1)) if m.group(1) else 1
    if b.startswith("heart_rate_filtered"):
        return 1
    return None

def read_apple_raw_times(path: str) -> pd.Series:
    """Return a UTC datetime Series of the raw Apple timestamps (no resampling)."""
    df = _safe_read_csv(path)
    tcol = _pick_time_col(df.columns, APPLE_PREFERRED_TIME_COLS)
    if not tcol:
        raise ValueError("Apple file has no recognizable time column.")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    return ts.dropna().sort_values().reset_index(drop=True)

def read_oura_raw_times(folder: str) -> pd.Series:
    """Return a UTC datetime Series of the raw Oura timestamps across all daily files."""
    paths = sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv")))
    frames = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else _pick_time_col(df.columns)
            if not tcol:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            frames.append(pd.DataFrame({"ts": ts}))
        except Exception as e:
            print(f"[WARN] Oura read fail {p}: {e}")
    if not frames:
        return pd.Series(dtype="datetime64[ns, UTC]")
    s = pd.concat(frames, ignore_index=True)["ts"].dropna().sort_values().reset_index(drop=True)
    return s

def sampling_stats_from_times(ts: pd.Series) -> dict:
    """Compute sampling cadence metrics from raw timestamps."""
    if ts.empty:
        return {
            "points": 0, "first_ts": pd.NaT, "last_ts": pd.NaT,
            "days_with_data": 0, "avg_points_per_day": 0.0,
            "interval_median_s": math.nan, "interval_mean_s": math.nan,
            "interval_p05_s": math.nan, "interval_p95_s": math.nan,
            "typical_interval_s": math.nan
        }
    # per-day counts for average points/day
    days = ts.dt.tz_convert("UTC").dt.date  # date in UTC
    days_with_data = days.nunique()
    avg_per_day = len(ts) / days_with_data if days_with_data > 0 else float("nan")

    # consecutive deltas
    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        med = mean = p05 = p95 = math.nan
    else:
        med = float(deltas.median())
        mean = float(deltas.mean())
        p05 = float(deltas.quantile(0.05))
        p95 = float(deltas.quantile(0.95))
    typical = float(round(med)) if deltas.size else math.nan

    return {
        "points": int(len(ts)),
        "first_ts": ts.iloc[0],
        "last_ts": ts.iloc[-1],
        "days_with_data": int(days_with_data),
        "avg_points_per_day": float(round(avg_per_day, 2)) if not math.isnan(avg_per_day) else math.nan,
        "interval_median_s": med,
        "interval_mean_s": mean,
        "interval_p05_s": p05,
        "interval_p95_s": p95,
        "typical_interval_s": typical
    }

def summarize_oura() -> pd.DataFrame:
    # detect participants from folder names P{n}OuraRing or P{nnn}OuraRing
    pids = []
    for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m:
            pids.append((int(m.group(1)), d))
    rows = []
    for pid, folder in sorted(set(pids), key=lambda x: x[0]):
        ts = read_oura_raw_times(folder)
        stats = sampling_stats_from_times(ts)
        rows.append({"participant": pid, **stats})
    return pd.DataFrame(rows).sort_values("participant")

def summarize_apple() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(APPLE_DIR, "heart_rate*_filtered_*.csv")))
    # map pid -> file (if multiple, you can extend to combine; here we take the last sorted)
    pid_to_file = {}
    for f in files:
        pid = parse_apple_pid(f)
        if pid is not None:
            pid_to_file[pid] = f
    rows = []
    for pid in sorted(pid_to_file.keys()):
        try:
            ts = read_apple_raw_times(pid_to_file[pid])
        except Exception as e:
            print(f"[WARN] Apple read fail {pid_to_file[pid]}: {e}")
            ts = pd.Series(dtype="datetime64[ns, UTC]")
        stats = sampling_stats_from_times(ts)
        rows.append({"participant": pid, **stats})
    return pd.DataFrame(rows).sort_values("participant")

def main():
    oura_df = summarize_oura()
    apple_df = summarize_apple()

    out_oura = os.path.join(OUT_DIR, "hr_sampling_summary_oura.csv")
    out_apple = os.path.join(OUT_DIR, "hr_sampling_summary_apple.csv")

    if not oura_df.empty:
        oura_df.to_csv(out_oura, index=False)
        print("Saved:", os.path.abspath(out_oura))
    else:
        print("No Oura data found.")

    if not apple_df.empty:
        apple_df.to_csv(out_apple, index=False)
        print("Saved:", os.path.abspath(out_apple))
    else:
        print("No Apple data found.")

if __name__ == "__main__":
    main()
