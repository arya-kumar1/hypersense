# sampling_rate_simple.py
# Reports how often (on average) Oura Ring and Apple Watch record heart rate.
# Output: prints a compact table and saves graphs/hr_average_freq.csv

import os, re, glob, math
import pandas as pd

APPLE_DIR = "./heart-rate-applewatch-ouraring-main"
OURA_DIR  = "./OuraRing"
OUT_CSV   = "./graphs/hr_average_freq.csv"
APPLE_PREFERRED_TIME_COLS = ["startDate", "endDate"]  # prefer in this order

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def _safe_read_csv(path: str) -> pd.DataFrame:
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

def parse_apple_pid(path: str):
    b = os.path.basename(path)
    m = re.match(r"heart_rate(\d*)_filtered_.*\.csv$", b)
    if m:
        return int(m.group(1)) if m.group(1) else 1
    if b.startswith("heart_rate_filtered"):
        return 1
    return None

def raw_times_apple(path: str) -> pd.Series:
    df = _safe_read_csv(path)
    tcol = _pick_time_col(df.columns, APPLE_PREFERRED_TIME_COLS)
    if not tcol: return pd.Series(dtype="datetime64[ns, UTC]")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    return ts.dropna().sort_values().reset_index(drop=True)

def raw_times_oura(folder: str) -> pd.Series:
    paths = sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv")))
    frames = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else _pick_time_col(df.columns)
            if not tcol: continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            frames.append(pd.DataFrame({"ts": ts}))
        except Exception:
            pass
    if not frames:
        return pd.Series(dtype="datetime64[ns, UTC]")
    s = pd.concat(frames, ignore_index=True)["ts"].dropna().sort_values().reset_index(drop=True)
    return s

def typical_interval_seconds(ts: pd.Series) -> float:
    if ts.empty: return math.nan
    deltas = ts.diff().dropna().dt.total_seconds()
    return float(deltas.median()) if not deltas.empty else math.nan

# Discover participants
apple_files = sorted(glob.glob(os.path.join(APPLE_DIR, "heart_rate*_filtered_*.csv")))
apple_map = {parse_apple_pid(f): f for f in apple_files if parse_apple_pid(f) is not None}

oura_ids = []
for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
    m = re.match(r".*P0*([0-9]+)OuraRing$", d)
    if m: oura_ids.append(int(m.group(1)))
oura_ids = sorted(set(oura_ids))

participants = sorted(set(apple_map.keys()).union(oura_ids))

rows = []
for pid in participants:
    # Apple
    ap_ts = raw_times_apple(apple_map.get(pid)) if pid in apple_map else pd.Series(dtype="datetime64[ns, UTC]")
    ap_med = typical_interval_seconds(ap_ts)
    # Oura
    oura_folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand): oura_folder = cand; break
    ou_ts = raw_times_oura(oura_folder) if oura_folder else pd.Series(dtype="datetime64[ns, UTC]")
    ou_med = typical_interval_seconds(ou_ts)

    def row(device, med):
        return {
            "participant": pid,
            "device": device,
            "typical_interval_s": round(med, 1) if pd.notna(med) else math.nan,
            "samples_per_min": round(60.0/med, 2) if med and med > 0 else math.nan,
            "samples_per_hour": round(3600.0/med, 1) if med and med > 0 else math.nan,
        }

    if not math.isnan(ap_med): rows.append(row("apple", ap_med))
    if not math.isnan(ou_med): rows.append(row("oura",  ou_med))

# Build result table
result = pd.DataFrame(rows).sort_values(["participant","device"])

# Print per-kid and overall medians
if not result.empty:
    print("\nPer participant (typical cadence):")
    print(result.to_string(index=False))

    overall = (result
               .groupby("device")["typical_interval_s"]
               .median()
               .rename("overall_typical_interval_s")
               .reset_index())
    overall["samples_per_min"]  = (60/overall["overall_typical_interval_s"]).round(2)
    overall["samples_per_hour"] = (3600/overall["overall_typical_interval_s"]).round(1)

    print("\nOverall (median across kids):")
    print(overall.to_string(index=False))

    # Save CSV
    result_out = result.copy()
    result_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {os.path.abspath(OUT_CSV)}")
else:
    print("No data found.")
