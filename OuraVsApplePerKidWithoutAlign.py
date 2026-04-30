"""
make_separate_hr_graphs.py
Creates two separate plots per participant:
  - Oura-only HR time series  -> graphs/oura/P###_oura_hr.png
  - Apple-only HR time series -> graphs/apple/P###_apple_hr.png

Assumes:
- Apple files: ./heart-rate-applewatch-ouraring-main/heart_rate{n}_filtered_*.csv
  (no {n} means participant 1). Apple HR column is "value" (count/min) or "bpm".
- Oura files:  ./OuraRing/P{n}OuraRing/HeartRate/*.csv  (one CSV per day, "bpm")
Times are converted to local PST before plotting.
"""

import os, re, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
import pytz

# -------- CONFIG --------
APPLE_DIR = "./heart-rate-applewatch-ouraring-main"
OURA_DIR  = "./OuraRing"
GRAPHS_DIR = "./graphs"
APPLE_OUT_DIR = os.path.join(GRAPHS_DIR, "apple")
OURA_OUT_DIR  = os.path.join(GRAPHS_DIR, "oura")
APPLE_PREFERRED_TIME_COLS = ["startDate", "endDate"]  # choose in this order
LOCAL_TZ = pytz.timezone("US/Pacific")
# ------------------------

os.makedirs(APPLE_OUT_DIR, exist_ok=True)
os.makedirs(OURA_OUT_DIR, exist_ok=True)

def parse_apple_pid(path: str):
    b = os.path.basename(path)
    m = re.match(r"heart_rate(\d*)_filtered_.*\.csv$", b)
    if m:
        return int(m.group(1)) if m.group(1) else 1
    if b.startswith("heart_rate_filtered"):
        return 1
    return None

def pick_time_col(cols):
    for c in APPLE_PREFERRED_TIME_COLS:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "date" in lc or "time" in lc or "iso" in lc or "timestamp" in lc:
            return c
    return None

def read_apple(path: str) -> pd.DataFrame:
    """Return minute-averaged Apple HR with UTC timestamps."""
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    tcol = pick_time_col(df.columns)
    if not tcol:
        raise ValueError("Apple file has no usable time column")

    # Apple’s HR is "value" (count/min); fall back to "bpm" if present
    hr_col = "value" if "value" in df.columns else ("bpm" if "bpm" in df.columns else None)
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    hr = pd.to_numeric(df.get(hr_col), errors="coerce")
    out = (
        pd.DataFrame({"timestamp_utc": ts.dt.floor("min"), "apple_bpm": hr})
        .dropna(subset=["timestamp_utc", "apple_bpm"])
        .groupby("timestamp_utc", as_index=False)["apple_bpm"].mean()
    )
    return out

def read_oura(folder: str) -> pd.DataFrame:
    """Return minute-averaged Oura HR with UTC timestamps from all daily files."""
    paths = sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv")))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
                df = pd.read_csv(p, sep=";")
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                for c in df.columns:
                    if isinstance(c, str) and ("T" in c and ("Z" in c or "+" in c)):
                        tcol = c; break
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            bpm = pd.to_numeric(df.get("bpm"), errors="coerce")
            frames.append(pd.DataFrame({"timestamp_utc": ts.dt.floor("min"), "oura_bpm": bpm}))
        except Exception as e:
            print(f"[WARN] Skipping Oura file {p}: {e}")

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "oura_bpm"])
    both = pd.concat(frames, ignore_index=True)
    both = both.dropna(subset=["timestamp_utc", "oura_bpm"])
    return both.groupby("timestamp_utc", as_index=False)["oura_bpm"].mean()

# ---- discover participants ----
apple_files = sorted(glob.glob(os.path.join(APPLE_DIR, "heart_rate*_filtered_*.csv")))
apple_map = {parse_apple_pid(f): f for f in apple_files if parse_apple_pid(f) is not None}

oura_ids = []
for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
    m = re.match(r".*P0*([0-9]+)OuraRing$", d)
    if m:
        oura_ids.append(int(m.group(1)))
oura_ids = sorted(set(oura_ids))

# union: we will make a separate graph if data exists for that kid on that device
participants_union = sorted(set(apple_map.keys()).union(oura_ids))

apple_summary, oura_summary = [], []

for pid in participants_union:
    # ----- OURA ONLY -----
    oura_folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            oura_folder = cand
            break
    if oura_folder:
        ou = read_oura(oura_folder)
        if not ou.empty:
            ou["time_local"] = ou["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
            plt.figure(figsize=(10,4.5))
            plt.plot(ou["time_local"], ou["oura_bpm"], linewidth=1.6)
            plt.title(f"Participant {pid} — Oura Heart Rate")
            plt.xlabel("Local Time (PST)")
            plt.ylabel("BPM")
            plt.tight_layout()
            out_png = os.path.join(OURA_OUT_DIR, f"P{pid:03d}_oura_hr.png")
            plt.savefig(out_png, dpi=160); plt.close()
            oura_summary.append({"participant": pid, "points": int(len(ou)), "plot": out_png})
        else:
            print(f"[INFO] Oura: no data for P{pid}")

    # ----- APPLE ONLY -----
    apple_path = apple_map.get(pid)
    if apple_path:
        ap = read_apple(apple_path)
        if not ap.empty:
            ap["time_local"] = ap["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
            # Light smoothing so minute-to-minute noise doesn’t dominate
            ap["apple_bpm_smooth"] = ap["apple_bpm"].rolling(3, min_periods=1).mean()
            plt.figure(figsize=(10,4.5))
            plt.plot(ap["time_local"], ap["apple_bpm_smooth"], linewidth=1.2)
            plt.title(f"Participant {pid} — Apple Watch Heart Rate")
            plt.xlabel("Local Time (PST)")
            plt.ylabel("BPM (count/min)")
            plt.tight_layout()
            out_png = os.path.join(APPLE_OUT_DIR, f"P{pid:03d}_apple_hr.png")
            plt.savefig(out_png, dpi=160); plt.close()
            apple_summary.append({"participant": pid, "points": int(len(ap)), "plot": out_png})
        else:
            print(f"[INFO] Apple: no data for P{pid}")

# Save summaries
if oura_summary:
    pd.DataFrame(oura_summary).sort_values("participant").to_csv(
        os.path.join(OURA_OUT_DIR, "summary_oura.csv"), index=False
    )
if apple_summary:
    pd.DataFrame(apple_summary).sort_values("participant").to_csv(
        os.path.join(APPLE_OUT_DIR, "summary_apple.csv"), index=False
    )

print("Done. Check:", os.path.abspath(OURA_OUT_DIR), "and", os.path.abspath(APPLE_OUT_DIR))
