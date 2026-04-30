# plot_apple_sdnn.py
# Graph Apple Watch HRV (SDNN, ms) per participant.
# Outputs (created if missing):
#   graphs/hrv_apple/P###_apple_sdnn_5min.csv
#   graphs/hrv_apple/P###_apple_sdnn_5min.png

import os, re, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plot saving
import matplotlib.pyplot as plt
import pytz
import numpy as np

# ---------- CONFIG ----------
APPLE_DIR = "./heart-rate-applewatch-ouraring-main"
OUT_DIR   = "./graphs/hrv_apple"
BIN_SECONDS = 300                 # 5-minute median bins (good for HRV)
TIMEZONE = "US/Pacific"           # plotting axis; alignment is done in UTC
APPLE_PREFERRED_TIME_COLS = ["endDate", "startDate"]  # prefer endDate if present
# Optional visual tweaks
ROLL_WINDOW_BINS = 3              # small rolling median to tidy noise (set 0 to disable)
CLIP_QUANTILES   = (0.01, 0.99)   # robust y-limits
# -----------------------------

os.makedirs(OUT_DIR, exist_ok=True)
local_tz = pytz.timezone(TIMEZONE)

def _safe_read_csv(path: str) -> pd.DataFrame:
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

def _tz_fix_to_utc(ts: pd.Series) -> pd.Series:
    """
    Localize naive timestamps to US/Pacific, else convert aware to UTC.
    This avoids the 'hours apart' issue.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize("US/Pacific", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    return ts.dt.tz_convert("UTC")

def parse_apple_pid(path: str):
    """
    Infer participant ID.
    Works with files like heart_rate_filtered_*.csv, heart_rate2_filtered_*.csv,
    or HRV-specific names that include a number.
    """
    b = os.path.basename(path)
    m = re.match(r"heart_rate(\d*)_filtered_.*\.csv$", b)
    if m:
        return int(m.group(1)) if m.group(1) else 1
    # Generic fallback for HRV files: ...hrv... or ...variability...
    m2 = re.match(r".*?(?:hrv|variability).*?(\d+)?[_-].*\.csv$", b, flags=re.I)
    if m2 and m2.group(1):
        return int(m2.group(1))
    if b.lower().startswith(("hrv", "variability", "heart_rate_filtered")):
        return 1
    return None

def find_apple_hrv_files() -> list[str]:
    """
    Search for likely Apple HRV exports.
    If your HRV is in the same folder as HR files, we'll still detect the HRV column.
    """
    patterns = [
        os.path.join(APPLE_DIR, "*hrv*.csv"),
        os.path.join(APPLE_DIR, "*variability*.csv"),
        os.path.join(APPLE_DIR, "heart_rate_variability*.csv"),
        os.path.join(APPLE_DIR, "*.csv"),
    ]
    files = []
    seen = set()
    for pat in patterns:
        for f in glob.glob(pat):
            if f not in seen:
                seen.add(f); files.append(f)
    return files

def read_apple_sdnn_resampled(path: str, bin_seconds: int) -> pd.DataFrame:
    """
    Return Apple HRV (SDNN, ms) indexed by UTC timestamps, resampled to BIN_SECONDS (median).
    Expects an HRV column in ms: usually 'value' for Apple HRV exports.
    """
    if not path:
        return pd.DataFrame(columns=["apple_sdnn_ms"])
    df = _safe_read_csv(path)

    # Identify HRV column (ms). Apple HRV exports typically store in 'value'.
    val_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("value", "sdnn", "sdnn_ms", "hrv", "hrv_ms"):
            val_col = c
            break
    if val_col is None:
        # Not an HRV file
        return pd.DataFrame(columns=["apple_sdnn_ms"])

    # Pick time column, prefer one that already has tz (endDate)
    tcol = _pick_time_col(df.columns, APPLE_PREFERRED_TIME_COLS)
    if not tcol:
        return pd.DataFrame(columns=["apple_sdnn_ms"])

    ts_utc = _tz_fix_to_utc(df[tcol])
    sdnn = pd.to_numeric(df[val_col], errors="coerce")

    s = pd.Series(sdnn.values, index=ts_utc).dropna().sort_index()
    # Median within each bin is typical for HRV
    out = s.resample(f"{bin_seconds}s").median().to_frame("apple_sdnn_ms")

    # Optional light smoothing to tidy spikiness
    if ROLL_WINDOW_BINS and ROLL_WINDOW_BINS > 1:
        out["apple_sdnn_ms_smooth"] = out["apple_sdnn_ms"].rolling(
            ROLL_WINDOW_BINS, center=True, min_periods=1
        ).median()
    else:
        out["apple_sdnn_ms_smooth"] = out["apple_sdnn_ms"]

    return out

def clip_limits(series: pd.Series, qlo=0.01, qhi=0.99):
    if series.dropna().empty:
        return (None, None)
    lo, hi = series.quantile([qlo, qhi])
    return float(lo), float(hi)

# ---- discover participants & pick best file per kid ----
apple_files = find_apple_hrv_files()
pid_to_file = {}
for f in apple_files:
    pid = parse_apple_pid(f)
    if pid is None:
        continue
    # keep the *largest* candidate per participant (usually the right export)
    if pid not in pid_to_file or os.path.getsize(f) > os.path.getsize(pid_to_file[pid]):
        pid_to_file[pid] = f

participants = sorted(pid_to_file.keys())
print("Participants with Apple HRV:", participants)

# ---- main loop ----
for pid in participants:
    fpath = pid_to_file[pid]
    df = read_apple_sdnn_resampled(fpath, BIN_SECONDS)
    if df.empty:
        print(f"[INFO] No SDNN detected for P{pid} in {os.path.basename(fpath)}")
        continue

    # Save aligned CSV (UTC)
    out_csv = os.path.join(OUT_DIR, f"P{pid:03d}_apple_sdnn_5min.csv")
    df[["apple_sdnn_ms"]].to_csv(out_csv, index_label="timestamp_utc")

    # Plot (local time axis)
    plot_df = df.copy()
    plot_df.index = plot_df.index.tz_convert(local_tz)

    # Robust y-limits
    lo, hi = clip_limits(plot_df["apple_sdnn_ms"])
    plt.figure(figsize=(11, 4.8))
    # draw raw and the smoothed line
    plt.plot(plot_df.index, plot_df["apple_sdnn_ms"], alpha=0.35, label="SDNN (ms) raw")
    plt.plot(plot_df.index, plot_df["apple_sdnn_ms_smooth"], linewidth=1.5, label="SDNN (ms) smoothed")
    plt.title(f"Participant {pid} — Apple HRV (SDNN, 5-min bins)")
    plt.xlabel(f"Local Time ({TIMEZONE})")
    plt.ylabel("SDNN (ms)")
    if lo is not None and hi is not None:
        plt.ylim(lo, hi)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"P{pid:03d}_apple_sdnn_5min.png")
    plt.savefig(out_png, dpi=160)
    plt.close()

    print(f"Saved: {os.path.abspath(out_csv)}")
    print(f"Saved: {os.path.abspath(out_png)}")

print("Done.")
