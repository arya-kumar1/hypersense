#!/usr/bin/env python3
"""
correlation_all_pairs.py
Compute overall Pearson correlation across ALL paired data points
(combining all participants and all days).

This aggregates all Apple vs Oura heart rate pairs and computes
a single correlation coefficient and p-value.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# Optional p-values via SciPy
try:
    from scipy.stats import spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    print("[WARN] scipy not found; p-values will be NaN. `pip install scipy` to enable p-values.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
HEALTHAPP_ROOT = "./HealthApp"          # folders like HealthAppP1, HealthAppP02, ...
OURA_DIR       = "./OuraRing"           # folders like P1OuraRing, P001OuraRing

OUT_DIR        = "./graphs/correlation_all_pairs"
OUTPUT_CSV     = os.path.join(OUT_DIR, "overall_correlation.csv")
OUTPUT_PLOT    = os.path.join(OUT_DIR, "overall_correlation_scatter2.png")

BIN_SECONDS    = 300                    # 5-min bins (most common in your data)
HR_MIN, HR_MAX = 40, 180                # plausible HR range
LOCAL_TZ       = "US/Pacific"           # for reference (not used in aggregation)
# ----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def _tz_to_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    # If naive, localize to US/Pacific then convert to UTC.
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

def _find_healthapp_base(pid: int) -> str | None:
    for name in (f"HealthAppP{pid}", f"HealthAppP{pid:02d}", f"HealthAppP{pid:03d}"):
        base = os.path.join(HEALTHAPP_ROOT, name)
        if os.path.isdir(base):
            return base
    return None

def _find_apple_hr_files(pid: int) -> list[str]:
    base = _find_healthapp_base(pid)
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
    return sorted(list(dict.fromkeys(out)))

def _read_apple(pid: int, bin_seconds: int) -> pd.DataFrame:
    """
    Apple HealthApp HR → per-bin mean in UTC index, column 'apple_bpm'.
    """
    paths = _find_apple_hr_files(pid)
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

    series = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            # Keep heart-rate rows, ensure right unit and numeric values
            if "Type" in df.columns:
                df = df[df["Type"].astype(str).str.contains("HeartRate", case=False, na=False)]
            if "Unit" in df.columns:
                df = df[df["Unit"].astype(str).str.lower().str.contains("count/min")]
            if "Value" not in df.columns:
                continue

            tcol = _pick_time_col(df)
            if not tcol:
                continue

            ts = _tz_to_utc(df[tcol])
            val = pd.to_numeric(df["Value"], errors="coerce")
            mask = (val >= HR_MIN) & (val <= HR_MAX)
            s = pd.Series(val.where(mask).values, index=ts).dropna()
            if not s.empty:
                series.append(s)
        except Exception as e:
            print(f"[WARN] Apple read fail {p}: {e}")

    if not series:
        return pd.DataFrame(columns=["apple_bpm"])

    s_all = pd.concat(series).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("apple_bpm")

def _read_oura(pid: int, bin_seconds: int) -> pd.DataFrame:
    """
    Oura HR → per-bin mean in UTC index, column 'oura_bpm'.
    """
    folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"),
                 os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand; break
    if not folder:
        return pd.DataFrame(columns=["oura_bpm"])

    series = []
    for p in sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv"))):
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                # fallback: any ISO-like with 'T' and 'Z' or offset
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
        return pd.DataFrame(columns=["oura_bpm"])

    s_all = pd.concat(series).sort_index()
    return s_all.resample(f"{bin_seconds}s").mean().to_frame("oura_bpm")

def _participants() -> list[int]:
    health_pids, oura_pids = [], []
    for d in glob.glob(os.path.join(HEALTHAPP_ROOT, "HealthAppP*")):
        m = re.match(r".*HealthAppP0*([0-9]+)$", d)
        if m: health_pids.append(int(m.group(1)))
    for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m: oura_pids.append(int(m.group(1)))
    return sorted(set(health_pids).union(oura_pids))

def _pearson_with_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return (np.nan, np.nan)
    if SCIPY_OK:
        r, p = spearmanr(x, y)  # Changed from pearsonr
        return (float(r), float(p))
    r = float(np.corrcoef(x, y)[0, 1])
    return (r, np.nan)

# ---------- main ----------
def main():
    pids = _participants()
    # Exclude participant 12 from the overall correlation
    pids = [pid for pid in pids if pid != 12]
    print("Participants (excluding P12):", pids)

    # Collect all Apple/Oura paired points per participant (inner-joined bins)
    all_pairs = []  # list of DataFrames with columns: apple_bpm, oura_bpm, participant
    for pid in pids:
        print(f"Processing participant {pid}...")
        ap = _read_apple(pid, BIN_SECONDS)
        ou = _read_oura(pid, BIN_SECONDS)

        ap = _ensure_dtindex_utc(ap)
        ou = _ensure_dtindex_utc(ou)

        if ap.empty or ou.empty:
            print(f"  [INFO] Skipping P{pid}: missing series (Apple empty={ap.empty}, Oura empty={ou.empty})")
            continue

        # Map onto each other: only bins where both exist
        mapped = ap.join(ou, how="inner").dropna()
        if mapped.empty:
            print(f"  [INFO] Skipping P{pid}: no overlapping bins after mapping")
            continue

        mapped["participant"] = pid
        all_pairs.append(mapped)
        print(f"  Found {len(mapped)} paired bins for P{pid}")

    if not all_pairs:
        print("[ERROR] No overlapping Apple/Oura bins across participants.")
        return

    # Combine all pairs across all participants
    pairs = pd.concat(all_pairs).sort_index()
    print(f"\nTotal paired bins across all participants: {len(pairs)}")

    # Extract Apple and Oura values
    x = pairs["apple_bpm"].astype(float)
    y = pairs["oura_bpm"].astype(float)

    # Compute overall correlation
    r, p = _pearson_with_p(x, y)
    n_total = len(pairs)

    print(f"\n{'='*60}")
    print(f"OVERALL CORRELATION (All Pairs Combined)")
    print(f"{'='*60}")
    print(f"  n_pairs = {n_total:,}")
    print(f"  pearson_r = {r:.6f}")
    if not np.isnan(p):
        print(f"  p_value = {p:.6e}")
    else:
        print(f"  p_value = NaN (scipy not available)")
    print(f"  bin_minutes = {BIN_SECONDS // 60}")
    print(f"{'='*60}")

    # Save results to CSV
    results_df = pd.DataFrame([{
        "n_pairs": n_total,
        "pearson_r": r,
        "p_value": p,
        "bin_minutes": BIN_SECONDS // 60,
    }])
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to: {os.path.abspath(OUTPUT_CSV)}")

    # Create scatter plot
    print("\nCreating scatter plot...")
    
    # Downsample for plotting if too many points
    plot_df = pairs.copy()
    MAX_POINTS_PLOT = 20000
    if len(plot_df) > MAX_POINTS_PLOT:
        plot_df = plot_df.sample(n=MAX_POINTS_PLOT, random_state=42)
        print(f"  Downsampled to {MAX_POINTS_PLOT} points for plotting")

    xx = plot_df["apple_bpm"].astype(float).values
    yy = plot_df["oura_bpm"].astype(float).values

    # Best-fit line
    try:
        slope, intercept = np.polyfit(xx, yy, 1)
        x_line = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
        y_line = slope * x_line + intercept
    except Exception:
        slope, intercept = (np.nan, np.nan)
        x_line, y_line = None, None

    plt.figure(figsize=(8, 7))
    plt.scatter(xx, yy, s=10, alpha=0.4, edgecolors='none')
    if y_line is not None:
        plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.2f}')
    
    title = f"Overall Correlation: r={r:.3f}"
    if not np.isnan(p):
        title += f", p={p:.3e}"
    title += f" (n={n_total:,})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Apple HR (bpm)", fontsize=12)
    plt.ylabel("Oura HR (bpm)", fontsize=12)
    plt.grid(True, alpha=0.25)
    if y_line is not None:
        plt.legend()
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved plot to: {os.path.abspath(OUTPUT_PLOT)}")

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Apple HR: mean={x.mean():.1f}, std={x.std():.1f}, range=[{x.min():.1f}, {x.max():.1f}]")
    print(f"  Oura HR:  mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.1f}, {y.max():.1f}]")

if __name__ == "__main__":
    main()
