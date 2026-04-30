#!/usr/bin/env python3
"""
bland_altman_plot.py
Create Bland-Altman plots to assess agreement between Apple Watch and Oura Ring
heart rate measurements.

Plots:
  - X-axis: mean of the two measurements, (Apple + Oura) / 2
  - Y-axis: difference, Apple - Oura
  - Bias: mean difference
  - Limits of agreement: mean ± 1.96 × SD of differences (95% expected range)
"""

import os
import re
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
HEALTHAPP_ROOT = "./HealthApp"
OURA_DIR       = "./OuraRing"

OUT_DIR        = "./graphs/bland_altman"
OUTPUT_CSV     = os.path.join(OUT_DIR, "bland_altman_stats.csv")
OUTPUT_PLOT    = os.path.join(OUT_DIR, "bland_altman_plot.png")

BIN_SECONDS    = 300
HR_MIN, HR_MAX = 40, 180
LOCAL_TZ       = "US/Pacific"

MAX_POINTS_PLOT = 20000  # downsample for plotting if needed
SCATTER_SIZE    = 10
SCATTER_ALPHA   = 0.85
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
    folder = None
    for cand in (os.path.join(OURA_DIR, f"P{pid}OuraRing"), os.path.join(OURA_DIR, f"P{pid:03d}OuraRing")):
        if os.path.isdir(cand):
            folder = cand
            break
    if not folder:
        return pd.DataFrame(columns=["oura_bpm"])

    series = []
    for p in sorted(glob.glob(os.path.join(folder, "HeartRate", "*.csv"))):
        try:
            df = _safe_read_csv(p)
            tcol = "Time_In_ISO" if "Time_In_ISO" in df.columns else None
            if not tcol:
                for c in df.columns:
                    if isinstance(c, str) and ("T" in c and ("Z" in c or "+" in c)):
                        tcol = c
                        break
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
        if m:
            health_pids.append(int(m.group(1)))
    for d in glob.glob(os.path.join(OURA_DIR, "P*OuraRing")):
        m = re.match(r".*P0*([0-9]+)OuraRing$", d)
        if m:
            oura_pids.append(int(m.group(1)))
    return sorted(set(health_pids).union(oura_pids))

# ---------- main ----------
def main():
    pids = _participants()
    print("Participants:", pids)

    all_pairs = []
    for pid in pids:
        ap = _read_apple(pid, BIN_SECONDS)
        ou = _read_oura(pid, BIN_SECONDS)
        ap = _ensure_dtindex_utc(ap)
        ou = _ensure_dtindex_utc(ou)
        if ap.empty or ou.empty:
            continue
        mapped = ap.join(ou, how="inner").dropna()
        if mapped.empty:
            continue
        mapped["participant"] = pid
        all_pairs.append(mapped)

    if not all_pairs:
        print("[ERROR] No overlapping Apple/Oura bins across participants.")
        return

    pairs = pd.concat(all_pairs).sort_index()

    # Bland-Altman: mean and difference
    apple = pairs["apple_bpm"].astype(float)
    oura = pairs["oura_bpm"].astype(float)
    mean_ba = (apple + oura) / 2
    diff = apple - oura

    # Statistics
    bias = diff.mean()
    std_diff = diff.std()
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff

    n = len(diff)
    print(f"\nBland-Altman Statistics (n = {n:,})")
    print(f"  Bias (mean difference): {bias:.3f} bpm")
    print(f"  SD of differences:      {std_diff:.3f} bpm")
    print(f"  Upper limit of agreement (+1.96 SD): {loa_upper:.3f} bpm")
    print(f"  Lower limit of agreement (-1.96 SD): {loa_lower:.3f} bpm")

    # Save stats
    stats_df = pd.DataFrame([{
        "n_pairs": n,
        "bias_bpm": bias,
        "sd_diff_bpm": std_diff,
        "loa_upper_bpm": loa_upper,
        "loa_lower_bpm": loa_lower,
        "bin_minutes": BIN_SECONDS // 60,
    }])
    stats_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved stats to: {os.path.abspath(OUTPUT_CSV)}")

    # Plot
    plot_mean = mean_ba.values
    plot_diff = diff.values
    if len(plot_mean) > MAX_POINTS_PLOT:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(plot_mean), size=MAX_POINTS_PLOT, replace=False)
        plot_mean = plot_mean[idx]
        plot_diff = plot_diff[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_mean, plot_diff, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, c="steelblue", edgecolors="none")

    # Reference lines
    x_min, x_max = plot_mean.min(), plot_mean.max()

    ax.axhline(bias, color="black", linewidth=2, linestyle="-", label=f"Bias: {bias:.2f} bpm")
    ax.axhline(loa_upper, color="gray", linewidth=1.5, linestyle="--", label=f"+1.96 SD: {loa_upper:.2f} bpm")
    ax.axhline(loa_lower, color="gray", linewidth=1.5, linestyle="--", label=f"-1.96 SD: {loa_lower:.2f} bpm")

    ax.set_xlabel("Mean HR (Apple + Oura) / 2 [bpm]", fontsize=12)
    ax.set_ylabel("Difference (Apple − Oura) [bpm]", fontsize=12)
    ax.set_title(f"Bland-Altman Plot: Apple Watch vs Oura Ring HR (n={n:,})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.75)
    ax.set_xlim(x_min, x_max)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved plot to: {os.path.abspath(OUTPUT_PLOT)}")


if __name__ == "__main__":
    main()
