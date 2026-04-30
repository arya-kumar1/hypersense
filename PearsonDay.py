# correlate_all_participants_by_day.py
# Aggregate across ALL participants per local day:
#   1) Read Apple (HealthApp) HR and Oura HR
#   2) Filter to valid HR + resample both to a common grid (5 min)
#   3) Map per-participant by inner-joining bins (Apple/Oura pairs)
#   4) Concatenate ALL participants' pairs per day
#   5) Compute Pearson r (and p-value) per day
#   6) Plot one scatter (Apple vs Oura) + best-fit line per day

import os, re, glob
import numpy as np
import pandas as pd

# Optional p-values via SciPy
try:
    from scipy.stats import pearsonr
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

OUT_ROOT   = "./graphs/correlations_by_day_all"
PLOTS_DIR  = os.path.join(OUT_ROOT, "plots")
SUMMARY_CSV = os.path.join(OUT_ROOT, "daily_summary.csv")

BIN_SECONDS    = 300                    # 5-min bins
HR_MIN, HR_MAX = 40, 180                # plausible HR range
LOCAL_TZ       = "US/Pacific"           # define "day" by local calendar day
MIN_PAIRS_DAY  = 30                     # minimum paired points to compute a daily r

# Plot controls
SCATTER_SIZE   = 10
SCATTER_ALPHA  = 0.55
MAX_POINTS_PER_PLOT = 20000             # downsample for plotting if needed
# ----------------------------------

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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
        os.path.join(base, "Labeled", "Record", "**", "*_HeartRate.csv"),
        os.path.join(base, "Record", "Record", "**", "*_HeartRate.csv"),  # safety
        os.path.join(base, "Record", "**", "*_HeartRate.csv"),
    ]
    out = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(list(dict.fromkeys(out)))

def _read_apple(pid: int, bin_seconds: int) -> pd.DataFrame:
    """
    Apple HealthApp HR → per-bin mean in UTC index, column 'apple_bpm'.
    Accepts files with columns like:
      class,Time_In_PST,time,CreationDate,EndDate,StartDate,Type,Unit,Value,ID,...
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
    Expects daily files with 'Time_In_ISO' and 'bpm'.
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
        r, p = pearsonr(x, y)
        return (float(r), float(p))
    r = float(np.corrcoef(x, y)[0, 1])
    return (r, np.nan)

# ---------- main ----------
def main():
    pids = _participants()
    print("Participants:", pids)

    # Collect all Apple/Oura paired points per participant (inner-joined bins)
    all_pairs = []  # list of DataFrames with columns: apple_bpm, oura_bpm, participant
    for pid in pids:
        ap = _read_apple(pid, BIN_SECONDS)
        ou = _read_oura(pid, BIN_SECONDS)

        ap = _ensure_dtindex_utc(ap)
        ou = _ensure_dtindex_utc(ou)

        if ap.empty or ou.empty:
            continue

        # Map onto each other: only bins where both exist
        mapped = ap.join(ou, how="inner").dropna()
        if mapped.empty:
            continue

        mapped["participant"] = pid
        all_pairs.append(mapped)

    if not all_pairs:
        print("[INFO] No overlapping Apple/Oura bins across participants.")
        return

    pairs = pd.concat(all_pairs).sort_index()
    # Add local date for grouping across ALL participants
    pairs = pairs.copy()
    pairs["local_date"] = pairs.index.tz_convert(LOCAL_TZ).date

    # Compute per-day stats across ALL participants combined
    rows = []
    for day, df in pairs.groupby("local_date"):
        if len(df) < MIN_PAIRS_DAY:
            continue

        x = df["apple_bpm"].astype(float)
        y = df["oura_bpm"].astype(float)

        # Pearson r, p
        r, p = _pearson_with_p(x, y)

        # Best-fit line: y = slope * x + intercept
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            slope, intercept = (np.nan, np.nan)

        rows.append({
            "date_local": str(day),
            "n_pairs": int(len(df)),
            "bin_minutes": BIN_SECONDS // 60,
            "pearson_r": float(r) if r == r else np.nan,
            "p_value": float(p) if p == p else np.nan,
            "slope_oura_vs_apple": float(slope) if slope == slope else np.nan,
            "intercept_oura_vs_apple": float(intercept) if intercept == intercept else np.nan,
        })

        # -------- Plot scatter (Apple on x, Oura on y) --------
        # Downsample for plotting if massive:
        plot_df = df
        if len(plot_df) > MAX_POINTS_PER_PLOT:
            plot_df = plot_df.sample(n=MAX_POINTS_PER_PLOT, random_state=42)

        xx = plot_df["apple_bpm"].astype(float).values
        yy = plot_df["oura_bpm"].astype(float).values

        # Line over x-range (if slope is finite)
        x_line = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
        if not np.isnan(slope) and not np.isnan(intercept):
            y_line = slope * x_line + intercept
        else:
            y_line = None

        plt.figure(figsize=(6.6, 5.6))
        plt.scatter(xx, yy, s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
        if y_line is not None:
            plt.plot(x_line, y_line, linewidth=2)
        title = f"{day} — r={r:.2f}" + (f", p={p:.3g}" if not np.isnan(p) else "")
        title += f" (n={len(df)})"
        plt.title(title)
        plt.xlabel("Apple HR (count/min)")
        plt.ylabel("Oura HR (bpm)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        out_png = os.path.join(PLOTS_DIR, f"{day}.png")
        plt.savefig(out_png, dpi=160)
        plt.close()

    if not rows:
        print("[INFO] No days met the minimum paired-points requirement.")
        return

    out_df = pd.DataFrame(rows).sort_values("date_local")
    out_df.to_csv(SUMMARY_CSV, index=False)
    print("Saved summary:", os.path.abspath(SUMMARY_CSV))
    print("Saved plots to:", os.path.abspath(PLOTS_DIR))

if __name__ == "__main__":
    main()