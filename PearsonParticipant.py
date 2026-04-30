

import os, re, glob
import numpy as np
import pandas as pd

# p-value support 
try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    

# ------------- CONFIG -------------
HEALTHAPP_ROOT = "./HealthApp"          
OURA_DIR       = "./OuraRing"
OUT_ROOT       = "./graphs/correlations_mapped"

BIN_SECONDS    = 300                    # 5 min bins 
HR_MIN, HR_MAX = 40, 180                # filter implausible values before resampling
LOCAL_TZ       = "US/Pacific"           # daily grouping is by local calendar day
MIN_PAIRS_DAY  = 12                    

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "aligned"), exist_ok=True)

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
        os.path.join(base, "Labeled", "Record", "**", "*_HeartRate.csv"),
        os.path.join(base, "Record", "Record", "**", "*_HeartRate.csv"), 
        os.path.join(base, "Record", "**", "*_HeartRate.csv"),
    ]
    out = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(list(dict.fromkeys(out)))

def _read_apple(pid: int, bin_seconds: int) -> pd.DataFrame:
    """
    Apple HealthApp HR → per-bin mean in UTC index, column 'apple_bpm'.
    Accepts files that look like:
      class,Time_In_PST,time,CreationDate,EndDate,StartDate,Type,Unit,Value,ID, ...
    """
    paths = _find_apple_hr_files(pid)
    if not paths:
        return pd.DataFrame(columns=["apple_bpm"])

    def _pick_time_col(df: pd.DataFrame) -> str | None:
        # Prefer columns that carry timezone offsets
        for c in ("CreationDate", "EndDate", "StartDate"):
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
    rows_part = []
    rows_day  = []

    pids = _participants()
    print("Participants:", pids)

    for pid in pids:
        ap = _read_apple(pid, BIN_SECONDS)
        ou = _read_oura(pid, BIN_SECONDS)

        ap = _ensure_dtindex_utc(ap)
        ou = _ensure_dtindex_utc(ou)

        if ap.empty or ou.empty:
            print(f"[INFO] Skipping P{pid}: missing series (Apple empty={ap.empty}, Oura empty={ou.empty})")
            continue

        mapped = ap.join(ou, how="inner").dropna()
        if mapped.empty:
            print(f"[INFO] Skipping P{pid}: no overlapping bins after mapping")
            continue

        # Save mapped series for audit
        mapped_path = os.path.join(OUT_ROOT, "aligned", f"P{pid:03d}_mapped_{BIN_SECONDS//60}min.csv")
        mapped.to_csv(mapped_path, index_label="timestamp_utc")

        #  Per participant correlation 
        r_all, p_all = _pearson_with_p(mapped["apple_bpm"], mapped["oura_bpm"])
        rows_part.append({
            "participant": pid,
            "n_pairs": int(len(mapped)),
            "bin_minutes": BIN_SECONDS // 60,
            "pearson_r": r_all,
            "p_value": p_all,
        })

        # ---- Per day correlation (local calendar day) ----
        mapped_local = mapped.copy()
        mapped_local["local_date"] = mapped_local.index.tz_convert(LOCAL_TZ).date
        for day, df in mapped_local.groupby("local_date"):
            if len(df) < MIN_PAIRS_DAY:
                continue
            r, p = _pearson_with_p(df["apple_bpm"], df["oura_bpm"])
            rows_day.append({
                "participant": pid,
                "date_local": str(day),
                "n_pairs": int(len(df)),
                "bin_minutes": BIN_SECONDS // 60,
                "pearson_r": r,
                "p_value": p,
            })

    part_df = pd.DataFrame(rows_part).sort_values("participant")
    day_df  = pd.DataFrame(rows_day).sort_values(["participant","date_local"])

    out_part = os.path.join(OUT_ROOT, "corr_per_participant.csv")
    out_day  = os.path.join(OUT_ROOT, "corr_per_day.csv")

    if not part_df.empty:
        part_df.to_csv(out_part, index=False)
        print("Saved:", os.path.abspath(out_part))
    else:
        print("[INFO] No participant-level correlations computed.")

    if not day_df.empty:
        day_df.to_csv(out_day, index=False)
        print("Saved:", os.path.abspath(out_day))
    else:
        print("[INFO] No per-day correlations computed (not enough overlapping bins).")

if __name__ == "__main__":
    main()