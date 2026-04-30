#!/usr/bin/env python3
"""
hr_by_class_and_time_heatmaps.py

Analyze how Apple Watch heart rate varies:
  1) By class × participant
  2) By time-of-day × participant

Heart-rate samples are mapped into class intervals (based on class-labeled
HealthApp exports) and into fixed 30‑minute time bins. Outputs are
annotated heatmaps similar to the existing data-coverage figures.
"""

from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------
HEALTHAPP_ROOT = "./HealthApp"
OURA_DIR       = "./OuraRing"

OUT_DIR = "./graphs/hr_by_class_and_time"
os.makedirs(OUT_DIR, exist_ok=True)

# HR filtering
HR_MIN, HR_MAX = 40, 180

# Time-of-day bins (30‑minute display windows)
TIME_BINS = [
    ("08:30", "09:00"),
    ("09:00", "09:30"),
    ("09:30", "10:00"),
    ("10:00", "10:30"),
    ("10:30", "11:00"),
    ("11:00", "11:30"),
    ("11:30", "12:00"),
    ("12:00", "12:30"),
    ("12:30", "13:00"),
    ("13:00", "13:30"),
    ("13:30", "14:00"),
    ("14:00", "14:30"),
    ("14:30", "15:00"),
]

LOCAL_TZ = "US/Pacific"
# ----------------------------------------


# ---------- helpers ----------
def canon_class_label(x: str) -> str:
    """
    Normalize class labels so variants are grouped together.
    Examples:
      - 'history', 'HISTORY' -> 'History'
      - 'ela', 'ELA/History' -> 'ELA/History'
    """
    s = str(x).strip()
    key = s.lower()
    mapping = {
        "history": "History",
        "ela": "ELA",
        "ela/history": "ELA",
        "ela & history": "ELA",
    }
    return mapping.get(key, s.title())


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.shape[1] == 1 and ";" in df.columns[0] and "," not in df.columns[0]:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            return None
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


def _find_healthapp_bases() -> List[Path]:
    root = Path(HEALTHAPP_ROOT)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("HealthAppP")],
                  key=lambda p: int(re.search(r"\d+", p.name).group()) if re.search(r"\d+", p.name) else 0)


def _pid_from_name(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def _pcode_from_name(name: str) -> str:
    pid = _pid_from_name(name)
    return f"P{pid:03d}" if pid >= 0 else name


# ---------- Apple HR (raw samples) ----------
def _find_apple_hr_files_for_dir(p_dir: Path) -> List[str]:
    pats = [
        p_dir / "Labeled" / "Record" / "**" / "HeartRate.csv",
        p_dir / "Labeled" / "Record" / "**" / "*HeartRate*.csv",
        p_dir / "Record" / "Record" / "**" / "HeartRate.csv",
        p_dir / "Record" / "Record" / "**" / "*HeartRate*.csv",
        p_dir / "Record" / "**" / "HeartRate.csv",
        p_dir / "Record" / "**" / "*HeartRate*.csv",
    ]
    out: List[str] = []
    for pat in pats:
        out.extend(glob.glob(str(pat), recursive=True))
    # deduplicate with order preserved
    return list(dict.fromkeys(out))


def _read_apple_raw_hr_for_dir(p_dir: Path) -> pd.DataFrame:
    paths = _find_apple_hr_files_for_dir(p_dir)
    if not paths:
        return pd.DataFrame(columns=["bpm"])

    def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
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
        df = _safe_read_csv(p)
        if df is None:
            continue
        if "Type" in df.columns:
            df = df[df["Type"].astype(str).str.contains("HeartRate", case=False, na=False)]
        if "Unit" in df.columns:
            df = df[df["Unit"].astype(str).str.lower().str.contains("count/min")]
        hr_col = "Value" if "Value" in df.columns else ("value" if "value" in df.columns else None)
        if not hr_col:
            continue
        tcol = _pick_time_col(df)
        if not tcol:
            continue

        ts = _tz_to_utc(df[tcol])
        bpm = pd.to_numeric(df[hr_col], errors="coerce")
        mask = (bpm >= HR_MIN) & (bpm <= HR_MAX)
        s = pd.Series(bpm.where(mask).values, index=ts).dropna()
        if not s.empty:
            series.append(s)

    if not series:
        return pd.DataFrame(columns=["bpm"])

    s_all = pd.concat(series).sort_index()
    out = s_all.to_frame("bpm")
    out = _ensure_dtindex_utc(out)
    return out


# ---------- Class intervals (Start/End + class) ----------
def _extract_class_intervals_for_dir(p_dir: Path) -> pd.DataFrame:
    """
    Scan Labeled/Record/*/*.csv for rows with 'class' (or similar) and
    Start/End timestamps, and return intervals in UTC.
    """
    rec_root = p_dir / "Labeled" / "Record"
    if not rec_root.exists():
        return pd.DataFrame(columns=["start_utc", "end_utc", "class_label"])

    all_rows = []
    for day_dir in sorted([d for d in rec_root.iterdir() if d.is_dir()]):
        for f in day_dir.glob("*.csv"):
            df = _safe_read_csv(str(f))
            if df is None or df.empty:
                continue
            cols = [c.strip() for c in df.columns]
            df.columns = cols

            # Flexible column detection
            class_col = None
            for cand in ("class", "Class", "label", "Label", "subject", "period", "activity"):
                if cand in df.columns:
                    class_col = cand
                    break
            start_col = None
            for cand in ("StartDate", "startDate", "Start", "start_time", "start"):
                if cand in df.columns:
                    start_col = cand
                    break
            end_col = None
            for cand in ("EndDate", "endDate", "End", "end_time", "end"):
                if cand in df.columns:
                    end_col = cand
                    break
            if not (class_col and start_col):
                continue

            sub = df[[class_col, start_col] + ([end_col] if end_col else [])].copy()
            sub.rename(columns={class_col: "class_label"}, inplace=True)
            sub["class_label"] = sub["class_label"].astype(str).apply(canon_class_label)

            sub["start_utc"] = _tz_to_utc(sub[start_col])
            if end_col:
                sub["end_utc"] = _tz_to_utc(sub[end_col])
            else:
                sub["end_utc"] = sub["start_utc"] + pd.Timedelta(minutes=5)

            sub = sub.dropna(subset=["class_label", "start_utc", "end_utc"])
            sub = sub[sub["end_utc"] > sub["start_utc"]]
            if sub.empty:
                continue

            all_rows.append(sub[["class_label", "start_utc", "end_utc"]])

    if not all_rows:
        return pd.DataFrame(columns=["start_utc", "end_utc", "class_label"])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values("start_utc")
    return out


# ---------- Oura HR (raw samples) ----------
def _read_oura_raw_hr_for_pid(pid: int) -> pd.DataFrame:
    """
    Return Oura raw HR samples with UTC datetime index and column 'bpm'.
    """
    folder = None
    for cand in (Path(OURA_DIR) / f"P{pid}OuraRing", Path(OURA_DIR) / f"P{pid:03d}OuraRing"):
        if cand.is_dir():
            folder = cand
            break
    if folder is None:
        return pd.DataFrame(columns=["bpm"])

    series = []
    for p in sorted(glob.glob(str(folder / "HeartRate" / "*.csv"))):
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
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

    if not series:
        return pd.DataFrame(columns=["bpm"])

    s_all = pd.concat(series).sort_index()
    out = s_all.to_frame("bpm")
    out = _ensure_dtindex_utc(out)
    return out


# ---------- Mapping HR samples to classes ----------
def _assign_class_to_hr(hr: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """
    For each class interval, take HR samples inside [start_utc, end_utc)
    and label them with that class.
    """
    if hr.empty or intervals.empty:
        return pd.DataFrame(columns=["timestamp_utc", "bpm", "class_label"])

    rows = []
    for _, row in intervals.iterrows():
        start = row["start_utc"]
        end = row["end_utc"]
        cls = row["class_label"]
        # Slice HR by index
        chunk = hr.loc[(hr.index >= start) & (hr.index < end), ["bpm"]].copy()
        if chunk.empty:
            continue
        chunk["timestamp_utc"] = chunk.index
        chunk["class_label"] = cls
        rows.append(chunk[["timestamp_utc", "bpm", "class_label"]])

    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "bpm", "class_label"])

    return pd.concat(rows, ignore_index=True)


# ---------- Aggregation helpers ----------
def _local_time_of_day_index(ts_utc: pd.Series) -> pd.Series:
    """Convert a UTC Series of timestamps to local tz-aware datetimes."""
    return ts_utc.dt.tz_convert(LOCAL_TZ)


def _time_bin_label(dt_local: pd.Series) -> pd.Series:
    """Label each timestamp with a human-readable time-bin like '08:30-09:00'."""
    # Extract time as HH:MM
    times = dt_local.dt.time
    labels: List[str] = []
    for t in times:
        h = t.hour
        m = t.minute
        label = None
        for start_str, end_str in TIME_BINS:
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            cur_min = h * 60 + m
            if start_min <= cur_min < end_min:
                label = f"{start_str}-{end_str}"
                break
        labels.append(label if label is not None else "")
    return pd.Series(labels, index=dt_local)


def _build_heatmap(df: pd.DataFrame, row_key: str, col_key: str, value_col: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Pivot to a matrix with rows=row_key, cols=col_key, values=value_col (mean).
    Returns (pivot_df, row_labels, col_labels) with sorted / stable ordering.
    """
    if df.empty:
        return pd.DataFrame(), [], []
    pivot = df.pivot_table(index=row_key, columns=col_key, values=value_col, aggfunc="mean")
    # Sort rows and columns sensibly
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    return pivot, row_labels, col_labels


def _plot_heatmap(matrix: pd.DataFrame, row_labels: List[str], col_labels: List[str],
                  title: str, ylabel: str, cbar_label: str, out_path: str,
                  vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    if matrix.empty:
        print(f"[WARN] No data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.7), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Participant")

    # Annotate each cell with value (one decimal) where finite
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved heatmap: {out_path}")


# ---------- main pipeline ----------
def main() -> None:
    bases = _find_healthapp_bases()
    if not bases:
        print(f"[ERROR] No HealthAppP* directories found under {HEALTHAPP_ROOT}")
        return

    all_class_hr_rows_apple = []
    all_time_hr_rows_apple = []
    all_class_hr_rows_oura = []
    all_time_hr_rows_oura = []

    for p_dir in bases:
        pid = _pid_from_name(p_dir.name)
        pcode = _pcode_from_name(p_dir.name)
        print(f"[INFO] Processing {p_dir.name} ({pcode}) …")

        ap_hr = _read_apple_raw_hr_for_dir(p_dir)
        if ap_hr.empty:
            print(f"   -> No Apple HR data, skipping participant for Apple plots.")
        pid = _pid_from_name(p_dir.name)
        ou_hr = _read_oura_raw_hr_for_pid(pid)
        if ou_hr.empty:
            print(f"   -> No Oura HR data, skipping participant for Oura plots.")

        intervals = _extract_class_intervals_for_dir(p_dir)
        if intervals.empty:
            print(f"   -> No class intervals found, skipping class-mapped plots for this participant.")
        else:
            if not ap_hr.empty:
                ap_with_class = _assign_class_to_hr(ap_hr, intervals)
                if not ap_with_class.empty:
                    ap_with_class["participant"] = pcode
                    all_class_hr_rows_apple.append(ap_with_class)
                    print(f"   -> {len(ap_with_class)} Apple HR samples mapped to classes.")
            if not ou_hr.empty:
                ou_with_class = _assign_class_to_hr(ou_hr, intervals)
                if not ou_with_class.empty:
                    ou_with_class["participant"] = pcode
                    all_class_hr_rows_oura.append(ou_with_class)
                    print(f"   -> {len(ou_with_class)} Oura HR samples mapped to classes.")

        # Time-of-day aggregation (independent of class)
        if not ap_hr.empty:
            ts_local_ap = _local_time_of_day_index(ap_hr.index.to_series())
            bin_labels_ap = _time_bin_label(ts_local_ap)
            ap_with_bins = ap_hr.copy()
            ap_with_bins["time_bin"] = bin_labels_ap.values
            ap_with_bins = ap_with_bins[ap_with_bins["time_bin"] != ""]
            if not ap_with_bins.empty:
                df_time_ap = ap_with_bins.reset_index(drop=False)[["bpm", "time_bin"]]
                df_time_ap["participant"] = pcode
                all_time_hr_rows_apple.append(df_time_ap)
                print(f"   -> {len(df_time_ap)} Apple HR samples assigned to time-of-day bins.")
            else:
                print("   -> No Apple HR samples within configured time-of-day bins.")

        if not ou_hr.empty:
            ts_local_ou = _local_time_of_day_index(ou_hr.index.to_series())
            bin_labels_ou = _time_bin_label(ts_local_ou)
            ou_with_bins = ou_hr.copy()
            ou_with_bins["time_bin"] = bin_labels_ou.values
            ou_with_bins = ou_with_bins[ou_with_bins["time_bin"] != ""]
            if not ou_with_bins.empty:
                df_time_ou = ou_with_bins.reset_index(drop=False)[["bpm", "time_bin"]]
                df_time_ou["participant"] = pcode
                all_time_hr_rows_oura.append(df_time_ou)
                print(f"   -> {len(df_time_ou)} Oura HR samples assigned to time-of-day bins.")
            else:
                print("   -> No Oura HR samples within configured time-of-day bins.")

    # ----- Class × participant heatmap (mean HR) -----
    if all_class_hr_rows_apple:
        class_df = pd.concat(all_class_hr_rows_apple, ignore_index=True)
        class_df["class_label"] = class_df["class_label"].astype(str)
        class_agg = (
            class_df.groupby(["class_label", "participant"], as_index=False)
                    .agg(mean_hr_bpm=("bpm", "mean"))
        )
        class_pivot, class_rows, class_cols = _build_heatmap(
            class_agg, row_key="class_label", col_key="participant", value_col="mean_hr_bpm"
        )
        out_png = os.path.join(OUT_DIR, "hr_by_class_heatmap_apple.png")
        _plot_heatmap(
            class_pivot, class_rows, class_cols,
            title="Mean Heart Rate by Class (Apple, 5‑min resolution)",
            ylabel="Class",
            cbar_label="Mean HR (bpm)",
            out_path=out_png,
        )
        class_agg.to_csv(os.path.join(OUT_DIR, "hr_by_class_table_apple.csv"), index=False)
        print(f"[INFO] Saved Apple class-level table.")
    else:
        print("[WARN] No Apple class‑mapped HR samples found; skipping Apple class heatmap.")

    if all_class_hr_rows_oura:
        class_df_o = pd.concat(all_class_hr_rows_oura, ignore_index=True)
        class_df_o["class_label"] = class_df_o["class_label"].astype(str)
        class_agg_o = (
            class_df_o.groupby(["class_label", "participant"], as_index=False)
                      .agg(mean_hr_bpm=("bpm", "mean"))
        )
        class_pivot_o, class_rows_o, class_cols_o = _build_heatmap(
            class_agg_o, row_key="class_label", col_key="participant", value_col="mean_hr_bpm"
        )
        out_png_o = os.path.join(OUT_DIR, "hr_by_class_heatmap_oura.png")
        _plot_heatmap(
            class_pivot_o, class_rows_o, class_cols_o,
            title="Mean Heart Rate by Class (Oura, 5‑min resolution)",
            ylabel="Class",
            cbar_label="Mean HR (bpm)",
            out_path=out_png_o,
        )
        class_agg_o.to_csv(os.path.join(OUT_DIR, "hr_by_class_table_oura.csv"), index=False)
        print(f"[INFO] Saved Oura class-level table.")
    else:
        print("[WARN] No Oura class‑mapped HR samples found; skipping Oura class heatmap.")

    # ----- Time-of-day × participant heatmap (mean HR) -----
    if all_time_hr_rows_apple:
        time_df = pd.concat(all_time_hr_rows_apple, ignore_index=True)
        time_agg = (
            time_df.groupby(["time_bin", "participant"], as_index=False)
                   .agg(mean_hr_bpm=("bpm", "mean"))
        )
        # Preserve the chronological order of time bins
        time_bin_order = [f"{s}-{e}" for (s, e) in TIME_BINS]
        time_agg["time_bin"] = pd.Categorical(time_agg["time_bin"], categories=time_bin_order, ordered=True)
        time_agg = time_agg.sort_values("time_bin")

        time_pivot, time_rows, time_cols = _build_heatmap(
            time_agg, row_key="time_bin", col_key="participant", value_col="mean_hr_bpm"
        )
        out_png2 = os.path.join(OUT_DIR, "hr_by_time_of_day_heatmap_apple.png")
        _plot_heatmap(
            time_pivot, time_rows, time_cols,
            title="Mean Heart Rate by Time of Day (Apple, 5‑min resolution)",
            ylabel="Time Interval (30‑min display)",
            cbar_label="Mean HR (bpm)",
            out_path=out_png2,
        )
        time_agg.to_csv(os.path.join(OUT_DIR, "hr_by_time_of_day_table_apple.csv"), index=False)
        print(f"[INFO] Saved Apple time‑of‑day table.")
    else:
        print("[WARN] No Apple HR samples within configured time-of-day bins; skipping Apple time heatmap.")

    if all_time_hr_rows_oura:
        time_df_o = pd.concat(all_time_hr_rows_oura, ignore_index=True)
        time_agg_o = (
            time_df_o.groupby(["time_bin", "participant"], as_index=False)
                     .agg(mean_hr_bpm=("bpm", "mean"))
        )
        time_bin_order = [f"{s}-{e}" for (s, e) in TIME_BINS]
        time_agg_o["time_bin"] = pd.Categorical(time_agg_o["time_bin"], categories=time_bin_order, ordered=True)
        time_agg_o = time_agg_o.sort_values("time_bin")

        time_pivot_o, time_rows_o, time_cols_o = _build_heatmap(
            time_agg_o, row_key="time_bin", col_key="participant", value_col="mean_hr_bpm"
        )
        out_png2_o = os.path.join(OUT_DIR, "hr_by_time_of_day_heatmap_oura.png")
        _plot_heatmap(
            time_pivot_o, time_rows_o, time_cols_o,
            title="Mean Heart Rate by Time of Day (Oura, 5‑min resolution)",
            ylabel="Time Interval (30‑min display)",
            cbar_label="Mean HR (bpm)",
            out_path=out_png2_o,
        )
        time_agg_o.to_csv(os.path.join(OUT_DIR, "hr_by_time_of_day_table_oura.csv"), index=False)
        print(f"[INFO] Saved Oura time‑of‑day table.")
    else:
        print("[WARN] No Oura HR samples within configured time-of-day bins; skipping Oura time heatmap.")

    print(f"[DONE] Outputs in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

