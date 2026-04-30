#!/usr/bin/env python3
"""
Correlate Oura mean HR (per class per participant) with Inattention and
Hyperactivity scores from OrHrAvgs.csv.

Pooled analysis merges every OrHrAvgs row (primary + expert raters) onto HR by
child id (P001, …) so each class×child HR is paired with each rater’s scores.

Participant-level and per-class analyses use the primary rater row only.

Correlations are Spearman rank only. Outputs: CSV + plots under --out-dir.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:
    print("This script requires scipy (pip install scipy).", file=sys.stderr)
    sys.exit(1)

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HR_PATH = os.path.join(
    ROOT, "graphs", "hr_by_class_and_time", "hr_by_class_table_oura.csv"
)
DEFAULT_OR_PATH = os.path.join(ROOT, "OrHrAvgs.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT, "graphs", "hr_by_class_and_time")

PRIMARY_PARTICIPANT = re.compile(r"^P\d+$", re.I)
BASE_PARTICIPANT = re.compile(r"^(P\d+)", re.I)

SUMMARY_CSV = "oura_hr_adhd_correlation_summary.csv"
PER_CLASS_CSV = "oura_hr_adhd_correlation_per_class.csv"
PLOT_POOLED = "oura_hr_adhd_scatter_pooled.png"
PLOT_PARTICIPANT = "oura_hr_adhd_scatter_participant.png"
PLOT_PER_CLASS = "oura_hr_adhd_spearman_by_class.png"


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_oura_hr(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _strip_cols(df)
    need = {"class_label", "participant", "mean_hr_bpm"}
    missing = need - set(c.lower() for c in df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    colmap = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(
        {
            "class_label": df[colmap["class_label"]],
            "participant": df[colmap["participant"]].astype(str).str.strip(),
            "mean_hr_bpm": pd.to_numeric(df[colmap["mean_hr_bpm"]], errors="coerce"),
        }
    )
    return out.dropna(subset=["mean_hr_bpm"])


def _base_participant_id(participant_cell: str) -> str | None:
    s = str(participant_cell).strip()
    m = BASE_PARTICIPANT.match(s)
    return m.group(1) if m else None


def load_orhr_primary(path: str) -> pd.DataFrame:
    """One row per child: primary rater only (P001, not P001 ExpertP)."""
    df = pd.read_csv(path)
    df = _strip_cols(df)
    if "Participant" not in df.columns:
        raise ValueError(f"{path}: expected column 'Participant'")
    pcol = "Participant"
    df = df[df[pcol].astype(str).str.strip().apply(lambda x: bool(PRIMARY_PARTICIPANT.match(x)))].copy()
    df["participant"] = df[pcol].astype(str).str.strip()

    for name in ("Inattention score", "Hyperactivity score"):
        if name not in df.columns:
            raise ValueError(f"{path}: missing column {name!r}")
        df[name] = pd.to_numeric(df[name], errors="coerce")

    keep = ["participant", "Inattention score", "Hyperactivity score"]
    return df[keep].drop_duplicates(subset=["participant"])


def load_orhr_all_raters(path: str) -> pd.DataFrame:
    """Primary + expert rows; `participant` is child id (P001); `participant_label` is full CSV label."""
    df = pd.read_csv(path)
    df = _strip_cols(df)
    if "Participant" not in df.columns:
        raise ValueError(f"{path}: expected column 'Participant'")
    pcol = "Participant"
    df["participant"] = df[pcol].map(_base_participant_id)
    df = df[df["participant"].notna()].copy()
    df["participant_label"] = df[pcol].astype(str).str.strip()

    for name in ("Inattention score", "Hyperactivity score"):
        if name not in df.columns:
            raise ValueError(f"{path}: missing column {name!r}")
        df[name] = pd.to_numeric(df[name], errors="coerce")

    keep = ["participant", "participant_label", "Inattention score", "Hyperactivity score"]
    return df[keep]


def corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "spearman_rho": np.nan, "spearman_p": np.nan}
    xx, yy = x[mask], y[mask]
    rs, ps = spearmanr(xx, yy)
    return {"n": n, "spearman_rho": rs, "spearman_p": ps}


def corr_report(stats: dict, label_x: str, label_y: str) -> str:
    if stats["n"] < 3:
        return f"{label_x} vs {label_y}: n={stats['n']} (need ≥3 for correlation)\n"
    return (
        f"{label_x} vs {label_y} (n={stats['n']}):\n"
        f"  Spearman rho = {stats['spearman_rho']:.4f}, p = {stats['spearman_p']:.4g}\n"
    )


def summary_row(
    analysis: str,
    x_label: str,
    y_label: str,
    stats: dict,
) -> dict:
    return {
        "analysis": analysis,
        "x_variable": x_label,
        "y_variable": y_label,
        "n": stats["n"],
        "spearman_rho": stats["spearman_rho"],
        "spearman_p": stats["spearman_p"],
    }


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares line on (x, y) for visualization (Spearman in the stats box is rank-based)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    xx, yy = x[mask], y[mask]
    if np.ptp(xx) < 1e-12:
        return None
    m, b = np.polyfit(xx, yy, 1)
    xs = np.linspace(np.nanmin(xx), np.nanmax(xx), 50)
    return xs, m * xs + b


def plot_scatter_pair(
    axs: np.ndarray,
    merged: pd.DataFrame,
    xcol: str,
    x_title: str,
    pooled: bool,
    label_col: str,
) -> None:
    titles = ("Inattention score", "Hyperactivity score")
    label_fs = 6.5 if pooled else 8
    for ax, ycol, title in zip(axs.flat, ("Inattention score", "Hyperactivity score"), titles):
        x = merged[xcol].to_numpy(dtype=float)
        y = merged[ycol].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.75, s=36, edgecolors="0.35", linewidths=0.4, zorder=2)
        line = _ols_line(x, y)
        if line is not None:
            xs, ys = line
            ax.plot(xs, ys, color="C3", linewidth=1.5, alpha=0.9, zorder=1)
        for _, row in merged.iterrows():
            ax.annotate(
                row[label_col],
                (row[xcol], row[ycol]),
                textcoords="offset points",
                xytext=(3, 2),
                fontsize=label_fs,
                color="0.25",
                zorder=3,
            )
        st = corr_stats(x, y)
        ann = f"Spearman rho={st['spearman_rho']:.3f}, p={st['spearman_p']:.3g}"
        ax.set_title(title)
        ax.set_xlabel(x_title)
        ax.set_ylabel("Score")
        ax.text(
            0.02,
            0.98,
            ann,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        ax.grid(True, alpha=0.35)


def plot_per_class_bars(per_class: pd.DataFrame, out_path: str) -> None:
    df = per_class.sort_values("class_label").reset_index(drop=True)
    labels = df["class_label"].tolist()
    y = np.arange(len(labels))
    h = 0.35
    ri = df["spearman_rho_inattention"].to_numpy(dtype=float)
    rh = df["spearman_rho_hyperactivity"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(labels) + 1.5)))
    ax.barh(y - h / 2, ri, height=h, label="Inattention", color="C0", alpha=0.85)
    ax.barh(y + h / 2, rh, height=h, label="Hyperactivity", color="C1", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman rho (mean HR vs score)")
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_title("Per-class correlation (primary rater; participants in that class)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hr-csv",
        default=DEFAULT_HR_PATH,
        help="hr_by_class_table_oura.csv path",
    )
    ap.add_argument(
        "--orhr-csv",
        default=DEFAULT_OR_PATH,
        help="OrHrAvgs.csv path",
    )
    ap.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV and PNG outputs",
    )
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    hr = load_oura_hr(args.hr_csv)
    orhr_primary = load_orhr_primary(args.orhr_csv)
    orhr_all = load_orhr_all_raters(args.orhr_csv)

    merged_primary = hr.merge(orhr_primary, on="participant", how="inner")
    merged_pooled = hr.merge(orhr_all, on="participant", how="inner")
    if merged_primary.empty:
        print("No overlapping participants between Oura HR table and OrHrAvgs (primary rows).")
        sys.exit(1)
    if merged_pooled.empty:
        print("No overlapping participants for pooled (all-rater) merge.")
        sys.exit(1)

    summary_rows: list[dict] = []

    # --- Pooled (all raters: primary + experts) ---
    print("=== Pooled (class × participant × rater) ===")
    print(
        "Note: rows are not independent — same child and HR appear with multiple rater scores; "
        "classes repeat per child.\n"
    )
    x = merged_pooled["mean_hr_bpm"].to_numpy()
    s_i = corr_stats(x, merged_pooled["Inattention score"].to_numpy())
    s_h = corr_stats(x, merged_pooled["Hyperactivity score"].to_numpy())
    print(corr_report(s_i, "mean_hr_bpm", "Inattention score"))
    print(corr_report(s_h, "mean_hr_bpm", "Hyperactivity score"))
    summary_rows.append(summary_row("pooled_long_all_raters", "mean_hr_bpm", "Inattention score", s_i))
    summary_rows.append(summary_row("pooled_long_all_raters", "mean_hr_bpm", "Hyperactivity score", s_h))

    # --- Participant-level (primary rater only) ---
    part = (
        merged_primary.groupby("participant", as_index=False)["mean_hr_bpm"]
        .mean()
        .rename(columns={"mean_hr_bpm": "mean_hr_across_classes_bpm"})
    )
    part = part.merge(orhr_primary, on="participant", how="inner")
    print("=== Participant-level (mean HR across classes for each participant) ===")
    x = part["mean_hr_across_classes_bpm"].to_numpy()
    sp_i = corr_stats(x, part["Inattention score"].to_numpy())
    sp_h = corr_stats(x, part["Hyperactivity score"].to_numpy())
    print(corr_report(sp_i, "mean_hr (across classes)", "Inattention score"))
    print(corr_report(sp_h, "mean_hr (across classes)", "Hyperactivity score"))
    summary_rows.append(
        summary_row("participant_mean_hr", "mean_hr_across_classes_bpm", "Inattention score", sp_i)
    )
    summary_rows.append(
        summary_row("participant_mean_hr", "mean_hr_across_classes_bpm", "Hyperactivity score", sp_h)
    )

    # --- Per class (primary rater only) ---
    print("=== Per class (HR vs scores; primary rater; participants in that class) ===")
    rows = []
    for cls, g in merged_primary.groupby("class_label"):
        xi = g["mean_hr_bpm"].to_numpy()
        yi = g["Inattention score"].to_numpy()
        yh = g["Hyperactivity score"].to_numpy()
        mask = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(yh)
        n = int(mask.sum())
        if n >= 3:
            rsi, psi = spearmanr(xi[mask], yi[mask])
            rsh, psh = spearmanr(xi[mask], yh[mask])
        else:
            rsi = psi = rsh = psh = np.nan
        rows.append(
            {
                "class_label": cls,
                "n_participants": n,
                "spearman_rho_inattention": rsi,
                "spearman_p_inattention": psi,
                "spearman_rho_hyperactivity": rsh,
                "spearman_p_hyperactivity": psh,
            }
        )
    per_class = pd.DataFrame(rows).sort_values("class_label")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(per_class.to_string(index=False))
    print()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, SUMMARY_CSV)
    per_class_path = os.path.join(out_dir, PER_CLASS_CSV)
    summary_df.to_csv(summary_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_class_path}")

    # --- Plots ---
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_scatter_pair(
        axs,
        merged_pooled,
        "mean_hr_bpm",
        "Mean HR (bpm), per class × child (all raters labeled)",
        pooled=True,
        label_col="participant_label",
    )
    fig.suptitle(
        "Pooled: class × child × rater (same HR repeated per expert/primary)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    pooled_png = os.path.join(out_dir, PLOT_POOLED)
    fig.savefig(pooled_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pooled_png}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    plot_scatter_pair(
        axs,
        part,
        "mean_hr_across_classes_bpm",
        "Mean HR (bpm), across classes",
        pooled=False,
        label_col="participant",
    )
    fig.suptitle("Participant-level: primary rater, one point per child", fontsize=11, y=1.02)
    fig.tight_layout()
    part_png = os.path.join(out_dir, PLOT_PARTICIPANT)
    fig.savefig(part_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {part_png}")

    per_class_png = os.path.join(out_dir, PLOT_PER_CLASS)
    plot_per_class_bars(per_class, per_class_png)
    print(f"Wrote {per_class_png}")


if __name__ == "__main__":
    main()
