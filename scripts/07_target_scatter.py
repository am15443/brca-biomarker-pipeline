#!/usr/bin/env python3
"""
07_target_scatter.py

The headline figure: efficacy vs. safety scatter.

  x-axis: log2 fold change in tumor (efficacy) -- right = more up in tumor
  y-axis: mean expression in healthy breast (safety) -- LOW = safer target

Quadrant logic (with x=FC_LINE and y=SAFE_LINE guides):
  bottom-right  = high efficacy + low healthy expression = GOOD TARGET zone
  top-right     = high efficacy + high healthy expression = DANGER zone
  left half     = not up in tumor = not relevant

Runs as plain python on the master node (small table ~20k rows). Reads the
target-score CSV out of HDFS, renders the scatter, writes a PNG, and copies it
to GCS so it can be downloaded.

Output: ~/biomarker/results/brca_target_scatter.png  (+ GCS copy)
"""
import subprocess
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = "am15443_nyu_edu"
HDFS_BASE = f"/user/{USER}/biomarker"
SCORE_CSV_GLOB = f"{HDFS_BASE}/results/brca_target_score_csv/part-*.csv"
LOCAL_OUT = f"/home/{USER}/biomarker/results/brca_target_scatter.png"
GCS_OUT = "gs://nyu-dataproc-temp/am15443_target_scatter.png"

# Thresholds for the quadrant guides and for calling a gene "up in tumor".
FC_LINE = 1.0          # log2FC > 1  => up in tumor (2x)
FDR_MAX = 0.05         # only trust genes significant in the DE step
SAFE_PCTILE = 25       # "low healthy expression" = below this percentile
N_LABEL_GOOD = 12      # label this many top good-target genes
N_LABEL_DANGER = 6     # label this many danger-zone genes


def hdfs_cat(path_glob: str) -> str:
    out = subprocess.run(["hdfs", "dfs", "-cat", path_glob],
                         capture_output=True, text=True, check=True)
    return out.stdout


def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

    df = pd.read_csv(io.StringIO(hdfs_cat(SCORE_CSV_GLOB)))
    print(f"genes loaded: {len(df):,}")

    # Only consider DE-significant genes for target calling; keep the rest as
    # faint background so the plot still shows the full cloud.
    sig = df["fdr"] < FDR_MAX
    x = df["log2fc"]
    y = df["healthy_mean_logcpm"]

    safe_thresh = np.percentile(df["healthy_mean_logcpm"], SAFE_PCTILE)
    print(f"'low healthy expression' threshold (p{SAFE_PCTILE}): "
          f"{safe_thresh:.2f} log2-CPM")

    # Quadrant membership
    up = sig & (df["log2fc"] > FC_LINE)
    good = up & (df["healthy_mean_logcpm"] <= safe_thresh)   # bottom-right
    danger = up & (df["healthy_mean_logcpm"] > safe_thresh)  # top-right
    print(f"GOOD target-zone genes:  {good.sum():,}")
    print(f"DANGER-zone genes:       {danger.sum():,}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 7.5))

    # background: everything, faint
    ax.scatter(x, y, s=5, c="#d9d9d9", alpha=0.35, linewidths=0,
               label="All genes", rasterized=True)
    # danger zone
    ax.scatter(df.loc[danger, "log2fc"], df.loc[danger, "healthy_mean_logcpm"],
               s=14, c="#d7301f", alpha=0.65, linewidths=0,
               label="Up in tumor, high in healthy (risk)")
    # good zone
    ax.scatter(df.loc[good, "log2fc"], df.loc[good, "healthy_mean_logcpm"],
               s=16, c="#238b45", alpha=0.75, linewidths=0,
               label="Up in tumor, low in healthy (candidate)")

    # quadrant guides
    ax.axvline(FC_LINE, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(safe_thresh, color="k", ls="--", lw=0.7, alpha=0.5)

    # ---- Labels ----
    # good candidates: label the highest-efficacy ones (they sit farthest right,
    # most spread out, least overlapping) rather than by score (which clusters).
    good_df = df[good].nlargest(N_LABEL_GOOD, "log2fc")
    for _, r in good_df.iterrows():
        ax.annotate(r["symbol"],
                    (r["log2fc"], r["healthy_mean_logcpm"]),
                    fontsize=7.5, fontweight="bold", color="#00441b",
                    xytext=(4, 2), textcoords="offset points")
    # danger genes: highest healthy expression among up-in-tumor
    danger_df = df[danger].nlargest(N_LABEL_DANGER, "healthy_mean_logcpm")
    for _, r in danger_df.iterrows():
        ax.annotate(r["symbol"],
                    (r["log2fc"], r["healthy_mean_logcpm"]),
                    fontsize=7.5, fontweight="bold", color="#7f0000",
                    xytext=(4, 2), textcoords="offset points")

    # quadrant captions
    ax.text(0.98, 0.02, "CANDIDATE TARGETS\n(up in tumor, quiet in healthy)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#238b45", fontweight="bold", alpha=0.8)
    ax.text(0.98, 0.98, "TOXICITY RISK\n(up in tumor, active in healthy)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#d7301f", fontweight="bold", alpha=0.8)

    ax.set_xlabel("Efficacy  \u2192  log2 fold change in tumor (vs. healthy)")
    ax.set_ylabel("Safety cost  \u2192  mean expression in healthy breast (log2-CPM)")
    ax.set_title("Efficacy vs. safety landscape for breast-cancer targets\n"
                 f"TCGA-BRCA differential expression vs. GTEx healthy breast "
                 f"({good.sum():,} candidate genes)")
    ax.legend(loc="upper left", frameon=True, fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT, dpi=150)
    print(f"wrote {LOCAL_OUT}")

    # copy to GCS for download
    try:
        subprocess.run(["gsutil", "cp", LOCAL_OUT, GCS_OUT], check=True)
        print(f"copied to {GCS_OUT}")
    except Exception as e:
        print(f"GCS copy skipped: {e}")


if __name__ == "__main__":
    main()
