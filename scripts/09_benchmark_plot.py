#!/usr/bin/env python3
"""
09_benchmark_plot.py

Plot the scalability benchmark: runtime vs. input size for the differential-
expression workload, with a linear fit and an inset showing per-row cost
flattening as fixed overhead amortizes. Annotates the extrapolation to the full
recount3 corpus.

Runs as plain python on the master node. Reads the benchmark CSV from HDFS,
renders the figure, writes a PNG, copies it to GCS for download.

Output: ~/biomarker/results/brca_benchmark.png (+ GCS copy)
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
BENCH_CSV_GLOB = f"{HDFS_BASE}/results/benchmark_csv/part-*.csv"
LOCAL_OUT = f"/home/{USER}/biomarker/results/brca_benchmark.png"
GCS_OUT = "gs://nyu-dataproc-temp/am15443_benchmark.png"

# Full recount3 human corpus, for extrapolation.
CORPUS_SAMPLES = 750_000
ROWS_PER_SAMPLE = 63_856   # genes per sample (constant annotation)


def hdfs_cat(path_glob: str) -> str:
    out = subprocess.run(["hdfs", "dfs", "-cat", path_glob],
                         capture_output=True, text=True, check=True)
    return out.stdout


def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

    df = pd.read_csv(io.StringIO(hdfs_cat(BENCH_CSV_GLOB)))
    df = df.sort_values("n_rows").reset_index(drop=True)
    print("benchmark points:")
    print(df.to_string(index=False))

    rows_m = df["n_rows"] / 1e6          # millions of rows
    secs = df["seconds"]
    per_row = secs / rows_m              # sec per million rows

    # Linear fit through the larger points (compute-dominated regime):
    # use points from 25% onward so startup overhead doesn't skew the slope.
    fit_mask = rows_m >= rows_m.iloc[1]
    slope, intercept = np.polyfit(rows_m[fit_mask], secs[fit_mask], 1)
    print(f"\nlinear fit (compute regime): {slope:.3f} sec/million rows "
          f"+ {intercept:.1f}s fixed")

    # Extrapolate to full corpus
    corpus_rows_m = CORPUS_SAMPLES * ROWS_PER_SAMPLE / 1e6
    corpus_secs = slope * corpus_rows_m + intercept
    corpus_hours = corpus_secs / 3600
    print(f"extrapolated full-corpus runtime on THIS cluster: "
          f"{corpus_hours:.1f} hours ({corpus_rows_m:,.0f}M rows)")

    # ---- Figure: main panel + inset ----
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # measured points
    ax.scatter(rows_m, secs, s=70, c="#2166ac", zorder=3,
               label="Measured runtime")
    ax.plot(rows_m, secs, c="#2166ac", lw=1.2, alpha=0.5, zorder=2)

    # linear fit line
    xfit = np.linspace(0, rows_m.max() * 1.05, 50)
    ax.plot(xfit, slope * xfit + intercept, "--", c="#b2182b", lw=1.5,
            label=f"Linear fit: {slope:.2f} s/M rows", zorder=1)

    for _, r in df.iterrows():
        ax.annotate(f"{int(r['n_samples'])} samples",
                    (r["n_rows"] / 1e6, r["seconds"]),
                    fontsize=7.5, xytext=(6, -4), textcoords="offset points",
                    color="#333333")

    ax.set_xlabel("Input size (millions of rows)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Differential-expression workload scales near-linearly\n"
                 "TCGA-BRCA, sample count varied on a fixed Dataproc cluster")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.15)

    # inset: per-row cost flattening
    axin = ax.inset_axes([0.58, 0.12, 0.38, 0.34])
    axin.plot(rows_m, per_row, "o-", c="#238b45", lw=1.3, ms=5)
    axin.set_title("Per-row cost amortizes", fontsize=8)
    axin.set_xlabel("M rows", fontsize=7)
    axin.set_ylabel("s / M rows", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.grid(True, alpha=0.2)

    # extrapolation caption
    ax.text(0.02, 0.72,
            f"Extrapolation to full recount3 corpus\n"
            f"(~{CORPUS_SAMPLES:,} samples \u2248 {corpus_rows_m/1000:,.1f}B rows):\n"
            f"\u2248 {corpus_hours:.0f} h on this cluster, "
            f"or ~{corpus_hours/10:.0f} h on a 10\u00D7 cluster",
            transform=ax.transAxes, fontsize=8, color="#555555",
            va="top", bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))

    fig.tight_layout()
    fig.savefig(LOCAL_OUT, dpi=150)
    print(f"\nwrote {LOCAL_OUT}")

    try:
        subprocess.run(["gsutil", "cp", LOCAL_OUT, GCS_OUT], check=True)
        print(f"copied to {GCS_OUT}")
    except Exception as e:
        print(f"GCS copy skipped: {e}")


if __name__ == "__main__":
    main()
