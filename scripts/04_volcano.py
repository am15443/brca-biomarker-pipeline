#!/usr/bin/env python3
"""
04_volcano.py

Build a volcano plot from the BRCA differential-expression results.

Runs as plain Python on the master node (NOT spark-submit): the DE results are
~20k rows, small enough to pull local. Reads the results CSV and the gene-symbol
map out of HDFS via `hdfs dfs -cat`, joins them, and renders a labeled volcano.

x-axis: log2 fold change (tumor vs normal)
y-axis: -log10(p-value)
Points passing both thresholds (|log2FC| and FDR) are colored; the strongest
few by significance are labeled with gene symbols.

Output: ~/biomarker/results/brca_volcano.png (local), also copied to HDFS.
"""
import subprocess
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no display on the cluster
import matplotlib.pyplot as plt

USER = "am15443_nyu_edu"
HDFS_BASE = f"/user/{USER}/biomarker"
DE_CSV_GLOB = f"{HDFS_BASE}/results/brca_diffexp_csv/part-*.csv"
GENE_MAP = f"{HDFS_BASE}/raw/gene_map.tsv"
LOCAL_OUT = f"/home/{USER}/biomarker/results/brca_volcano.png"

# Thresholds for calling a gene "significant" on the plot.
FC_THRESH = 1.0        # |log2 fold change| >= 1  (2x change)
FDR_THRESH = 0.05
N_LABEL = 8            # label this many top genes by significance, each side


def hdfs_cat(path_glob: str) -> str:
    """Return the concatenated text of HDFS file(s) matching path_glob."""
    out = subprocess.run(
        ["hdfs", "dfs", "-cat", path_glob],
        capture_output=True, text=True, check=True,
    )
    return out.stdout


def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

    # ---- Load DE results (single-part CSV written with header) ----
    de_txt = hdfs_cat(DE_CSV_GLOB)
    de = pd.read_csv(io.StringIO(de_txt))
    print(f"DE rows loaded: {len(de):,}")

    # ---- Load gene symbol map ----
    map_txt = hdfs_cat(GENE_MAP)
    gmap = pd.read_csv(io.StringIO(map_txt), sep="\t",
                       header=None, names=["gene_id", "symbol"])
    print(f"gene map rows: {len(gmap):,}")

    df = de.merge(gmap, on="gene_id", how="left")
    df["symbol"] = df["symbol"].fillna(df["gene_id"])

    # ---- Compute plot coordinates ----
    # Guard against p==0 underflow: floor tiny p at the smallest positive double.
    tiny = np.nextafter(0, 1)
    df["pvalue"] = df["pvalue"].clip(lower=tiny)
    df["neglog10p"] = -np.log10(df["pvalue"])

    sig = (df["fdr"] < FDR_THRESH) & (df["log2fc"].abs() >= FC_THRESH)
    up = sig & (df["log2fc"] > 0)
    down = sig & (df["log2fc"] < 0)
    print(f"significant up:   {up.sum():,}")
    print(f"significant down: {down.sum():,}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(df.loc[~sig, "log2fc"], df.loc[~sig, "neglog10p"],
               s=6, c="#b0b0b0", alpha=0.4, linewidths=0, label="Not significant")
    ax.scatter(df.loc[down, "log2fc"], df.loc[down, "neglog10p"],
               s=10, c="#2c7fb8", alpha=0.7, linewidths=0, label="Down in tumor")
    ax.scatter(df.loc[up, "log2fc"], df.loc[up, "neglog10p"],
               s=10, c="#d7301f", alpha=0.7, linewidths=0, label="Up in tumor")

    # threshold guides
    ax.axvline(FC_THRESH, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(-FC_THRESH, color="k", ls="--", lw=0.6, alpha=0.5)

    # ---- Label the strongest genes on each side ----
    # Rank by a combined score (significance * effect size) so labels go to the
    # genes that stand out in BOTH dimensions -- these sit at the plot edges and
    # collide least. Then add the few most extreme fold changes (far corners).
    df["score"] = df["neglog10p"] * df["log2fc"].abs()
    top_up = df[up].nlargest(N_LABEL, "score")
    top_down = df[down].nlargest(N_LABEL, "score")
    extreme = pd.concat([df[up].nlargest(4, "log2fc"),
                         df[down].nsmallest(4, "log2fc")])
    to_label = pd.concat([top_up, top_down, extreme]).drop_duplicates("gene_id")
    for _, r in to_label.iterrows():
        ax.annotate(r["symbol"], (r["log2fc"], r["neglog10p"]),
                    fontsize=7.5, fontweight="bold",
                    xytext=(4, 2), textcoords="offset points",
                    color="#111111")

    ax.set_xlabel("log2 fold change (tumor vs. normal)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("TCGA-BRCA differential expression: tumor vs. healthy breast\n"
                 f"{up.sum():,} up / {down.sum():,} down at FDR<{FDR_THRESH}, "
                 f"|log2FC|>={FC_THRESH}")
    ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=8)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT, dpi=150)
    print(f"wrote {LOCAL_OUT}")


if __name__ == "__main__":
    main()
