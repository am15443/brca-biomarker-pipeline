#!/usr/bin/env python3
"""12_pca_pc3.py - compare PC1-PC2 vs PC1-PC3 (both colored by tissue)."""
import subprocess, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = "am15443_nyu_edu"
HDFS_BASE = f"hdfs://nyu-dataproc-m:8020/user/{USER}/biomarker"
SVD_PARQUET = f"{HDFS_BASE}/parquet/model/svd_features"
LOCAL_OUT = f"/home/{USER}/biomarker/results/pca_pc3.png"
GCS_OUT = "gs://nyu-dataproc-temp/am15443_pca_pc3.png"


def load_svd():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("pca_pc3_read").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.parquet(SVD_PARQUET).toPandas()
    spark.stop()
    return df

def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
    df = load_svd()
    print(f"samples: {len(df):,}")
    tissues = sorted(df["tissue"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tissues)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    for pair, ax in [((0, 1), ax1), ((0, 2), ax2)]:
        px, py = pair
        for t, c in zip(tissues, cmap):
            m = df["tissue"] == t
            ax.scatter(df[f"svd_{px}"][m], df[f"svd_{py}"][m],
                       s=10, color=c, alpha=0.6, linewidths=0, label=t)
        ax.set_xlabel(f"PC{px+1}"); ax.set_ylabel(f"PC{py+1}")
        ax.set_title(f"PC{px+1} vs PC{py+1}")
        ax.grid(True, alpha=0.12)
    ax1.legend(loc="best", fontsize=8, markerscale=2)
    fig.suptitle("Tissue separation: PC1-PC2 vs PC1-PC3 "
                 "(does PC3 split breast/lung?)", fontsize=12)
    fig.tight_layout(); fig.savefig(LOCAL_OUT, dpi=150)
    print(f"wrote {LOCAL_OUT}")
    try:
        subprocess.run(["gsutil", "cp", LOCAL_OUT, GCS_OUT], check=True)
        print(f"copied to {GCS_OUT}")
    except Exception as e:
        print(f"GCS copy skipped: {e}")

if __name__ == "__main__":
    main()
