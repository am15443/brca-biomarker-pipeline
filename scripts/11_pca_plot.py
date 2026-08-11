#!/usr/bin/env python3
"""11_pca_plot.py - PCA projection of SVD features, two panels."""
import subprocess, io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = "am15443_nyu_edu"
HDFS_BASE = f"hdfs://nyu-dataproc-m:8020/user/{USER}/biomarker"
SVD_PARQUET = f"{HDFS_BASE}/parquet/model/svd_features"
LOCAL_OUT = f"/home/{USER}/biomarker/results/pca_projection.png"
GCS_OUT = "gs://nyu-dataproc-temp/am15443_pca_projection.png"
PC_X, PC_Y = 0, 1

def load_svd():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("pca_read").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.parquet(SVD_PARQUET).toPandas()
    spark.stop()
    return df

def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
    df = load_svd()
    print(f"samples: {len(df):,}")
    x, y = df[f"svd_{PC_X}"], df[f"svd_{PC_Y}"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    tissues = sorted(df["tissue"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tissues)))
    for t, c in zip(tissues, cmap):
        m = df["tissue"] == t
        ax1.scatter(x[m], y[m], s=10, color=c, alpha=0.6, linewidths=0, label=t)
    ax1.set_title("Colored by tissue")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(loc="best", fontsize=8, markerscale=2); ax1.grid(True, alpha=0.12)
    for lab, c, name in [(1, "#d7301f", "Tumor"), (0, "#238b45", "Healthy")]:
        m = df["is_tumor"] == lab
        ax2.scatter(x[m], y[m], s=10, color=c, alpha=0.5, linewidths=0, label=name)
    ax2.set_title("Colored by tumor vs healthy")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.legend(loc="best", fontsize=9, markerscale=2); ax2.grid(True, alpha=0.12)
    fig.suptitle("PCA of SVD feature space: 7,144 samples, 6 tissues")
    fig.tight_layout(); fig.savefig(LOCAL_OUT, dpi=150)
    print(f"wrote {LOCAL_OUT}")
    try:
        subprocess.run(["gsutil", "cp", LOCAL_OUT, GCS_OUT], check=True)
        print(f"copied to {GCS_OUT}")
    except Exception as e:
        print(f"GCS copy skipped: {e}")

if __name__ == "__main__":
    main()
