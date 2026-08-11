"""
10_combine_svd.py  [N_GENES]

Modeling layer, step 1: combine all cancers + healthy tissues into one matrix,
select the most variable genes, pivot to a sample-by-gene feature matrix, and
run distributed truncated SVD (Spark MLlib) to reduce dimensionality.

  N_GENES (optional, default 2000): number of top-variance genes to keep.

Reads all per-cancer / per-tissue long tables:
    parquet/tcga_long/<CANCER>/   (has: gene_id, sample_id, count, sample_type, cancer)
    parquet/gtex_long/<TISSUE>/   (has: gene_id, sample_id, count, sample_type, tissue)

Writes:
    parquet/model/svd_features/   one row per sample:
        sample_id, group_label (tissue+tumor/normal), tissue, is_tumor,
        svd_0 ... svd_{k-1}
    parquet/model/sample_labels/  sample_id -> labels (for plotting/clustering)

Pipeline:
  1. Union all tables into (gene_id, sample_id, count, group_label, tissue, is_tumor).
  2. Per-sample CPM + log2 normalization.
  3. Select top-N most variable genes (variance of log-CPM across samples).
  4. Pivot to wide: sample x gene, assemble MLlib feature vector.
  5. Standardize features, run truncated SVD (via RowMatrix.computeSVD).
  6. Write reduced feature matrix + labels.
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.linalg.distributed import RowMatrix

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"

TCGA = ["BRCA", "LUAD", "THCA", "PRAD", "COAD", "KIRC"]
GTEX = ["BREAST", "LUNG", "THYROID", "PROSTATE", "COLON", "KIDNEY"]
# map each GTEx tissue to the cancer's tissue name for grouping
TISSUE_OF_CANCER = {"BRCA": "BREAST", "LUAD": "LUNG", "THCA": "THYROID",
                    "PRAD": "PROSTATE", "COAD": "COLON", "KIRC": "KIDNEY"}

SVD_K = 50            # number of SVD components to keep
DEFAULT_N_GENES = 2000


def main():
    n_genes = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_GENES

    spark = (SparkSession.builder
             .appName(f"combine_svd_{n_genes}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. Union all tables into a common schema ----
    frames = []
    for c in TCGA:
        df = spark.read.parquet(f"{BASE}/parquet/tcga_long/{c}")
        tissue = TISSUE_OF_CANCER[c]
        df = df.select(
            "gene_id", "sample_id", "count",
            F.lit(tissue).alias("tissue"),
            # tumor vs normal within TCGA
            F.when(F.col("sample_type") == "Solid Tissue Normal", 0)
             .otherwise(1).alias("is_tumor"))
        frames.append(df)
    for tis in GTEX:
        df = spark.read.parquet(f"{BASE}/parquet/gtex_long/{tis}")
        df = df.select(
            "gene_id", "sample_id", "count",
            F.lit(tis).alias("tissue"),
            F.lit(0).alias("is_tumor"))   # GTEx = healthy
        frames.append(df)

    long_all = frames[0]
    for f in frames[1:]:
        long_all = long_all.unionByName(f)

    # Strip the version suffix from gene IDs (ENSG0000042832.11 -> ENSG0000042832).
    # Dots are column-path separators in Spark SQL, so they break the pivot that
    # turns gene IDs into column names. Stripping the version is also standard
    # practice for cross-database joins.
    long_all = long_all.withColumn(
        "gene_id", F.regexp_replace(F.col("gene_id"), r"\..*$", ""))

    # group label used for coloring plots, e.g. "BREAST-tumor" / "BREAST-healthy"
    long_all = long_all.withColumn(
        "group_label",
        F.concat_ws("-", F.col("tissue"),
                    F.when(F.col("is_tumor") == 1, "tumor").otherwise("healthy")))

    # ---- 2. Per-sample CPM + log2 ----
    lib = long_all.groupBy("sample_id").agg(F.sum("count").alias("lib"))
    norm = (long_all.join(F.broadcast(lib), on="sample_id")
            .withColumn("logcpm",
                        F.log2(F.col("count") / F.col("lib") * F.lit(1e6) + F.lit(1.0))))

    n_samples = norm.select("sample_id").distinct().count()
    print(f"total samples across all tissues: {n_samples:,}")

    # ---- 3. Top-N most variable genes (variance of log-CPM across samples) ----
    # Collapse any duplicate gene_ids created by version stripping (sum first
    # would double-count; here each (gene,sample) should be unique, but guard
    # by averaging logcpm across any collisions).
    norm = (norm.groupBy("gene_id", "sample_id", "group_label", "tissue", "is_tumor")
            .agg(F.mean("logcpm").alias("logcpm")))

    gene_var = (norm.groupBy("gene_id")
                .agg(F.variance("logcpm").alias("v"))
                .orderBy(F.col("v").desc())
                .limit(n_genes))
    top_genes = [r["gene_id"] for r in gene_var.collect()]
    print(f"selected top {len(top_genes)} variable genes")

    norm = norm.join(F.broadcast(spark.createDataFrame(
        [(g,) for g in top_genes], ["gene_id"])), on="gene_id")

    # ---- 4. Pivot to wide (sample x gene) ----
    # collect labels per sample first (small)
    labels = (norm.select("sample_id", "group_label", "tissue", "is_tumor")
              .distinct())

    wide = (norm.groupBy("sample_id")
            .pivot("gene_id", top_genes)
            .agg(F.first("logcpm")))
    wide = wide.na.fill(0.0)

    # assemble feature vector
    assembler = VectorAssembler(inputCols=top_genes, outputCol="features_raw",
                                handleInvalid="keep")
    feat = assembler.transform(wide).select("sample_id", "features_raw")

    # ---- 5. Standardize + truncated SVD ----
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    feat = scaler.fit(feat).transform(feat).select("sample_id", "features")

    # to RowMatrix for SVD (preserve sample_id order)
    indexed = feat.rdd.map(lambda r: (r["sample_id"], r["features"])).zipWithIndex()
    id_order = indexed.map(lambda x: (x[1], x[0][0]))  # (idx, sample_id)
    rows = indexed.map(lambda x: MLLibVectors.dense(x[0][1].toArray()))
    mat = RowMatrix(rows)

    svd = mat.computeSVD(SVD_K, computeU=True)
    U = svd.U   # rows = samples, cols = components

    # attach sample_id back by index
    id_map = dict(id_order.collect())
    svd_rows = (U.rows.zipWithIndex()
                .map(lambda x: (id_map[x[1]],) + tuple(float(v) for v in x[0].toArray())))
    cols = ["sample_id"] + [f"svd_{i}" for i in range(SVD_K)]
    svd_df = spark.createDataFrame(svd_rows, cols)

    result = svd_df.join(labels, on="sample_id")

    # ---- 6. Write ----
    (result.write.mode("overwrite")
     .parquet(f"{BASE}/parquet/model/svd_features"))
    (labels.write.mode("overwrite")
     .parquet(f"{BASE}/parquet/model/sample_labels"))

    print("=== SVD complete ===")
    print(f"samples: {result.count():,}   components: {SVD_K}")
    print("singular values (top 10):",
          [round(float(s), 1) for s in svd.s.toArray()[:10]])
    print("=== samples per group ===")
    result.groupBy("group_label").count().orderBy("group_label").show(30, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
