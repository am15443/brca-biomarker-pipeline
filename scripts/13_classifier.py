#!/usr/bin/env python3
"""
13_classifier.py  [N_GENES]

Modeling layer, step 2: Random Forest classifier predicting tumor vs. healthy
from gene expression, across all six tissues combined. Outputs accuracy,
per-class precision/recall, a confusion matrix, and a ranked gene-importance
table -- a second, independent biomarker ranking to cross-check against the
differential-expression (t-test) results.

  N_GENES (optional, default 2000): number of top-variance genes to use.

Reads the same per-cancer / per-tissue long tables as script 10, rebuilds the
wide sample-by-gene matrix on the top-variance genes, then trains/evaluates.

Writes:
    parquet/model/rf_importance/   ranked gene_id + importance
    results/rf_importance_csv/      same, as CSV for inspection
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
                                   BinaryClassificationEvaluator)

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"

TCGA = ["BRCA", "LUAD", "THCA", "PRAD", "COAD", "KIRC"]
GTEX = ["BREAST", "LUNG", "THYROID", "PROSTATE", "COLON", "KIDNEY"]
TISSUE_OF_CANCER = {"BRCA": "BREAST", "LUAD": "LUNG", "THCA": "THYROID",
                    "PRAD": "PROSTATE", "COAD": "COLON", "KIRC": "KIDNEY"}

DEFAULT_N_GENES = 2000
N_TREES = 100
SEED = 42


def main():
    n_genes = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_GENES
    spark = (SparkSession.builder.appName(f"rf_classifier_{n_genes}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. Union all tables, label is_tumor ----
    frames = []
    for c in TCGA:
        df = spark.read.parquet(f"{BASE}/parquet/tcga_long/{c}")
        frames.append(df.select(
            "gene_id", "sample_id", "count",
            F.when(F.col("sample_type") == "Solid Tissue Normal", 0)
             .otherwise(1).alias("is_tumor")))
    for tis in GTEX:
        df = spark.read.parquet(f"{BASE}/parquet/gtex_long/{tis}")
        frames.append(df.select(
            "gene_id", "sample_id", "count", F.lit(0).alias("is_tumor")))
    long_all = frames[0]
    for f in frames[1:]:
        long_all = long_all.unionByName(f)

    # strip version suffix so gene ids are valid column names
    long_all = long_all.withColumn(
        "gene_id", F.regexp_replace(F.col("gene_id"), r"\..*$", ""))

    # ---- 2. Normalize (CPM, log2) ----
    lib = long_all.groupBy("sample_id").agg(F.sum("count").alias("lib"))
    norm = (long_all.join(F.broadcast(lib), on="sample_id")
            .withColumn("logcpm",
                        F.log2(F.col("count") / F.col("lib") * F.lit(1e6) + F.lit(1.0))))
    # collapse any duplicate (gene,sample) from version stripping
    norm = (norm.groupBy("gene_id", "sample_id", "is_tumor")
            .agg(F.mean("logcpm").alias("logcpm")))

    # ---- 3. Top-N variable genes ----
    gene_var = (norm.groupBy("gene_id").agg(F.variance("logcpm").alias("v"))
                .orderBy(F.col("v").desc()).limit(n_genes))
    top_genes = [r["gene_id"] for r in gene_var.collect()]
    print(f"using top {len(top_genes)} genes")
    norm = norm.join(F.broadcast(spark.createDataFrame(
        [(g,) for g in top_genes], ["gene_id"])), on="gene_id")

    # ---- 4. Pivot to wide (sample x gene) + label ----
    labels = norm.select("sample_id", "is_tumor").distinct()
    wide = (norm.groupBy("sample_id").pivot("gene_id", top_genes)
            .agg(F.first("logcpm")).na.fill(0.0))
    wide = wide.join(labels, on="sample_id")

    assembler = VectorAssembler(inputCols=top_genes, outputCol="features",
                               handleInvalid="keep")
    data = assembler.transform(wide).select("sample_id", "features",
                                            F.col("is_tumor").alias("label"))

    # ---- 5. Train / test split ----
    train, test = data.randomSplit([0.7, 0.3], seed=SEED)
    print(f"train: {train.count():,}   test: {test.count():,}")

    rf = RandomForestClassifier(numTrees=N_TREES, seed=SEED,
                                featuresCol="features", labelCol="label")
    model = rf.fit(train)
    pred = model.transform(test)

    # ---- 6. Metrics ----
    acc = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", metricName="f1").evaluate(pred)
    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC").evaluate(pred)
    print("=== classifier performance (tumor vs healthy) ===")
    print(f"accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")
    print(f"ROC AUC : {auc:.4f}")

    print("=== confusion matrix (rows=actual, cols=predicted) ===")
    (pred.groupBy("label").pivot("prediction").count()
     .orderBy("label").show())

    print("=== per-class precision / recall ===")
    for m in ["precisionByLabel", "recallByLabel"]:
        for lbl in [0.0, 1.0]:
            v = MulticlassClassificationEvaluator(
                labelCol="label", metricName=m, metricLabel=lbl).evaluate(pred)
            print(f"  {m} (label={int(lbl)}): {v:.4f}")

    # ---- 7. Gene importances (the second biomarker ranking) ----
    importances = model.featureImportances.toArray()
    imp_rows = sorted(zip(top_genes, importances),
                      key=lambda x: x[1], reverse=True)
    imp_df = spark.createDataFrame(
        [(g, float(i)) for g, i in imp_rows], ["gene_id", "importance"])

    (imp_df.write.mode("overwrite").parquet(f"{BASE}/parquet/model/rf_importance"))
    (imp_df.coalesce(1).write.mode("overwrite").option("header", True)
     .csv(f"{BASE}/results/rf_importance_csv"))

    print("=== top 25 tumor-vs-healthy discriminating genes ===")
    imp_df.limit(25).show(25, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
