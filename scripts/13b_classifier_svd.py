#!/usr/bin/env python3
"""
13b_classifier_svd.py

Random Forest classifier, tumor vs. healthy, trained on the precomputed SVD
features (50 components) instead of 2,000 raw genes. This avoids the wide
pivot and the memory blowup that made the gene-level RF fail on this cluster:
the SVD already reduced dimensionality, so training is fast and light.

Reads:  parquet/model/svd_features   (sample_id, svd_0..svd_49, tissue, is_tumor)
Writes: results/rf_svd_metrics_csv   (accuracy, f1, auc)

The reportable result is the ACCURACY of predicting malignancy from expression
across six cancer types. Gene-level biomarker rankings come from the
differential-expression analysis (script 03), not from this classifier.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
                                   BinaryClassificationEvaluator)

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
SVD_IN = f"{BASE}/parquet/model/svd_features"
OUT_CSV = f"{BASE}/results/rf_svd_metrics_csv"

N_COMPONENTS = 50
SEED = 42


def main():
    spark = SparkSession.builder.appName("rf_svd").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(SVD_IN)
    feat_cols = [f"svd_{i}" for i in range(N_COMPONENTS)]

    assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
    data = (assembler.transform(df)
            .select("features", F.col("is_tumor").alias("label")))

    train, test = data.randomSplit([0.7, 0.3], seed=SEED)
    print(f"train: {train.count():,}   test: {test.count():,}")

    rf = RandomForestClassifier(numTrees=100, maxDepth=10, seed=SEED,
                                featuresCol="features", labelCol="label")
    model = rf.fit(train)
    pred = model.transform(test)

    acc = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", metricName="f1").evaluate(pred)
    auc = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC").evaluate(pred)

    print("=== classifier performance (tumor vs healthy, SVD features) ===")
    print(f"accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")
    print(f"ROC AUC : {auc:.4f}")

    print("=== confusion matrix (rows=actual, cols=predicted) ===")
    pred.groupBy("label").pivot("prediction").count().orderBy("label").show()

    print("=== per-class precision / recall ===")
    for m in ["precisionByLabel", "recallByLabel"]:
        for lbl in [0.0, 1.0]:
            v = MulticlassClassificationEvaluator(
                labelCol="label", metricName=m, metricLabel=lbl).evaluate(pred)
            print(f"  {m} (label={int(lbl)}): {v:.4f}")

    metrics = spark.createDataFrame(
        [("accuracy", float(acc)), ("f1", float(f1)), ("auc", float(auc))],
        ["metric", "value"])
    metrics.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_CSV)
    print("wrote metrics to", OUT_CSV)

    spark.stop()


if __name__ == "__main__":
    main()
