"""
08_benchmark.py

Scalability benchmark for the differential-expression workload.

Runs the core DE computation on increasing fractions of the BRCA long table
(10%, 25%, 50%, 75%, 100% of samples) and records wall-clock runtime and input
row count for each. The output is a small table you plot as runtime vs. data
size -- the evidence that the pipeline scales, and the basis for extrapolating
cost to the full recount3 corpus.

Design:
  - We subsample SAMPLES, not rows, because differential expression is per-gene
    across samples -- halving samples halves the statistical work per gene in a
    way that mirrors how the real corpus would grow (more studies = more
    samples). Subsampling random rows would break the per-gene structure.
  - Each fraction is run END TO END (filter -> normalize -> t-test -> collect)
    and timed. A .count() forces execution so we time real work, not lazy
    planning.
  - Spark's fixed overhead (JVM, scheduling) dominates at small sizes; the
    meaningful signal is the slope at the larger fractions. The output includes
    rows so you can plot runtime vs. actual input size, not just fraction.

Note: on a shared cluster, other jobs cause timing noise. Run this when the
cluster is quiet if possible, and treat absolute numbers as approximate -- the
trend across sizes is the result, not any single measurement.
"""
import time
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
IN = f"{BASE}/parquet/brca_long"
OUT_CSV = f"{BASE}/results/benchmark_csv"

TUMOR = "Primary Tumor"
NORMAL = "Solid Tissue Normal"
MIN_CPM = 1.0
MIN_FRAC = 0.20

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
SEED = 42


def run_de(df):
    """The DE computation, returning a small collected result to force execution.
    Mirrors 03 but trimmed to the compute-heavy core (no FDR/plotting)."""
    # per-sample library size -> CPM -> log2
    lib = df.groupBy("sample_id").agg(F.sum("count").alias("lib_size"))
    d = (df.join(F.broadcast(lib), on="sample_id")
         .withColumn("cpm", F.col("count") / F.col("lib_size") * F.lit(1e6))
         .withColumn("logcpm", F.log2(F.col("cpm") + F.lit(1.0))))

    # gene filter
    n_samples = df.select("sample_id").distinct().count()
    min_samples = int(round(MIN_FRAC * n_samples))
    expressed = (d.withColumn("is_expr", (F.col("cpm") > F.lit(MIN_CPM)).cast("int"))
                 .groupBy("gene_id").agg(F.sum("is_expr").alias("n_expr")))
    keep = expressed.filter(F.col("n_expr") >= F.lit(min_samples)).select("gene_id")
    d = d.join(F.broadcast(keep), on="gene_id")

    # per-gene, per-group summary -> Welch t-stat
    grp = (d.groupBy("gene_id", "sample_type")
           .agg(F.mean("logcpm").alias("mean"),
                F.variance("logcpm").alias("var"),
                F.count("logcpm").alias("n")))
    t = grp.filter(F.col("sample_type") == TUMOR).select(
        "gene_id", F.col("mean").alias("mt"), F.col("var").alias("vt"),
        F.col("n").alias("nt"))
    nrm = grp.filter(F.col("sample_type") == NORMAL).select(
        "gene_id", F.col("mean").alias("mn"), F.col("var").alias("vn"),
        F.col("n").alias("nn"))
    stats = (t.join(nrm, on="gene_id")
             .withColumn("se2", F.col("vt") / F.col("nt") + F.col("vn") / F.col("nn"))
             .filter(F.col("se2") > F.lit(0.0))
             .withColumn("t_stat",
                         (F.col("mt") - F.col("mn")) / F.sqrt(F.col("se2"))))
    # force execution
    return stats.count()


def main():
    spark = (SparkSession.builder
             .appName("brca_benchmark")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    full = spark.read.parquet(IN)

    # distinct sample list to subsample by sample (not by row)
    all_samples = [r["sample_id"] for r in
                   full.select("sample_id").distinct().collect()]
    n_all = len(all_samples)
    print(f"total samples available: {n_all}")

    import random
    random.seed(SEED)

    rows = []
    for frac in FRACTIONS:
        k = max(2, int(round(frac * n_all)))
        subset_samples = set(random.sample(all_samples, k))
        sub = full.filter(F.col("sample_id").isin(subset_samples))

        # cache so the timed run doesn't re-read Parquet mid-timing
        sub = sub.cache()
        n_rows = sub.count()  # materialize cache before timing

        t0 = time.time()
        n_genes = run_de(sub)
        elapsed = time.time() - t0

        sub.unpersist()
        print(f"frac={frac:.2f}  samples={k:4d}  rows={n_rows:>12,}  "
              f"genes_tested={n_genes:>6,}  seconds={elapsed:7.2f}")
        rows.append((frac, k, n_rows, n_genes, round(elapsed, 2)))

    # write results table
    result = spark.createDataFrame(
        rows, schema=["fraction", "n_samples", "n_rows", "n_genes_tested", "seconds"])
    (result.coalesce(1).write.mode("overwrite")
     .option("header", True).csv(OUT_CSV))

    print("=== benchmark complete ===")
    result.orderBy("fraction").show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
