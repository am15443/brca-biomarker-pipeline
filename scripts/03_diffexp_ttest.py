"""
03_diffexp_ttest.py

Differential expression for TCGA-BRCA: tumor vs. normal, all genes at once,
using a Welch's t-test on log2-CPM computed with pure Spark aggregations.

Pipeline:
  1. Read the long-format Parquet table (gene_id, sample_id, count, sample_type).
  2. Compute per-sample library size (total counts) -> CPM normalization.
  3. log2(CPM + 1) transform.
  4. Filter low-expression genes (expressed in too few samples).
  5. Per gene, compute group means/variances for tumor and normal, then
     Welch's t-statistic, degrees of freedom, and log2 fold change.
  6. Two-sided p-value from the t distribution; Benjamini-Hochberg FDR.
  7. Write a ranked results table to Parquet + a small CSV for plotting.

Design notes:
  - Everything through step 5 is Spark-native groupBy aggregation: all ~63,856
    genes are tested in a single distributed pass, no per-gene Python loop.
  - The p-value (step 6) needs the t-CDF, which Spark SQL lacks, so it is
    applied with a small pandas UDF over the per-gene summary table (only
    ~n_genes rows, tiny) -- not over the 79M-row long table.
"""
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
IN = f"{BASE}/parquet/brca_long"
OUT_PARQUET = f"{BASE}/parquet/brca_diffexp"
OUT_CSV = f"{BASE}/results/brca_diffexp_csv"

TUMOR = "Primary Tumor"
NORMAL = "Solid Tissue Normal"

# Gene filter: keep genes with CPM > MIN_CPM in at least MIN_FRAC of samples.
MIN_CPM = 1.0
MIN_FRAC = 0.20


def main():
    spark = (SparkSession.builder
             .appName("brca_diffexp_ttest")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    long_df = spark.read.parquet(IN)

    # ---- 1. Per-sample library size -> CPM ----
    lib = (long_df.groupBy("sample_id")
           .agg(F.sum("count").alias("lib_size")))

    df = (long_df.join(F.broadcast(lib), on="sample_id", how="inner")
          .withColumn("cpm", F.col("count") / F.col("lib_size") * F.lit(1_000_000.0))
          .withColumn("logcpm", F.log2(F.col("cpm") + F.lit(1.0))))

    # ---- 2. Gene filter: expressed (CPM > MIN_CPM) in >= MIN_FRAC of samples ----
    n_samples = long_df.select("sample_id").distinct().count()
    min_samples = int(round(MIN_FRAC * n_samples))
    print(f"Total samples: {n_samples}; gene must exceed CPM>{MIN_CPM} "
          f"in >= {min_samples} samples")

    expressed = (df.withColumn("is_expr", (F.col("cpm") > F.lit(MIN_CPM)).cast("int"))
                 .groupBy("gene_id")
                 .agg(F.sum("is_expr").alias("n_expr")))
    keep_genes = expressed.filter(F.col("n_expr") >= F.lit(min_samples)).select("gene_id")
    n_keep = keep_genes.count()
    print(f"Genes passing filter: {n_keep:,}")

    df = df.join(F.broadcast(keep_genes), on="gene_id", how="inner")

    # ---- 3. Per-gene, per-group summary statistics ----
    # Welch's t-test needs mean, variance, and n for each group.
    grp = (df.groupBy("gene_id", "sample_type")
           .agg(F.mean("logcpm").alias("mean"),
                F.variance("logcpm").alias("var"),
                F.count("logcpm").alias("n")))

    tumor = (grp.filter(F.col("sample_type") == TUMOR)
             .select("gene_id",
                     F.col("mean").alias("mean_t"),
                     F.col("var").alias("var_t"),
                     F.col("n").alias("n_t")))
    normal = (grp.filter(F.col("sample_type") == NORMAL)
              .select("gene_id",
                      F.col("mean").alias("mean_n"),
                      F.col("var").alias("var_n"),
                      F.col("n").alias("n_n")))

    stats = tumor.join(normal, on="gene_id", how="inner")

    # ---- 4. Welch's t-statistic, Welch-Satterthwaite df, log2FC ----
    # se2 = var_t/n_t + var_n/n_n
    stats = stats.withColumn("se2",
                             F.col("var_t") / F.col("n_t")
                             + F.col("var_n") / F.col("n_n"))
    # Guard against zero-variance genes producing division by zero.
    stats = stats.filter(F.col("se2") > F.lit(0.0))

    stats = stats.withColumn("t_stat",
                             (F.col("mean_t") - F.col("mean_n")) / F.sqrt(F.col("se2")))

    # Welch-Satterthwaite degrees of freedom
    num = F.pow(F.col("se2"), F.lit(2.0))
    den = (F.pow(F.col("var_t") / F.col("n_t"), F.lit(2.0)) / (F.col("n_t") - F.lit(1.0))
           + F.pow(F.col("var_n") / F.col("n_n"), F.lit(2.0)) / (F.col("n_n") - F.lit(1.0)))
    stats = stats.withColumn("df", num / den)

    # log2 fold change: tumor mean minus normal mean (already in log2 space)
    stats = stats.withColumn("log2fc", F.col("mean_t") - F.col("mean_n"))

    # ---- 5. p-value from t distribution (pandas UDF over the small stats table) ----
    # This table has ~n_keep rows (thousands), so a pandas UDF is cheap here.
    @F.pandas_udf(T.DoubleType())
    def t_sf_pvalue(t_stat: pd.Series, df_: pd.Series) -> pd.Series:
        from scipy import stats as ss
        # two-sided p-value: 2 * survival function of |t|
        return pd.Series(2.0 * ss.t.sf(t_stat.abs().values, df_.values))

    stats = stats.withColumn("pvalue", t_sf_pvalue(F.col("t_stat"), F.col("df")))

    # ---- 6. Benjamini-Hochberg FDR ----
    # BH: sort by p ascending, rank r, adjusted = p * m / r, then enforce monotonicity.
    m = stats.count()
    w_rank = Window.orderBy(F.col("pvalue").asc())
    ranked = stats.withColumn("rank", F.row_number().over(w_rank))
    ranked = ranked.withColumn("bh_raw",
                               F.col("pvalue") * F.lit(m) / F.col("rank"))
    # Enforce monotonic non-decreasing from the largest p downward via cumulative min
    # over descending rank. Implemented as running min of bh_raw ordered by rank desc.
    w_cummin = Window.orderBy(F.col("rank").desc()).rowsBetween(
        Window.unboundedPreceding, Window.currentRow)
    ranked = ranked.withColumn("fdr",
                               F.least(F.lit(1.0), F.min("bh_raw").over(w_cummin)))

    results = ranked.select(
        "gene_id", "log2fc", "t_stat", "df", "pvalue", "fdr",
        "mean_t", "mean_n", "n_t", "n_n", "rank",
    ).orderBy(F.col("pvalue").asc())

    # ---- 7. Write outputs ----
    (results.write.mode("overwrite").parquet(OUT_PARQUET))
    (results.coalesce(1).write.mode("overwrite")
     .option("header", True).csv(OUT_CSV))

    # ---- 8. Report ----
    print("=== differential expression complete ===")
    print(f"genes tested: {m:,}")
    n_sig = results.filter(F.col("fdr") < F.lit(0.05)).count()
    print(f"significant at FDR<0.05: {n_sig:,}")
    print("=== top 20 by p-value ===")
    (results.select("gene_id", "log2fc", "t_stat", "pvalue", "fdr")
            .limit(20).show(truncate=False))
    print("=== strongest UP in tumor (log2fc desc, among FDR<0.05) ===")
    (results.filter(F.col("fdr") < F.lit(0.05))
            .orderBy(F.col("log2fc").desc())
            .select("gene_id", "log2fc", "pvalue", "fdr")
            .limit(10).show(truncate=False))
    print("=== strongest DOWN in tumor (log2fc asc, among FDR<0.05) ===")
    (results.filter(F.col("fdr") < F.lit(0.05))
            .orderBy(F.col("log2fc").asc())
            .select("gene_id", "log2fc", "pvalue", "fdr")
            .limit(10).show(truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
