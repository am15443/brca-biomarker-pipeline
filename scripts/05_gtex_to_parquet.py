"""
05_gtex_to_parquet.py

Reshape the GTEx healthy-breast gene_sums matrix into the same long-format
Parquet layout used for TCGA-BRCA, so the two can be joined for the
tissue-specificity / safety analysis.

Difference from the TCGA script (02): GTEx breast is uniformly healthy tissue,
so there is no per-sample tumor/normal label to join. Every sample gets a
constant label sample_type = "GTEx Healthy Breast".

Input  (HDFS):
    raw/gtex.gene_sums.BREAST.G026.gz   comment lines (##), header
                                        (gene_id + 482 GTEx sample IDs),
                                        then 63,856 gene rows.

Output (HDFS):
    parquet/gtex_breast_long/           columns: gene_id, sample_id, count,
                                        sample_type, gene_bucket
                                        partitioned by gene_bucket
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
RAW = f"{BASE}/raw"
OUT = f"{BASE}/parquet/gtex_breast_long"

LABEL = "GTEx Healthy Breast"
N_GENE_BUCKETS = 16  # match the TCGA table's partitioning


def main():
    spark = (SparkSession.builder
             .appName("gtex_breast_to_parquet")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. Read gene matrix as raw text, strip comment lines ----
    raw = spark.read.text(f"{RAW}/gtex.gene_sums.BREAST.G026.gz")
    raw = raw.filter(~F.col("value").startswith("##"))

    header_row = raw.filter(F.col("value").startswith("gene_id")).first()
    if header_row is None:
        print("ERROR: no header row found", file=sys.stderr)
        sys.exit(1)
    header = header_row["value"].split("\t")
    sample_ids = header[1:]
    n_samples = len(sample_ids)
    print(f"Header parsed: {n_samples} samples")

    body = raw.filter(~F.col("value").startswith("gene_id"))

    # ---- 2. Split each line into gene_id + count array ----
    parts = F.split(F.col("value"), "\t")
    with_cols = body.select(
        parts.getItem(0).alias("gene_id"),
        parts.alias("all_parts"),
    )
    counts_arr = F.slice(F.col("all_parts"), 2, n_samples).alias("counts")
    with_counts = with_cols.select("gene_id", counts_arr)

    # ---- 3. Explode array -> long format ----
    long_df = with_counts.select(
        "gene_id",
        F.posexplode(F.col("counts")).alias("pos", "count_str"),
    )

    lookup = spark.createDataFrame(
        [(i, sid) for i, sid in enumerate(sample_ids)],
        schema=T.StructType([
            T.StructField("pos", T.IntegerType(), False),
            T.StructField("sample_id", T.StringType(), False),
        ]),
    )
    long_df = (long_df
               .join(F.broadcast(lookup), on="pos", how="inner")
               .select(
                   "gene_id",
                   "sample_id",
                   F.col("count_str").cast(T.LongType()).alias("count"),
               ))

    # ---- 4. Constant healthy-tissue label (no metadata join needed) ----
    labeled = long_df.withColumn("sample_type", F.lit(LABEL))

    # ---- 5. Gene bucket for partitioning (same scheme as TCGA table) ----
    labeled = labeled.withColumn(
        "gene_bucket",
        F.pmod(F.hash(F.col("gene_id")), F.lit(N_GENE_BUCKETS)),
    )

    # ---- 6. Write Parquet ----
    (labeled.write
     .mode("overwrite")
     .partitionBy("gene_bucket")
     .parquet(OUT))

    # ---- 7. Report ----
    written = spark.read.parquet(OUT)
    total = written.count()
    n_genes = written.select("gene_id").distinct().count()
    n_kept_samples = written.select("sample_id").distinct().count()
    print("=== written GTEx breast long table ===")
    print(f"rows          : {total:,}")
    print(f"distinct genes: {n_genes:,}")
    print(f"samples       : {n_kept_samples:,}")
    print("=== sample of rows ===")
    written.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
