"""
02_to_parquet.py

Reshape the TCGA-BRCA gene_sums matrix (genes-as-rows, samples-as-columns)
into a long-format Parquet table: one row per (gene, sample, count), with the
tumor/normal label carried alongside so downstream differential expression
needs no re-join.

Input  (HDFS):
    raw/tcga.gene_sums.BRCA.G026.gz   comment lines (##), then header
                                      (gene_id + 1256 sample UUIDs), then
                                      63,856 gene rows.
    raw/brca_labels.tsv               external_id <TAB> sample_type

Output (HDFS):
    parquet/brca_long/                columns: gene_id, sample_id, count,
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
OUT = f"{BASE}/parquet/brca_long"

# Sample types we keep for a clean binary contrast. Drop Metastatic (7) and NA (10).
KEEP_TYPES = ("Primary Tumor", "Solid Tissue Normal")
N_GENE_BUCKETS = 16  # partition count for the long table


def main():
    spark = (SparkSession.builder
             .appName("brca_to_parquet")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. Read the gene matrix as raw text, strip comment lines ----
    raw = spark.read.text(f"{RAW}/tcga.gene_sums.BRCA.G026.gz")
    raw = raw.filter(~F.col("value").startswith("##"))

    # The header is the one line starting with "gene_id".
    header_row = raw.filter(F.col("value").startswith("gene_id")).first()
    if header_row is None:
        print("ERROR: no header row found", file=sys.stderr)
        sys.exit(1)
    header = header_row["value"].split("\t")
    sample_ids = header[1:]              # drop the "gene_id" label
    n_samples = len(sample_ids)
    print(f"Header parsed: {n_samples} samples")

    # Body rows = everything that is not a comment and not the header.
    body = raw.filter(~F.col("value").startswith("gene_id"))

    # ---- 2. Split each line into gene_id + count array ----
    # split() on the whole line; first element is gene_id, rest are counts.
    parts = F.split(F.col("value"), "\t")
    with_cols = body.select(
        parts.getItem(0).alias("gene_id"),
        parts.alias("all_parts"),
    )

    # Build an array of counts (skip index 0 which is the gene_id).
    # slice() is 1-indexed in Spark SQL; start at 2 to skip gene_id.
    counts_arr = F.slice(F.col("all_parts"), 2, n_samples).alias("counts")
    with_counts = with_cols.select("gene_id", counts_arr)

    # ---- 3. Explode array -> long format ----
    # posexplode gives (position, value); position maps back to sample_ids.
    long_df = with_counts.select(
        "gene_id",
        F.posexplode(F.col("counts")).alias("pos", "count_str"),
    )

    # Map position -> sample_id via a small broadcast lookup DataFrame.
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

    # ---- 4. Attach labels, keep only the two contrast classes ----
    labels = (spark.read.option("sep", "\t").csv(f"{RAW}/brca_labels.tsv")
              .toDF("sample_id", "sample_type"))

    labeled = (long_df
               .join(F.broadcast(labels), on="sample_id", how="inner")
               .filter(F.col("sample_type").isin(*KEEP_TYPES)))

    # ---- 5. Add a gene bucket for partitioning ----
    # Deterministic hash of gene_id -> [0, N_GENE_BUCKETS). pmod keeps it non-negative.
    labeled = labeled.withColumn(
        "gene_bucket",
        F.pmod(F.hash(F.col("gene_id")), F.lit(N_GENE_BUCKETS)),
    )

    # ---- 6. Write Parquet, partitioned by gene_bucket ----
    (labeled.write
     .mode("overwrite")
     .partitionBy("gene_bucket")
     .parquet(OUT))

    # ---- 7. Report ----
    written = spark.read.parquet(OUT)
    total = written.count()
    n_genes = written.select("gene_id").distinct().count()
    n_kept_samples = written.select("sample_id").distinct().count()
    print("=== written long table ===")
    print(f"rows          : {total:,}")
    print(f"distinct genes: {n_genes:,}")
    print(f"kept samples  : {n_kept_samples:,}")
    print("=== label split (sample-level) ===")
    (written.select("sample_id", "sample_type").distinct()
            .groupBy("sample_type").count().show(truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
