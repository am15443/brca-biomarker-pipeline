"""
Load the TCGA-BRCA gene count matrix, transpose to samples-as-rows,
join tumor/normal labels, and report basic dimensions.
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = (SparkSession.builder
         .appName("brca_load")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

RAW = "hdfs:///user/am15443_nyu_edu/biomarker/raw"

# The gene_sums file has 2 comment lines (##...) then a header row.
# Spark's CSV reader can't skip leading comment lines cleanly, so we
# read as text, drop comment lines, and parse manually.
raw = spark.read.text(f"{RAW}/tcga.gene_sums.BRCA.G026.gz")
raw = raw.filter(~F.col("value").startswith("##"))

# First surviving row is the header (gene_id + sample UUIDs)
header = raw.first()["value"].split("\t")
sample_ids = header[1:]
print(f"Genes file header parsed: {len(sample_ids)} samples")

# Parse the remaining rows into (gene_id, [counts...])
rows = raw.filter(~F.col("value").startswith("gene_id"))
print(f"Gene rows: {rows.count()}")

# Load labels
labels = (spark.read
          .option("sep", "\t")
          .csv(f"{RAW}/brca_labels.tsv")
          .toDF("external_id", "sample_type"))
print("=== label distribution ===")
labels.groupBy("sample_type").count().show(truncate=False)

spark.stop()
