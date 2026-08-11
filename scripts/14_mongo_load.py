#!/usr/bin/env python3
"""
14_mongo_load.py

Serving layer, load step: read the finished per-gene target-score results and
write them to MongoDB Atlas as one document per gene. This is the "results
store" -- analysis is done; MongoDB serves finished results for fast lookup.

Each document:
    {
      gene_id, symbol,
      efficacy: { log2fc, fdr },
      safety:   { healthy_mean_logcpm },
      target_score,
      verdict            # "candidate" | "danger" | "not_significant"
    }

Connection string is read from the MONGO_URI environment variable so the
password never lands in source control. Set it before running:
    export MONGO_URI='mongodb+srv://am15443_db_user:PASSWORD@cluster0.pxl8wvh.mongodb.net/?appName=Cluster0'

Reads:  results/brca_target_score_csv  (from HDFS, via hdfs dfs -cat)
Writes: MongoDB  biomarker.genes  collection
"""
import os
import sys
import io
import subprocess
import pandas as pd

HDFS_CSV = "/user/am15443_nyu_edu/biomarker/results/brca_target_score_csv/part-*.csv"
DB_NAME = "biomarker"
COLLECTION = "genes"

FDR_MAX = 0.05
FC_MIN = 1.0
SAFE_MAX = 2.01   # p25 healthy-expression threshold from the target-score step


def hdfs_cat(glob):
    out = subprocess.run(["hdfs", "dfs", "-cat", glob],
                         capture_output=True, text=True, check=True)
    return out.stdout


def verdict(row):
    if row["fdr"] >= FDR_MAX or row["log2fc"] <= FC_MIN:
        return "not_significant"
    if row["healthy_mean_logcpm"] <= SAFE_MAX:
        return "candidate"
    return "danger"


def main():
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("ERROR: set MONGO_URI environment variable first", file=sys.stderr)
        sys.exit(1)

    # pymongo import here so the error is clear if it's missing
    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pymongo not installed. Run: pip install --user pymongo",
              file=sys.stderr)
        sys.exit(1)

    # ---- read results from HDFS ----
    df = pd.read_csv(io.StringIO(hdfs_cat(HDFS_CSV)))
    print(f"loaded {len(df):,} genes from HDFS results")

    # ---- build one document per gene ----
    docs = []
    for _, r in df.iterrows():
        docs.append({
            "gene_id": r["gene_id"],
            "symbol": r["symbol"],
            "efficacy": {
                "log2fc": float(r["log2fc"]),
                "fdr": float(r["fdr"]),
            },
            "safety": {
                "healthy_mean_logcpm": float(r["healthy_mean_logcpm"]),
            },
            "target_score": float(r["target_score"]),
            "verdict": verdict(r),
        })

    # ---- connect + insert ----
    client = MongoClient(uri)
    coll = client[DB_NAME][COLLECTION]

    coll.delete_many({})                     # idempotent: clear then reload
    coll.insert_many(docs)
    coll.create_index("symbol")              # fast lookup by gene symbol
    coll.create_index("target_score")        # fast ranked queries

    n = coll.count_documents({})
    print(f"inserted {n:,} gene documents into {DB_NAME}.{COLLECTION}")

    # ---- quick sanity: show the top candidate ----
    top = coll.find_one({"verdict": "candidate"}, sort=[("target_score", -1)])
    print("top candidate document:")
    print(f"  {top['symbol']}  score={top['target_score']:.2f}  "
          f"log2fc={top['efficacy']['log2fc']:.2f}  "
          f"healthy={top['safety']['healthy_mean_logcpm']:.2f}")

    # counts by verdict
    print("verdict breakdown:")
    for v in ["candidate", "danger", "not_significant"]:
        print(f"  {v}: {coll.count_documents({'verdict': v}):,}")

    client.close()


if __name__ == "__main__":
    main()
