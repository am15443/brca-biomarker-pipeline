#!/usr/bin/env python3
"""
15_mongo_query.py  [GENE_SYMBOL]

Serving layer, query step: demonstrate fast single-document lookup from the
MongoDB results store. Given a gene symbol, fetch its complete profile in one
indexed read -- no joins, no recomputation. Also shows ranked queries (top
candidates, danger genes) that the indexed store answers instantly.

    python3 15_mongo_query.py COL10A1     # look up one gene
    python3 15_mongo_query.py             # demo: top candidates + a danger gene

Reads MONGO_URI from the environment (same as the loader).
"""
import os
import sys


def main():
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("ERROR: set MONGO_URI environment variable first", file=sys.stderr)
        sys.exit(1)
    from pymongo import MongoClient

    client = MongoClient(uri)
    coll = client["biomarker"]["genes"]

    # ---- single-gene lookup (the core serving pattern) ----
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        doc = coll.find_one({"symbol": symbol})
        if not doc:
            print(f"gene '{symbol}' not found")
            client.close()
            return
        print(f"=== {symbol} profile (single indexed lookup) ===")
        print(f"  gene_id      : {doc['gene_id']}")
        print(f"  verdict      : {doc['verdict']}")
        print(f"  target_score : {doc['target_score']:.3f}")
        print(f"  efficacy     : log2fc={doc['efficacy']['log2fc']:.3f}  "
              f"fdr={doc['efficacy']['fdr']:.2e}")
        print(f"  safety       : healthy_logcpm="
              f"{doc['safety']['healthy_mean_logcpm']:.3f}")
        client.close()
        return

    # ---- demo mode: ranked queries the indexed store answers instantly ----
    print("=== TOP 10 CANDIDATE TARGETS (up in tumor, quiet in healthy) ===")
    for d in coll.find({"verdict": "candidate"}).sort("target_score", -1).limit(10):
        print(f"  {d['symbol']:10} score={d['target_score']:6.2f}  "
              f"log2fc={d['efficacy']['log2fc']:5.2f}  "
              f"healthy={d['safety']['healthy_mean_logcpm']:5.2f}")

    print("\n=== TOP 5 DANGER GENES (up in tumor BUT high in healthy) ===")
    for d in coll.find({"verdict": "danger"}).sort(
            "safety.healthy_mean_logcpm", -1).limit(5):
        print(f"  {d['symbol']:10} score={d['target_score']:6.2f}  "
              f"log2fc={d['efficacy']['log2fc']:5.2f}  "
              f"healthy={d['safety']['healthy_mean_logcpm']:5.2f}")

    print("\n=== example single lookups ===")
    for sym in ["COL10A1", "COL1A1", "MMP11"]:
        d = coll.find_one({"symbol": sym})
        if d:
            print(f"  {sym:10} -> {d['verdict']:15} (score {d['target_score']:.2f})")

    print("\n=== collection stats ===")
    print(f"  total genes served : {coll.count_documents({}):,}")
    print(f"  candidates         : {coll.count_documents({'verdict': 'candidate'}):,}")
    print(f"  danger             : {coll.count_documents({'verdict': 'danger'}):,}")

    client.close()


if __name__ == "__main__":
    main()
