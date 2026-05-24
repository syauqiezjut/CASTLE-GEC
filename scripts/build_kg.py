"""
build_kg.py
Build CASTLE Knowledge Graph from IGED dataset
================================================
Run this BEFORE training to construct the knowledge graph.

Usage (with DeepSeek API key, recommended):
  python scripts/build_kg.py \\
    --csv data/IGED_stratified_dataset.csv \\
    --output data/castle_kg.pkl \\
    --deepseek_key sk-xxxxxxxx \\
    --max_neural 50000

Usage (without API, rule-based only):
  python scripts/build_kg.py \\
    --csv data/IGED_stratified_dataset.csv \\
    --output data/castle_kg.pkl

For Kaggle/Colab (fast test with 10K samples):
  python scripts/build_kg.py \\
    --csv data/IGED_stratified_dataset.csv \\
    --output data/castle_kg.pkl \\
    --max_samples 10000
"""

import sys
import os
import logging
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from knowledge_graph import CASTLEKnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("build_kg.log"),
    ],
)
logger = logging.getLogger(__name__)


def normalize_sample(row) -> dict:
    """Normalize column names from IGED CSV."""
    # Try common column names
    source = (row.get("source") or row.get("src") or
              row.get("incorrect") or row.get("input") or row.get("error") or "")
    target = (row.get("target") or row.get("tgt") or
              row.get("correct") or row.get("output") or row.get("correction") or "")
    category = str(
        row.get("category") or row.get("error_type") or
        row.get("cat") or row.get("type") or "syntax"
    ).lower().strip()
    return {"source": str(source), "target": str(target), "category": category}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to IGED_stratified_dataset.csv")
    parser.add_argument("--output", default="data/castle_kg.pkl")
    parser.add_argument("--deepseek_key", default=None,
                        help="DeepSeek API key for neural stage (40%% of relations)")
    parser.add_argument("--max_neural", type=int, default=None,
                        help="Limit neural API calls (e.g. 50000 to control cost)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use only first N samples (for testing)")
    parser.add_argument("--export_json", default=None,
                        help="Also export as JSON for inspection")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="DeepSeek API batch size (paper: k=10)")
    parser.add_argument("--cache_dir", default="data/kg_cache",
                        help="Cache DeepSeek results for resumability")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading dataset from {args.csv}")
    df = pd.read_csv(args.csv)
    logger.info(f"Total rows: {len(df):,}")

    if args.max_samples:
        df = df.head(args.max_samples)
        logger.info(f"Using first {args.max_samples:,} samples")

    samples = [normalize_sample(row) for row in df.to_dict("records")]
    # Remove empty samples
    samples = [s for s in samples if s["source"] and s["target"]]
    logger.info(f"Valid samples: {len(samples):,}")

    # Build KG
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    kg = CASTLEKnowledgeGraph()
    kg.build(
        samples,
        deepseek_api_key=args.deepseek_key,
        batch_size=args.batch_size,
        max_neural_samples=args.max_neural,
        cache_dir=args.cache_dir,
    )

    kg.save(args.output)

    if args.export_json:
        kg.export_json(args.export_json)
        logger.info(f"JSON exported to {args.export_json}")

    # Final summary
    G = kg.graph
    logger.info("\n" + "=" * 50)
    logger.info("Knowledge Graph Summary:")
    logger.info(f"  Nodes:     {G.number_of_nodes():,}")
    logger.info(f"  Edges:     {G.number_of_edges():,}")
    logger.info(f"  Saved to:  {args.output}")
    logger.info("\nPaper reference (Table 5):")
    logger.info("  Nodes: 46,784 | Edges: 233,286")
    logger.info("=" * 50)
