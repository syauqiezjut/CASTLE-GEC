"""
prune_kg.py — Prune full KG to match paper's ~46K nodes / ~233K edges

Strategy:
  1. Score every node: score = sum of all incident edge weights (in + out)
     → Nodes most central to GEC error/correction patterns score highest
  2. Keep top --top_nodes nodes by score (default: 50_000)
  3. Remove edges with weight < min_edge_weight
  4. Cap outgoing edges per node (keep highest-weight)
  5. Remove isolated nodes

Expected output: ~46K nodes, ~233K edges (matching ESWA 299 paper)

Usage:
    python scripts/prune_kg.py \
        --input  data/castle_kg.pkl \
        --output data/castle_kg_pruned.pkl \
        --top_nodes 50000 \
        --min_edge_weight 0.3 \
        --max_edges_per_node 10

Author: Syauqie (CASTLE paper)
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict


def prune_kg(input_path: str, output_path: str,
             top_nodes: int = 50_000,
             min_edge_weight: float = 0.3,
             max_edges_per_node: int = 10):

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"[1/6] Loading KG from {input_path} ...", flush=True)
    with open(input_path, "rb") as f:
        kg = pickle.load(f)

    G             = kg["graph"]
    error_nodes   = kg.get("error_nodes",   set())
    correct_nodes = kg.get("correct_nodes", set())
    stem_nodes    = kg.get("stem_nodes",    set())

    print(f"      Full graph    : {G.number_of_nodes():>10,} nodes  {G.number_of_edges():>12,} edges")
    print(f"      error_nodes   : {len(error_nodes):>10,}")
    print(f"      correct_nodes : {len(correct_nodes):>10,}")
    print(f"      stem_nodes    : {len(stem_nodes):>10,}")

    # ── 2. Score every node ──────────────────────────────────────────────────
    print("[2/6] Scoring nodes by total incident edge weight ...", flush=True)
    scores = defaultdict(float)
    total_edges = G.number_of_edges()
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        if i % 500_000 == 0:
            print(f"      Processed {i:,}/{total_edges:,} edges ...", flush=True)
        w = d.get("weight", 0.0)
        scores[u] += w
        scores[v] += w

    # ── 3. Priority boost: paper's categorical nodes score 10× ──────────────
    #   This ensures error/correct/stem words from IGED are preferred
    paper_nodes = error_nodes | correct_nodes | stem_nodes
    for node in paper_nodes:
        if node in scores:
            scores[node] *= 10.0

    # ── 4. Keep top-N nodes ──────────────────────────────────────────────────
    print(f"[3/6] Selecting top {top_nodes:,} nodes ...", flush=True)
    sorted_nodes = sorted(scores, key=lambda n: -scores[n])
    kept_nodes   = set(sorted_nodes[:top_nodes])

    # Stats: how many from each category survived
    kept_error   = len(error_nodes   & kept_nodes)
    kept_correct = len(correct_nodes & kept_nodes)
    kept_stem    = len(stem_nodes    & kept_nodes)
    print(f"      Kept error_nodes   : {kept_error:,}")
    print(f"      Kept correct_nodes : {kept_correct:,}")
    print(f"      Kept stem_nodes    : {kept_stem:,}")

    sub = G.subgraph(kept_nodes).copy()
    print(f"      After node filter  : {sub.number_of_nodes():,} nodes  {sub.number_of_edges():,} edges")

    # ── 5. Edge filters ──────────────────────────────────────────────────────
    print(f"[4/6] Filtering edges (weight >= {min_edge_weight}) ...", flush=True)
    low_w = [(u, v) for u, v, d in sub.edges(data=True)
             if d.get("weight", 0.0) < min_edge_weight]
    sub.remove_edges_from(low_w)
    print(f"      After weight filter : {sub.number_of_edges():,} edges")

    if max_edges_per_node > 0:
        print(f"[5/6] Capping at {max_edges_per_node} outgoing edges per node ...", flush=True)
        to_remove = []
        for node in sub.nodes():
            out_e = sorted(sub.out_edges(node, data=True),
                           key=lambda x: -x[2].get("weight", 0.0))
            if len(out_e) > max_edges_per_node:
                to_remove.extend([(u, v) for u, v, _ in out_e[max_edges_per_node:]])
        sub.remove_edges_from(to_remove)
        print(f"      After edge cap      : {sub.number_of_edges():,} edges")

    isolated = [n for n in list(sub.nodes()) if sub.degree(n) == 0]
    sub.remove_nodes_from(isolated)
    print(f"      Isolated removed    : {len(isolated):,}")

    # ── 6. Build fast lookup & save ──────────────────────────────────────────
    print(f"[6/6] Building fast lookup & saving ...", flush=True)
    fast_lookup = {}
    for node in sub.nodes():
        edges = []
        for src, dst, data in sub.out_edges(node, data=True):
            edges.append((src, dst, data.get("relation", ""), data.get("weight", 0.0)))
        for src, dst, data in sub.in_edges(node, data=True):
            edges.append((src, dst, data.get("relation", ""), data.get("weight", 0.0)))
        edges.sort(key=lambda x: -x[3])
        fast_lookup[node] = edges[:20]

    final_nodes = set(sub.nodes())
    pruned = {
        "graph":         sub,
        "vocab":         kg.get("vocab", set()) & final_nodes,
        "error_nodes":   error_nodes   & final_nodes,
        "correct_nodes": correct_nodes & final_nodes,
        "stem_nodes":    stem_nodes    & final_nodes,
        "_fast_lookup":  fast_lookup,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pruned, f, protocol=4)

    size_mb = Path(output_path).stat().st_size / 1e6
    print()
    print("=" * 60)
    print("  PRUNING COMPLETE")
    print(f"  Nodes  : {sub.number_of_nodes():,}   (paper: ~46,784)")
    print(f"  Edges  : {sub.number_of_edges():,}   (paper: ~233,000)")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"  Output : {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",              default="data/castle_kg.pkl")
    parser.add_argument("--output",             default="data/castle_kg_pruned.pkl")
    parser.add_argument("--top_nodes",          type=int,   default=50_000)
    parser.add_argument("--min_edge_weight",    type=float, default=0.3)
    parser.add_argument("--max_edges_per_node", type=int,   default=10)
    args = parser.parse_args()

    prune_kg(args.input, args.output,
             args.top_nodes, args.min_edge_weight, args.max_edges_per_node)


if __name__ == "__main__":
    main()
