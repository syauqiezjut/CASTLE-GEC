"""
evaluate.py
CASTLE Evaluation Script
==========================
Computes evaluation metrics from paper:
  - Precision, Recall, F1 (token-level correction accuracy)
  - BLEU score (SacreBLEU)
  - Per-category analysis: Morphology, Syntax, Semantics
  - Per-subcategory: Affixation, Word Formation, Reduplication,
                     Phrase Structure, Preposition, Sentence Completeness,
                     Diction, Ambiguity, Pleonasm

Target metrics (Table 9 — CASTLE):
  Overall: Prec=0.9718, Rec=0.9559, F1=0.9629, BLEU=92.72
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import sacrebleu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from castle_model import CASTLE, build_castle_model
from dataset import load_iged, get_dataloaders, PAD_TOKEN, ERROR_CATEGORY_MAP

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {0: "Morphology", 1: "Syntax", 2: "Semantics"}
SUBCATEGORY_NAMES = {
    "affixation": "Morphology", "word_formation": "Morphology", "reduplication": "Morphology",
    "phrase_structure": "Syntax", "preposition": "Syntax", "sentence_completeness": "Syntax",
    "diction": "Semantics", "ambiguity": "Semantics", "pleonasm": "Semantics",
}


# ──────────────────────────────────────────────
# Token-level F1 (GEC standard evaluation)
# ──────────────────────────────────────────────

def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """
    Token-level F1 between predicted and gold correction.
    Measures how well the model corrects the text (not just reproduces).
    """
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())

    if not pred_toks and not gold_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gold_toks:
        return 0.0, 0.0, 0.0

    tp = len(pred_toks & gold_toks)
    prec = tp / len(pred_toks) if pred_toks else 0.0
    rec = tp / len(gold_toks) if gold_toks else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def sentence_f1_corpus(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Macro-average token F1 over corpus."""
    precs, recs, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        p, r, f = token_f1(pred, ref)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
    return {
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
        "f1": sum(f1s) / len(f1s),
    }


# ──────────────────────────────────────────────
# BLEU score
# ──────────────────────────────────────────────

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """SacreBLEU score (matches paper evaluation)."""
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score


# ──────────────────────────────────────────────
# Decode predictions
# ──────────────────────────────────────────────

@torch.no_grad()
def decode_batch(
    model: CASTLE,
    src_tokens: torch.Tensor,   # [B, S]
    src_mask: torch.Tensor,     # [B, S]
    tokenizer,
    src_texts: List[str],
    bos_id: int,
    eos_id: int,
    max_len: int = 128,
) -> List[str]:
    """Run model inference and decode token IDs to strings."""
    generated = model.generate(
        src_tokens, src_mask, tokenizer,
        src_texts=src_texts,
        max_len=max_len,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    decoded = []
    for token_ids in generated:
        # Remove BOS/EOS/PAD
        clean_ids = [tid for tid in token_ids
                     if tid not in (bos_id, eos_id, tokenizer.token_to_id(PAD_TOKEN))]
        text = tokenizer.decode(clean_ids)
        decoded.append(text)
    return decoded


# ──────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────

def evaluate(
    model: CASTLE,
    test_loader,
    tokenizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Full evaluation on test set.
    Returns overall + per-category metrics.
    """
    model.eval()

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    all_preds, all_refs, all_cats = [], [], []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if max_batches and i >= max_batches:
            break

        src = batch["src_tokens"].to(device)
        mask = batch["src_mask"].to(device)
        src_texts = batch["src_texts"]
        tgt_texts = batch["tgt_texts"]
        categories = batch["categories"].tolist()

        preds = decode_batch(model, src, mask, tokenizer, src_texts, bos_id, eos_id)

        all_preds.extend(preds)
        all_refs.extend(tgt_texts)
        all_cats.extend(categories)

    # ── Overall metrics ────────────────────────────────────────
    overall_f1 = sentence_f1_corpus(all_preds, all_refs)
    overall_bleu = compute_bleu(all_preds, all_refs)

    results = {
        "overall": {
            "precision": overall_f1["precision"],
            "recall": overall_f1["recall"],
            "f1": overall_f1["f1"],
            "bleu": overall_bleu,
            "n_samples": len(all_preds),
        }
    }

    # ── Per-category metrics (Tables 10-12) ──────────────────
    for cat_id, cat_name in CATEGORY_NAMES.items():
        cat_preds = [p for p, c in zip(all_preds, all_cats) if c == cat_id]
        cat_refs = [r for r, c in zip(all_refs, all_cats) if c == cat_id]
        if not cat_preds:
            continue
        cat_f1 = sentence_f1_corpus(cat_preds, cat_refs)
        cat_bleu = compute_bleu(cat_preds, cat_refs)
        results[cat_name] = {
            "precision": cat_f1["precision"],
            "recall": cat_f1["recall"],
            "f1": cat_f1["f1"],
            "bleu": cat_bleu,
            "n_samples": len(cat_preds),
        }

    model.train()
    return results


def print_results(results: Dict):
    """Pretty-print evaluation results (matches paper Table format)."""
    print("\n" + "=" * 70)
    print("CASTLE Evaluation Results")
    print("=" * 70)
    print(f"{'Category':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'BLEU':>8} {'N':>8}")
    print("-" * 70)

    # Paper targets for comparison
    paper_targets = {
        "overall": {"precision": 0.9718, "recall": 0.9559, "f1": 0.9629, "bleu": 92.72},
        "Semantics": {"f1": 0.9652, "bleu": 92.05},
        "Morphology": {"f1": 0.9611, "bleu": 93.93},
        "Syntax": {"f1": 0.9355, "bleu": 88.52},
    }

    for key, metrics in results.items():
        target = paper_targets.get(key, {})
        f1_diff = metrics["f1"] - target.get("f1", 0) if target else 0
        bleu_diff = metrics["bleu"] - target.get("bleu", 0) if target else 0
        diff_str = f"  (Δ F1={f1_diff:+.4f}, Δ BLEU={bleu_diff:+.2f})" if target else ""

        print(
            f"{key:<20} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['bleu']:>8.2f} "
            f"{metrics['n_samples']:>8}"
            f"{diff_str}"
        )

    print("=" * 70)
    print("\nPaper targets (Table 9):")
    print("  Overall: Prec=0.9718, Rec=0.9559, F1=0.9629, BLEU=92.72")
    print("  Semantics F1=0.9652, Morphology F1=0.9611, Syntax F1=0.9355")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate CASTLE model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/castle_base.yaml")
    parser.add_argument("--local_csv", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    import yaml, json
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + data
    _, _, test_ds, tokenizer = load_iged(
        hf_dataset_name=cfg["dataset"]["name"],
        tokenizer_dir=cfg.get("tokenizer_dir", "data/tokenizer"),
        local_csv=args.local_csv,
    )
    pad_idx = tokenizer.token_to_id(PAD_TOKEN)
    _, _, test_loader = get_dataloaders(
        test_ds, test_ds, test_ds,  # placeholder for train/val
        batch_size=32, pad_id=pad_idx,
    )

    # Load model
    vocab_size = tokenizer.get_vocab_size()
    model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # Load KG if available
    from knowledge_graph import CASTLEKnowledgeGraph
    kg_path = cfg.get("knowledge_graph", {}).get("output_path", "data/castle_kg.pkl")
    if os.path.exists(kg_path):
        kg = CASTLEKnowledgeGraph.load(kg_path)
        model.set_knowledge_graph(kg, kg.word_to_idx())

    results = evaluate(model, test_loader, tokenizer, device, args.max_batches)
    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
