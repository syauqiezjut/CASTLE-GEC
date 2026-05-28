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
from dataset import load_iged, get_dataloaders, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, ERROR_CATEGORY_MAP

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


def wordpiece_decode(token_str: str) -> str:
    """
    Join WordPiece ## continuation tokens back to surface form.
    e.g. "ber ##jalan ke ##luar" → "berjalan keluar"
    Also normalise common punctuation spacing artifacts.
    """
    import re
    text = token_str.replace(' ##', '')
    text = re.sub(r'\s+([.,!?;:»\)])', r'\1', text)
    text = re.sub(r'([\(«])\s+', r'\1', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


# ──────────────────────────────────────────────
# Decode predictions
# ──────────────────────────────────────────────

def ids_to_token_string(token_ids: List[int], tokenizer, bos_id: int, eos_id: int) -> str:
    """
    Convert token IDs → space-separated token string (WordPiece level).
    e.g. [101, 1234, 119, 102] → "saya pergi ##."
    This matches the format of the original Fairseq --remove-bpe evaluation,
    where BLEU is computed at the token level, not surface form.
    """
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    skip = {bos_id, eos_id, pad_id}
    tokens = []
    for tid in token_ids:
        if tid in skip:
            continue
        tok = tokenizer.id_to_token(tid)
        if tok is not None:
            tokens.append(tok)
    return ' '.join(tokens)


def tokenize_to_string(text: str, tokenizer) -> str:
    """
    Tokenize surface-form reference text → space-separated token string.
    Strips BOS/EOS tokens added by the template processor.
    Matches format produced by ids_to_token_string().
    """
    enc = tokenizer.encode(text)
    tokens = [t for t in enc.tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
    return ' '.join(tokens)


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
    eval_mode: str = 'tokenized',
    length_penalty: float = 1.0,
    max_len_a: float = 1.2,
    max_len_b: int = 10,
) -> List[str]:
    """
    Run model inference and decode token IDs to strings.

    eval_mode='tokenized' (default / Option A):
        Returns space-separated WordPiece token strings.
        Reference must also be tokenized before BLEU/F1 — avoids
        decode artifacts like "Jakarta ." vs "Jakarta.".
    eval_mode='surface':
        Returns fully decoded surface form (original behaviour).
        Use only if references are also in surface form.
    """
    generated = model.generate(
        src_tokens, src_mask, tokenizer,
        src_texts=src_texts,
        max_len=max_len,
        bos_id=bos_id,
        eos_id=eos_id,
        length_penalty=length_penalty,
        max_len_a=max_len_a,
        max_len_b=max_len_b,
    )
    decoded = []
    for token_ids in generated:
        if eval_mode == 'tokenized':
            text = ids_to_token_string(token_ids, tokenizer, bos_id, eos_id)
        else:
            # surface mode: original behaviour
            pad_id = tokenizer.token_to_id(PAD_TOKEN)
            clean_ids = [tid for tid in token_ids
                         if tid not in (bos_id, eos_id, pad_id)]
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
    eval_mode: str = 'tokenized',
    max_len: int = 128,
    length_penalty: float = 1.0,
    max_len_a: float = 1.2,
    max_len_b: int = 10,
    max_src_len: Optional[int] = None,
) -> Dict:
    """
    Full evaluation on test set.
    Returns overall + per-category metrics.

    eval_mode='tokenized' (default / Option A):
        Both predictions and references are evaluated at WordPiece token level.
        Matches the original Fairseq --remove-bpe evaluation methodology.
        Eliminates decode artifacts ("Jakarta ." vs "Jakarta.").
    eval_mode='surface':
        Original behaviour — surface form comparison.
    max_src_len: if set, skip samples whose source token count exceeds this value.
        Matches Fairseq --skip-invalid-size-inputs-valid-test behaviour.
        Set to 128 to replicate the paper's evaluation exactly.
    """
    model.eval()

    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    all_preds, all_refs, all_cats = [], [], []
    skipped_total = 0

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if max_batches and i >= max_batches:
            break

        src = batch["src_tokens"].to(device)
        mask = batch["src_mask"].to(device)
        src_texts = batch["src_texts"]
        tgt_texts = batch["tgt_texts"]
        categories = batch["categories"].tolist()

        # ── Skip-invalid filter (--skip-invalid-size-inputs-valid-test) ──
        if max_src_len is not None:
            src_lens = mask.sum(dim=1)                        # real token count per sample
            valid = (src_lens <= max_src_len).cpu().tolist()  # True = keep
            if not any(valid):
                skipped_total += len(src_texts)
                continue
            # Filter to valid samples only
            keep_idx = [j for j, v in enumerate(valid) if v]
            skipped_total += len(src_texts) - len(keep_idx)
            src       = src[keep_idx]
            mask      = mask[keep_idx]
            src_texts = [src_texts[j] for j in keep_idx]
            tgt_texts = [tgt_texts[j] for j in keep_idx]
            categories = [categories[j] for j in keep_idx]

        preds = decode_batch(
            model, src, mask, tokenizer, src_texts, bos_id, eos_id,
            eval_mode=eval_mode,
            max_len=max_len,
            length_penalty=length_penalty,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
        )

        all_preds.extend(preds)

        # Option A: tokenize references to same format as predictions
        if eval_mode == 'tokenized':
            tokenized_refs = [tokenize_to_string(ref, tokenizer) for ref in tgt_texts]
            all_refs.extend(tokenized_refs)
        else:
            all_refs.extend(tgt_texts)

        all_cats.extend(categories)

    if skipped_total > 0:
        logger.info(f"Skipped {skipped_total} samples with src_len > {max_src_len} "
                    f"(--skip-invalid-size-inputs-valid-test). Evaluated {len(all_preds)} samples.")

    # ── Save hyp/ref files (untuk investigate_bleu_gap.py --from_files) ──
    try:
        import yaml
        metrics_dir = Path("checkpoints/castle9/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "hyp.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(all_preds))
        with open(metrics_dir / "ref.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(all_refs))
        logger.info(f"Saved hyp/ref files to {metrics_dir}/")
    except Exception as e:
        logger.warning(f"Could not save hyp/ref files: {e}")

    # ── Overall metrics ────────────────────────────────────────
    overall_f1 = sentence_f1_corpus(all_preds, all_refs)
    overall_bleu = compute_bleu(all_preds, all_refs)

    # Surface BLEU (WordPiece detokenized) — investigasi gap ke paper
    if eval_mode == 'tokenized':
        all_preds_surf = [wordpiece_decode(p) for p in all_preds]
        all_refs_surf  = [wordpiece_decode(r) for r in all_refs]
        overall_bleu_surface = compute_bleu(all_preds_surf, all_refs_surf)
    else:
        overall_bleu_surface = None

    results = {
        "overall": {
            "precision": overall_f1["precision"],
            "recall": overall_f1["recall"],
            "f1": overall_f1["f1"],
            "bleu": overall_bleu,
            "bleu_surface": overall_bleu_surface,
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
    print(f"{'Category':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'BLEU(tok)':>10} {'BLEU(srf)':>10} {'N':>8}")
    print("-" * 80)

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
        diff_str = f"  (ΔF1={f1_diff:+.4f}, ΔBLEUtok={bleu_diff:+.2f})" if target else ""

        bleu_surf = metrics.get("bleu_surface")
        surf_str = f"{bleu_surf:>10.2f}" if bleu_surf is not None else f"{'—':>10}"

        print(
            f"{key:<20} "
            f"{metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} "
            f"{metrics['f1']:>8.4f} "
            f"{metrics['bleu']:>10.2f} "
            f"{surf_str} "
            f"{metrics['n_samples']:>8}"
            f"{diff_str}"
        )

    print("=" * 80)
    # Surface BLEU for overall if available
    overall = results.get("overall", {})
    if overall.get("bleu_surface") is not None:
        print(f"\nBLEU(tok)={overall['bleu']:.2f}  BLEU(surface)={overall['bleu_surface']:.2f}  "
              f"→ WordPiece decode gain: {overall['bleu_surface'] - overall['bleu']:+.2f}")
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
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Eval batch size (beam search pakai lebih banyak VRAM, default=8)")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256,
                        help="Max generation length. Default 256 (was 128 — too short for some sentences)")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Beam search length penalty. 1.0=Fairseq default. 0.6 caused over-generation.")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--max_len_a", type=float, default=1.2,
                        help="Dynamic max output = max_len_a * src_len + max_len_b "
                             "(Fairseq convention). Set 0.0 to use fixed --max_len.")
    parser.add_argument("--max_len_b", type=int, default=10,
                        help="Additive term for dynamic max output length.")
    parser.add_argument("--max_src_len", type=int, default=None,
                        help="Skip samples with src token count > this value "
                             "(replicates --skip-invalid-size-inputs-valid-test). "
                             "Set to 128 for exact paper comparison.")
    parser.add_argument(
        "--eval_mode", default="tokenized", choices=["tokenized", "surface"],
        help=(
            "tokenized (default/Option A): evaluasi di level WordPiece token — "
            "menghilangkan decode artifacts, sesuai cara paper (Fairseq --remove-bpe). "
            "surface: surface form comparison (perilaku lama)."
        ),
    )
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
        batch_size=args.batch_size, pad_id=pad_idx,
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

    logger.info(
        f"Evaluation mode: {args.eval_mode} | "
        f"max_len_a={args.max_len_a} max_len_b={args.max_len_b} | "
        f"length_penalty={args.length_penalty} | "
        f"max_src_len={args.max_src_len}"
    )
    results = evaluate(
        model, test_loader, tokenizer, device,
        max_batches=args.max_batches,
        eval_mode=args.eval_mode,
        max_len=args.max_len,
        length_penalty=args.length_penalty,
        max_len_a=args.max_len_a,
        max_len_b=args.max_len_b,
        max_src_len=args.max_src_len,
    )
    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
