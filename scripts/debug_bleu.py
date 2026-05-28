"""
debug_bleu.py
Diagnostic script to understand the BLEU gap.
Shows: source | pred_tokens | ref_tokens | match analysis

Usage:
  cd /root/autodl-tmp/CASTLE-GEC
  python scripts/debug_bleu.py \
    --checkpoint checkpoints/castle9/checkpoint_best.pt \
    --config configs/castle_base.yaml \
    --n_samples 30
"""

import sys, os, argparse, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import load_iged, get_dataloaders, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from castle_model import build_castle_model
from evaluate import ids_to_token_string, tokenize_to_string
import sacrebleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/castle_base.yaml")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--local_csv", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_ds, tokenizer = load_iged(
        hf_dataset_name=cfg["dataset"]["name"],
        tokenizer_dir=cfg.get("tokenizer_dir", "data/tokenizer"),
        local_csv=args.local_csv,
    )
    pad_idx = tokenizer.token_to_id(PAD_TOKEN)
    bos_id  = tokenizer.token_to_id(BOS_TOKEN)
    eos_id  = tokenizer.token_to_id(EOS_TOKEN)

    _, _, test_loader = get_dataloaders(
        test_ds, test_ds, test_ds,
        batch_size=8, pad_id=pad_idx,
    )

    # Load model
    vocab_size = tokenizer.get_vocab_size()
    model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # Load KG
    try:
        from knowledge_graph import CASTLEKnowledgeGraph
        kg_path = cfg.get("knowledge_graph", {}).get("output_path", "data/castle_kg.pkl")
        if os.path.exists(kg_path):
            kg = CASTLEKnowledgeGraph.load(kg_path)
            model.set_knowledge_graph(kg, kg.word_to_idx())
            print("KG loaded.")
    except Exception as e:
        print(f"KG not loaded: {e}")

    collected = 0
    all_preds_tok, all_refs_tok = [], []

    print(f"\n{'='*90}")
    print(f"{'SRC':<30} | {'PRED (tokens)':<30} | {'REF (tokens)':<30}")
    print(f"{'='*90}")

    with torch.no_grad():
        for batch in test_loader:
            if collected >= args.n_samples:
                break

            src = batch["src_tokens"].to(device)
            mask = batch["src_mask"].to(device)
            src_texts  = batch["src_texts"]
            tgt_texts  = batch["tgt_texts"]

            generated = model.generate(
                src, mask, tokenizer,
                src_texts=src_texts,
                max_len=128,
                bos_id=bos_id,
                eos_id=eos_id,
            )

            for i, token_ids in enumerate(generated):
                if collected >= args.n_samples:
                    break

                pred_str = ids_to_token_string(token_ids, tokenizer, bos_id, eos_id)
                ref_str  = tokenize_to_string(tgt_texts[i], tokenizer)
                src_str  = src_texts[i]

                all_preds_tok.append(pred_str)
                all_refs_tok.append(ref_str)

                # Token-level analysis
                pred_toks = set(pred_str.lower().split())
                ref_toks  = set(ref_str.lower().split())
                tp = len(pred_toks & ref_toks)
                extra = pred_toks - ref_toks   # over-correction candidates
                missed = ref_toks - pred_toks  # under-correction candidates

                exact_match = (pred_str.lower() == ref_str.lower())

                print(f"\n[Sample {collected+1}]")
                print(f"  SRC : {src_str[:80]}")
                print(f"  PRED: {pred_str[:80]}")
                print(f"  REF : {ref_str[:80]}")
                print(f"  → Exact match: {exact_match}")
                print(f"  → Extra tokens (pred not in ref): {list(extra)[:8]}")
                print(f"  → Missed tokens (ref not in pred): {list(missed)[:8]}")
                print(f"  → Pred len: {len(pred_str.split())}  |  Ref len: {len(ref_str.split())}")

                collected += 1

    # Corpus-level BLEU + detail
    bleu = sacrebleu.corpus_bleu(all_preds_tok, [all_refs_tok])
    print(f"\n{'='*90}")
    print(f"BLEU on {collected} samples: {bleu.score:.2f}")
    print(f"  BP={bleu.bp:.4f}  ratio={bleu.sys_len/bleu.ref_len:.4f}  "
          f"sys_len={bleu.sys_len}  ref_len={bleu.ref_len}")
    print(f"  1-gram={bleu.precisions[0]:.2f}  2-gram={bleu.precisions[1]:.2f}  "
          f"3-gram={bleu.precisions[2]:.2f}  4-gram={bleu.precisions[3]:.2f}")

    exact_matches = sum(p.lower()==r.lower() for p,r in zip(all_preds_tok, all_refs_tok))
    print(f"  Exact match: {exact_matches}/{collected} ({100*exact_matches/collected:.1f}%)")

    # Length analysis
    pred_lens = [len(p.split()) for p in all_preds_tok]
    ref_lens  = [len(r.split()) for r in all_refs_tok]
    print(f"  Avg pred len: {sum(pred_lens)/len(pred_lens):.1f}  "
          f"Avg ref len: {sum(ref_lens)/len(ref_lens):.1f}")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()
