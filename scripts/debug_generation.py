"""
debug_generation.py
Melihat contoh output aktual model — src, gold, prediction.
Jalankan: python scripts/debug_generation.py --checkpoint checkpoints/castle4/checkpoint_best.pt
"""
import sys, os, yaml, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from castle_model import CASTLE, build_castle_model
from dataset import load_iged, get_dataloaders, PAD_TOKEN
from knowledge_graph import CASTLEKnowledgeGraph

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config", default="configs/castle_base.yaml")
parser.add_argument("--n_samples", type=int, default=10)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer + data
_, _, test_ds, tokenizer = load_iged(
    hf_dataset_name=cfg["dataset"]["name"],
    tokenizer_dir=cfg.get("tokenizer_dir", "data/tokenizer"),
)
pad_idx = tokenizer.token_to_id(PAD_TOKEN)
_, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds, batch_size=8, pad_id=pad_idx)

# Test decode correctness
test_str = "saya pergi ke sekolah"
enc = tokenizer.encode(test_str)
dec = tokenizer.decode(enc.ids)
print(f"\n=== TOKENIZER DECODE TEST ===")
print(f"  Input:    '{test_str}'")
print(f"  Token IDs: {enc.ids[:10]}...")
print(f"  Decoded:  '{dec}'")
print(f"  IDs include BOS/EOS: {tokenizer.token_to_id('<s>')} / {tokenizer.token_to_id('</s>')}")

# Load model
vocab_size = tokenizer.get_vocab_size()
model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
ckpt = torch.load(args.checkpoint, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

# Load KG
kg_path = cfg.get("knowledge_graph", {}).get("output_path", "data/castle_kg.pkl")
if os.path.exists(kg_path):
    kg = CASTLEKnowledgeGraph.load(kg_path)
    model.set_knowledge_graph(kg, kg.word_to_idx())
    model.build_token_id_cache(tokenizer, vocab_size=vocab_size, pad_idx=pad_idx)

bos_id = tokenizer.token_to_id("<s>")
eos_id  = tokenizer.token_to_id("</s>")

print(f"\n=== GENERATION EXAMPLES (n={args.n_samples}) ===\n")

n_shown = 0
for batch in test_loader:
    src = batch["src_tokens"].to(device)
    mask = batch["src_mask"].to(device)
    src_texts = batch["src_texts"]
    tgt_texts = batch["tgt_texts"]

    with torch.no_grad():
        generated = model.generate(src, mask, tokenizer, src_texts=src_texts,
                                   max_len=128, bos_id=bos_id, eos_id=eos_id)

    for b in range(len(src_texts)):
        if n_shown >= args.n_samples:
            break

        raw_ids = generated[b]
        clean_ids = [tid for tid in raw_ids
                     if tid not in (bos_id, eos_id, pad_idx)]
        pred_text = tokenizer.decode(clean_ids)

        src_t  = src_texts[b]
        gold_t = tgt_texts[b]
        same   = (pred_text.strip() == gold_t.strip())

        print(f"--- Sample {n_shown+1} ---")
        print(f"  SRC  : {src_t[:120]}")
        print(f"  GOLD : {gold_t[:120]}")
        print(f"  PRED : {pred_text[:120]}")
        print(f"  GEN  : {raw_ids[:20]}  (len={len(raw_ids)}, EOS hit={'yes' if eos_id in raw_ids else 'NO'})")
        print(f"  MATCH: {'✓ EXACT' if same else '✗'}")

        # Quick word-level stats
        pred_words = set(pred_text.lower().split())
        gold_words = set(gold_t.lower().split())
        tp = len(pred_words & gold_words)
        prec = tp / len(pred_words) if pred_words else 0
        rec  = tp / len(gold_words) if gold_words else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"  F1   : prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}  "
              f"pred_words={len(pred_words)} gold_words={len(gold_words)}")
        print()
        n_shown += 1

    if n_shown >= args.n_samples:
        break

print("=== DONE ===")
