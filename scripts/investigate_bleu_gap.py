"""
investigate_bleu_gap.py
=======================
Investigasi sisa BLEU gap (-18.35) dengan membandingkan:
  1. Tokenized BLEU  — token WordPiece level (current castle9 approach)
  2. Surface BLEU    — setelah WordPiece detokenization (## joined, space normalized)
  3. Ratio & BP analysis
  4. Per-ngram breakdown
  5. Sample-level inspection

Bisa dijalankan dua cara:
  A. Dari file hyp/ref yang sudah ada (cepat, tidak perlu GPU):
       python scripts/investigate_bleu_gap.py --from_files \
         --hyp checkpoints/castle9/metrics/hyp.txt \
         --ref checkpoints/castle9/metrics/ref.txt

  B. Dari model langsung (perlu GPU, beam search ulang, N_SAMPLES sample):
       python scripts/investigate_bleu_gap.py \
         --checkpoint checkpoints/castle9/checkpoint_best.pt \
         --config configs/castle_base.yaml \
         --n_samples 200

Usage (Mode A, paling cepat):
  cd /root/autodl-tmp/CASTLE-GEC
  python scripts/investigate_bleu_gap.py --from_files \
    --hyp checkpoints/castle9/metrics/hyp.txt \
    --ref checkpoints/castle9/metrics/ref.txt
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import sacrebleu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wordpiece_decode(token_str: str) -> str:
    """
    Join WordPiece ## continuation tokens back to surface form.
    e.g. "ber ##jalan ke ##luar" → "berjalan keluar"
    Also normalises common punctuation spacing artifacts.
    """
    # Step 1: join ## subwords
    text = token_str.replace(' ##', '')
    # Step 2: common punctuation spacing (WordPiece splits these)
    #   "kata ." → "kata."   "( BNN )" → "(BNN)"
    import re
    text = re.sub(r'\s+([.,!?;:»\)])', r'\1', text)
    text = re.sub(r'([\(«])\s+', r'\1', text)
    # Step 3: collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


def ngram_overlap_detail(hyps, refs):
    """Return per-order precision array from sacrebleu."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu


def print_separator(char='=', n=80):
    print(char * n)


# ---------------------------------------------------------------------------
# Mode A: from existing hyp/ref files
# ---------------------------------------------------------------------------

def from_files(hyp_file, ref_file, n_inspect=20):
    with open(hyp_file, encoding='utf-8') as f:
        hyps_tok = [l.strip() for l in f]
    with open(ref_file, encoding='utf-8') as f:
        refs_tok = [l.strip() for l in f]

    assert len(hyps_tok) == len(refs_tok), \
        f"Line count mismatch: hyp={len(hyps_tok)}, ref={len(refs_tok)}"

    hyps_surf = [wordpiece_decode(h) for h in hyps_tok]
    refs_surf = [wordpiece_decode(r) for r in refs_tok]

    run_analysis(hyps_tok, refs_tok, hyps_surf, refs_surf, n_inspect)


# ---------------------------------------------------------------------------
# Mode B: from model (requires GPU)
# ---------------------------------------------------------------------------

def from_model(args):
    import yaml, torch
    from dataset import load_iged, get_dataloaders, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
    from castle_model import build_castle_model
    from evaluate import ids_to_token_string, tokenize_to_string

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    vocab_size = tokenizer.get_vocab_size()
    model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    hyps_tok, refs_tok = [], []
    collected = 0

    with torch.no_grad():
        for batch in test_loader:
            if collected >= args.n_samples:
                break
            src  = batch["src_tokens"].to(device)
            mask = batch["src_mask"].to(device)

            generated = model.generate(
                src, mask, tokenizer,
                src_texts=batch["src_texts"],
                max_len=256, bos_id=bos_id, eos_id=eos_id,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
                max_len_a=1.2, max_len_b=10,
            )

            for i, token_ids in enumerate(generated):
                if collected >= args.n_samples:
                    break
                hyps_tok.append(ids_to_token_string(token_ids, tokenizer, bos_id, eos_id))
                refs_tok.append(tokenize_to_string(batch["tgt_texts"][i], tokenizer))
                collected += 1

    hyps_surf = [wordpiece_decode(h) for h in hyps_tok]
    refs_surf = [wordpiece_decode(r) for r in refs_tok]
    run_analysis(hyps_tok, refs_tok, hyps_surf, refs_surf, n_inspect=args.n_inspect)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_analysis(hyps_tok, refs_tok, hyps_surf, refs_surf, n_inspect=20):
    N = len(hyps_tok)

    bleu_tok  = sacrebleu.corpus_bleu(hyps_tok, [refs_tok])
    bleu_surf = sacrebleu.corpus_bleu(hyps_surf, [refs_surf])

    print_separator()
    print(f"BLEU GAP INVESTIGATION  ({N:,} sentences)")
    print_separator()

    print(f"\n{'':40s}  {'BLEU':>8}  {'BP':>6}  {'ratio':>7}  {'1g':>7}  {'2g':>7}  {'3g':>7}  {'4g':>7}")
    print("-" * 90)

    def fmt_row(label, b):
        ratio = b.sys_len / b.ref_len if b.ref_len > 0 else 0
        return (f"{label:<40s}  {b.score:>8.2f}  {b.bp:>6.4f}  {ratio:>7.4f}  "
                f"{b.precisions[0]:>7.2f}  {b.precisions[1]:>7.2f}  "
                f"{b.precisions[2]:>7.2f}  {b.precisions[3]:>7.2f}")

    print(fmt_row("Tokenized (current castle9)", bleu_tok))
    print(fmt_row("Surface (wordpiece_decode)", bleu_surf))
    print(fmt_row("PAPER target", type('B', (), {
        'score': 92.72, 'bp': 1.0,
        'sys_len': 1, 'ref_len': 1,
        'precisions': ['-', '-', '-', '-']
    })()))

    # Exact match
    exact_tok  = sum(h.lower() == r.lower() for h, r in zip(hyps_tok, refs_tok))
    exact_surf = sum(h.lower() == r.lower() for h, r in zip(hyps_surf, refs_surf))
    print(f"\nExact match (tokenized): {exact_tok}/{N} ({100*exact_tok/N:.1f}%)")
    print(f"Exact match (surface  ): {exact_surf}/{N} ({100*exact_surf/N:.1f}%)")

    # Length stats
    pred_lens_tok  = [len(h.split()) for h in hyps_tok]
    ref_lens_tok   = [len(r.split()) for r in refs_tok]
    pred_lens_surf = [len(h.split()) for h in hyps_surf]
    ref_lens_surf  = [len(r.split()) for r in refs_surf]

    print(f"\nLength (tokenized): avg pred={sum(pred_lens_tok)/N:.1f}  avg ref={sum(ref_lens_tok)/N:.1f}  "
          f"ratio={sum(pred_lens_tok)/max(sum(ref_lens_tok),1):.4f}")
    print(f"Length (surface  ): avg pred={sum(pred_lens_surf)/N:.1f}  avg ref={sum(ref_lens_surf)/N:.1f}  "
          f"ratio={sum(pred_lens_surf)/max(sum(ref_lens_surf),1):.4f}")

    # Samples with biggest improvement surface vs tokenized
    print_separator('-')
    print(f"\nSample inspection (first {n_inspect} sentences):")
    print_separator('-')

    for i in range(min(n_inspect, N)):
        h_tok, r_tok = hyps_tok[i], refs_tok[i]
        h_sur, r_sur = hyps_surf[i], refs_surf[i]
        exact = (h_tok.lower() == r_tok.lower())
        print(f"\n[{i+1}]")
        print(f"  TOK PRED: {h_tok[:100]}")
        print(f"  TOK REF : {r_tok[:100]}")
        print(f"  SRF PRED: {h_sur[:100]}")
        print(f"  SRF REF : {r_sur[:100]}")
        print(f"  Exact (tok): {exact}  |  pred_len_tok={len(h_tok.split())}  ref_len_tok={len(r_tok.split())}")

    # Divergence: sentences where surface ≠ tokenized match
    surface_better = []
    for i, (h_tok, r_tok, h_sur, r_sur) in enumerate(zip(hyps_tok, refs_tok, hyps_surf, refs_surf)):
        tok_match  = (h_tok.lower() == r_tok.lower())
        surf_match = (h_sur.lower() == r_sur.lower())
        if surf_match and not tok_match:
            surface_better.append(i)

    print_separator('-')
    print(f"\nSurface match but NOT token match: {len(surface_better)} sentences")
    print("(These are cases where WordPiece detokenization resolves the mismatch)")
    for i in surface_better[:10]:
        print(f"  [{i+1}] TOK_PRED: {hyps_tok[i][:80]}")
        print(f"       TOK_REF : {refs_tok[i][:80]}")
        print(f"       SRF_PRED: {hyps_surf[i][:80]}")
        print(f"       SRF_REF : {refs_surf[i][:80]}")

    print_separator()
    print("SUMMARY:")
    print(f"  Tokenized BLEU : {bleu_tok.score:.2f}")
    print(f"  Surface BLEU   : {bleu_surf.score:.2f}  (Δ={bleu_surf.score - bleu_tok.score:+.2f})")
    print(f"  Paper BLEU     : 92.72")
    print(f"  Remaining gap  : {92.72 - bleu_surf.score:.2f} (surface) / {92.72 - bleu_tok.score:.2f} (tokenized)")
    print_separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate remaining BLEU gap")
    parser.add_argument("--from_files", action="store_true",
                        help="Read from existing hyp/ref files (no GPU needed)")
    parser.add_argument("--hyp", default="checkpoints/castle9/metrics/hyp.txt",
                        help="Path to hypothesis file (tokenized)")
    parser.add_argument("--ref", default="checkpoints/castle9/metrics/ref.txt",
                        help="Path to reference file (tokenized)")
    parser.add_argument("--n_inspect", type=int, default=20,
                        help="Number of samples to print in detail")
    # Mode B args
    parser.add_argument("--checkpoint", default="checkpoints/castle9/checkpoint_best.pt")
    parser.add_argument("--config", default="configs/castle_base.yaml")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--local_csv", default=None)
    args = parser.parse_args()

    if args.from_files:
        from_files(args.hyp, args.ref, n_inspect=args.n_inspect)
    else:
        from_model(args)
