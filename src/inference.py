"""
inference.py
CASTLE Inference — Correct Indonesian Grammar
===============================================
Load a trained CASTLE checkpoint and run correction on:
  - Single sentences (interactive)
  - Batch from file

Example usage:
  python src/inference.py \\
    --checkpoint checkpoints/castle/checkpoint_best.pt \\
    --sentence "Saya sudah pergi ke sana kemarin hari."

  python src/inference.py \\
    --checkpoint checkpoints/castle/checkpoint_best.pt \\
    --input_file sentences.txt \\
    --output_file corrected.txt
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from castle_model import CASTLE, build_castle_model
from dataset import build_wordpiece_tokenizer, PAD_TOKEN
from knowledge_graph import CASTLEKnowledgeGraph

logger = logging.getLogger(__name__)


class CASTLECorrector:
    """
    High-level interface for CASTLE grammar correction.

    Usage:
        corrector = CASTLECorrector.from_checkpoint(
            checkpoint_path="checkpoints/castle/checkpoint_best.pt",
            config_path="configs/castle_base.yaml",
            kg_path="data/castle_kg.pkl",
        )
        result = corrector.correct("Saya sudah pergi ke sana kemarin hari.")
        print(result)
    """

    def __init__(
        self,
        model: CASTLE,
        tokenizer,
        device: torch.device,
        max_len: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/castle_base.yaml",
        tokenizer_dir: str = "data/tokenizer",
        kg_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "CASTLECorrector":
        """Load CASTLE from checkpoint file."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Load tokenizer
        tokenizer = build_wordpiece_tokenizer(save_dir=tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
        pad_idx = tokenizer.token_to_id(PAD_TOKEN)

        # Build and load model
        model = build_castle_model(cfg, vocab_size=vocab_size, pad_idx=pad_idx)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()

        logger.info(f"Model loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

        # Load KG if available
        kg_path = kg_path or cfg.get("knowledge_graph", {}).get("output_path", "data/castle_kg.pkl")
        if kg_path and os.path.exists(kg_path):
            kg = CASTLEKnowledgeGraph.load(kg_path)
            model.set_knowledge_graph(kg, kg.word_to_idx())
            logger.info(f"Knowledge graph attached ({kg.graph.number_of_nodes():,} nodes)")
        else:
            logger.warning("No KG found — running without knowledge graph integration")

        return cls(model, tokenizer, device, max_len=cfg["model"].get("max_len", 128))

    def correct(self, sentence: str) -> str:
        """Correct a single Indonesian sentence."""
        return self.correct_batch([sentence])[0]

    def correct_batch(self, sentences: List[str]) -> List[str]:
        """Correct a batch of Indonesian sentences."""
        # Tokenize
        encoded = [self.tokenizer.encode(s) for s in sentences]
        max_src_len = max(len(e.ids) for e in encoded)
        max_src_len = min(max_src_len, self.max_len)

        B = len(sentences)
        src_tokens = torch.full((B, max_src_len), self.pad_id, dtype=torch.long)
        src_mask = torch.zeros(B, max_src_len, dtype=torch.bool)

        for i, enc in enumerate(encoded):
            ids = enc.ids[:max_src_len]
            src_tokens[i, :len(ids)] = torch.tensor(ids)
            src_mask[i, :len(ids)] = True

        src_tokens = src_tokens.to(self.device)
        src_mask = src_mask.to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                src_tokens, src_mask, self.tokenizer,
                src_texts=sentences,
                max_len=self.max_len,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                beam_size=5,
                length_penalty=1.0,
                max_len_a=1.0,   # GEC: output length ≈ input length
                max_len_b=3,     # allow up to src_len + 3 extra tokens
            )

        # Decode
        results = []
        for token_ids in generated:
            clean_ids = [
                tid for tid in token_ids
                if tid not in (self.bos_id, self.eos_id, self.pad_id)
            ]
            text = self.tokenizer.decode(clean_ids)
            results.append(text)

        return results

    def correct_file(self, input_path: str, output_path: str, batch_size: int = 32):
        """Correct all sentences in a file, one per line."""
        with open(input_path, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        logger.info(f"Correcting {len(sentences)} sentences from {input_path}")

        corrected = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            corrected.extend(self.correct_batch(batch))
            if (i // batch_size) % 10 == 0:
                logger.info(f"  Processed {min(i + batch_size, len(sentences))}/{len(sentences)}")

        with open(output_path, "w", encoding="utf-8") as f:
            for line in corrected:
                f.write(line + "\n")

        logger.info(f"Corrections written to {output_path}")
        return corrected


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="CASTLE Indonesian Grammar Error Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correct a single sentence
  python src/inference.py \\
    --checkpoint checkpoints/castle/checkpoint_best.pt \\
    --sentence "Saya sudah pergi ke sana kemarin hari."

  # Interactive mode
  python src/inference.py \\
    --checkpoint checkpoints/castle/checkpoint_best.pt \\
    --interactive

  # Correct file
  python src/inference.py \\
    --checkpoint checkpoints/castle/checkpoint_best.pt \\
    --input_file errors.txt \\
    --output_file corrected.txt
        """
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--config", default="configs/castle_base.yaml")
    parser.add_argument("--tokenizer_dir", default="data/tokenizer")
    parser.add_argument("--kg_path", default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sentence", type=str, help="Single sentence to correct")
    mode.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    mode.add_argument("--input_file", type=str, help="File with one sentence per line")

    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Load corrector
    corrector = CASTLECorrector.from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tokenizer_dir=args.tokenizer_dir,
        kg_path=args.kg_path,
        device=args.device,
    )

    if args.sentence:
        result = corrector.correct(args.sentence)
        print(f"\nInput:  {args.sentence}")
        print(f"Output: {result}")

    elif args.interactive:
        print("\nCASTLE Indonesian Grammar Corrector")
        print("Type a sentence and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                sentence = input("Input: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if sentence.lower() in ("quit", "exit", "q"):
                break
            if not sentence:
                continue
            result = corrector.correct(sentence)
            print(f"Output: {result}\n")

    elif args.input_file:
        out = args.output_file or args.input_file.replace(".txt", "_corrected.txt")
        corrector.correct_file(args.input_file, out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
