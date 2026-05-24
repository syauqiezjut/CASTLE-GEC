"""
dataset.py
IGED Dataset Loader for CASTLE
================================
Loads the Indonesian Grammar Error correction Dataset (IGED) from HuggingFace Hub
(syauqie/IGED) and prepares it for sequence-to-sequence training.

Dataset statistics (from paper):
  - 1,345,096 total samples
  - 80% train / 10% val / 10% test
  - Error distribution: Morphological 35%, Syntactic 40%, Semantic 25%
  - Avg words: 21.3, Avg chars: 154.3
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

ERROR_CATEGORY_MAP = {
    "morphology": 0,
    "syntax": 1,
    "semantics": 2,
    # Subcategories
    "affixation": 0, "word_formation": 0, "reduplication": 0,
    "phrase_structure": 1, "preposition": 1, "sentence_completeness": 1,
    "diction": 2, "ambiguity": 2, "pleonasm": 2,
}

SEMANTIC_SUBCATEGORIES = {"diction", "ambiguity", "pleonasm",
                           "diksi", "ambigu", "pleonasme"}


# ──────────────────────────────────────────────
# Tokenizer builder
# ──────────────────────────────────────────────

def build_wordpiece_tokenizer(
    vocab_size: int = 10_000,
    train_files: Optional[List[str]] = None,
    save_dir: str = "data/tokenizer",
    min_frequency: int = 2,
) -> BertWordPieceTokenizer:
    """
    Build or load a WordPiece tokenizer.
    Paper uses 10K vocabulary (WordPiece) as the best tokenization scheme.
    """
    vocab_file = os.path.join(save_dir, "vocab.txt")

    if os.path.exists(vocab_file):
        logger.info(f"Loading existing tokenizer from {save_dir}")
        tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=False)
    else:
        assert train_files, "train_files required to train tokenizer from scratch"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Training WordPiece tokenizer (vocab_size={vocab_size})")
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        tokenizer.train(
            files=train_files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
                            "[CLS]", "[SEP]", "[MASK]"],
        )
        tokenizer.save_model(save_dir)
        logger.info(f"Tokenizer saved to {save_dir}")

    # Add post-processor for BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    tokenizer.enable_padding(pad_token=PAD_TOKEN,
                             pad_id=tokenizer.token_to_id(PAD_TOKEN))
    return tokenizer


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────

class IGEDDataset(Dataset):
    """
    PyTorch Dataset for IGED.

    Each sample is a dict with:
      - src_ids:    LongTensor  [src_len]   — tokenized erroneous sentence
      - tgt_ids:    LongTensor  [tgt_len]   — tokenized correct sentence
      - category:   int                     — error category (0=morph, 1=syn, 2=sem)
      - src_text:   str                     — raw erroneous sentence
      - tgt_text:   str                     — raw correct sentence
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: BertWordPieceTokenizer,
        max_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        src_text = sample["source"]
        tgt_text = sample["target"]
        category_str = sample.get("category", "syntax").lower()

        # Resolve category int
        category = ERROR_CATEGORY_MAP.get(category_str, 1)

        # Tokenize
        src_enc = self.tokenizer.encode(src_text)
        tgt_enc = self.tokenizer.encode(tgt_text)

        src_ids = src_enc.ids[: self.max_length]
        tgt_ids = tgt_enc.ids[: self.max_length]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "category": category,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


# ──────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────

def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict:
    """Pad sequences in batch to same length."""
    src_lengths = [len(s["src_ids"]) for s in batch]
    tgt_lengths = [len(s["tgt_ids"]) for s in batch]

    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)

    bsz = len(batch)
    src_tokens = torch.full((bsz, max_src), pad_id, dtype=torch.long)
    tgt_tokens = torch.full((bsz, max_tgt), pad_id, dtype=torch.long)
    src_mask = torch.zeros(bsz, max_src, dtype=torch.bool)

    for i, s in enumerate(batch):
        L_src = len(s["src_ids"])
        L_tgt = len(s["tgt_ids"])
        src_tokens[i, :L_src] = s["src_ids"]
        tgt_tokens[i, :L_tgt] = s["tgt_ids"]
        src_mask[i, :L_src] = True

    categories = torch.tensor([s["category"] for s in batch], dtype=torch.long)
    src_texts = [s["src_text"] for s in batch]
    tgt_texts = [s["tgt_text"] for s in batch]

    return {
        "src_tokens": src_tokens,       # [B, S]
        "tgt_tokens": tgt_tokens,       # [B, T]
        "src_mask": src_mask,           # [B, S] — True where not pad
        "categories": categories,       # [B]
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
    }


# ──────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────

def load_iged(
    hf_dataset_name: str = "syauqie/IGED",
    tokenizer: Optional[BertWordPieceTokenizer] = None,
    tokenizer_dir: str = "data/tokenizer",
    cache_dir: str = "data/hf_cache",
    max_length: int = 128,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    local_csv: Optional[str] = None,
) -> Tuple[IGEDDataset, IGEDDataset, IGEDDataset, BertWordPieceTokenizer]:
    """
    Load IGED dataset and return (train, val, test, tokenizer).

    Args:
        hf_dataset_name: HuggingFace dataset repo ID (default: syauqie/IGED)
        tokenizer: Pre-built tokenizer (optional, will build/load if None)
        tokenizer_dir: Directory to save/load tokenizer vocab
        cache_dir: HuggingFace cache directory
        max_length: Maximum sequence length (tokens)
        train_ratio: Fraction for training (0.80)
        val_ratio: Fraction for validation (0.10)
        seed: Random seed for reproducible splits
        local_csv: Path to local CSV file (alternative to HF Hub)
    """
    logger.info("=" * 60)
    logger.info("Loading IGED dataset")
    logger.info("=" * 60)

    # ── Column name resolver ───────────────────────────────────
    # Tries many variations for source / target / category columns
    SRC_COLS = ("source", "src", "incorrect", "error", "input",
                "kalimat_salah", "salah", "noisy")
    TGT_COLS = ("target", "tgt", "correct", "correction", "output",
                "kalimat_benar", "benar", "clean", "reference")
    CAT_COLS = ("category", "cat", "error_type", "type", "label",
                "error_category", "kategori")

    def _resolve(ex: dict, candidates: tuple, default: str = "") -> str:
        for k in candidates:
            if k in ex and ex[k] is not None and str(ex[k]).strip():
                return str(ex[k]).strip()
            # case-insensitive fallback
            for ek in ex:
                if ek.lower() == k and ex[ek] is not None and str(ex[ek]).strip():
                    return str(ex[ek]).strip()
        return default

    def _normalize_df_cols(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in SRC_COLS:
                col_map[col] = "source"
            elif cl in TGT_COLS:
                col_map[col] = "target"
            elif cl in CAT_COLS:
                col_map[col] = "category"
        df = df.rename(columns=col_map)
        logger.info(f"CSV columns after normalize: {list(df.columns)}")
        return df

    # ── Load raw data ──────────────────────────────────────────
    if local_csv and os.path.exists(local_csv):
        logger.info(f"Loading from local CSV: {local_csv}")
        df = pd.read_csv(local_csv)
        logger.info(f"CSV columns (raw): {list(df.columns)}")
        df = _normalize_df_cols(df)
        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError(
                f"Cannot find source/target columns in CSV.\n"
                f"Columns found: {list(df.columns)}\n"
                f"Expected one of: src={SRC_COLS}, tgt={TGT_COLS}"
            )
        if "category" not in df.columns:
            df["category"] = "syntax"
        df = df[["source", "target", "category"]].dropna(subset=["source", "target"])
        df = df[df["source"].str.strip().astype(bool) & df["target"].str.strip().astype(bool)]
        samples_raw = df.to_dict("records")
        logger.info(f"Valid samples after dropna: {len(samples_raw):,}")
    else:
        logger.info(f"Loading from HuggingFace: {hf_dataset_name}")
        try:
            hf_data = load_dataset(hf_dataset_name, cache_dir=cache_dir)
            logger.info(f"HF dataset splits: {list(hf_data.keys())}")

            # Show first example to debug column names
            first_split = list(hf_data.values())[0]
            if len(first_split) > 0:
                ex0 = first_split[0]
                logger.info(f"First example keys: {list(ex0.keys())}")

            def _to_records(split):
                rows = []
                for ex in split:
                    src = _resolve(ex, SRC_COLS)
                    tgt = _resolve(ex, TGT_COLS)
                    cat = _resolve(ex, CAT_COLS, "syntax")
                    if src and tgt:
                        rows.append({"source": src, "target": tgt, "category": cat})
                return rows

            # Load all available splits
            all_records = []
            for split_name, split_data in hf_data.items():
                records = _to_records(split_data)
                logger.info(f"  Split '{split_name}': {len(records):,} records")
                all_records.extend(records)

            samples_raw = all_records
            train_records = val_records = test_records = None

        except Exception as e:
            raise RuntimeError(
                f"Failed to load {hf_dataset_name}. "
                f"Pass local_csv='/path/to/IGED_stratified_dataset.csv' instead.\n"
                f"Original error: {e}"
            )

    # ── Manual stratified split ────────────────────────────────
    if "train_records" not in dir() or train_records is None:
        import random
        random.seed(seed)
        random.shuffle(samples_raw)
        N = len(samples_raw)
        n_train = int(N * train_ratio)
        n_val = int(N * val_ratio)
        train_records = samples_raw[:n_train]
        val_records = samples_raw[n_train: n_train + n_val]
        test_records = samples_raw[n_train + n_val:]
        logger.info(
            f"Split sizes — train: {len(train_records)}, "
            f"val: {len(val_records)}, test: {len(test_records)}"
        )

    # ── Build tokenizer if needed ──────────────────────────────
    if tokenizer is None:
        vocab_file = os.path.join(tokenizer_dir, "vocab.txt")
        if not os.path.exists(vocab_file):
            # Write temp corpus for tokenizer training
            logger.info("Writing corpus for tokenizer training...")
            corpus_path = os.path.join(tokenizer_dir, "corpus.txt")
            os.makedirs(tokenizer_dir, exist_ok=True)
            with open(corpus_path, "w", encoding="utf-8") as f:
                for rec in tqdm(train_records, desc="Building corpus"):
                    f.write(rec["source"] + "\n")
                    f.write(rec["target"] + "\n")
            train_files = [corpus_path]
        else:
            train_files = None

        tokenizer = build_wordpiece_tokenizer(
            vocab_size=10_000,
            train_files=train_files,
            save_dir=tokenizer_dir,
        )

    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Tokenizer vocab size: {vocab_size}")

    # ── Create datasets ────────────────────────────────────────
    train_ds = IGEDDataset(train_records, tokenizer, max_length)
    val_ds = IGEDDataset(val_records, tokenizer, max_length)
    test_ds = IGEDDataset(test_records, tokenizer, max_length)

    logger.info(
        f"Datasets created — train: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, tokenizer


def get_dataloaders(
    train_ds: IGEDDataset,
    val_ds: IGEDDataset,
    test_ds: IGEDDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pad_id: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, val, test splits."""
    _collate = lambda batch: collate_fn(batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Test IGED dataset loading")
    parser.add_argument("--csv", type=str, default=None, help="Local CSV path")
    parser.add_argument("--hf", type=str, default="syauqie/IGED", help="HF dataset name")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer")
    args = parser.parse_args()

    train_ds, val_ds, test_ds, tok = load_iged(
        hf_dataset_name=args.hf,
        tokenizer_dir=args.tokenizer_dir,
        local_csv=args.csv,
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=4,
        pad_id=tok.token_to_id(PAD_TOKEN),
    )

    batch = next(iter(train_loader))
    print("\nSample batch:")
    print(f"  src_tokens shape : {batch['src_tokens'].shape}")
    print(f"  tgt_tokens shape : {batch['tgt_tokens'].shape}")
    print(f"  categories       : {batch['categories']}")
    print(f"  src[0]: {batch['src_texts'][0]}")
    print(f"  tgt[0]: {batch['tgt_texts'][0]}")
