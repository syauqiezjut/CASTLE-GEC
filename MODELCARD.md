---
language: id
tags:
  - grammatical-error-correction
  - gec
  - indonesian
  - transformer
  - knowledge-graph
  - seq2seq
license: mit
datasets:
  - syauqie/IGED
metrics:
  - bleu
  - f1
pipeline_tag: text2text-generation
---

# castle-gec

**CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement**
Indonesian Grammatical Error Correction

**Paper:** [Expert Systems with Applications, Vol. 299 (2026), 130233](https://doi.org/10.1016/j.eswa.2025.130233)
**Code:** [github.com/syauqie/castle-gec](https://github.com/syauqie/castle-gec)
**Dataset:** [syauqie/IGED](https://huggingface.co/datasets/syauqie/IGED)

---

## Model Description

CASTLE is a seq2seq transformer for correcting grammatical errors in Indonesian text. The model was trained on [IGED](https://huggingface.co/datasets/syauqie/IGED), covering three error categories: morphology, syntax, and semantics.

Architecture: 4-layer encoder-decoder transformer (d=256, 8 heads, FFN=2048) with WordPiece tokenization (vocab=10K). Key feature: linked attention mechanism that conditions each layer's attention on the previous layer's attention scores.

This checkpoint is a **PyTorch reconstruction** of the original Fairseq-based model described in the paper. See the [GitHub README](https://github.com/syauqie/castle-gec#reconstruction-notes) for full transparency on what differs from the paper.

---

## Evaluation Results

On the IGED test set (134,025 sentence pairs):

| Category | F1 | BLEU |
|---|---|---|
| **Overall** | **0.9444** | **88.95** |
| Morphology | 0.9682 | — |
| Syntax | 0.9343 | — |
| Semantics | 0.9413 | — |

BLEU is evaluated at WordPiece token level with length-constrained decoding (`max_len = src_len + 3`). The paper reports F1=0.9629, BLEU=92.72 — see the repository README for a detailed explanation of the gap.

---

## How to Use

### Load and run inference

```python
import torch
from huggingface_hub import snapshot_download

# Download model files
local_dir = snapshot_download(repo_id="syauqie/castle-gec")

# Load corrector
import sys
sys.path.insert(0, local_dir)
from src.inference import CASTLECorrector

corrector = CASTLECorrector.from_checkpoint(
    checkpoint_path=f"{local_dir}/checkpoint_best.pt",
    config_path=f"{local_dir}/configs/castle_base.yaml",
    tokenizer_dir=f"{local_dir}/data/tokenizer",
)

# Correct a sentence
result = corrector.correct("Saya sudah pergi ke sana kemarin hari.")
print(result)
# → "Saya sudah pergi ke sana kemarin."
```

### Batch correction

```python
sentences = [
    "Para mahasiswa-mahasiswa itu berdiskusi dengan aktif.",
    "Dia mempermasalahkan tentang hal tersebut.",
    "Mobil itu sangat cepat sekali.",
]
results = corrector.correct_batch(sentences)
for src, tgt in zip(sentences, results):
    print(f"  Error: {src}")
    print(f"  Fixed: {tgt}")
    print()
```

---

## Intended Use

This model is intended for:
- Automated correction of Indonesian grammatical errors
- Research on Indonesian NLP and GEC
- Educational tools for Indonesian language learners

The model performs best on formal written Indonesian. It may not handle very informal or colloquial text well.

---

## Repository Files

| File | Description |
|---|---|
| `checkpoint_best.pt` | Model weights (PyTorch) |
| `configs/castle_base.yaml` | Model configuration |
| `data/tokenizer/` | WordPiece tokenizer (vocab=10K) |
| `src/` | Full source code |

---

## Training Details

- **Dataset:** IGED (80/10/10 train/val/test split)
- **Optimizer:** Adam (β₁=0.9, β₂=0.98, ε=1e-8)
- **Learning rate:** 5e-4 with inverse sqrt schedule, 4000 warmup steps
- **Batch size:** ~40 samples, update_freq=2 (effective batch = 80)
- **Training:** 10 epochs (~150K gradient steps)
- **Loss:** Label-smoothed cross-entropy (ε=0.1)
- **Hardware:** NVIDIA RTX 3090 24GB

---

## Citation

```bibtex
@article{castle2026,
  title   = {{CASTLE}: Context-Aware Semantic Transformer with Knowledge Graph
             Enhancement for Indonesian Grammatical Error Correction},
  author  = {Abdurrahman, M. Syauqi and others},
  journal = {Expert Systems with Applications},
  volume  = {299},
  pages   = {130233},
  year    = {2026},
  doi     = {10.1016/j.eswa.2025.130233}
}
```
