# CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement

[![Paper](https://img.shields.io/badge/Paper-ESWA%202026-blue)](https://doi.org/10.1016/j.eswa.2025.130233)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/syauqie/IGED)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official code for:

> **CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement for Low-Resource Grammar Correction**  
> Syauqie Muhammad Marier, Xiangjie Kong, Linan Zhu, Xiangfan Chen, Abdulloh Badruzzaman, I Nyoman Apraz Ramatryana  
> *Expert Systems with Applications, Vol. 299, 2026*

## Results (IGED Test Set)

| Model | Prec | Rec | F1 | BLEU |
|-------|------|-----|----|------|
| Baseline (Whitespace) | 0.8957 | 0.8923 | 0.8934 | 73.69 |
| + WordPiece | 0.9667 | 0.9513 | 0.9579 | 92.30 |
| **CASTLE (ours)** | **0.9718** | **0.9559** | **0.9629** | **92.72** |
| BART-large (finetuned) | 0.9641 | 0.9558 | 0.9592 | 90.83 |

**Key advantages**: 34.7M params vs 406M (BART-large) — 91.5% fewer parameters, 2.3× faster inference.

## Architecture

CASTLE extends a Transformer encoder-decoder (4 layers, 8 heads, d=256) with three innovations:

```
Input sentence
      │
   [Encoder]
   ┌─────────────────────────────────────────────┐
   │  Layer n:                                    │
   │  X̃ = (1-g_kg)·X + g_kg·W_kg·X  [Eq. 6]    │  ← KG gate
   │  LinkedAttn_n(X̃) = MHA_n(X̃) +             │  ← Linked attention
   │                    a_n · MHA_{n-1}(X̃)       │    (Eq. 1)
   │  CastleAttn = LinkedAttn + γ·LN(W_res·X̃)   │  ← Residual (Eq. 7)
   └─────────────────────────────────────────────┘
      │
 [Knowledge Graph]  ← g_kg = σ((c_n - c^l)/ε) · Σ w_c · p_c  [Eq. 5]
      │
   [Decoder]  + LinkedCrossAttn (Eq. 2)
      │
 [KG-Guided Decoding]  ← ℓ'[w] += Σ w_ij·ψ(r_ij)·𝟙[w=surface(v_j)]  [Eq. 8]
      │
Output (corrected sentence)
```

## Dataset

**IGED** (Indonesian Grammar Error correction Dataset) — 1.3M sentence pairs:
- 📊 [HuggingFace: syauqie/IGED](https://huggingface.co/datasets/syauqie/IGED)
- 35% morphological errors (affixation, word formation, reduplication)
- 40% syntactic errors (phrase structure, preposition, sentence completeness)
- 25% semantic errors (diction, ambiguity, pleonasm)

## Installation

```bash
git clone https://github.com/syauqie/castle-gec
cd castle-gec
pip install -r requirements.txt

# Install Sastrawi stemmer (Indonesian morphological analysis)
pip install PySastrawi
```

## Quick Start

### 1. Build Knowledge Graph

```bash
# With DeepSeek API (recommended — full neural relations)
export DEEPSEEK_API_KEY="sk-your-key-here"
python scripts/build_kg.py \
    --csv data/IGED_stratified_dataset.csv \
    --output data/castle_kg.pkl \
    --deepseek_key $DEEPSEEK_API_KEY \
    --max_neural 100000

# Without API (rule-based only — faster, slightly lower semantic performance)
python scripts/build_kg.py \
    --csv data/IGED_stratified_dataset.csv \
    --output data/castle_kg.pkl
```

### 2. Train

```bash
# Full training (uses HuggingFace dataset automatically)
python src/train.py --config configs/castle_base.yaml

# With local CSV
python src/train.py \
    --config configs/castle_base.yaml \
    --local_csv data/IGED_stratified_dataset.csv

# All-in-one script
bash scripts/run_train.sh
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --checkpoint checkpoints/castle/checkpoint_best.pt \
    --config configs/castle_base.yaml
```

### 4. Inference

```bash
# Single sentence
python src/inference.py \
    --checkpoint checkpoints/castle/checkpoint_best.pt \
    --sentence "Saya sudah pergi ke sana kemarin hari."

# Interactive mode
python src/inference.py \
    --checkpoint checkpoints/castle/checkpoint_best.pt \
    --interactive

# Batch file
python src/inference.py \
    --checkpoint checkpoints/castle/checkpoint_best.pt \
    --input_file errors.txt \
    --output_file corrected.txt
```

## Kaggle / Google Colab (Student-Friendly)

For resource-constrained environments:

```python
# 1. Install dependencies
!pip install PySastrawi sacrebleu tokenizers datasets openai networkx -q

# 2. Download dataset from HuggingFace
from datasets import load_dataset
ds = load_dataset("syauqie/IGED")

# 3. Quick KG build (subset for testing)
!python scripts/build_kg.py \
    --csv /path/to/IGED_stratified_dataset.csv \
    --output data/castle_kg.pkl \
    --max_samples 100000  # start small

# 4. Train with reduced batch size for T4 GPU
!python src/train.py \
    --config configs/castle_base.yaml \
    --local_csv /path/to/IGED_stratified_dataset.csv
```

**Recommended compute options (cheapest first):**
| Platform | GPU | Cost | Notes |
|----------|-----|------|-------|
| Kaggle | T4/P100 | Free | 30h/week GPU |
| Google Colab | T4 | Free | Limited runtime |
| Vast.ai | RTX 3090 | ~$0.30/hr | Full training ≈ $3 |
| Runpod | RTX 4090 | ~$0.44/hr | Fastest option |
| Colab Pro | V100/A100 | $10/mo | Good for iteration |

## File Structure

```
castle-gec/
├── src/
│   ├── castle_model.py    # CASTLE architecture (Eq. 1-8)
│   ├── knowledge_graph.py # KG construction (Algorithm 1)
│   ├── dataset.py         # IGED loader + WordPiece tokenizer
│   ├── train.py           # Training loop (Eq. 9-11)
│   ├── evaluate.py        # F1, BLEU, per-category metrics
│   └── inference.py       # Inference interface
├── scripts/
│   ├── build_kg.py        # Build KG from IGED CSV
│   └── run_train.sh       # Full pipeline script
├── configs/
│   └── castle_base.yaml   # All hyperparameters (Table 8)
└── requirements.txt
```

## Hyperparameters (Table 8)

| Parameter | Value |
|-----------|-------|
| Encoder/Decoder Layers | 4 |
| Attention Heads | 8 |
| Embedding Dim | 256 |
| FFN Dim | 2048 |
| Dropout | 0.3 |
| Optimizer | Adam (β₁=0.9, β₂=0.98) |
| Learning Rate | 5e-4 (inverse sqrt decay) |
| Warmup Steps | 4000 |
| Label Smoothing | 0.1 |
| Batch Size | 128 (update_freq=2) |
| Max Epochs | 10 (patience=5) |
| Tokenizer | WordPiece, 10K vocab |
| Total Parameters | ~34.7M |

## Citation

```bibtex
@article{marier2026castle,
  title={CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement
         for Low-Resource Grammar Correction},
  author={Marier, Syauqie Muhammad and Kong, Xiangjie and Zhu, Linan and
          Chen, Xiangfan and Badruzzaman, Abdulloh and Ramatryana, I Nyoman Apraz},
  journal={Expert Systems with Applications},
  volume={299},
  pages={130233},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2025.130233}
}
```

## License

MIT License — see [LICENSE](LICENSE).

---

*This work was supported by the National Natural Science Foundation of China (62176234, 62476247, 62072409) and Zhejiang Provincial Natural Science Foundation (LR21F020003).*
