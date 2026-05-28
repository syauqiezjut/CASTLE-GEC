"""
castle_model.py
CASTLE Model Architecture (v2 — matches original training code)
================================================================
Context-Aware Semantic Transformer with Knowledge Graph Enhancement.

Implementation notes (based on original fairseq_extensions analysis):

  Eq. (1)  LinkedSelfAttn_n: pre-softmax score linking, MLP content-dependent gate,
           DISABLED during inference (eval mode) — matches original Fairseq extension
           behavior with incremental_state check.

  Eq. (2)  LinkedCrossAttn_n: output-level linking with scalar gate β_n (unchanged).

  Eq. (3-6) KG encoder gate: was a placeholder (no-op) in original code — NOT implemented.

  Eq. (7)  Residual γ·LayerNorm(W_res·X) retained in encoder.

  Eq. (8)  KG logit bias at decoding: the ONLY active KG component at inference.

  Loss: standard cross-entropy + label smoothing ONLY (Eq. 10-11 auxiliary losses
        were not used in original training — lambda_kg=0, lambda_reg=0).

  Decoding: beam search with beam_size=5, length_penalty=0.6.

Model config (Table 8):
  Encoder/Decoder layers: 4
  Attention heads:        8
  Embedding dim:          256
  FFN dim:                2048
  Dropout:                0.3
  Total params:           ~34.7M
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants from paper
# ──────────────────────────────────────────────

# Knowledge-guided decoding bias weights ψ(r) — Eq. (8)
PSI = {
    "corrected_as_diksi":    0.15,
    "corrected_as_ambigu":   0.12,
    "corrected_as_pleonasme": 0.10,
    "alternative":           0.05,
    "stem_of":               0.05,
    "collocates":            0.02,
}


# ──────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend PE on-the-fly for sequences longer than pre-computed buffer.
            # Uses same sinusoidal formula — no learnable params, safe to compute.
            d_model = self.pe.size(2)
            pos = torch.arange(seq_len, device=x.device).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2, device=x.device).float()
                * (-math.log(10000.0) / d_model)
            )
            pe_ext = torch.zeros(1, seq_len, d_model, device=x.device)
            pe_ext[0, :, 0::2] = torch.sin(pos * div)
            pe_ext[0, :, 1::2] = torch.cos(pos * div)
            return self.dropout(x + pe_ext)
        return self.dropout(x + self.pe[:, :seq_len])


# ──────────────────────────────────────────────
# Gated Linked Self-Attention (Eq. 1)
# ──────────────────────────────────────────────

class GatedLinkedSelfAttention(nn.Module):
    """
    Implements Equation (1) — pre-softmax score linking with MLP gate.

    During training:
        scores_n = Q_n @ K_n^T / sqrt(d)
        scores = scores_n + gate(Q_n) * (Q_{n-1} @ K_{n-1}^T / sqrt(d))
        attn = softmax(scores) @ V

    During inference (eval mode):
        Standard multi-head attention (no linking).
        Matches original code behavior: incremental_state → super().forward()

    Gate: content-dependent MLP over query, shape [B, S, 1].
    Paper reports learned gate values: {0.3000, 0.1415, 0.1383, 0.1285} for layers 0-3.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Manual Q/K/V projections (needed to intercept scores pre-softmax)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Content-dependent MLP gate (original code: Linear→ReLU→Linear→Sigmoid)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """[B, L, D] → [B*H, L, head_dim]"""
        B = x.size(0)
        x = x.view(B, seq_len, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)           # [B, H, L, head_dim]
        return x.reshape(B * self.n_heads, seq_len, self.head_dim)

    def _merge_heads(self, x: torch.Tensor, B: int, seq_len: int) -> torch.Tensor:
        """[B*H, L, head_dim] → [B, L, D]"""
        x = x.view(B, self.n_heads, seq_len, self.head_dim)
        x = x.permute(0, 2, 1, 3)           # [B, L, H, head_dim]
        return x.reshape(B, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,                              # [B, S, D]
        prev_q: Optional[torch.Tensor] = None,        # [B*H, S, head_dim] from layer n-1
        prev_k: Optional[torch.Tensor] = None,        # [B*H, S, head_dim] from layer n-1
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True=valid
        attn_mask: Optional[torch.Tensor] = None,         # [S, S] True=mask-out (causal)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            out:   [B, S, D] — attention output with residual+norm
            Q:     [B*H, S, head_dim] — query projections (for next layer linking)
            K:     [B*H, S, head_dim] — key projections (for next layer linking)
        """
        B, S, D = x.shape
        BH = B * self.n_heads

        Q = self._split_heads(self.q_proj(x), S)   # [B*H, S, head_dim]
        K = self._split_heads(self.k_proj(x), S)
        V = self._split_heads(self.v_proj(x), S)

        # Scaled dot-product scores [B*H, S, S]
        scale = math.sqrt(self.head_dim)
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale

        # Pre-softmax linking (TRAINING ONLY — disabled during eval/inference)
        if self.training and prev_q is not None and prev_k is not None:
            gate = self.gate_proj(x)                         # [B, S, 1]
            gate_bh = (gate.unsqueeze(1)
                       .expand(B, self.n_heads, S, 1)
                       .reshape(BH, S, 1))                   # [B*H, S, 1]
            prev_scores = torch.bmm(prev_q, prev_k.transpose(1, 2)) / scale
            scores = scores + gate_bh * prev_scores

        # Apply causal mask (decoder self-attention)
        if attn_mask is not None:
            # attn_mask: [S, S] True=mask-out
            scores = scores.masked_fill(attn_mask.unsqueeze(0), float("-inf"))

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] True=valid → invert to True=pad
            pad_mask = (~key_padding_mask)             # [B, S] True=pad
            pad_mask_bh = (pad_mask
                           .unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
                           .expand(B, self.n_heads, S, S)
                           .reshape(BH, S, S))
            scores = scores.masked_fill(pad_mask_bh, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: [B*H, S, head_dim]
        attn_out_bh = torch.bmm(attn_weights, V)
        attn_out = self._merge_heads(attn_out_bh, B, S)   # [B, S, D]
        attn_out = self.out_proj(attn_out)

        out = self.norm(x + self.dropout(attn_out))
        return out, Q, K


# ──────────────────────────────────────────────
# Gated Linked Cross-Attention (Eq. 2)
# ──────────────────────────────────────────────

class GatedLinkedCrossAttention(nn.Module):
    """
    Implements Equation (2) — output-level linking with scalar gate β_n:
      LinkedCrossAttn_n(Q,K,V) = CrossAttn_n(Q,K,V) + β_n · CrossAttn_{n-1}(Q,K,V)

    Note: cross-attention uses standard nn.MultiheadAttention with output-level linking
    (not pre-softmax), consistent with paper Eq. 2 description.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)
        # β_n scalar gate — initialized to 0.3
        self.gate = nn.Parameter(torch.tensor(0.3))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,                  # [B, T, D] decoder query
        key: torch.Tensor,                    # [B, S, D] encoder key
        value: torch.Tensor,                  # [B, S, D] encoder value
        prev_cross_attn: Optional[torch.Tensor],  # [B, T, D] from layer n-1
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # key_padding_mask convention: True=valid → invert for nn.MultiheadAttention
        kpm = (~key_padding_mask) if key_padding_mask is not None else None
        cross_out, _ = self.attn(query, key, value, key_padding_mask=kpm)

        # Eq. (2): output-level linking
        if prev_cross_attn is not None:
            linked = cross_out + self.gate * prev_cross_attn
        else:
            linked = cross_out

        out = self.norm(query + self.dropout(linked))
        return out, cross_out


# ──────────────────────────────────────────────
# Feed-Forward Network
# ──────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# ──────────────────────────────────────────────
# CASTLE Encoder Layer
# ──────────────────────────────────────────────

class CASTLEEncoderLayer(nn.Module):
    """
    Encoder layer:
      1. Gated linked self-attention (Eq. 1, pre-softmax MLP gate)
      2. Residual γ·LayerNorm(W_res·X)  [Eq. 7, simplified without KG gate]
      3. Feed-forward network
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.linked_self_attn = GatedLinkedSelfAttention(d_model, n_heads,
                                                         dropout, attn_dropout)
        # Learnable residual weight γ (Eq. 7) — initialized to 0.3
        self.gamma = nn.Parameter(torch.tensor(0.3))
        self.W_res = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm_res = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,                       # [B, S, D]
        prev_q: Optional[torch.Tensor],        # [B*H, S, head_dim]
        prev_k: Optional[torch.Tensor],        # [B*H, S, head_dim]
        src_mask: Optional[torch.Tensor],      # [B, S] True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Linked self-attention (Eq. 1)
        attn_out, Q, K = self.linked_self_attn(x, prev_q, prev_k,
                                                key_padding_mask=src_mask)
        # Residual (Eq. 7 simplified: KG gate is no-op → X̃ = X)
        out = attn_out + self.gamma * self.layer_norm_res(self.W_res(x))
        out = self.ffn(out)
        return out, Q, K


# ──────────────────────────────────────────────
# CASTLE Decoder Layer
# ──────────────────────────────────────────────

class CASTLEDecoderLayer(nn.Module):
    """
    Decoder layer:
      1. Masked linked self-attention (Eq. 1)
      2. Linked cross-attention with encoder (Eq. 2)
      3. Learnable residual γ
      4. Feed-forward network
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        # Masked self-attention (linked, Eq. 1)
        self.self_attn = GatedLinkedSelfAttention(d_model, n_heads,
                                                   dropout, attn_dropout)
        # Cross-attention with encoder (linked output-level, Eq. 2)
        self.cross_attn = GatedLinkedCrossAttention(d_model, n_heads,
                                                     dropout, attn_dropout)
        # Learnable residual
        self.gamma = nn.Parameter(torch.tensor(0.3))
        self.W_res = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm_res = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, D]
        enc_out: torch.Tensor,                    # [B, S, D]
        prev_self_q: Optional[torch.Tensor],      # [B*H, T, head_dim]
        prev_self_k: Optional[torch.Tensor],      # [B*H, T, head_dim]
        prev_cross_attn: Optional[torch.Tensor],  # [B, T, D]
        tgt_mask: Optional[torch.Tensor] = None,  # [T, T] causal mask
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Masked self-attention (Eq. 1)
        x, self_q, self_k = self.self_attn(x, prev_self_q, prev_self_k,
                                            attn_mask=tgt_mask)

        # Cross-attention (Eq. 2)
        x, cross_out = self.cross_attn(x, enc_out, enc_out, prev_cross_attn,
                                        key_padding_mask=src_key_padding_mask)

        # Residual
        x = x + self.gamma * self.layer_norm_res(self.W_res(x))
        x = self.ffn(x)

        return x, self_q, self_k, cross_out


# ──────────────────────────────────────────────
# Full CASTLE Model
# ──────────────────────────────────────────────

class CASTLE(nn.Module):
    """
    CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement.

    Architecture (Table 8):
      - 4 encoder + 4 decoder layers
      - 8 attention heads
      - d_model = 256, d_ff = 2048
      - Dropout = 0.3, Attention dropout = 0.1
      - WordPiece tokenizer, 10K vocab
      - ~34.7M parameters
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 2048,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
        kg_enabled: bool = True,
        kg_bias_weights: Optional[Dict[str, float]] = None,
        kg_similarity_threshold: float = 0.8,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.kg_enabled = kg_enabled
        self.kg_similarity_threshold = kg_similarity_threshold
        self.kg_bias_weights = kg_bias_weights or PSI

        # Embeddings
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            CASTLEEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            CASTLEDecoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Knowledge graph (attached externally)
        self.kg = None
        self.kg_vocab_to_idx: Optional[Dict[str, int]] = None
        self._kg_token_id_cache: Optional[Dict[str, List[int]]] = None

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers; scaled normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def set_knowledge_graph(self, kg, vocab_to_idx: Optional[Dict[str, int]] = None):
        """Attach a CASTLEKnowledgeGraph for KG logit bias at decoding (Eq. 8)."""
        self.kg = kg
        self.kg_vocab_to_idx = vocab_to_idx
        self._kg_token_id_cache = None

    def build_token_id_cache(self, tokenizer, vocab_size: int, pad_idx: int):
        """
        Precompute token IDs for all KG nodes once.
        Eliminates tokenizer.encode() from inner decode loop.
        Call once before training/inference loop.
        """
        if self.kg is None or not hasattr(self.kg, "_fast_lookup"):
            return
        cache = {}
        for node in self.kg._fast_lookup:
            try:
                enc = tokenizer.encode(node)
                ids = [tid for tid in enc.ids if tid != pad_idx and 0 <= tid < vocab_size]
                if ids:
                    cache[node] = ids
            except Exception:
                pass
        self._kg_token_id_cache = cache
        logger.info(f"KG token ID cache built: {len(cache):,} nodes mapped")

    # ──────────────────────────────────────────
    # Encode / Decode
    # ──────────────────────────────────────────

    def encode(self, src_tokens: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run encoder with linked self-attention.
        Returns [B, S, D].
        """
        x = self.pos_enc(self.src_embed(src_tokens) * math.sqrt(self.d_model))
        prev_q = prev_k = None
        for layer in self.encoder_layers:
            x, prev_q, prev_k = layer(x, prev_q, prev_k, src_mask)
        return self.enc_norm(x)

    def decode(
        self,
        tgt_tokens: torch.Tensor,   # [B, T]
        enc_out: torch.Tensor,      # [B, S, D]
        src_mask: torch.Tensor,     # [B, S] True=valid
    ) -> torch.Tensor:
        """Run decoder. Returns logits [B, T, V]."""
        T = tgt_tokens.size(1)
        # Causal mask: True = mask-out (upper triangle, excluding diagonal)
        tgt_mask = torch.triu(
            torch.ones(T, T, device=tgt_tokens.device), diagonal=1
        ).bool()

        x = self.pos_enc(self.tgt_embed(tgt_tokens) * math.sqrt(self.d_model))
        prev_self_q = prev_self_k = prev_cross = None
        for layer in self.decoder_layers:
            x, prev_self_q, prev_self_k, prev_cross = layer(
                x, enc_out, prev_self_q, prev_self_k, prev_cross,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_mask,
            )
        x = self.dec_norm(x)
        return self.output_proj(x)  # [B, T, V]

    # ──────────────────────────────────────────
    # Forward (training)
    # ──────────────────────────────────────────

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: torch.Tensor,
        src_texts: Optional[List[str]] = None,
        tokenizer=None,
        kg_edges_per_sample: Optional[List[List]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass (teacher forcing).

        Returns dict:
          logits:    [B, T, V]  raw decoder output
          kg_logits: [B, T, V]  after KG logit bias (Eq. 8)
          enc_out:   [B, S, D]
        """
        enc_out = self.encode(src_tokens, src_mask)
        logits = self.decode(tgt_tokens, enc_out, src_mask)

        # Eq. (8): Knowledge-guided decoding bias
        kg_logits = self._apply_kg_bias(logits, src_texts, tokenizer, kg_edges_per_sample)

        return {
            "logits": logits,
            "kg_logits": kg_logits,
            "enc_out": enc_out,
        }

    # ──────────────────────────────────────────
    # KG Logit Bias (Eq. 8)
    # ──────────────────────────────────────────

    def _precompute_kg_bias(
        self,
        B: int,
        src_texts: Optional[List[str]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Build [B, V] KG bias tensor from source texts.
        Returns None if KG not available.
        Uses numpy bincount vectorized scatter-add (~100× faster than Python loops).
        """
        if self.kg is None or src_texts is None or self._kg_token_id_cache is None:
            return None

        b_idx, t_idx, vals = [], [], []
        for b, src_text in enumerate(src_texts):
            edges = self.kg.get_edge_set_for_sequence(
                src_text.lower().split(), self.kg_similarity_threshold
            )
            for (_, dst_node, relation, weight) in edges:
                psi = self.kg_bias_weights.get(relation, 0.0)
                if psi == 0.0:
                    continue
                for tid in self._kg_token_id_cache.get(dst_node, []):
                    b_idx.append(b)
                    t_idx.append(tid)
                    vals.append(float(weight * psi))

        if not vals:
            return None

        flat_idx = (np.array(b_idx, dtype=np.int64) * self.vocab_size
                    + np.array(t_idx, dtype=np.int64))
        flat = np.bincount(flat_idx,
                           weights=np.array(vals, dtype=np.float32),
                           minlength=B * self.vocab_size)
        return torch.from_numpy(flat.reshape(B, self.vocab_size).astype(np.float32)).to(device)

    def _apply_kg_bias(
        self,
        logits: torch.Tensor,
        src_texts: Optional[List[str]],
        tokenizer,
        kg_edges_per_sample: Optional[List[List]] = None,
    ) -> torch.Tensor:
        """
        Equation (8): ℓ_t'[w] = ℓ_t[w] + Σ w_ij · ψ(r_ij) · 𝟙[w=surface(v_j)]

        Bias is constant across all T decoder steps (depends only on source).
        Single GPU transfer via [B, V] broadcast over T.
        """
        if self.kg is None and kg_edges_per_sample is None:
            return logits
        if src_texts is None and kg_edges_per_sample is None:
            return logits

        B, T, V = logits.shape
        device = logits.device

        b_idx, t_idx, vals = [], [], []
        for b in range(B):
            if kg_edges_per_sample is not None:
                edges = kg_edges_per_sample[b]
            elif src_texts is not None and self.kg is not None:
                edges = self.kg.get_edge_set_for_sequence(
                    src_texts[b].lower().split(), self.kg_similarity_threshold
                )
            else:
                continue

            for (_, dst_node, relation, weight) in edges:
                psi = self.kg_bias_weights.get(relation, 0.0)
                if psi == 0.0:
                    continue
                if self._kg_token_id_cache is not None:
                    token_ids = self._kg_token_id_cache.get(dst_node, [])
                else:
                    try:
                        enc = tokenizer.encode(dst_node)
                        token_ids = [tid for tid in enc.ids if 0 <= tid < V]
                    except Exception:
                        token_ids = []
                for tid in token_ids:
                    b_idx.append(b)
                    t_idx.append(tid)
                    vals.append(float(weight * psi))

        if not vals:
            return logits

        flat_idx = (np.array(b_idx, dtype=np.int64) * V
                    + np.array(t_idx, dtype=np.int64))
        flat_bias = np.bincount(flat_idx,
                                weights=np.array(vals, dtype=np.float32),
                                minlength=B * V)
        bias_gpu = (torch.from_numpy(flat_bias.reshape(B, V).astype(np.float32))
                    .to(device=device, dtype=logits.dtype)
                    .unsqueeze(1))  # [B, 1, V] → broadcast over T
        return logits + bias_gpu

    # ──────────────────────────────────────────
    # Beam Search Generation
    # ──────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        src_tokens: torch.Tensor,     # [B, S]
        src_mask: torch.Tensor,       # [B, S]
        tokenizer,
        src_texts: Optional[List[str]] = None,
        max_len: int = 128,
        bos_id: int = 2,
        eos_id: int = 3,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        max_len_a: float = 1.2,
        max_len_b: int = 10,
    ) -> List[List[int]]:
        """
        Beam search decoding with KG logit bias (Eq. 8).

        max_len_a / max_len_b: dynamic per-sample length limit, matching Fairseq convention.
            max_output = max_len_a * src_len + max_len_b  (e.g. 1.2 * 30 + 10 = 46)
            If max_len_a == 0.0, falls back to fixed max_len for all samples.
        length_penalty: Google length penalty exponent. 1.0 = Fairseq default.
        Returns list of token-ID lists (one per batch item).
        Note: model is in eval() mode here, so linked attention is disabled.
        """
        B = src_tokens.size(0)
        device = src_tokens.device
        V = self.vocab_size
        S = src_tokens.size(1)

        # ── Per-sample dynamic max length (Fairseq: max_len_a / max_len_b) ──
        if max_len_a > 0.0:
            # src_mask: True = real token; count real tokens per sample
            src_lens = src_mask.sum(dim=1).float()           # [B]
            per_sample_max = (max_len_a * src_lens + max_len_b).long()  # [B]
            per_sample_max = per_sample_max.clamp(min=1)
            max_len = int(per_sample_max.max().item())        # global loop limit
        else:
            per_sample_max = torch.full((B,), max_len, dtype=torch.long, device=device)

        # Encode once [B, S, D]
        enc_out = self.encode(src_tokens, src_mask)
        D = enc_out.size(2)

        # Precompute KG bias [B, V]
        kg_bias = self._precompute_kg_bias(B, src_texts, device)

        # Expand encoder outputs for beam search: [B*beam, S, D]
        enc_out_exp = (enc_out.unsqueeze(1)
                       .expand(B, beam_size, S, D)
                       .contiguous().view(B * beam_size, S, D))
        src_mask_exp = (src_mask.unsqueeze(1)
                        .expand(B, beam_size, S)
                        .contiguous().view(B * beam_size, S))

        # Expand KG bias: [B*beam, V]
        if kg_bias is not None:
            kg_bias_exp = (kg_bias.unsqueeze(1)
                           .expand(B, beam_size, V)
                           .contiguous().view(B * beam_size, V))
        else:
            kg_bias_exp = None

        # Beam scores [B, beam] — log space; only first beam is active
        beam_scores = torch.full((B, beam_size), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0

        # Sequences [B, beam, max_len+1], initialized with BOS
        max_seq = max_len + 1
        seqs = torch.full((B, beam_size, max_seq), self.pad_idx,
                          dtype=torch.long, device=device)
        seqs[:, :, 0] = bos_id

        done = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            cur_len = step + 1  # length of seqs so far

            # Build input [B*beam, cur_len]
            tgt = seqs[:, :, :cur_len].view(B * beam_size, cur_len)

            # Decode [B*beam, cur_len, V] → take last step
            logits = self.decode(tgt, enc_out_exp, src_mask_exp)
            step_logits = logits[:, -1, :]  # [B*beam, V]

            # Apply KG bias
            if kg_bias_exp is not None:
                step_logits = step_logits + kg_bias_exp

            log_probs = F.log_softmax(step_logits.float(), dim=-1)  # [B*beam, V]
            log_probs = log_probs.view(B, beam_size, V)              # [B, beam, V]

            # For finished beams: only EOS is allowed (score 0), rest -inf
            eos_filler = torch.full((V,), float("-inf"), device=device)
            eos_filler[eos_id] = 0.0
            log_probs = torch.where(
                done.unsqueeze(-1),            # [B, beam, 1]
                eos_filler.view(1, 1, V),
                log_probs,
            )

            # Candidate scores: [B, beam, V] → [B, beam*V]
            candidate_scores = beam_scores.unsqueeze(-1) + log_probs
            candidate_scores = candidate_scores.view(B, beam_size * V)

            # Select top-beam_size candidates
            top_scores, top_indices = candidate_scores.topk(beam_size, dim=-1)
            beam_ids = top_indices // V   # [B, beam] — which beam they came from
            token_ids = top_indices % V   # [B, beam] — which token

            # Reorder sequences according to selected beams
            batch_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
            new_seqs = seqs[batch_idx, beam_ids]                      # [B, beam, max_seq]
            new_seqs[:, :, cur_len] = token_ids
            seqs = new_seqs

            # Reorder done flags and update
            done = done[batch_idx, beam_ids]  # [B, beam]
            done = done | (token_ids == eos_id)

            # Per-sample length limit: force done for samples that hit their max
            step_limit = ((step + 1) >= per_sample_max.unsqueeze(1))  # [B, 1]
            done = done | step_limit

            beam_scores = top_scores  # [B, beam]

            if done.all():
                break

        # Select best sequence (length-normalized score)
        results = []
        for b in range(B):
            best_score = float("-inf")
            best_seq: List[int] = [bos_id]
            for k in range(beam_size):
                seq = seqs[b, k].tolist()
                # Trim at EOS
                try:
                    eos_pos = seq.index(eos_id)
                    seq = seq[: eos_pos + 1]
                except ValueError:
                    pass  # no EOS found — keep full sequence
                length = max(len(seq), 1)
                # Google length penalty: lp = ((5 + len) / 6)^α
                lp = ((5 + length) / 6) ** length_penalty
                score = beam_scores[b, k].item() / lp
                if score > best_score:
                    best_score = score
                    best_seq = seq
            results.append(best_seq)

        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────

def build_castle_model(cfg: Dict, vocab_size: int, pad_idx: int = 0) -> CASTLE:
    """Build CASTLE model from config dict (matches castle_base.yaml)."""
    m = cfg.get("model", cfg)
    model = CASTLE(
        vocab_size=vocab_size,
        d_model=m.get("encoder_embed_dim", 256),
        n_heads=m.get("encoder_attention_heads", 8),
        d_ff=m.get("encoder_ffn_embed_dim", 2048),
        n_enc_layers=m.get("encoder_layers", 4),
        n_dec_layers=m.get("decoder_layers", 4),
        dropout=m.get("dropout", 0.3),
        attn_dropout=m.get("attention_dropout", 0.1),
        pad_idx=pad_idx,
        kg_enabled=m.get("kg_enabled", True),
        kg_bias_weights={
            "corrected_as_diksi":    m.get("kg_bias_diksi", 0.15),
            "corrected_as_ambigu":   m.get("kg_bias_ambigu", 0.12),
            "corrected_as_pleonasme": m.get("kg_bias_pleonasm", 0.10),
            "alternative":           m.get("kg_bias_alternative", 0.05),
        },
        kg_similarity_threshold=m.get("kg_string_similarity_threshold", 0.8),
    )
    n_params = model.count_parameters()
    logger.info(f"CASTLE model built: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    return model


# ──────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    model = CASTLE(vocab_size=10_000, d_model=256, n_heads=8,
                   d_ff=2048, n_enc_layers=4, n_dec_layers=4,
                   dropout=0.3, attn_dropout=0.1)

    B, S, T = 4, 20, 15
    src = torch.randint(4, 10_000, (B, S))
    tgt = torch.randint(4, 10_000, (B, T))
    src_mask = torch.ones(B, S, dtype=torch.bool)

    # Training mode (linked attention active)
    model.train()
    out = model(src, tgt, src_mask)
    print(f"[train] logits: {out['logits'].shape}")     # [4, 15, 10000]
    print(f"[train] kg_logits: {out['kg_logits'].shape}")

    # Eval mode (linked attention disabled)
    model.eval()
    bos_id = tokenizer_bos = 2
    eos_id = 3
    gen = model.generate(src, src_mask, tokenizer=None,
                         max_len=20, bos_id=bos_id, eos_id=eos_id, beam_size=5)
    print(f"[eval] generated sequences: {[len(g) for g in gen]}")
    print(f"Total params: {model.count_parameters():,}")
