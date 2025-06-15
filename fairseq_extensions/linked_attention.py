import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq.modules.multihead_attention import MultiheadAttention


class LinkedMultiheadAttention(MultiheadAttention):
    """Multi-headed attention with linked attention mechanism"""

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                         add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention)
        
        # Simpan layer sebelumnya - akan diisi saat transformer dibuat
        self.prev_layer = None
    
    def set_prev_layer(self, prev_layer):
        self.prev_layer = prev_layer
    
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Implementasi dari linked attention"""
        
        # Jika tidak ada prev_layer, gunakan attention biasa
        if self.prev_layer is None:
            return super().forward(
                query, key, value, key_padding_mask, incremental_state,
                need_weights, static_kv, attn_mask, before_softmax, need_head_weights
            )
        
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim, \
            f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            # Hanya gunakan attention normal jika incremental_state digunakan
            return super().forward(
                query, key, value, key_padding_mask, incremental_state,
                need_weights, static_kv, attn_mask, before_softmax, need_head_weights
            )
        
        # Mendapatkan proyeksi Q, K, V dari layer saat ini
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Mendapatkan proyeksi Q, K dari layer sebelumnya
        prev_q = self.prev_layer.q_proj(query)
        
        if self.self_attention:
            # Untuk self-attention, K juga diambil dari prev_layer
            prev_k = self.prev_layer.k_proj(key)
        else:
            # Untuk cross-attention, gunakan K yang sama (dari encoder)
            prev_k = self.prev_layer.k_proj(key)
        
        # Reshape untuk multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        prev_q = prev_q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        prev_k = prev_k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Hitung attention scores
        current_attn_weights = torch.bmm(q, k.transpose(1, 2))
        prev_attn_weights = torch.bmm(prev_q, prev_k.transpose(1, 2))
        
        # Gabungkan current dan prev attention scores
        attn_weights = current_attn_weights + prev_attn_weights
        
        # Scaling by head dimension
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # Reshape key_padding_mask untuk broadcast
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v
            
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        # Gunakan dropout_module bukan dropout
        attn_probs = self.dropout_module(attn_weights)
        
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len)
            if need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.permute(0, 2, 3, 1)
            else:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)
                
            return attn, attn_weights
        else:
            return attn, None