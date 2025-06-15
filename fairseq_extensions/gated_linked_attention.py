import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import MultiheadAttention

class AdaptiveLinkedAttention(MultiheadAttention):
    """Simplified Enhanced Linked Attention - dimension safe"""
    
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                        add_bias_kv, add_zero_attn, self_attention,
                        encoder_decoder_attention)
        
        # Simplified learnable parameters
        self.gate_weight = nn.Parameter(torch.ones(1) * 0.3)
        self.semantic_weight = nn.Parameter(torch.ones(1) * 0.8)
        self.morphological_weight = nn.Parameter(torch.ones(1) * 0.6)
        
        # Simple projection for context
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.prev_layer_output = None
        
    def set_prev_layer(self, prev_attention):
        """Set reference to previous layer"""
        self.prev_attention = prev_attention
    
    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, before_softmax=False,
                need_head_weights=False):
        
        # Standard attention computation
        attn_output, attn_weights = super().forward(
            query, key, value, key_padding_mask, incremental_state,
            need_weights, static_kv, attn_mask, before_softmax, need_head_weights
        )
        
        # Simple linking mechanism
        if hasattr(self, 'prev_attention') and self.prev_layer_output is not None:
            # Ensure same dimensions
            if self.prev_layer_output.size() == attn_output.size():
                # Simple weighted combination
                prev_context = self.context_proj(self.prev_layer_output)
                prev_context = self.layer_norm(prev_context)
                
                # Learnable gating
                enhanced_output = attn_output + self.gate_weight * prev_context
                
                # Update for next layer
                self.prev_layer_output = enhanced_output.detach()
                return enhanced_output, attn_weights
        
        # Update and return standard output
        self.prev_layer_output = attn_output.detach() if attn_output is not None else None
        return attn_output, attn_weights

# Backward compatibility
GatedLinkedMultiheadAttention = AdaptiveLinkedAttention
