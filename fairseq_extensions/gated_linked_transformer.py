# gated_linked_transformer.py
import torch
from torch import nn
import torch.nn.functional as F

from fairseq.models.transformer import (
    TransformerModel, 
    TransformerEncoder, 
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    base_architecture
)

from fairseq.models import (
    register_model,
    register_model_architecture
)

from .gated_linked_attention import GatedLinkedMultiheadAttention


class GatedLinkedTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer that uses GatedLinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention with GatedLinkedMultiheadAttention
        self.self_attn = GatedLinkedMultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class GatedLinkedTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer that uses GatedLinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention with GatedLinkedMultiheadAttention
        self.self_attn = GatedLinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        
        # Override encoder-decoder attention with GatedLinkedMultiheadAttention
        self.encoder_attn = GatedLinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )


class GatedLinkedTransformerEncoder(TransformerEncoder):
    """Transformer encoder using GatedLinkedTransformerEncoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        # Override layers with GatedLinkedTransformerEncoderLayer
        self.layers = nn.ModuleList([
            GatedLinkedTransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)


class GatedLinkedTransformerDecoder(TransformerDecoder):
    """Transformer decoder using GatedLinkedTransformerDecoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # Override layers with GatedLinkedTransformerDecoderLayer
        self.layers = nn.ModuleList([
            GatedLinkedTransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)
            if getattr(self.layers[i], "encoder_attn", None) is not None:
                self.layers[i].encoder_attn.set_prev_layer(self.layers[i-1].encoder_attn)


@register_model("gated_linked_transformer")
class GatedLinkedTransformerModel(TransformerModel):
    """
    Transformer model with GatedLinkedMultiheadAttention.
    """
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return GatedLinkedTransformerEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return GatedLinkedTransformerDecoder(args, tgt_dict, embed_tokens)


# Register model architecture
@register_model_architecture("gated_linked_transformer", "gated_linked_transformer")
def gated_linked_transformer_architecture(args):
    # Use base architecture from standard transformer
    base_architecture(args)