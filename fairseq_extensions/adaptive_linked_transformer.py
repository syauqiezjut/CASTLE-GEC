# adaptive_linked_transformer.py
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

from .adaptive_linked_attention import AdaptiveLinkedMultiheadAttention


class AdaptiveLinkedTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer that uses AdaptiveLinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention with AdaptiveLinkedMultiheadAttention
        self.self_attn = AdaptiveLinkedMultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class AdaptiveLinkedTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer that uses AdaptiveLinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention with AdaptiveLinkedMultiheadAttention
        self.self_attn = AdaptiveLinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        
        # Override encoder-decoder attention with AdaptiveLinkedMultiheadAttention
        self.encoder_attn = AdaptiveLinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )


class AdaptiveLinkedTransformerEncoder(TransformerEncoder):
    """Transformer encoder using AdaptiveLinkedTransformerEncoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        # Override layers with AdaptiveLinkedTransformerEncoderLayer
        self.layers = nn.ModuleList([
            AdaptiveLinkedTransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)


class AdaptiveLinkedTransformerDecoder(TransformerDecoder):
    """Transformer decoder using AdaptiveLinkedTransformerDecoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # Override layers with AdaptiveLinkedTransformerDecoderLayer
        self.layers = nn.ModuleList([
            AdaptiveLinkedTransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)
            if getattr(self.layers[i], "encoder_attn", None) is not None:
                self.layers[i].encoder_attn.set_prev_layer(self.layers[i-1].encoder_attn)


@register_model("adaptive_linked_transformer")
class AdaptiveLinkedTransformerModel(TransformerModel):
    """
    Transformer model with AdaptiveLinkedMultiheadAttention.
    """
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return AdaptiveLinkedTransformerEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AdaptiveLinkedTransformerDecoder(args, tgt_dict, embed_tokens)


# Register model architecture
@register_model_architecture("adaptive_linked_transformer", "adaptive_linked_transformer")
def adaptive_linked_transformer_architecture(args):
    # Use base architecture from standard transformer
    base_architecture(args)