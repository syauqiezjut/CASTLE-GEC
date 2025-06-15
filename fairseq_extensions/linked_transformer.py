# linked_transformer.py
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

from .linked_attention import LinkedMultiheadAttention


class LinkedTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer yang menggunakan LinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention dengan LinkedMultiheadAttention
        self.self_attn = LinkedMultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class LinkedTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer yang menggunakan LinkedMultiheadAttention"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # Override self attention dengan LinkedMultiheadAttention
        self.self_attn = LinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        
        # Override encoder-decoder attention dengan LinkedMultiheadAttention
        self.encoder_attn = LinkedMultiheadAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )


class LinkedTransformerEncoder(TransformerEncoder):
    """Transformer encoder yang menggunakan LinkedTransformerEncoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        
        # Override layers dengan LinkedTransformerEncoderLayer
        self.layers = nn.ModuleList([
            LinkedTransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
        # Set previous layer untuk setiap layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)


class LinkedTransformerDecoder(TransformerDecoder):
    """Transformer decoder yang menggunakan LinkedTransformerDecoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # Override layers dengan LinkedTransformerDecoderLayer
        self.layers = nn.ModuleList([
            LinkedTransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
        # Set previous layer untuk setiap layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)
            self.layers[i].encoder_attn.set_prev_layer(self.layers[i-1].encoder_attn)


# Pastikan pendaftaran model dan arsitektur terlihat seperti ini
@register_model("linked_transformer")
class LinkedTransformerModel(TransformerModel):
    """
    Transformer model dengan LinkedMultiheadAttention.
    """
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LinkedTransformerEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LinkedTransformerDecoder(args, tgt_dict, embed_tokens)


# Daftarkan arsitektur model
@register_model_architecture("linked_transformer", "linked_transformer")
def linked_transformer_architecture(args):
    # Gunakan arsitektur dasar dari transformer standar
    base_architecture(args)