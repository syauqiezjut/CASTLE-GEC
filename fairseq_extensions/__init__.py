from . import linked_attention
from . import linked_transformer

# Import both names for backward compatibility
from .gated_linked_attention import AdaptiveLinkedAttention, GatedLinkedMultiheadAttention
from .gated_linked_transformer import (
    GatedLinkedTransformerModel,
    GatedLinkedTransformerEncoder,
    GatedLinkedTransformerDecoder,
    GatedLinkedTransformerEncoderLayer,
    GatedLinkedTransformerDecoderLayer,
    gated_linked_transformer_architecture
)

# KG enhanced implementations
from .selective_kg_attention import SelectiveKGAttention
from .kg_transformer_implementation import (
    CastleTransformerModel,
    castle_transformer_architecture
)

__all__ = [
    'AdaptiveLinkedAttention',
    'GatedLinkedMultiheadAttention',  # Backward compatibility
    'GatedLinkedTransformerModel',
    'GatedLinkedTransformerEncoder',
    'GatedLinkedTransformerDecoder',
    'GatedLinkedTransformerEncoderLayer',
    'GatedLinkedTransformerDecoderLayer',
    'gated_linked_transformer_architecture',
    'SelectiveKGAttention',
    'CastleTransformerModel',
    'castle_transformer_architecture'
]
