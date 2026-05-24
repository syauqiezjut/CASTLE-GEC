"""
CASTLE: Context-Aware Semantic Transformer with Knowledge Graph Enhancement
for Low-Resource Grammar Correction

Paper: Expert Systems with Applications, Vol. 299, 2026
DOI: https://doi.org/10.1016/j.eswa.2025.130233
"""

from .castle_model import CASTLE, build_castle_model
from .knowledge_graph import CASTLEKnowledgeGraph
from .dataset import load_iged, get_dataloaders, build_wordpiece_tokenizer
from .inference import CASTLECorrector

__version__ = "1.0.0"
__all__ = [
    "CASTLE",
    "build_castle_model",
    "CASTLEKnowledgeGraph",
    "load_iged",
    "get_dataloaders",
    "build_wordpiece_tokenizer",
    "CASTLECorrector",
]
