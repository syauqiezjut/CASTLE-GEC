# CASTLE-GEC

## Overview
CASTLE (Context-Aware Semantic Transformer with Knowledge Graph Enhancement) is a novel framework for grammatical error correction in low-resource languages, specifically designed for Indonesian. The model integrates external linguistic knowledge graphs with linked attention mechanisms to achieve superior performance in morphological and semantic error correction.
Key Features

Knowledge Graph Integration: Heterogeneous semantic knowledge graph construction from error-correction pairs
Gated Linked Attention: Cross-layer information propagation with adaptive learned gating
Semantic Error Focus: Specialized handling of diksi (diction), ambigu (ambiguity), and pleonasm errors
Efficient Architecture: 91.5% fewer parameters than BART-large while maintaining competitive performance
Comprehensive Dataset: IGED dataset with 1.55M Indonesian error-correction pairs

Performance
ModelF1 ScoreBLEUParametersBaseline (Whitespace)0.893473.69718.2MWordPiece Baseline0.957992.3027.0MBART-large0.959290.83406.0MCASTLE0.962992.7234.7M
