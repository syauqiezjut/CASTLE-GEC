import torch
from torch import nn
import torch.nn.functional as F

from fairseq.models.transformer import (
    TransformerModel,
    base_architecture
)

from fairseq.models import (
    register_model,
    register_model_architecture
)

from .gated_linked_transformer import (
    GatedLinkedTransformerEncoder,
    GatedLinkedTransformerDecoder,
    GatedLinkedTransformerModel,
    GatedLinkedTransformerEncoderLayer,
    GatedLinkedTransformerDecoderLayer
)

from .residual_kg_attention import ResidualKGAttention
from .selective_kg_attention import SelectiveKGAttention


class ResidualKGEncoderLayer(GatedLinkedTransformerEncoderLayer):
    """Encoder layer with Residual KG Attention"""
    
    def __init__(self, args):
        super(GatedLinkedTransformerEncoderLayer, self).__init__(args)
        
        # Override self attention with ResidualKGAttention
        self.self_attn = ResidualKGAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class ResidualKGDecoderLayer(GatedLinkedTransformerDecoderLayer):
    """Decoder layer with Residual KG Attention"""
    
    def __init__(self, args):
        super(GatedLinkedTransformerDecoderLayer, self).__init__(args)
        
        # Override self attention with ResidualKGAttention
        self.self_attn = ResidualKGAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        
        # Override encoder-decoder attention with ResidualKGAttention
        self.encoder_attn = ResidualKGAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )


class SelectiveKGEncoderLayer(GatedLinkedTransformerEncoderLayer):
    """Encoder layer with Selective KG Attention"""
    
    def __init__(self, args):
        super(GatedLinkedTransformerEncoderLayer, self).__init__(args)
        
        # Override self attention with SelectiveKGAttention
        self.self_attn = SelectiveKGAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )


class SelectiveKGDecoderLayer(GatedLinkedTransformerDecoderLayer):
    """Decoder layer with Selective KG Attention"""
    
    def __init__(self, args):
        super(GatedLinkedTransformerDecoderLayer, self).__init__(args)
        
        # Override self attention with SelectiveKGAttention
        self.self_attn = SelectiveKGAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        
        # Override encoder-decoder attention with SelectiveKGAttention
        self.encoder_attn = SelectiveKGAttention(
            self.embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

    def apply_semantic_bias(self, logits, input_tokens, kg):
        """Apply semantic bias to logits based on knowledge graph"""
        # Get vocabulary
        vocab = self.dictionary
        batch_size, seq_len, vocab_size = logits.size()
        
        # Get tokens as words
        input_words = []
        for i in range(batch_size):
            sentence = []
            for j in range(input_tokens.size(1)):
                token_idx = input_tokens[i, j].item()
                if token_idx != self.dictionary.pad() and token_idx != self.dictionary.eos():
                    word = self.dictionary[token_idx]
                    sentence.append(word)
            input_words.append(sentence)
        
        # Process each item in the batch
        for i in range(batch_size):
            # Check input words against KG
            for word in input_words[i]:
                # Get potential corrections
                corrections = kg.get_correction(word.lower())
                
                if corrections:
                    # Apply bias to logits
                    for correction in corrections:
                        correct_word = correction['word']
                        category = correction['category']
                        weight = correction['weight']
                        
                        # Get index in vocab
                        try:
                            word_idx = vocab.index(correct_word)
                            
                            # Apply category-specific bias
                            bias = 0.0
                            if 'diksi' in category.lower():
                                bias = 0.15
                            elif 'ambigu' in category.lower():
                                bias = 0.12
                            elif 'pleonasme' in category.lower():
                                bias = 0.10
                            
                            # Apply bias across all positions
                            # Bias is higher for more probable positions
                            logits[:, :, word_idx] += bias * weight
                        except:
                            # Word not in vocab
                            continue
                
                # Gunakan collocations dari KG hanya sebagai fitur tambahan,
                # bukan sebagai kategori semantik terpisah
                collocations = kg.get_collocations(word.lower())
                if collocations:
                    for collocation in collocations:
                        collocate_word = collocation['word']
                        weight = collocation['weight']
                        
                        try:
                            word_idx = vocab.index(collocate_word)
                            
                            # Apply small collocation bias
                            logits[:, :, word_idx] += 0.05 * weight  # Lebih rendah dari bias kategori semantik
                        except:
                            continue
        
        return logits


class ResidualKGEncoder(GatedLinkedTransformerEncoder):
    """Transformer encoder using ResidualKGEncoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens):
        super(GatedLinkedTransformerEncoder, self).__init__(args, dictionary, embed_tokens)
        
        # Override layers with ResidualKGEncoderLayer
        self.layers = nn.ModuleList([
            ResidualKGEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)


class ResidualKGDecoder(GatedLinkedTransformerDecoder):
    """Transformer decoder using ResidualKGDecoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(GatedLinkedTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # Override layers with ResidualKGDecoderLayer
        self.layers = nn.ModuleList([
            ResidualKGDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)
            if getattr(self.layers[i], "encoder_attn", None) is not None:
                self.layers[i].encoder_attn.set_prev_layer(self.layers[i-1].encoder_attn)


class SelectiveKGEncoder(GatedLinkedTransformerEncoder):
    """Transformer encoder using SelectiveKGEncoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens):
        super(GatedLinkedTransformerEncoder, self).__init__(args, dictionary, embed_tokens)
        
        # Override layers with SelectiveKGEncoderLayer
        self.layers = nn.ModuleList([
            SelectiveKGEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)


class SelectiveKGDecoder(GatedLinkedTransformerDecoder):
    """Transformer decoder using SelectiveKGDecoderLayer"""
    
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(GatedLinkedTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        # Override layers with SelectiveKGDecoderLayer
        self.layers = nn.ModuleList([
            SelectiveKGDecoderLayer(args)
            for _ in range(args.decoder_layers)
        ])
        
        # Set previous layer for each layer
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.set_prev_layer(self.layers[i-1].self_attn)
            if getattr(self.layers[i], "encoder_attn", None) is not None:
                self.layers[i].encoder_attn.set_prev_layer(self.layers[i-1].encoder_attn)


@register_model("residual_kg_transformer")
class ResidualKGTransformerModel(GatedLinkedTransformerModel):
    """
    Transformer model with Residual KG-enhanced attention.
    """
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ResidualKGEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ResidualKGDecoder(args, tgt_dict, embed_tokens)


@register_model("selective_kg_transformer")
class SelectiveKGTransformerModel(GatedLinkedTransformerModel):
    """
    Transformer model with Selective KG-enhanced attention.
    """
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SelectiveKGEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SelectiveKGDecoder(args, tgt_dict, embed_tokens)

@register_model("castle_transformer")
class CastleTransformerModel(GatedLinkedTransformerModel):
    """
    CASTLE: Contrastive-Attention Semantic Transformer with Linked Dynamic Embedding
    untuk Low-Resource Language Error Correction
    """
    
    @staticmethod
    def add_args(parser):
        # Add model-specific arguments
        super(CastleTransformerModel, CastleTransformerModel).add_args(parser)
        parser.add_argument('--kg-path', type=str, default='/ssd-data1/sq2023/belajar/semantic_kg_deepseek.json',
                           help='Path to knowledge graph JSON file')
        parser.add_argument('--semantic-weight', type=float, default=0.7,
                           help='Weight for semantic errors')
        
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SelectiveKGEncoder(args, src_dict, embed_tokens)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SelectiveKGDecoder(args, tgt_dict, embed_tokens)

# Register architectures
@register_model_architecture("residual_kg_transformer", "residual_kg_transformer")
def residual_kg_transformer_architecture(args):
    base_architecture(args)


@register_model_architecture("selective_kg_transformer", "selective_kg_transformer")
def selective_kg_transformer_architecture(args):
    base_architecture(args)

@register_model_architecture("castle_transformer", "castle_transformer")
def castle_transformer_architecture(args):
    base_architecture(args)


 
    

#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 fairseq-train /ssd-data1/sq2023/belajar/data-bin-wordpiece \
#     --arch castle_transformer \
#     --user-dir /ssd-data1/sq2023/belajar/fairseq_extensions \
#     --kg-path /ssd-data1/sq2023/belajar/semantic_kg_deepseek.json \
#     --semantic-weight 0.7 \
#     --encoder-layers 4 \
#     --decoder-layers 4 \
#     --encoder-attention-heads 8 \
#     --decoder-attention-heads 8 \
#     --encoder-embed-dim 256 \
#     --decoder-embed-dim 256 \
#     --encoder-ffn-embed-dim 2048 \
#     --decoder-ffn-embed-dim 2048 \
#     --dropout 0.3 \
#     --attention-dropout 0.1 \
#     --share-decoder-input-output-embed \
#     --optimizer adam \
#     --adam-betas '(0.9, 0.98)' \
#     --clip-norm 0.1 \
#     --lr 5e-4 \
#     --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --max-tokens 1024 \
#     --batch-size 128 \
#     --update-freq 2 \
#     --save-dir /ssd-data1/sq2023/belajar/checkpoints/castle \
#     --max-epoch 10 \
#     --patience 5 \
#     --keep-best-checkpoints 2 \
#     --save-interval 1 \
#     --validate-interval 1 \
#     --no-epoch-checkpoints \
#     --log-format json \
#     --log-interval 50 \
#     --seed 42 \
#     --no-save-optimizer-state \
#     --skip-invalid-size-inputs-valid-test \
#     --best-checkpoint-metric loss

# CUDA_VISIBLE_DEVICES=5 fairseq-train /ssd-data1/sq2023/belajar/data-bin-wordpiece-semantic \
#     --arch castle_transformer \
#     --user-dir /ssd-data1/sq2023/belajar/fairseq_extensions \
#     --kg-path /ssd-data1/sq2023/belajar/semantic_kg_deepseek.json \
#     --semantic-weight 0.9 \
#     --criterion label_smoothed_cross_entropy \
#     --encoder-layers 4 \
#     --decoder-layers 4 \
#     --encoder-attention-heads 8 \
#     --decoder-attention-heads 8 \
#     --encoder-embed-dim 256 \
#     --decoder-embed-dim 256 \
#     --encoder-ffn-embed-dim 2048 \
#     --decoder-ffn-embed-dim 2048 \
#     --dropout 0.2 \
#     --attention-dropout 0.1 \
#     --share-decoder-input-output-embed \
#     --optimizer adam \
#     --adam-betas '(0.9, 0.98)' \
#     --clip-norm 0.1 \
#     --lr 1e-4 \
#     --lr-scheduler inverse_sqrt \
#     --warmup-updates 4000 \
#     --label-smoothing 0.1 \
#     --max-tokens 1024 \
#     --batch-size 128 \
#     --update-freq 4 \
#     --save-dir /ssd-data1/sq2023/belajar/checkpoints/castle_improved \
#     --max-epoch 15 \
#     --patience 7 \
#     --keep-best-checkpoints 2 \
#     --save-interval 1 \
#     --validate-interval 1 \
#     --no-epoch-checkpoints \
#     --log-format json \
#     --log-interval 50 \
#     --seed 42 \
#     --no-save-optimizer-state \
#     --skip-invalid-size-inputs-valid-test \
#     --best-checkpoint-metric loss \
#     > /ssd-data1/sq2023/belajar/checkpoints/castle_improved/training.log 2>&1


# CUDA_VISIBLE_DEVICES=5 fairseq-generate /ssd-data1/sq2023/belajar/data-bin-wordpiece \
#     --path /ssd-data1/sq2023/belajar/checkpoints/castle/checkpoint_best.pt \
#     --user-dir /ssd-data1/sq2023/belajar/fairseq_extensions \
#     --gen-subset test \
#     --beam 5 \
#     --remove-bpe \
#     --batch-size 32 \
#     --skip-invalid-size-inputs-valid-test \
#     --task translation \
#     --source-lang source \
#     --target-lang target \
#     > /ssd-data1/sq2023/belajar/checkpoints/castle/generate-test.txt

# python -c "
# import sys
# import os
# sys.path.append('/ssd-data1/sq2023/belajar')
# # from evaluate_model_new import extract_texts, calculate_metrics
# from train_with_eval_adaptive import extract_texts, calculate_word_f1, calculate_gleu, calculate_sacrebleu


# # Define paths
# output_dir = '/ssd-data1/sq2023/belajar/checkpoints/castle/metrics'
# os.makedirs(output_dir, exist_ok=True)
# gen_output = os.path.join(output_dir, 'generate-test.txt')
# src_file = os.path.join(output_dir, 'src.txt')
# ref_file = os.path.join(output_dir, 'ref.txt')
# hyp_file = os.path.join(output_dir, 'hyp.txt')

# # Langkah 1: Ekstrak teks sumber, referensi dan prediksi
# print("Extracting text...")
# extraction_success = extract_texts(gen_output, src_file, ref_file, hyp_file)
# if not extraction_success:
#     print("Failed to extract texts from generation output")
#     sys.exit(1)

# # Langkah 2: Hitung metrik
# print("Calculating metrics...")
# precision, recall, f1 = calculate_word_f1(hyp_file, ref_file, output_dir)

# try:
#     from nltk.translate.gleu_score import corpus_gleu
#     gleu = calculate_gleu(hyp_file, ref_file, output_dir)
# except ImportError:
#     print("NLTK not available for GLEU calculation")
#     gleu = None

# bleu = calculate_sacrebleu(hyp_file, ref_file, output_dir)

# # Langkah 3: Simpan metrik ke JSON
# import json
# metrics = {
#     'precision': precision,
#     'recall': recall,
#     'f1': f1,
#     'gleu': gleu,
#     'bleu': bleu
# }

# with open(os.path.join(output_dir, 'final_metrics.json'), 'w') as f:
#     json.dump(metrics, f, indent=4)

# print(f"Evaluation completed. Results saved to {os.path.join(output_dir, 'final_metrics.json')}")
# print(f"Evaluation results: Precision={precision}, Recall={recall}, F1={f1}, GLEU={gleu}, BLEU={bleu}")
# "