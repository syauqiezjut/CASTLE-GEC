import torch
from torch import nn
import torch.nn.functional as F
import math
import json
from collections import defaultdict
import os

from .gated_linked_attention import GatedLinkedMultiheadAttention

class KGSemanticGraph:
    """Knowledge Graph untuk koreksi semantik Bahasa Indonesia"""
    
    def __init__(self, kg_path):
        print(f"Loading knowledge graph from {kg_path}")
        self.kg_path = kg_path
        self.nodes = {}
        self.edges = {}
        self.corrections = {}
        self.word_embeddings = {}
        
        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
                
            # Process nodes and build corrections dictionary
            for node in kg_data.get('nodes', []):
                node_id = node.get('id', '')
                node_type = node.get('type', '')
                label = node.get('label', '')
                
                self.nodes[node_id] = node
                
                # Track error words and their corrections
                if node_type == 'error_word':
                    self.corrections[label] = []
            
            # Process edges to find corrections
            for edge in kg_data.get('edges', []):
                source = edge.get('source', '')
                target = edge.get('target', '')
                relation = edge.get('relation', '')
                weight = edge.get('weight', 1)
                
                self.edges[edge.get('id', '')] = edge
                
                # Find correction relationships
                if relation.startswith('corrected_as_'):
                    # Extract source word (error) and target (correction)
                    source_node = self.nodes.get(source, {})
                    target_node = self.nodes.get(target, {})
                    
                    if source_node and target_node:
                        error_word = source_node.get('label', '')
                        correction = target_node.get('label', '')
                        
                        if error_word in self.corrections:
                            self.corrections[error_word].append({
                                'word': correction,
                                'weight': weight,
                                'category': relation.replace('corrected_as_', '')
                            })
            
            print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            print(f"Found {len(self.corrections)} error words with corrections")
            
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
    
    def get_correction(self, word):
        """Get potential corrections for a word"""
        return self.corrections.get(word.lower(), [])
    
    def get_category_weight(self, category):
        """Get weight based on semantic category"""
        category = category.lower() if category else ""
        
        # Set weights based on error category
        if "diksi" in category:
            return 0.8
        elif "ambigu" in category:
            return 0.7
        elif "pleonasme" in category:
            return 0.6
        else:
            return 0.3  # Default weight for other/unknown categories


class ResidualKGAttention(GatedLinkedMultiheadAttention):
    """Gated Linked Attention dengan Residual KG connection"""
    
    # def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True,
    #              add_bias_kv=False, add_zero_attn=False, self_attention=False,
    #              encoder_decoder_attention=False):
    #     super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
    #                      add_bias_kv, add_zero_attn, self_attention,
    #                      encoder_decoder_attention)
        
    #     # Load KG if available
    #     kg_path = "/ssd-data1/sq2023/belajar/semantic_kg.json"
    #     self.kg = None
    #     if os.path.exists(kg_path):
    #         try:
    #             self.kg = KGSemanticGraph(kg_path)
    #             print(f"KG loaded successfully for ResidualKGAttention")
                
    #             # Parameter untuk mengontrol pengaruh KG (trainable)
    #             self.kg_weight = nn.Parameter(torch.ones(1) * 0.3)
                
    #             # Proyeksi untuk representasi KG
    #             self.kg_projection = nn.Sequential(
    #                 nn.Linear(embed_dim, embed_dim),
    #                 nn.LayerNorm(embed_dim),
    #                 nn.ReLU(),
    #                 nn.Linear(embed_dim, embed_dim)
    #             )
                
    #             # Gate untuk mengontrol pengaruh KG per token
    #             self.kg_gate = nn.Sequential(
    #                 nn.Linear(embed_dim, embed_dim // 4),
    #                 nn.ReLU(),
    #                 nn.Linear(embed_dim // 4, 1),
    #                 nn.Sigmoid()
    #             )
                
    #             # Proyeksi untuk query/key dengan KG
    #             self.kg_q_proj = nn.Linear(embed_dim, embed_dim)
    #             self.kg_k_proj = nn.Linear(embed_dim, embed_dim)
                
    #             # Cache untuk mengurangi perhitungan berulang
    #             self.kg_cache = {}
                
    #         except Exception as e:
    #             print(f"Error initializing KG: {e}")
    #             self.kg = None

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True,
             add_bias_kv=False, add_zero_attn=False, self_attention=False,
             encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                        add_bias_kv, add_zero_attn, self_attention,
                        encoder_decoder_attention)
        
        # Load KG if available
        kg_path = "/ssd-data1/sq2023/belajar/semantic_kg_deepseek.json"  # Sesuaikan dengan path Anda
        self.kg = None
        if os.path.exists(kg_path):
            try:
                self.kg = KGSemanticGraph(kg_path)
                print(f"KG loaded successfully for ResidualKGAttention")
                
                # Parameter untuk mengontrol pengaruh KG (trainable)
                self.kg_weight = nn.Parameter(torch.ones(1) * 0.3)
                self.diksi_weight = nn.Parameter(torch.ones(1) * 0.8)
                self.ambigu_weight = nn.Parameter(torch.ones(1) * 0.7)
                self.pleonasme_weight = nn.Parameter(torch.ones(1) * 0.6)
                
                # Proyeksi untuk representasi KG
                self.kg_projection = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim)
                )
                
                # Gate untuk mengontrol pengaruh KG per token
                self.kg_gate = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 4),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 4, 1),
                    nn.Sigmoid()
                )
                
                # Proyeksi untuk query/key dengan KG
                self.kg_q_proj = nn.Linear(embed_dim, embed_dim)
                self.kg_k_proj = nn.Linear(embed_dim, embed_dim)
                
                # Semantic error classifier - untuk mendeteksi kesalahan semantik
                self.semantic_classifier = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, 3)  # 3 kategori semantik (diksi, ambigu, pleonasme)
                )
                
                # Context scorer - untuk menilai konteks
                self.context_scorer = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, 1),
                    nn.Sigmoid()
                )
                
                # Cache untuk mengurangi perhitungan berulang
                self.kg_cache = {}
                
            except Exception as e:
                print(f"Error initializing KG: {e}")
                self.kg = None
    
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
        token_embeddings=None,
    ):
        """Implementation of KG-enhanced gated linked attention dengan residual connection"""
        
        # Jika tidak ada KG, gunakan implementasi standar
        if self.kg is None:
            return super().forward(
                query, key, value, key_padding_mask, incremental_state,
                need_weights, static_kv, attn_mask, before_softmax, need_head_weights
            )
        
        # Simpan query/key original untuk residual connection
        original_query = query
        original_key = key
        
        # Hitung gate value berdasarkan konten
        # Shape: (tgt_len, bsz, 1)
        gate_value = self.kg_gate(query)
        
        # Clip gate value to ensure stability
        gate_value = torch.clamp(gate_value * self.kg_weight, 0.0, 0.8)
        
        # Get KG-enhanced query dan key
        kg_query = self.kg_q_proj(query)
        kg_key = self.kg_k_proj(key)
        
        # Combine dengan residual connection
        enhanced_query = original_query * (1 - gate_value) + kg_query * gate_value
        enhanced_key = original_key * (1 - gate_value) + kg_key * gate_value
        
        # Gunakan enhanced query/key untuk forward pass
        return super().forward(
            enhanced_query, enhanced_key, value, key_padding_mask, incremental_state,
            need_weights, static_kv, attn_mask, before_softmax, need_head_weights
        )