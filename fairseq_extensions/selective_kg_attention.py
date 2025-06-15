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
        self.semantic_categories = ['diksi', 'ambigu', 'pleonasme']  # Kategori yang ada
        
        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
                
            # Process nodes
            for node in kg_data.get('nodes', []):
                node_id = node.get('id', '')
                node_type = node.get('type', '')
                label = node.get('label', '')
                
                self.nodes[node_id] = node
                
                # Track error words
                if node_type == 'error_word':
                    self.corrections[label] = []
            
            # Process edges
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
                        category = relation.replace('corrected_as_', '')
                        
                        if error_word in self.corrections:
                            self.corrections[error_word].append({
                                'word': correction,
                                'weight': weight,
                                'category': category
                            })
            
            # Relasi collocates juga ada, tapi tidak sebagai kategori kesalahan
            # Simpan sebagai informasi tambahan saja
            self.collocations = defaultdict(list)
            for edge_id, edge in self.edges.items():
                if edge.get('relation') == 'collocates':
                    source_id = edge.get('source', '')
                    target_id = edge.get('target', '')
                    weight = edge.get('weight', 1)
                    
                    if source_id in self.nodes and target_id in self.nodes:
                        source_word = self.nodes[source_id].get('label', '')
                        target_word = self.nodes[target_id].get('label', '')
                        
                        self.collocations[source_word].append({
                            'word': target_word,
                            'weight': weight
                        })
            
            print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            print(f"Found {len(self.corrections)} error words with corrections")
            
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
    
    def get_correction(self, word):
        """Get potential corrections for a word"""
        return self.corrections.get(word.lower(), [])
    
    def get_collocations(self, word):
        """Get words that collocate with this word (informasi tambahan)"""
        return self.collocations.get(word.lower(), [])
    
    def get_category_weight(self, category):
        """Get weight based on semantic category"""
        category = category.lower() if category else ""
        
        # Bobot berdasarkan kategori kesalahan
        if "diksi" in category:
            return 0.8
        elif "ambigu" in category:
            return 0.7
        elif "pleonasme" in category:
            return 0.6
        else:
            return 0.3  # Default weight
    
    def is_semantic_category(self, category):
        """Check if category is semantic"""
        if not category:
            return False
        return any(sem_cat in category.lower() for sem_cat in self.semantic_categories)

class SelectiveKGAttention(GatedLinkedMultiheadAttention):
    """Gated Linked Attention with selective KG integration"""
    
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True,
             add_bias_kv=False, add_zero_attn=False, self_attention=False,
             encoder_decoder_attention=False, kg_path=None):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                        add_bias_kv, add_zero_attn, self_attention,
                        encoder_decoder_attention)

        # Load KG
        kg_path = "/ssd-data1/sq2023/belajar/semantic_kg.json"
        self.kg = None
        if os.path.exists(kg_path):
            try:
                self.kg = KGSemanticGraph(kg_path)
                print(f"KG loaded successfully for SelectiveKGAttention")
                
                # Confidence predictor - determine when to use KG
                self.confidence_predictor = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 1),
                    nn.Sigmoid()
                )
                
                # Proyeksi untuk query/key
                self.kg_q_proj = nn.Linear(embed_dim, embed_dim)
                self.kg_k_proj = nn.Linear(embed_dim, embed_dim)
                
                # Category predictor - predict error category
                self.category_predictor = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim // 2, 3)  # 3 kategori semantik
                )
                
                # # Category-specific weights (trainable)
                # self.diksi_weight = nn.Parameter(torch.ones(1) * 0.6)
                # self.ambigu_weight = nn.Parameter(torch.ones(1) * 0.5)
                # self.pleonasme_weight = nn.Parameter(torch.ones(1) * 0.4)

                self.diksi_weight = nn.Parameter(torch.ones(1) * 0.9)  # Increase from 0.7
                self.ambigu_weight = nn.Parameter(torch.ones(1) * 0.8)  # Increase from 0.6
                self.pleonasme_weight = nn.Parameter(torch.ones(1) * 0.7)  # Increase from 0.5
                
                # Threshold untuk selective activation
                self.confidence_threshold = 0.5
                
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
        """Implementation of selectively KG-enhanced gated linked attention"""
        
        # Jika tidak ada KG, gunakan implementasi standar
        if self.kg is None:
            return super().forward(
                query, key, value, key_padding_mask, incremental_state,
                need_weights, static_kv, attn_mask, before_softmax, need_head_weights
            )
        
        # Shape: [tgt_len, bsz, embed_dim]
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        # Simpan type dan device dari tensors
        dtype = query.dtype
        device = query.device
        
        # Predict confidence of whether KG should be used
        # Shape: [tgt_len, bsz, 1]
        confidence = self.confidence_predictor(query)
        
        # Only use KG when confidence is high
        # Shape: [tgt_len, bsz, 1]
        kg_mask = (confidence > self.confidence_threshold).float()
        
        # Predict category untuk category-specific weighting
        # Shape: [tgt_len, bsz, 3] - hanya 3 kategori semantik (diksi, ambigu, pleonasme)
        category_logits = self.category_predictor(query)
        category_probs = F.softmax(category_logits, dim=-1)
        
        # Compute category-specific weights
        # Shape: [tgt_len, bsz, 1]
        category_weights = (
            category_probs[:, :, 0:1] * self.diksi_weight + 
            category_probs[:, :, 1:2] * self.ambigu_weight + 
            category_probs[:, :, 2:3] * self.pleonasme_weight
        )
        
        # Enhanced gate that combines confidence and category weights
        # Shape: [tgt_len, bsz, 1]
        selective_gate = kg_mask * category_weights
        
        # Ensure gate is clamped to reasonable values
        selective_gate = torch.clamp(selective_gate, 0.0, 0.7)
        
        # Project original query/key
        kg_query = self.kg_q_proj(query)
        kg_key = self.kg_k_proj(key)
        
        # CRITICAL FIX: Pastikan dimensi sesuai untuk operasi perkalian elemen
        # Reshape selective_gate untuk broadcast jika perlu
        if query.size(0) != key.size(0):
            # Untuk kasus encoder-decoder attention, di mana dimensi query dan key berbeda
            # Gunakan standard attention tanpa KG enhancement
            return super().forward(
                query, key, value, key_padding_mask, incremental_state,
                need_weights, static_kv, attn_mask, before_softmax, need_head_weights
            )
        
        # Apply selective integration
        enhanced_query = query * (1 - selective_gate) + kg_query * selective_gate
        enhanced_key = key * (1 - selective_gate) + kg_key * selective_gate
        
        # Use enhanced representations
        return super().forward(
            enhanced_query, enhanced_key, value, key_padding_mask, incremental_state,
            need_weights, static_kv, attn_mask, before_softmax, need_head_weights
        )
    # def forward(
    #     self,
    #     query,
    #     key,
    #     value,
    #     key_padding_mask=None,
    #     incremental_state=None,
    #     need_weights=True,
    #     static_kv=False,
    #     attn_mask=None,
    #     before_softmax=False,
    #     need_head_weights=False,
    #     token_embeddings=None,
    # ):
    #     """Implementation of selectively KG-enhanced gated linked attention"""
        
    #     # Jika tidak ada KG, gunakan implementasi standar
    #     if self.kg is None:
    #         return super().forward(
    #             query, key, value, key_padding_mask, incremental_state,
    #             need_weights, static_kv, attn_mask, before_softmax, need_head_weights
    #         )
        
    #     # Shape: [tgt_len, bsz, embed_dim]
    #     tgt_len, bsz, embed_dim = query.size()
    #     src_len = key.size(0)
        
    #     # Simpan type dan device dari tensors
    #     dtype = query.dtype
    #     device = query.device
        
    #     # Predict confidence of whether KG should be used
    #     # Shape: [tgt_len, bsz, 1]
    #     confidence = self.confidence_predictor(query)
        
    #     # Only use KG when confidence is high
    #     # Shape: [tgt_len, bsz, 1]
    #     kg_mask = (confidence > self.confidence_threshold).float()
        
    #     # Predict category for category-specific weighting
    #     # Shape: [tgt_len, bsz, 3]
    #     category_logits = self.category_predictor(query)
    #     category_probs = F.softmax(category_logits, dim=-1)
        
    #     # Compute category-specific weights
    #     # Shape: [tgt_len, bsz, 1]
    #     category_weights = (
    #         category_probs[:, :, 0:1] * self.diksi_weight + 
    #         category_probs[:, :, 1:2] * self.ambigu_weight + 
    #         category_probs[:, :, 2:3] * self.pleonasme_weight
    #     )
        
    #     # Enhanced gate that combines confidence and category weights
    #     # Shape: [tgt_len, bsz, 1]
    #     selective_gate = kg_mask * category_weights
        
    #     # Ensure gate is clamped to reasonable values
    #     selective_gate = torch.clamp(selective_gate, 0.0, 0.7)
        
    #     # Project original query/key
    #     kg_query = self.kg_q_proj(query)
    #     kg_key = self.kg_k_proj(key)
        
    #     # CRITICAL FIX: Pastikan dimensi sesuai untuk operasi perkalian elemen
    #     # Reshape selective_gate untuk broadcast jika perlu
    #     if query.size(0) != key.size(0):
    #         # Untuk kasus encoder-decoder attention, di mana dimensi query dan key berbeda
    #         # Gunakan standard attention tanpa KG enhancement
    #         return super().forward(
    #             query, key, value, key_padding_mask, incremental_state,
    #             need_weights, static_kv, attn_mask, before_softmax, need_head_weights
    #         )
        
    #     # Apply selective integration
    #     enhanced_query = query * (1 - selective_gate) + kg_query * selective_gate
    #     enhanced_key = key * (1 - selective_gate) + kg_key * selective_gate
        
    #     # Use enhanced representations
    #     return super().forward(
    #         enhanced_query, enhanced_key, value, key_padding_mask, incremental_state,
    #         need_weights, static_kv, attn_mask, before_softmax, need_head_weights
    #     )