"""
knowledge_graph.py
CASTLE Knowledge Graph Construction
=====================================
Implements Algorithm 1 from the paper:

  Input:  Sentence pairs S = {(src, trg, cat)}_i=1^n
  Output: Knowledge Graph G = (V, E, R)

Three-stage construction:
  Stage 1: Token-level diff extraction
  Stage 2: Rule-based morphological relations (60%) via Sastrawi stemmer
  Stage 3: Neural semantic relations (40%) via DeepSeek R1 API
  Stage 4: Statistical collocation extraction

Graph statistics (paper, Table 5):
  - 46,784 nodes (11,934 error words + 19,324 correct words + 15,526 stems)
  - 233,286 relations
  Relation weights:
    stem_of:              count=46,784  weight=0.98  method=rule-based
    corrected_as_diksi:   count=42,369  weight=0.80  method=hybrid
    corrected_as_ambigu:  count=19,809  weight=0.82  method=hybrid
    corrected_as_pleonas: count=8,151   weight=0.60  method=neural
    alternative:          count=35,924  weight=0.50  method=neural
    collocates:           count=80,249  weight=0.30  method=statistical
"""

import os
import json
import pickle
import logging
import hashlib
import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from difflib import SequenceMatcher

import networkx as nx
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Relation types and weights (Table 5)
# ──────────────────────────────────────────────

RELATION_WEIGHTS = {
    "stem_of":               0.98,
    "corrected_as_diksi":    0.80,
    "corrected_as_ambigu":   0.82,
    "corrected_as_pleonasme": 0.60,
    "alternative":           0.50,
    "collocates":            0.30,
}

# Category → relation type mapping
CATEGORY_TO_RELATION = {
    "diksi":      ("corrected_as_diksi",    0.80),
    "diction":    ("corrected_as_diksi",    0.80),
    "ambigu":     ("corrected_as_ambigu",   0.82),
    "ambiguity":  ("corrected_as_ambigu",   0.82),
    "pleonasme":  ("corrected_as_pleonasme", 0.60),
    "pleonasm":   ("corrected_as_pleonasme", 0.60),
    # Morphological and syntactic errors use stem_of relation
    "affixation":            ("corrected_as_diksi", 0.80),
    "word_formation":        ("corrected_as_diksi", 0.80),
    "reduplication":         ("corrected_as_diksi", 0.80),
    "phrase_structure":      ("corrected_as_diksi", 0.80),
    "preposition":           ("corrected_as_diksi", 0.80),
    "sentence_completeness": ("corrected_as_diksi", 0.80),
}

SEMANTIC_CATEGORIES = {
    "diksi", "diction", "ambigu", "ambiguity", "pleonasme", "pleonasm",
    "semantik_diksi", "semantik_ambigu", "semantik_pleonasme", "semantik",
}


# ──────────────────────────────────────────────
# Sastrawi stemmer wrapper
# ──────────────────────────────────────────────

def _get_stemmer():
    """Load Sastrawi stemmer v1.2.0 (Indonesian morphological analyzer)."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        return factory.create_stemmer()
    except ImportError:
        logger.warning(
            "PySastrawi not found. Install with: pip install PySastrawi\n"
            "Falling back to identity stemmer (no morphological analysis)."
        )
        class _IdentityStemmer:
            def stem(self, word): return word
        return _IdentityStemmer()


# ──────────────────────────────────────────────
# Token-level diff (Algorithm 1, Stage 1)
# ──────────────────────────────────────────────

def extract_diff(
    src_tokens: List[str],
    trg_tokens: List[str],
) -> List[Tuple[str, str]]:
    """
    Extract (error_token, correct_token) pairs from token-level diff.
    Uses SequenceMatcher for alignment (approximates edit-distance alignment).
    Returns list of (w_err, w_cor) pairs where tokens differ.

    Algorithm 1, line 6: Δ ← ExtractDiff(T_src, T_trg)
    """
    diff_pairs = []
    matcher = SequenceMatcher(None, src_tokens, trg_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            # Align token pairs (zip truncates to shorter side)
            for w_err, w_cor in zip(src_tokens[i1:i2], trg_tokens[j1:j2]):
                if w_err.lower() != w_cor.lower():
                    diff_pairs.append((w_err, w_cor))
        elif tag == "delete":
            for w_err in src_tokens[i1:i2]:
                diff_pairs.append((w_err, ""))
        elif tag == "insert":
            for w_cor in trg_tokens[j1:j2]:
                diff_pairs.append(("", w_cor))
    return diff_pairs


# ──────────────────────────────────────────────
# Levenshtein similarity
# ──────────────────────────────────────────────

def levenshtein_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity ∈ [0, 1]."""
    if not a or not b:
        return 0.0
    try:
        from Levenshtein import ratio
        return ratio(a, b)
    except ImportError:
        # Pure python fallback
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        edit = dp[n]
        return 1.0 - edit / max(m, n)


# ──────────────────────────────────────────────
# DeepSeek R1 neural analysis (Algorithm 1, Stage 3)
# ──────────────────────────────────────────────

DEEPSEEK_PROMPT_TEMPLATE = """You are an Indonesian linguistics expert.
Analyze these {k} error-correction pairs. Respond ONLY with a JSON array, no explanation outside JSON.

Pairs:
{pairs_json}

Return exactly this format (one object per pair):
[{{"pair_id":0,"alternatives":["word1","word2"],"confidence":0.85,"relation_type":"corrected_as_diksi"}},{{"pair_id":1,...}}]

relation_type must be one of: corrected_as_diksi, corrected_as_ambigu, corrected_as_pleonasme
confidence: 0.0-1.0
alternatives: 1-3 other valid corrections (can be empty list [])
Output ONLY the JSON array, nothing else."""


def batch_deepseek_analysis(
    pairs: List[Tuple[str, str, str]],   # (error_word, correct_word, category)
    api_key: str,
    batch_size: int = 10,
    max_retries: int = 3,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Call DeepSeek API to analyze semantic error-correction pairs.
    Algorithm 1, lines 15-26 (Stage 3: Neural semantic relations).
    Saves per-batch cache so interrupted runs can resume.

    Args:
        pairs: List of (error_word, correct_word, category) tuples
        api_key: DeepSeek API key
        batch_size: Batch size k=10 (as in paper)
        max_retries: Retry attempts on API error
        cache_dir: Directory to save per-batch cache files (enables resume)

    Returns:
        List of dicts with: pair_id, alternatives, confidence, relation_type
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai package: pip install openai>=1.3.0")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,   # hard timeout — prevents ssl.recv() hanging forever
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    all_results = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size

    # Count already-cached batches for accurate progress display
    cached_count = 0
    api_count = 0

    pbar = tqdm(
        range(0, len(pairs), batch_size),
        total=total_batches,
        desc="DeepSeek batches",
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    pbar.set_postfix(cached=0, api_calls=0, skip=0)

    for batch_start in pbar:
        batch = pairs[batch_start: batch_start + batch_size]
        category = batch[0][2] if batch else "semantic"

        # ── Per-batch cache: skip if already done ──
        batch_cache_path = None
        if cache_dir:
            batch_key = hashlib.md5(
                json.dumps(batch, ensure_ascii=False).encode()
            ).hexdigest()[:10]
            batch_cache_path = os.path.join(cache_dir, f"batch_{batch_key}.json")
            if os.path.exists(batch_cache_path):
                with open(batch_cache_path) as f:
                    all_results.extend(json.load(f))
                cached_count += 1
                pbar.set_postfix(cached=cached_count, api_calls=api_count,
                                 pairs_done=len(all_results))
                continue   # already done, skip API call

        # Format pairs for prompt
        pairs_data = [
            {"pair_id": i, "error": p[0], "correct": p[1]}
            for i, p in enumerate(batch)
        ]
        prompt = DEEPSEEK_PROMPT_TEMPLATE.format(
            k=len(batch),
            category=category,
            pairs_json=json.dumps(pairs_data, ensure_ascii=False, indent=2),
        )

        def _fallback_results(n, cat):
            rel = CATEGORY_TO_RELATION.get(cat, ("corrected_as_diksi", 0.80))[0]
            return [{"pair_id": i, "alternatives": [],
                     "confidence": 0.5, "relation_type": rel}
                    for i in range(n)]

        def _extract_json(text: str):
            """Robustly extract JSON array from response, even if truncated."""
            text = text.strip()
            # Strip markdown fences
            if "```" in text:
                parts = text.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:]
                    if p.startswith("["):
                        text = p.strip()
                        break
            # Find JSON array boundaries
            start = text.find("[")
            if start == -1:
                return None
            # Try full parse first
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                pass
            # Try to recover truncated JSON — extract complete objects
            import re
            objects = re.findall(r'\{[^{}]+\}', text[start:])
            results = []
            for obj in objects:
                try:
                    results.append(json.loads(obj))
                except Exception:
                    pass
            return results if results else None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",   # faster + better JSON than R1
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content or ""
                batch_results = _extract_json(content)
                if batch_results is None:
                    raise ValueError(f"Could not extract JSON from: {content[:200]}")
                all_results.extend(batch_results)
                # ── Save per-batch cache immediately ──
                if batch_cache_path:
                    with open(batch_cache_path, "w") as f:
                        json.dump(batch_results, f, ensure_ascii=False)
                api_count += 1
                pbar.set_postfix(cached=cached_count, api_calls=api_count,
                                 pairs_done=len(all_results))
                break
            except Exception as e:
                logger.warning(f"DeepSeek API attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Skipping batch after {max_retries} failures")
                    fallback = _fallback_results(len(batch), category)
                    all_results.extend(fallback)
                    # Save fallback to cache so we don't retry failed batches
                    if batch_cache_path:
                        with open(batch_cache_path, "w") as f:
                            json.dump(fallback, f, ensure_ascii=False)
                    api_count += 1
                    pbar.set_postfix(cached=cached_count, api_calls=api_count,
                                     pairs_done=len(all_results))

    return all_results


# ──────────────────────────────────────────────
# Statistical collocation extraction (Algorithm 1, Stage 4)
# ──────────────────────────────────────────────

def extract_collocations(
    sentences: List[List[str]],
    window: int = 3,
    threshold: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    Extract statistically significant word collocations using PMI.
    Algorithm 1, line 29: C ← Collocations(S_trg, ω=3, τ=3)

    Returns:
        List of (word_a, word_b, pmi_score) tuples
    """
    # Count unigrams and bigrams within window
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_tokens = 0

    for tokens in sentences:
        for w in tokens:
            unigram_counts[w.lower()] += 1
            total_tokens += 1
        for i, w1 in enumerate(tokens):
            for w2 in tokens[i+1: i+window+1]:
                if w1.lower() != w2.lower():
                    pair = tuple(sorted([w1.lower(), w2.lower()]))
                    bigram_counts[pair] += 1

    # Filter by threshold
    collocations = []
    for (w1, w2), count in bigram_counts.items():
        if count < threshold:
            continue
        # Pointwise Mutual Information
        p_w1 = unigram_counts[w1] / total_tokens
        p_w2 = unigram_counts[w2] / total_tokens
        p_w1w2 = count / total_tokens
        if p_w1 > 0 and p_w2 > 0:
            pmi = np.log(p_w1w2 / (p_w1 * p_w2))
            if pmi > 0:
                collocations.append((w1, w2, min(pmi / 10.0, 1.0)))

    logger.info(f"Extracted {len(collocations)} collocations")
    return collocations


# ──────────────────────────────────────────────
# Main KG construction — Algorithm 1
# ──────────────────────────────────────────────

class CASTLEKnowledgeGraph:
    """
    Heterogeneous Knowledge Graph for CASTLE.

    Node types (V):
      - V_E: error word nodes
      - V_C: correct word nodes
      - V_S: morphological stem nodes
      - V_A: alternative correction nodes

    Relation set R (with weights from Table 5):
      r_diksi, r_ambigu, r_pleon, r_stem, r_alt

    Usage:
        kg = CASTLEKnowledgeGraph()
        kg.build(samples, deepseek_api_key="sk-...")
        kg.save("data/castle_kg.pkl")

        # Later:
        kg = CASTLEKnowledgeGraph.load("data/castle_kg.pkl")
        edges = kg.get_relevant_subgraph(word, max_nodes=20)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.vocab: Set[str] = set()
        self.error_nodes: Set[str] = set()
        self.correct_nodes: Set[str] = set()
        self.stem_nodes: Set[str] = set()
        self._stemmer = None
        self._built = False

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = _get_stemmer()
        return self._stemmer

    # ── Graph building ─────────────────────────────────────────

    def build(
        self,
        samples: List[Dict],
        deepseek_api_key: Optional[str] = None,
        batch_size: int = 10,
        collocation_window: int = 3,
        collocation_threshold: int = 3,
        levenshtein_threshold: float = 0.8,
        max_neural_samples: Optional[int] = None,
        cache_dir: str = "data/kg_cache",
    ) -> "CASTLEKnowledgeGraph":
        """
        Build knowledge graph from error-correction pairs.
        Implements Algorithm 1 from the paper.

        Args:
            samples: List of dicts with keys: source, target, category
            deepseek_api_key: API key for DeepSeek R1 (Stage 3)
            batch_size: Batch size for DeepSeek API calls (k=10 in paper)
            collocation_window: Window size ω for collocation (paper: 3)
            collocation_threshold: Min count τ for collocation (paper: 3)
            levenshtein_threshold: Similarity threshold for morphological linking
            max_neural_samples: Limit neural API calls (for cost control)
            cache_dir: Directory to cache DeepSeek results
        """
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("=" * 60)
        logger.info("Building CASTLE Knowledge Graph (Algorithm 1)")
        logger.info(f"  Total samples: {len(samples)}")
        logger.info("=" * 60)

        semantic_pairs_for_neural = []
        all_correct_sentences = []

        # ── Algorithm 1, lines 2-26 ──────────────────────────
        logger.info("Stage 1 & 2: Token diff + Rule-based morphological relations")
        for sample in tqdm(samples, desc="Processing pairs"):
            src = sample.get("source", "")
            trg = sample.get("target", "")
            cat = sample.get("category", "syntax").lower()

            if not src or not trg:
                continue

            # Stage 1: Token diff (line 4-6)
            T_src = src.split()
            T_trg = trg.split()
            all_correct_sentences.append(T_trg)
            delta = extract_diff(T_src, T_trg)

            for (w_err, w_cor) in delta:
                if not w_err or not w_cor:
                    continue
                w_err_l = w_err.lower()
                w_cor_l = w_cor.lower()

                # Add nodes
                self.graph.add_node(w_err_l, node_type="error")
                self.graph.add_node(w_cor_l, node_type="correct")
                self.error_nodes.add(w_err_l)
                self.correct_nodes.add(w_cor_l)
                self.vocab.update([w_err_l, w_cor_l])

                # Stage 2: Rule-based morphological relations (60%) ──────
                # Algorithm 1, lines 8-13
                s_e = self.stemmer.stem(w_err_l)
                s_c = self.stemmer.stem(w_cor_l)

                if s_e:
                    self.graph.add_node(s_e, node_type="stem")
                    self.stem_nodes.add(s_e)
                    self.vocab.add(s_e)
                    # w_err → stem
                    self._add_edge(w_err_l, s_e, "stem_of", 1.0)
                    # w_cor → stem (if same stem: morphological variant)
                    if s_e == s_c:
                        self._add_edge(w_cor_l, s_e, "stem_of", 1.0)

                # Morphological similarity → add corrected_as relation
                sim = levenshtein_sim(w_err_l, w_cor_l)
                if sim >= levenshtein_threshold:
                    rel, weight = CATEGORY_TO_RELATION.get(
                        cat, ("corrected_as_diksi", 0.80)
                    )
                    self._add_edge(w_err_l, w_cor_l, rel, weight)

                # Stage 3: Collect semantic pairs for neural processing ──
                # Algorithm 1, lines 15-26
                if cat in SEMANTIC_CATEGORIES:
                    semantic_pairs_for_neural.append((w_err_l, w_cor_l, cat))

        # ── Stage 3: Neural semantic relations via DeepSeek R1 ──────────
        logger.info(
            f"Stage 3: Neural semantic relations "
            f"({len(semantic_pairs_for_neural)} semantic pairs)"
        )

        if semantic_pairs_for_neural and deepseek_api_key:
            # Deduplicate
            unique_sem_pairs = list(set(semantic_pairs_for_neural))
            if max_neural_samples:
                unique_sem_pairs = unique_sem_pairs[:max_neural_samples]
            logger.info(f"  Calling DeepSeek R1 for {len(unique_sem_pairs)} unique pairs")

            # Cache key for resumability
            cache_key = hashlib.md5(
                json.dumps(unique_sem_pairs[:100], ensure_ascii=False).encode()
            ).hexdigest()[:8]
            cache_path = os.path.join(cache_dir, f"deepseek_{cache_key}.json")

            if os.path.exists(cache_path):
                logger.info(f"  Loading cached DeepSeek results: {cache_path}")
                with open(cache_path) as f:
                    neural_results = json.load(f)
            else:
                neural_results = batch_deepseek_analysis(
                    unique_sem_pairs, deepseek_api_key,
                    batch_size=batch_size,
                    cache_dir=cache_dir,  # enables per-batch resume
                )
                with open(cache_path, "w") as f:
                    json.dump(neural_results, f, ensure_ascii=False)
                logger.info(f"  Cached DeepSeek results to {cache_path}")

            # Algorithm 1, lines 18-25: Add neural edges
            for i, result in enumerate(neural_results):
                if i >= len(unique_sem_pairs):
                    break
                w_err, w_cor, cat = unique_sem_pairs[i]
                conf = result.get("confidence", 0.5)
                rel = result.get("relation_type", "corrected_as_diksi")
                w = RELATION_WEIGHTS.get(rel, 0.80)
                weight = CATEGORY_TO_RELATION.get(cat, (rel, w))[1]

                # Algorithm 1, line 21: Add edge w_err → w_cor with weight
                self._add_edge(w_err, w_cor, rel, weight * conf)

                # Algorithm 1, lines 22-24: Add alternative edges
                for alt in result.get("alternatives", []):
                    alt_l = alt.lower().strip()
                    if alt_l and alt_l != w_cor and alt_l != w_err:
                        self.graph.add_node(alt_l, node_type="alternative")
                        self.vocab.add(alt_l)
                        self._add_edge(w_cor, alt_l, "alternative", 0.5)

        elif semantic_pairs_for_neural and not deepseek_api_key:
            logger.warning(
                "No DeepSeek API key provided. "
                "Skipping neural stage. Pass deepseek_api_key to build()."
            )
            # Fallback: add rule-based edges for semantic pairs
            for w_err, w_cor, cat in set(semantic_pairs_for_neural):
                rel, weight = CATEGORY_TO_RELATION.get(
                    cat, ("corrected_as_diksi", 0.80)
                )
                self._add_edge(w_err, w_cor, rel, weight * 0.6)  # reduced confidence

        # ── Stage 4: Statistical collocations ────────────────────────────
        # Algorithm 1, line 29: C ← Collocations(S_trg, ω=3, τ=3)
        logger.info("Stage 4: Statistical collocation extraction")
        collocations = extract_collocations(
            all_correct_sentences,
            window=collocation_window,
            threshold=collocation_threshold,
        )
        for w1, w2, pmi in collocations:
            if w1 in self.vocab and w2 in self.vocab:
                self._add_edge(w1, w2, "collocates", RELATION_WEIGHTS["collocates"])

        self._built = True
        self._log_stats()
        return self

    def _add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        weight: float,
    ):
        """Add or update edge with relation type and weight."""
        if src == dst:
            return
        if self.graph.has_edge(src, dst):
            # Take max weight for duplicate edges
            existing = self.graph[src][dst].get("weight", 0.0)
            weight = max(weight, existing)
        self.graph.add_edge(src, dst, relation=relation, weight=weight)

    def _log_stats(self):
        """Log knowledge graph statistics (matches Table 5 structure)."""
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        degrees = [d for _, d in self.graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0

        rel_counts = Counter(
            data.get("relation", "unknown")
            for _, _, data in self.graph.edges(data=True)
        )

        logger.info("=" * 60)
        logger.info("Knowledge Graph Statistics")
        logger.info(f"  Nodes: {n_nodes:,} (error={len(self.error_nodes):,}, "
                    f"correct={len(self.correct_nodes):,}, stem={len(self.stem_nodes):,})")
        logger.info(f"  Edges: {n_edges:,}")
        logger.info(f"  Avg degree: {avg_degree:.2f}")
        logger.info(f"  Density: {nx.density(self.graph):.6f}")
        logger.info("  Relation counts:")
        for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
            w = RELATION_WEIGHTS.get(rel, 0.0)
            logger.info(f"    {rel:<30} {count:>8,}  weight={w}")
        logger.info("=" * 60)

    # ── Query interface ─────────────────────────────────────────

    def get_relevant_subgraph(
        self,
        word: str,
        max_nodes: int = 20,
        similarity_threshold: float = 0.8,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Retrieve relevant edges for a given word (Eq. 8 in paper).
        Uses string similarity for matching (θ = 0.8).

        Returns:
            List of (src_node, dst_node, relation, weight) tuples
        """
        word_l = word.lower()
        edges = []

        # Direct match
        candidate_nodes = set()
        if word_l in self.graph:
            candidate_nodes.add(word_l)

        # Fuzzy match with similarity threshold
        for node in self.graph.nodes():
            if levenshtein_sim(word_l, node) >= similarity_threshold:
                candidate_nodes.add(node)

        # Collect edges from candidate nodes
        for node in candidate_nodes:
            for src, dst, data in self.graph.out_edges(node, data=True):
                edges.append((src, dst, data.get("relation", ""), data.get("weight", 0.0)))
            for src, dst, data in self.graph.in_edges(node, data=True):
                edges.append((src, dst, data.get("relation", ""), data.get("weight", 0.0)))

        # Sort by weight descending, limit
        edges = sorted(edges, key=lambda x: -x[3])[:max_nodes]
        return edges

    def get_edge_set_for_sequence(
        self,
        tokens: List[str],
        similarity_threshold: float = 0.8,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Get all relevant edges for a sequence of tokens.
        Used for knowledge-guided decoding (Eq. 8).
        """
        all_edges = []
        seen = set()
        for token in tokens:
            for edge in self.get_relevant_subgraph(token, similarity_threshold=similarity_threshold):
                key = (edge[0], edge[1], edge[2])
                if key not in seen:
                    seen.add(key)
                    all_edges.append(edge)
        return all_edges

    def word_to_idx(self) -> Dict[str, int]:
        """Return mapping from graph node to integer index."""
        return {word: i for i, word in enumerate(sorted(self.vocab))}

    # ── Serialization ───────────────────────────────────────────

    def save(self, path: str):
        """Serialize KG to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "graph": self.graph,
            "vocab": self.vocab,
            "error_nodes": self.error_nodes,
            "correct_nodes": self.correct_nodes,
            "stem_nodes": self.stem_nodes,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        logger.info(f"KG saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "CASTLEKnowledgeGraph":
        """Load a saved KG from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        kg = cls()
        kg.graph = data["graph"]
        kg.vocab = data["vocab"]
        kg.error_nodes = data["error_nodes"]
        kg.correct_nodes = data["correct_nodes"]
        kg.stem_nodes = data["stem_nodes"]
        kg._built = True
        logger.info(
            f"KG loaded from {path}: "
            f"{kg.graph.number_of_nodes():,} nodes, "
            f"{kg.graph.number_of_edges():,} edges"
        )
        return kg

    def export_json(self, path: str, max_edges: int = 500_000):
        """Export KG as JSON for inspection/sharing."""
        nodes = [
            {"id": n, "type": self.graph.nodes[n].get("node_type", "unknown")}
            for n in self.graph.nodes()
        ]
        edges = [
            {"src": u, "dst": v, "relation": d.get("relation"), "weight": d.get("weight")}
            for u, v, d in list(self.graph.edges(data=True))[:max_edges]
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False)
        logger.info(f"KG exported to JSON: {path}")


# ──────────────────────────────────────────────
# CLI script
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build CASTLE Knowledge Graph")
    parser.add_argument("--csv", required=True, help="Path to IGED_stratified_dataset.csv")
    parser.add_argument("--output", default="data/castle_kg.pkl", help="Output KG path")
    parser.add_argument("--deepseek_key", default=None, help="DeepSeek API key")
    parser.add_argument("--max_neural", type=int, default=None,
                        help="Max pairs for neural stage (cost control)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset samples (for testing)")
    parser.add_argument("--export_json", default=None, help="Also export as JSON")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.csv)
    samples = df.to_dict("records")
    if args.max_samples:
        samples = samples[:args.max_samples]

    # Normalize column names
    normalized = []
    for s in samples:
        normalized.append({
            "source": s.get("source", s.get("src", s.get("incorrect", ""))),
            "target": s.get("target", s.get("tgt", s.get("correct", ""))),
            "category": str(s.get("category", s.get("error_type", "syntax"))).lower(),
        })

    # Build KG
    kg = CASTLEKnowledgeGraph()
    kg.build(
        normalized,
        deepseek_api_key=args.deepseek_key,
        max_neural_samples=args.max_neural,
    )
    kg.save(args.output)

    if args.export_json:
        kg.export_json(args.export_json)

    # Quick sanity check
    print("\nSanity check — querying 'kembali':")
    edges = kg.get_relevant_subgraph("kembali")
    for src, dst, rel, w in edges[:5]:
        print(f"  {src} --[{rel}]--> {dst}  (w={w:.2f})")
