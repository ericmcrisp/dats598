"""
Enhanced EvidenceRetriever with query optimization, result reranking,
and better deduplication strategies.
"""

from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.claim import Claim
from app.models.evidence import Evidence, EvidenceSource
from app.utils.faiss_vecdb import FaissVecDB
from app.core.config import settings


class ImprovedEvidenceRetriever:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        self.vector_db = FaissVecDB()
        self.vector_db.load(self.cfg.FAISS_INDEX_PATH)
        
        # Load encoder for reranking
        self.encoder = SentenceTransformer(self.cfg.EMBEDDING_MODEL_NAME)
        
        # Configuration
        self.top_k_per_query = self.cfg.EVIDENCE_TOP_K
        self.final_top_k = self.cfg.EVIDENCE_TOP_K
        self.min_similarity = self.cfg.EVIDENCE_MIN_SIMILARITY
        
        # Quality thresholds
        self.diversity_weight = 0.3  # Balance between relevance and diversity

    def retrieve_evidence_for_claim(
        self, 
        claim: Claim,
        top_k: int = None,
        min_similarity: float = None
    ) -> List[Evidence]:
        """
        Enhanced evidence retrieval with query optimization and reranking.
        """
        top_k = top_k or self.final_top_k
        min_similarity = min_similarity or self.min_similarity
        
        # Step 1: Optimize and select best queries
        optimized_queries = self._optimize_queries(claim)
        
        # Step 2: Retrieve evidence from multiple queries
        raw_evidence = self._retrieve_from_queries(
            optimized_queries, 
            claim.text,
            top_k_per_query=self.top_k_per_query * 2  # Get more for filtering
        )
        
        # Step 3: Deduplicate intelligently
        deduplicated_evidence = self._smart_deduplicate(raw_evidence)
        
        # Step 4: Rerank by relevance and quality
        reranked_evidence = self._rerank_evidence(
            claim.text, 
            deduplicated_evidence
        )
        
        # Step 5: Apply diversity filtering
        diverse_evidence = self._apply_diversity_filter(
            reranked_evidence, 
            top_k
        )
        
        # Step 6: Filter by minimum similarity
        final_evidence = [
            e for e in diverse_evidence 
            if e.relevance_score >= min_similarity
        ]
        
        return final_evidence[:top_k]

    def _optimize_queries(self, claim: Claim) -> List[str]:
        """
        Select and optimize search queries to avoid redundancy.
        Returns the most diverse and effective queries.
        """
        if not claim.search_queries:
            return [claim.text]
        
        queries = claim.search_queries
        
        # If we have many queries, select the most diverse ones
        if len(queries) <= 3:
            return queries
        
        # Use embeddings to measure query diversity
        query_embeddings = self.encoder.encode(queries)
        
        # Select diverse queries using maximal marginal relevance (MMR)
        selected = [0]  # Always include first query (usually the full claim)
        selected_embeddings = [query_embeddings[0]]
        
        while len(selected) < min(5, len(queries)):
            # Find query most different from already selected
            max_min_distance = -1
            best_idx = -1
            
            for i, emb in enumerate(query_embeddings):
                if i in selected:
                    continue
                
                # Calculate minimum similarity to selected queries
                similarities = cosine_similarity([emb], selected_embeddings)[0]
                min_similarity = np.min(similarities)
                
                if min_similarity > max_min_distance:
                    max_min_distance = min_similarity
                    best_idx = i
            
            if best_idx != -1:
                selected.append(best_idx)
                selected_embeddings.append(query_embeddings[best_idx])
        
        return [queries[i] for i in selected]

    def _retrieve_from_queries(
        self, 
        queries: List[str], 
        claim_text: str,
        top_k_per_query: int
    ) -> List[Evidence]:
        """Retrieve evidence from multiple optimized queries"""
        all_evidence = []
        
        for query in queries:
            results = self.vector_db.search(query, top_k=top_k_per_query)
            
            for result in results:
                evidence = Evidence(
                    text=result['text'],
                    source=EvidenceSource(
                        type='wikipedia',
                        title=result['metadata'].get('title', 'Unknown'),
                        url=result['metadata'].get('url', '')
                    ),
                    relevance_score=result['similarity'],
                    claim_text=claim_text,
                    retrieval_query=query  # Track which query retrieved this
                )
                all_evidence.append(evidence)
        
        return all_evidence

    def _smart_deduplicate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """
        Intelligent deduplication that handles near-duplicates and
        keeps the highest quality version.
        """
        if not evidence_list:
            return []
        
        # Group similar evidence
        deduplicated = []
        seen_fingerprints = set()
        
        # Sort by relevance first (keep best versions)
        sorted_evidence = sorted(
            evidence_list, 
            key=lambda e: e.relevance_score, 
            reverse=True
        )
        
        for evidence in sorted_evidence:
            # Create multiple fingerprints for robust deduplication
            fingerprints = self._create_fingerprints(evidence.text)
            
            # Check if we've seen something very similar
            is_duplicate = any(fp in seen_fingerprints for fp in fingerprints)
            
            if not is_duplicate:
                deduplicated.append(evidence)
                seen_fingerprints.update(fingerprints)
        
        return deduplicated

    def _create_fingerprints(self, text: str) -> Set[str]:
        """
        Create multiple fingerprints for robust duplicate detection.
        Uses different text lengths and normalization strategies.
        """
        # Normalize text
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        fingerprints = set()
        
        # Multiple fingerprint strategies
        fingerprints.add(normalized[:150])  # First 150 chars
        fingerprints.add(normalized[:100])  # First 100 chars
        fingerprints.add(normalized[-100:])  # Last 100 chars
        
        # Middle section
        mid_start = len(normalized) // 3
        fingerprints.add(normalized[mid_start:mid_start + 100])
        
        return fingerprints

    def _rerank_evidence(
        self, 
        claim_text: str, 
        evidence_list: List[Evidence]
    ) -> List[Evidence]:
        """
        Rerank evidence using semantic similarity between claim and evidence.
        This is more accurate than the initial retrieval similarity.
        """
        if not evidence_list:
            return []
        
        # Encode claim
        claim_embedding = self.encoder.encode([claim_text])[0]
        
        # Encode all evidence
        evidence_texts = [e.text for e in evidence_list]
        evidence_embeddings = self.encoder.encode(evidence_texts)
        
        # Calculate semantic similarity
        similarities = cosine_similarity([claim_embedding], evidence_embeddings)[0]
        
        # Update relevance scores with reranked similarity
        for evidence, new_similarity in zip(evidence_list, similarities):
            # Blend original and reranked scores
            evidence.relevance_score = (
                0.4 * evidence.relevance_score + 
                0.6 * float(new_similarity)
            )
        
        # Sort by updated relevance
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return evidence_list

    def _apply_diversity_filter(
        self, 
        evidence_list: List[Evidence], 
        top_k: int
    ) -> List[Evidence]:
        """
        Select evidence that balances relevance with source diversity.
        Uses Maximal Marginal Relevance (MMR) algorithm.
        """
        if len(evidence_list) <= top_k:
            return evidence_list
        
        selected = []
        remaining = evidence_list.copy()
        
        # Always select the most relevant first
        selected.append(remaining.pop(0))
        
        # Encode all evidence for similarity comparison
        all_texts = [e.text for e in evidence_list]
        all_embeddings = self.encoder.encode(all_texts)
        selected_indices = [0]
        
        while len(selected) < top_k and remaining:
            best_score = -1
            best_idx = -1
            
            for i, evidence in enumerate(remaining):
                # Original index in full list
                orig_idx = evidence_list.index(evidence)
                
                # Relevance score (from reranking)
                relevance = evidence.relevance_score
                
                # Diversity score (distance from already selected)
                selected_embeddings = [all_embeddings[idx] for idx in selected_indices]
                current_embedding = all_embeddings[orig_idx]
                
                similarities = cosine_similarity(
                    [current_embedding], 
                    selected_embeddings
                )[0]
                max_similarity = np.max(similarities)
                diversity = 1 - max_similarity
                
                # Source diversity bonus
                source_bonus = 0
                if evidence.source.title not in [s.source.title for s in selected]:
                    source_bonus = 0.1
                
                # MMR score: balance relevance and diversity
                mmr_score = (
                    (1 - self.diversity_weight) * relevance + 
                    self.diversity_weight * diversity +
                    source_bonus
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected_evidence = remaining.pop(best_idx)
                selected.append(selected_evidence)
                selected_indices.append(evidence_list.index(selected_evidence))
        
        return selected

    def retrieve_evidence_for_claims(
        self, 
        claims: List[Claim]
    ) -> List[Tuple[Claim, List[Evidence]]]:
        """
        Retrieve evidence for multiple claims.
        Returns list of (claim, evidence_list) tuples.
        """
        results = []
        for claim in claims:
            evidence = self.retrieve_evidence_for_claim(claim)
            results.append((claim, evidence))
        return results

    def get_evidence_summary(self, evidence_list: List[Evidence]) -> Dict:
        """Enhanced evidence summary with quality metrics"""
        if not evidence_list:
            return {
                'status': 'NO_EVIDENCE',
                'evidence_count': 0,
                'avg_relevance': 0.0,
                'unique_sources': 0,
                'top_relevance': 0.0,
                'quality_score': 0.0
            }

        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        unique_sources = len(set(e.source.title for e in evidence_list))
        top_relevance = evidence_list[0].relevance_score if evidence_list else 0.0
        
        # Quality score: combines relevance, diversity, and coverage
        quality_score = (
            0.5 * avg_relevance +
            0.3 * (unique_sources / len(evidence_list)) +
            0.2 * top_relevance
        )

        return {
            'status': 'EVIDENCE_FOUND',
            'evidence_count': len(evidence_list),
            'unique_sources': unique_sources,
            'avg_relevance': float(avg_relevance),
            'top_relevance': float(top_relevance),
            'quality_score': float(quality_score),
            'source_distribution': self._get_source_distribution(evidence_list)
        }

    def _get_source_distribution(self, evidence_list: List[Evidence]) -> Dict[str, int]:
        """Get distribution of evidence across sources"""
        distribution = defaultdict(int)
        for evidence in evidence_list:
            distribution[evidence.source.title] += 1
        return dict(distribution)