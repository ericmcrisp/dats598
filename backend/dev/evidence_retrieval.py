"""
RAG using existing FaissVecDB
"""
import sys
import os
import numpy as np

from typing import List, Dict
from faiss_vecdb import FaissVecDB
from evidence_model import Evidence, EvidenceSource
from config import Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class EvidenceRetriever:

    def __init__(self):
        self.vector_db = FaissVecDB()
        self.vector_db.load(Config.FAISS_INDEX_PATH)
        self.top_k_queries = Config.EVIDENCE_TOP_K
        
        print(f"Loaded {len(self.vector_db.documents)} documents from faiss index.")

    # er for a single claim.
    def retrieve_evidence_for_claim(self, claim_dict: Dict, top_k=Config.EVIDENCE_TOP_K, min_similarity=Config.EVIDENCE_MIN_SIMILARITY) -> List[Evidence]:
        claim_text = claim_dict['text']
        search_queries = claim_dict.get('search_queries', [claim_text])
        # collect evidence from multiple queries
        all_evidence = []
        seen_texts = set()
        # loop over search queries
        n = min(len(search_queries), self.top_k_queries)
        for query in search_queries[:n]:
            results = self.vector_db.search(query, top_k=top_k)

            # only process similar results - pretty sure this is done in the evidence gathering
            for result in results:
                # Skip if similarity too low
                if result['similarity'] < min_similarity:
                    continue

                # Deduplicate by first 100 chars
                text_key = result['text'][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)

                    # create evidence object
                    evidence = Evidence(
                        text=result['text'],
                        source=EvidenceSource(
                            type='wikipedia',
                            title=result['metadata'].get('title', 'Unknown'),
                            url=result['metadata'].get('url', '')
                        ),
                        relevance_score=result['similarity'],
                        claim_text=claim_text
                    )
                    all_evidence.append(evidence)

        # sort by relevance score
        all_evidence.sort(key=lambda x: x.relevance_score, reverse=True)

        return all_evidence[:top_k]

    # handle all the claims
    def retrieve_evidence_for_claims(self, claims: List[Dict]) -> Dict[str, List[Evidence]]:
        results = {}
        for claim in claims:
            evidence = self.retrieve_evidence_for_claim(claim)
            results[claim['text']] = evidence
        return results

    # summarize stats for claims
    def get_evidence_summary(self, evidence_list: List[Evidence]) -> Dict:
        if not evidence_list:
            return {
                'status': 'NO_EVIDENCE',
                'evidence_count': 0,
                'avg_relevance': 0.0
            }

        avg_relevance = sum(e.relevance_score for e in evidence_list) / len(evidence_list)
        unique_sources = len(set(e.source.title for e in evidence_list))

        # what else would be good ...

        return {
            'status': 'EVIDENCE_FOUND',
            'evidence_count': len(evidence_list),
            'unique_sources': unique_sources,
            'avg_relevance': avg_relevance,
            'top_relevance': evidence_list[0].relevance_score if evidence_list else 0.0
        }