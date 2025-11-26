"""
RAG using existing FaissVecDB
"""
from typing import List, Dict
from app.models.claim import Claim, ClaimComponent
from app.models.evidence import Evidence, EvidenceSource
from app.utils.faiss_vecdb import FaissVecDB
from app.core.config import settings


class EvidenceRetriever:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        self.vector_db = FaissVecDB(self.cfg)
        self.vector_db.load(self.cfg.FAISS_INDEX_PATH)
        if self.vector_db.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty")
        self.top_k_queries = self.cfg.EVIDENCE_TOP_K

    # er for a single claim
    def retrieve_evidence_for_claim(self, claim: Claim,
                                    top_k: float = None,
                                    min_similarity: float = None) -> List[Evidence]:

        # more configuration parameters
        top_k = max(0, top_k or self.cfg.EVIDENCE_TOP_K)
        min_similarity = min_similarity or self.cfg.EVIDENCE_MIN_SIMILARITY
        # pull out the original claim text
        claim_text = claim.text
        search_queries = claim.queries if claim.queries else claim_text
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
                        similarity=result['similarity'],
                        rank=-1
                    )
                    all_evidence.append(evidence)

        # sort by relevance score
        all_evidence.sort(key=lambda x: x.similarity, reverse=True)
        # update rank based on similarity score (max)
        for i, evidence in enumerate(all_evidence, start=1):
            evidence.rank = i

        return all_evidence[:top_k]

    # handle all the claims
    def retrieve_evidence_for_claims(self, claims: List[Claim]) -> List[Claim]:
        for claim in claims:
            evidence_list = self.retrieve_evidence_for_claim(claim)
            claim.evidence = evidence_list
        return claims

    # summarize stats for claims
    def get_evidence_summary(self, evidence_list: List[Evidence]) -> Dict:
        if not evidence_list:
            return {
                'status': 'NO_EVIDENCE',
                'evidence_count': 0,
                'avg_relevance': 0.0
            }

        avg_relevance = sum(e.similarity for e in evidence_list) / len(evidence_list)
        unique_sources = len(set(e.source.title for e in evidence_list))

        # what else would be good ...

        return {
            'status': 'EVIDENCE_FOUND',
            'evidence_count': len(evidence_list),
            'unique_sources': unique_sources,
            'avg_relevance': avg_relevance,
            'top_relevance': evidence_list[0].similarity if evidence_list else 0.0
        }