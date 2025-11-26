"""
Class determine the process for verifying factual claims against evidence pulled from vec db
"""

import numpy as np
import spacy
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.models.evidence import Evidence, EvidenceSource
from app.models.claim import Claim
from app.models.verification import Verification
from app.core.config import settings


class FactVerifier:

    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)
        # create caching for model loading 
        self.encoder = SentenceTransformer(self.cfg.EMBEDDING_MODEL_NAME)
        # determine thresholds for verdict determination
        self.SUPPORTS_THRESHOLD = self.cfg.SUPPORTS_THRESHOLD
        self.REFUTES_THRESHOLD = 1 - self.SUPPORTS_THRESHOLD
        self.evidence_report_limit = 500

    # handle a single claim (the helper function below loops through claims)
    def verify_claim(self, claim: Claim, evidence_list: List[Evidence]) -> Verification:
        if not evidence_list:
            return Verification(
                claim=claim,
                verdict="NOT_ENOUGH_EVIDENCE",
                confidence=0.0,
                evidence_count=0,
                max_similarity=0.0,
                avg_similarity=0.0,
                best_evidence=Evidence(),
                all_evidence=[],
                explanation="No evidence retrieved from the knowledge base."
            )
        # grab the claim info
        text = claim.text
        # encode claim
        claim_embedding = self.encoder.encode([text])
        # encode all evidence passages
        evidence_texts = [evidence.text for evidence in evidence_list]
        evidence_embeddings = self.encoder.encode(evidence_texts)
        # calculate similarities
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]
        # get statistics
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        # assumes that the best evidence is tracked with highest similiarity
        best_evidence_idx = int(np.argmax(similarities))
        # determine verdict
        confidence = max_similarity
        if max_similarity >= self.SUPPORTS_THRESHOLD:
            verdict = 'SUPPORTS'
            explanation = f"Evidence supports this claim based on similarities (max : {max_similarity:.2f}, avg : {avg_similarity:.2f})"
        elif max_similarity <= self.REFUTES_THRESHOLD:
            verdict = 'REFUTES'
            explanation = f"Evidence contradicts this claim based on similarities (max : {max_similarity:.2f}, avg: {avg_similarity:.2f})"
        else:
            verdict = 'NOT_ENOUGH_INFO' 
            explanation = f"Evidence is unclear or insufficient based on similarities (max : {max_similarity:.2f}, avg: {avg_similarity:.2f})"

        return Verification(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence_count=len(evidence_list),
            max_similarity=max_similarity,
            avg_similarity=avg_similarity,
            best_evidence=evidence_list[best_evidence_idx],
            all_evidence=evidence_list,
            explanation=explanation
        )

    def verify_claims(self, claims_with_evidence: List[Claim]) -> List[Verification]: 
        results = []
        for claim in claims_with_evidence:
            results.append(self.verify_claim(claim, claim.evidence))
        return results

    def get_overall_assessment(self, verification_results: List[Dict]) -> Dict:
        if not verification_results:
            return {'status': 'NO_CLAIMS'}
        supports = sum(1 for r in verification_results if r['verdict'] == 'SUPPORTS')
        refutes = sum(1 for r in verification_results if r['verdict'] == 'REFUTES')
        not_enough = sum(1 for r in verification_results if r['verdict'] == 'NOT_ENOUGH_INFO')
        total = len(verification_results)
        avg_confidence = sum(r['confidence'] for r in verification_results) / total

        return {
            'total_claims': total,
            'supports': supports,
            'refutes': refutes,
            'not_enough_info': not_enough,
            'avg_confidence': avg_confidence,
            'accuracy_rate': supports / total if total > 0 else 0.0
        }