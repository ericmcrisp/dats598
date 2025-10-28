"""
Class determine the process for verifying factual claims against evidence pulled from vec db
"""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os

from evidence_model import Evidence
from config import Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class FactVerifier:

    def __init__(self):
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        # determine thresholds for verdict determination
        self.SUPPORTS_THRESHOLD = Config.SUPPORTS_THRESHOLD
        self.REFUTES_THRESHOLD = 1 - self.SUPPORTS_THRESHOLD
        self.evidence_report_limit = 10

    def verify_claim(self, claim: str, evidence_list: List[Evidence]) -> Dict:
        if not evidence_list:
            return {
                'claim': claim,
                'verdict': 'NOT_ENOUGH_INFO',
                'confidence': 0.0,
                'evidence_count': 0,
                'evidence_used': [],
                'explanation': 'No evidence found for this claim.'
            }

        # encode claim
        claim_embedding = self.encoder.encode([claim])
        # encode all evidence passages
        evidence_texts = [e.text for e in evidence_list]
        evidence_embeddings = self.encoder.encode(evidence_texts)
        # calculate similarities
        similarities = cosine_similarity(claim_embedding, evidence_embeddings)[0]
        # get statistics
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        best_evidence_idx = int(np.argmax(similarities))
        # determine verdict
        if max_similarity >= self.SUPPORTS_THRESHOLD:
            verdict = 'SUPPORTS'
            confidence = max_similarity
            explanation = f"Evidence supports this claim based on similarities (max : {max_similarity:.2f}, avg : {avg_similarity:.2f})"
        elif max_similarity <= self.REFUTES_THRESHOLD:
            verdict = 'REFUTES'
            confidence = 1.0 - max_similarity
            explanation = f"Evidence contradicts this claim based on similarities (max : {max_similarity:.2f}, avg: {avg_similarity:.2f})"
        else:
            verdict = 'NOT_ENOUGH_INFO'
            confidence = 0.5
            explanation = f"Evidence is unclear or insufficient based on similarities (max : {max_similarity:.2f}, avg: {avg_similarity:.2f})"

        evidence_used = []
        for i, (evidence, sim) in enumerate(zip(evidence_list, similarities)):
            evidence_used.append({
                'text': evidence.text[:self.evidence_report_limit] + '...' if len(evidence.text) > self.evidence_report_limit else evidence.text,
                'source': evidence.source.title,
                'url': evidence.source.url,
                'similarity': float(sim),
                'rank': i + 1
            })

        return {
            'claim': claim,
            'verdict': verdict,
            'confidence': confidence,
            'evidence_count': len(evidence_list),
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'best_evidence': evidence_used[best_evidence_idx],
            'all_evidence': evidence_used,
            'explanation': explanation
        }

    def verify_claims(self, claims_with_evidence: Dict[str, List[Evidence]]) -> List[Dict]:
        results = []
        for claim_text, evidence in claims_with_evidence.items():
            result = self.verify_claim(claim_text, evidence)
            results.append(result)
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