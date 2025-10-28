"""
test_complete_pipeline.py - Test complete fact-checking pipeline
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claim_pipe import ClaimDetectionPipeline as cdp
from evidence_retrieval import EvidenceRetriever
from fact_verification import FactVerifier


def test_complete_pipeline():
    """Test the complete fact-checking pipeline."""

    print("="*70)
    print("Complete Fact-Checking Pipeline Test")
    print("="*70)

    text = """
    The Eiffel Tower was built in 1889 for the World's Fair in Paris.
    It stands 324 meters tall and was designed by Gustave Eiffel.
    The tower is located in London, England and weighs 50,000 tons.
    Albert Einstein was born in 1879 in Germany.
    """

    print(f"\nInput Text:\n{text}\n")

    # Step 1: Detect claims
    print("\n" + "="*70)
    print("STEP 1: Claim Detection")
    print("="*70)

    claim_pipeline = cdp()
    claims = claim_pipeline.process_text(text)


    print(f"\n✓ Detected {len(claims)} claims:\n")
    for i, claim in enumerate(claims, 1):
        print(f"{i}. {claim['text']}")
        print(f"   Confidence: {claim['confidence']:.2f} | Type: {claim['type']}")
        print(f"   Search Queries: {claim['search_queries'][:2]}\n")

    # Step 2: Retrieve evidence
    print("\n" + "="*70)
    print("STEP 2: Evidence Retrieval")
    print("="*70)

    evidence_retriever = EvidenceRetriever(index_path='../data/wikipedia/faiss_index')
    claims_with_evidence = evidence_retriever.retrieve_evidence_for_claims(claims)

    for claim_text, evidence_list in claims_with_evidence.items():
        print(f"\nClaim: {claim_text}")
        print(f"Evidence Retrieved: {len(evidence_list)} passages")
        
        if evidence_list:
            top_evidence = evidence_list[0]
            print(f"\nTop Evidence:")
            print(f"  Source: {top_evidence.source.title}")
            print(f"  Relevance: {top_evidence.relevance_score:.3f}")
            print(f"  Text: {top_evidence.text[:200]}...")
        else:
            print("  No evidence found!")
    
    # Step 3: Verify claims
    print("\n" + "="*70)
    print("STEP 3: Fact Verification")
    print("="*70)
    
    verifier = FactVerifier()
    verification_results = verifier.verify_claims(claims_with_evidence)
    
    for result in verification_results:
        print(f"\n{'='*70}")
        print(f"Claim: {result['claim']}")
        print(f"{'='*70}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Evidence Count: {result['evidence_count']}")
        print(f"Max Similarity: {result['max_similarity']:.3f}")
        print(f"Avg Similarity: {result['avg_similarity']:.3f}")
        print(f"\nExplanation: {result['explanation']}")

        if result['best_evidence']:
            print(f"\nBest Supporting Evidence:")
            print(f"  Source: {result['best_evidence']['source']}")
            print(f"  Similarity: {result['best_evidence']['similarity']:.3f}")
            print(f"  Text: {result['best_evidence']['text'][:250]}...")

    # Step 4: Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    assessment = verifier.get_overall_assessment(verification_results)

    print(f"\nTotal Claims: {assessment['total_claims']}")
    print(f"  ✓ Supports: {assessment['supports']}")
    print(f"  ✗ Refutes: {assessment['refutes']}")
    print(f"  ? Not Enough Info: {assessment['not_enough_info']}")
    print(f"\nAverage Confidence: {assessment['avg_confidence']:.2f}")
    print(f"Accuracy Rate: {assessment['accuracy_rate']:.1%}")


if __name__ == "__main__":
    test_complete_pipeline()