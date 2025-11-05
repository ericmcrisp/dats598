"""
This class is the pipeline that takes a text input and preprocesses it such that the claims in a text can be analyzed with search.
"""

from app.models.claim import Claim, ClaimComponent
from claim_detection import ClaimDetector
from claim_extraction import ClaimExtractor
from text_preprocessing import TextPreprocessor
from evidence_retrieval import EvidenceRetriever
from fact_verification import FactVerifier

from app.core.config import settings

import pandas as pd


class FactCheckPipe:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        # subclasses/processes need to share the cfg
        self.preprocessor = TextPreprocessor(self.cfg)
        self.nlp = self.preprocessor.nlp
        self.detector = ClaimDetector(self.cfg, nlp=self.nlp)
        self.extractor = ClaimExtractor(self.cfg, nlp=self.nlp)
        self.evidence = EvidenceRetriever(self.cfg, nlp=self.nlp)
        self.verify = FactVerifier(self.cfg, nlp=self.nlp)
        # params
        self.cleaned_text = None
        self.sentences = None
        self.detected_claims = None
        self.df = None
        self.claims_with_evidence = None
        self.verification_results = None
        self.assessment = None

    # process the text and prep it to be embedded for search
    def process(self, text):
        # step 1: clean text
        cleaned_text = self.preprocessor.clean_text(text)
        # step 2: split into sentences
        sentences = self.preprocessor.segment_sentences(cleaned_text)
        # step 3: detect claims
        detected_claims = []
        for sentence in sentences:
            is_claim, confidence, claim_type = self.detector.is_factual_claim(sentence)

            if is_claim and confidence > self.detector.claim_threshold:
                # step 4: extract claim components
                components = self.extractor.extract_claim_components(sentence)
                # step 5: generate search queries
                search_queries = self.extractor.generate_search_queries(components)

                detected_claims.append(
                    Claim(
                        text=sentence,
                        confidence=confidence,
                        type=claim_type,
                        components=ClaimComponent(**components),
                        search_queries=search_queries
                    )
                )

        # update object state
        self.cleaned_text = cleaned_text
        self.sentences = sentences
        self.detected_claims = detected_claims

        # step 6: encode and embed the claims to find similar documents
        self.claims_with_evidence = self.evidence.retrieve_evidence_for_claims(self.detected_claims)
        # step7: verify is evidence confirms or denies
        self.verification_results = self.verify.verify_claims(self.claims_with_evidence)
        # step 8: asses the alignment
        self.assessment = self.verify.get_overall_assessment(self.verification_results)

        return self.assessment

    def claims_to_dataframe(self, claims=None):
        if claims is None and self.detected_claims is not None:
            claims = self.detected_claims
        data = []
        for claim in claims:
            data.append({
                'claim_text': claim['text'],
                'confidence': claim['confidence'],
                'type': claim['type'],
                'entities': ', '.join([e['text'] for e in claim['components']['entities']]),
                'primary_query': claim['search_queries'][0] if claim['search_queries'] else None
            })
        # update object state
        self.df = pd.DataFrame(data)
        return self.df
