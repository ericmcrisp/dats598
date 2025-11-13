"""
This class is the pipeline that takes a text input and preprocesses it such that the claims in a text can be analyzed with search.
"""

#pydantic models for structured data
from app.models.claim import Claim
from app.models.verification import Verification
from app.models.evidence import Evidence
from app.models.factcheck import FactCheckResponse

# features
from app.features.claim_detection import ClaimDetector
from app.features.claim_extraction import ClaimExtractor
from app.features.text_preprocessing import TextPreprocessor
from app.features.evidence_retrieval import EvidenceRetriever
from app.features.fact_verification import FactVerifier

# config
from app.core.config import settings

# export data
import pandas as pd


class FactCheckPipe:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        # subclasses/processes need to share the cfg
        self.preprocessor = TextPreprocessor(self.cfg)
        self.nlp = self.preprocessor.nlp
        self.detector = ClaimDetector(self.cfg, nlp=self.nlp)
        self.extractor = ClaimExtractor(self.cfg, nlp=self.nlp)
        self.evidence = EvidenceRetriever(self.cfg)
        self.verify = FactVerifier(self.cfg, nlp=self.nlp)
        # params
        self.original_text = None
        self.cleaned_text = None
        self.sentences = None
        self.detected_claims = None
        self.claims_with_evidence = None
        self.verification_results = None
        self.assessment = None
        self.df = None

    # step 1: clean
    def clean(self, text):
        self.cleaned_text = self.preprocessor.clean_text(text)
        return self.cleaned_text

    # step 2: split into sentences
    def segment_sentences(self, text=None):
        text = text or self.cleaned_text
        self.sentences = self.preprocessor.segment_sentences(text)
        return self.sentences

    # step 3, 4, 5: detect claims
    def detect_claims(self, sentences=None):
        sentences = sentences or self.sentences
        self.detected_claims = []
        # might need to refactor this because what if a claim spans multiple sentences
        for sentence in sentences:
            #   bool,      float,        str
            is_claim, confidence, claim_type = self.detector.is_factual_claim(sentence)
            # if a claim: 
            if is_claim and confidence > self.detector.claim_threshold:
                # step 4: extract claim components
                components = self.extractor.extract_claim_components(sentence)
                # step 5: generate search queries
                search_queries = self.extractor.generate_search_queries(components)

                self.detected_claims.append(
                    Claim(
                        text=sentence,
                        confidence=confidence,
                        type=claim_type,
                        components=components,
                        search_queries=search_queries
                    )
                )
        return self.detected_claims

    # step 6: retrieve relevant evidence
    def retrieve_evidence(self, claims=None):
        claims = claims or self.detected_claims
        self.claims_with_evidence = self.evidence.retrieve_evidence_for_claims(claims)
        return self.claims_with_evidence

    # step 7: verify claims
    def verify_claims(self, claims_with_evidence=None):
        claims_with_evidence = claims_with_evidence or self.claims_with_evidence
        self.verification_results = self.verify.verify_claims(claims_with_evidence)
        return self.verification_results

    # step 8: assess whether evidence supports or refutes claim
    def assess(self, verification_results=None):
        verification_results = verification_results or self.verification_results
        self.assessment = self.verify.get_overall_assessment(verification_results)
        return self.assessment

    # process the text and return fact checking result
    def process(self, text):
        self.original_text = text
        # step 1: clean text
        self.clean(self.original_text)
        # step 2: split into sentences
        self.segment_sentences()
        # step 3, 4, 5: detect claims
        self.detect_claims()
        # step 6: encode and embed the claims to find similar documents
        self.retrieve_evidence()
        # step7: verify is evidence confirms or denies
        self.verify_claims()
        # step 8: assess the alignment
        self.assess()
        # return the results as a structured response
        return FactCheckResponse(
            claims=self.verification_results,
            summary=self.assessment,
            config=self.cfg.dict()
        )

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
