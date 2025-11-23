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
from app.features.claim_segmentation import ClaimSegmentator
# from app.features.coreference import CoreferenceResolver
from app.features.text_preprocessing import TextPreprocessor
from app.features.evidence_retrieval import EvidenceRetriever
from app.features.fact_verification import FactVerifier
# adding in the LLM version of these processes
# from app.features.llm_claim_analysis import LLMClaimAnalysis

# config
from app.core.config import settings

# export data
import pandas as pd


class FactCheckPipe:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        # subclasses/processes need to share the cfg
        # self.extractor = LLMClaimAnalysis()
        self.preprocessor = TextPreprocessor(self.cfg)
        self.nlp = self.preprocessor.nlp
        self.segmentator = ClaimSegmentator(self.cfg, nlp=self.nlp)
        # self.coref = CoreferenceResolver(nlp=self.nlp)
        self.detector = ClaimDetector(self.cfg, nlp=self.nlp)
        self.extractor = ClaimExtractor(self.cfg, nlp=self.nlp)
        self.evidence = EvidenceRetriever(self.cfg)
        self.verify = FactVerifier(self.cfg, nlp=self.nlp)
        # params
        self.original_text = None
        self.cleaned_text = None
        self.sentences = None
        self.segments = None
        self.detected_claims = None
        self.claims_with_evidence = None
        self.verification_results = None
        self.assessment = None
        self.df = None

    # step 1: clean
    def clean(self, text):
        self.cleaned_text = self.preprocessor.clean_text(text)
        return self.cleaned_text

    # step 2a: split into sentences
    def segment_sentences(self, text=None):
        text = text or self.cleaned_text
        self.sentences = self.preprocessor.segment_sentences(text)
        return self.sentences

    # step 2b: segment sentences into claims
    def segment_claims(self, sentences=None):
        self.segment_sentences()
        self.segments = self.segmentator.segment(self.sentences)
        return self.segments

    # step 3, 4, 5: detect claims
    def detect_claims(self, sentences=None):
        if sentences is None:
            sentences = self.segment_sentences()

        self.detected_claims = []

        # simple: treat each sentence as a candidate claim
        if self.cfg.CLAIM_MODE == 'simple':
            for sent_idx, sentence in enumerate(sentences):
                is_claim, confidence, claim_type = self.detector.is_factual_claim(sentence)
                if is_claim and confidence > self.detector.claim_threshold:
                    components = self.extractor.extract_claim_components(sentence)
                    queries = self.extractor.generate_search_queries(components)
                    self.detected_claims.append(
                        Claim(
                            text=sentence,
                            confidence=confidence,
                            type=claim_type,
                            components=components,
                            queries=queries,
                            start_sentence_idx=sent_idx,
                            end_sentence_idx=sent_idx,
                            clause_index=0,
                            context_text=self.cleaned_text
                        )
                    )
        # advanced: treat each sentence as if it could contain multiple claims with context
        elif self.cfg.CLAIM_MODE == 'advanced':
            segmentator = ClaimSegmentator(cfg=self.cfg, nlp=self.nlp)
            claim_units = segmentator.segment(sentences)

            for cu in claim_units:
                is_claim, confidence, claim_type = self.detector.is_factual_claim(cu['text'])
                if is_claim and confidence > self.detector.claim_threshold:
                    components = self.extractor.extract_claim_components(cu['text'])
                    queries = self.extractor.generate_search_queries(components)
                    self.detected_claims.append(
                        Claim(
                            text=cu['text'],
                            confidence=confidence,
                            type=claim_type,
                            components=components,
                            queries=queries,
                            start_sentence_idx=cu['start_sentence_idx'],
                            end_sentence_idx=cu['end_sentence_idx'],
                            clause_index=cu['clause_index'],
                            context_text=self.cleaned_text
                        )
                    )
        else:
            raise ValueError(f"Unknown CLAIM_MODE: {self.cfg.CLAIM_MODE}")
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
        # step 2, 3, 4, 5: detect claims
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
