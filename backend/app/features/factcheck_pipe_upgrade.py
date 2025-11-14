"""
upgraded the FactCheckPipe so user can switch claim extraction strategies.
Supports rules-based and LLM-based methods
"""

from typing import Optional
import pandas as pd

# models
from app.models.claim import Claim
from app.models.verification import Verification
from app.models.evidence import Evidence
from app.models.factcheck import FactCheckResponse

# Features
from app.features.text_preprocessing import TextPreprocessor
from app.features.evidence_retrieval import EvidenceRetriever
from app.features.fact_verification import FactVerifier

# Strategy pattern for claim extraction
from app.features.claim_extraction_upgrade import ClaimExtractorFactory

from app.core.config import settings


class FactCheckPipe:
    """
    Fact-checking pipeline with pluggable claim extraction.
    
    Usage:
        # Rules-based (default)
        pipe = FactCheckPipe()
        
        # LLM-based
        pipe = FactCheckPipe(extraction_method="llm")
        
        # Hybrid
        pipe = FactCheckPipe(extraction_method="hybrid")
    """
    
    def __init__(self, cfg=None, extraction_method: str = "rules"):
        self.cfg = cfg or settings
        self.extraction_method = extraction_method
        
        # Text preprocessing (always needed)
        self.preprocessor = TextPreprocessor(self.cfg)
        self.nlp = self.preprocessor.nlp
        
        # get the extraction method
        self.claim_extractor = ClaimExtractorFactory.create(
            method=extraction_method,
            cfg=self.cfg,
            nlp=self.nlp
        )
        # do the ER and verification
        self.evidence = EvidenceRetriever(self.cfg)
        self.verify = FactVerifier(self.cfg, nlp=self.nlp)
        
        # states
        self.original_text = None
        self.cleaned_text = None
        self.sentences = None
        self.detected_claims = None
        self.claims_with_evidence = None
        self.verification_results = None
        self.assessment = None
        self.df = None

    def clean(self, text: str) -> str:
        """Step 1: clean and normalize text"""
        self.cleaned_text = self.preprocessor.clean_text(text)
        return self.cleaned_text

    def segment_sentences(self, text: Optional[str] = None) -> list:
        """Step 2: split text into sentences"""
        text = text or self.cleaned_text
        self.sentences = self.preprocessor.segment_sentences(text)
        return self.sentences

    def detect_claims(self, text: Optional[str] = None, sentences: Optional[list] = None) -> list:
        """
        Step 3-5: detect and extract claims using the selected strategy.
        """
        text = text or self.cleaned_text
        sentences = sentences or self.sentences
        
        # Use the strategy to extract claims
        self.detected_claims = self.claim_extractor.extract_claims(text, sentences)
        return self.detected_claims

    def retrieve_evidence(self, claims: Optional[list] = None):
        """Step 6: ER relevant evidence for each claim"""
        claims = claims or self.detected_claims
        self.claims_with_evidence = self.evidence.retrieve_evidence_for_claims(claims)
        return self.claims_with_evidence

    def verify_claims(self, claims_with_evidence: Optional[list] = None):
        """Step 7: Verify claims against evidence"""
        claims_with_evidence = claims_with_evidence or self.claims_with_evidence
        self.verification_results = self.verify.verify_claims(claims_with_evidence)
        return self.verification_results

    def assess(self, verification_results: Optional[list] = None):
        """Step 8: Generate overall assessment"""
        verification_results = verification_results or self.verification_results
        self.assessment = self.verify.get_overall_assessment(verification_results)
        return self.assessment

    def process(self, text: str) -> FactCheckResponse:
        """
        Process text through the full fact-checking pipeline.
        
        Args:
            text: Input text to fact-check
            
        Returns:
            FactCheckResponse with claims, verdicts, and summary
        """
        self.original_text = text
        # Step 1: Clean text
        self.clean(self.original_text)
        # Step 2: Split into sentences
        self.segment_sentences()
        # Step 3-5: Detect and extract claims (using selected strategy)
        self.detect_claims()
        # Step 6: Retrieve evidence
        self.retrieve_evidence()
        # Step 7: Verify claims
        self.verify_claims()
        # Step 8: Generate assessment
        self.assess()
        
        # Add metadata about extraction method
        config_with_method = self.cfg.dict()
        config_with_method['extraction_method'] = self.claim_extractor.get_method_name()
        config_with_method['claims_detected'] = len(self.detected_claims)
        
        # Return structured response
        return FactCheckResponse(
            claims=self.verification_results,
            summary=self.assessment,
            config=config_with_method
        )

    # ==================== UTILITY METHODS ====================

    def change_extraction_method(self, method: str):
        """
        Change the claim extraction method on the fly.
        
        Args:
            method: "rules", "llm", or "hybrid"
        """
        self.extraction_method = method
        self.claim_extractor = ClaimExtractorFactory.create(
            method=method,
            cfg=self.cfg,
            nlp=self.nlp
        )

    def get_extraction_method(self) -> str:
        """Get the current extraction method name"""
        return self.claim_extractor.get_method_name()

    def get_available_methods(self) -> list:
        """Get list of available extraction methods"""
        return ClaimExtractorFactory.get_available_methods()

    def claims_to_dataframe(self, claims: Optional[list] = None) -> pd.DataFrame:
        """Convert claims to pandas DataFrame for analysis"""
        if claims is None and self.detected_claims is not None:
            claims = self.detected_claims
        
        data = []
        for claim in claims:
            data.append({
                'claim_text': claim.text,
                'confidence': claim.confidence,
                'type': claim.type,
                'entities': ', '.join([e['text'] for e in claim.components.entities]),
                'primary_query': claim.searches[0] if claim.searches else None,
                'num_search_queries': len(claim.searches) if claim.searches else 0
            })
        
        self.df = pd.DataFrame(data)
        return self.df

    def get_pipeline_info(self) -> dict:
        """Get information about the current pipeline configuration"""
        return {
            'extraction_method': self.get_extraction_method(),
            'available_methods': self.get_available_methods(),
            'config': self.cfg.dict(),
            'claims_detected': len(self.detected_claims) if self.detected_claims else 0,
            'evidence_retrieved': len(self.claims_with_evidence) if self.claims_with_evidence else 0,
            'claims_verified': len(self.verification_results) if self.verification_results else 0
        }