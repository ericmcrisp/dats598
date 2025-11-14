""""
Improved ClaimDetector with better confidence scoring, context awareness,
and more sophisticated claim type detection.
"""

import re
import spacy
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from app.core.config import settings


@dataclass
class ClaimFeatures:
    """Features extracted from a sentence for claim detection"""
    has_temporal: bool = False
    has_numerical: bool = False
    has_relational: bool = False
    has_definitional: bool = False
    has_event: bool = False
    has_causal: bool = False
    has_comparative: bool = False
    entity_count: int = 0
    entity_types: List[str] = None
    verb_tense: Optional[str] = None
    has_strong_verbs: bool = False
    sentence_length: int = 0
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = []


class ImprovedClaimDetector:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)
        self.claim_threshold = settings.CLAIM_CONFIDENCE_THRESHOLD
        
        # Weights for different claim indicators (tunable)
        self.feature_weights = {
            'temporal': 0.20,
            'numerical': 0.18,
            'relational': 0.15,
            'definitional': 0.12,
            'event': 0.15,
            'causal': 0.18,
            'comparative': 0.15,
            'has_entities': 0.25,
            'strong_verbs': 0.15,
            'appropriate_length': 0.10
        }

    def is_factual_claim(self, sentence: str) -> Tuple[bool, float, str]:
        """
        Detect if a sentence contains a factual claim with improved scoring.
        
        Returns:
            (is_claim: bool, confidence: float, claim_type: str)
        """
        doc = self.nlp(sentence)
        
        # First check if it's definitely not a claim
        if self._is_non_claim(sentence, doc):
            return False, 0.0, None
        
        # Extract all features
        features = self._extract_features(sentence, doc)
        
        # Calculate weighted confidence score
        confidence = self._calculate_confidence(features)
        
        # Determine primary claim type
        claim_type = self._get_primary_claim_type(features)
        
        is_claim = confidence >= self.claim_threshold
        
        return is_claim, confidence, claim_type

    def _extract_features(self, sentence: str, doc) -> ClaimFeatures:
        """Extract all relevant features from the sentence"""
        features = ClaimFeatures()
        
        # Pattern-based features
        features.has_temporal = self._has_temporal_claim(sentence, doc)
        features.has_numerical = self._has_numerical_claim(sentence, doc)
        features.has_relational = self._has_relational_claim(sentence, doc)
        features.has_definitional = self._has_definitional_claim(sentence, doc)
        features.has_event = self._has_event_claim(sentence, doc)
        features.has_causal = self._has_causal_claim(sentence, doc)
        features.has_comparative = self._has_comparative_claim(sentence, doc)
        
        # Entity-based features
        features.entity_count = len(doc.ents)
        features.entity_types = [ent.label_ for ent in doc.ents]
        
        # Verb features
        features.verb_tense = self._get_verb_tense(doc)
        features.has_strong_verbs = self._has_strong_verbs(doc)
        
        # Structural features
        features.sentence_length = len(doc)
        
        return features

    def _calculate_confidence(self, features: ClaimFeatures) -> float:
        """Calculate weighted confidence score based on features"""
        score = 0.0
        
        # Add scores for each detected pattern
        if features.has_temporal:
            score += self.feature_weights['temporal']
        if features.has_numerical:
            score += self.feature_weights['numerical']
        if features.has_relational:
            score += self.feature_weights['relational']
        if features.has_definitional:
            score += self.feature_weights['definitional']
        if features.has_event:
            score += self.feature_weights['event']
        if features.has_causal:
            score += self.feature_weights['causal']
        if features.has_comparative:
            score += self.feature_weights['comparative']
        
        # Entity bonus (scaled by count)
        if features.entity_count > 0:
            entity_score = min(features.entity_count * 0.1, self.feature_weights['has_entities'])
            score += entity_score
        
        # Strong verb bonus
        if features.has_strong_verbs:
            score += self.feature_weights['strong_verbs']
        
        # Sentence length appropriateness (too short or too long is suspicious)
        if 5 <= features.sentence_length <= 40:
            score += self.feature_weights['appropriate_length']
        
        # Normalize to 0-1 range
        return min(score, 1.0)

    def _get_primary_claim_type(self, features: ClaimFeatures) -> Optional[str]:
        """Determine the primary type of claim based on features"""
        claim_types = []
        
        if features.has_temporal:
            claim_types.append(('temporal', self.feature_weights['temporal']))
        if features.has_numerical:
            claim_types.append(('numerical', self.feature_weights['numerical']))
        if features.has_causal:
            claim_types.append(('causal', self.feature_weights['causal']))
        if features.has_event:
            claim_types.append(('event', self.feature_weights['event']))
        if features.has_comparative:
            claim_types.append(('comparative', self.feature_weights['comparative']))
        if features.has_relational:
            claim_types.append(('relational', self.feature_weights['relational']))
        if features.has_definitional:
            claim_types.append(('definitional', self.feature_weights['definitional']))
        
        if claim_types:
            # Return the highest weighted claim type
            return max(claim_types, key=lambda x: x[1])[0]
        
        return None

    # ==================== PATTERN DETECTION METHODS ====================
    
    def _has_temporal_claim(self, sentence: str, doc) -> bool:
        """Enhanced temporal claim detection"""
        temporal_patterns = [
            r'\b(in|on|during|since|until|by|from|between)\s+\d{4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(was|were|is|are)\s+\w+\s+(in|on|during)\s+\d{4}\b',
            r'\b(born|died|founded|established|created|built|opened|closed|started|ended)\s+(in|on)\s+\d{4}\b',
            r'\b\d{4}\s*[-–]\s*\d{4}\b',  # Year ranges
            r'\b(early|mid|late)\s+\d{4}s?\b',  # "early 1990s"
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Also check for DATE entities from spaCy
        return any(ent.label_ == 'DATE' for ent in doc.ents)

    def _has_numerical_claim(self, sentence: str, doc) -> bool:
        """Enhanced numerical claim detection"""
        numerical_patterns = [
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(meters?|feet|ft|miles?|mi|kilometers?|km|kgs?|kilograms?|pounds?|lbs?|tons?|tonnes?|degrees?|celsius|fahrenheit)\b',
            r'\b\d+(?:\.\d+)?\s*(?:percent|%)\b',
            r'\b(population|area|height|width|weight|length|distance|speed|temperature|cost|price|revenue|profit)\s+(?:of|is|was|equals?|reaches?)\s+\d+',
            r'\bmeasures?\s+\d+',
            r'\bstands?\s+(?:at\s+)?\d+',
            r'\bweighs?\s+\d+',
            r'\b\$\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion)?\b',  # Money
            r'\b\d+(?:,\d{3})*\s+(?:people|citizens|residents|inhabitants)\b',  # Population
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Also check for QUANTITY, PERCENT, or MONEY entities
        return any(ent.label_ in ['QUANTITY', 'PERCENT', 'MONEY', 'CARDINAL'] for ent in doc.ents)

    def _has_relational_claim(self, sentence: str, doc) -> bool:
        """Enhanced relational claim detection"""
        relational_patterns = [
            r'\b(located|situated|found|positioned|placed)\s+in\b',
            r'\b(capital|president|king|queen|leader|founder|CEO|director|chairman|minister|mayor)\s+of\b',
            r'\b(married|divorced|related|connected|linked)\s+to\b',
            r'\bis\s+(?:the|a)\s+(?:\w+\s+)?(?:of|in|from)\b',
            r'\b(born|raised|grew up|lives?|resided?)\s+in\b',
            r'\b(belongs?|owned|possessed|controlled)\s+(?:to|by)\b',
            r'\b(part|member|component|element)\s+of\b',
        ]
        
        for pattern in relational_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False

    def _has_definitional_claim(self, sentence: str, doc) -> bool:
        """Enhanced definitional claim detection"""
        definitional_patterns = [
            r'\b(is|are|was|were)\s+(?:a|an|the)\s+\w+',
            r'\b(defined|known|called|termed|referred|described|characterized)\s+as\b',
            r'\b(means?|represents?|signifies?|denotes?)\b',
            r'\bconsists?\s+of\b',
        ]
        
        for pattern in definitional_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Must have named entities to be factual
                if any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT'] 
                       for ent in doc.ents):
                    return True
        
        return False

    def _has_event_claim(self, sentence: str, doc) -> bool:
        """Enhanced event claim detection"""
        # Look for past tense verbs with named entities
        has_past_verb = any(token.tag_ in ['VBD', 'VBN'] for token in doc)
        has_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'NORP'] 
                        for ent in doc.ents)
        
        event_keywords = [
            'won', 'defeated', 'conquered', 'discovered', 'invented', 
            'wrote', 'painted', 'directed', 'signed', 'achieved', 
            'completed', 'launched', 'released', 'published', 'announced',
            'declared', 'proclaimed', 'awarded', 'received', 'won',
            'fought', 'battled', 'negotiated', 'agreed', 'treaty'
        ]
        has_event_keyword = any(keyword in sentence.lower() 
                               for keyword in event_keywords)
        
        return (has_past_verb and has_entity) or (has_event_keyword and has_entity)

    def _has_causal_claim(self, sentence: str, doc) -> bool:
        """NEW: Detect causal claims (X causes Y)"""
        causal_patterns = [
            r'\b(causes?|caused|causing)\b',
            r'\b(leads?\s+to|led\s+to|leading\s+to)\b',
            r'\b(results?\s+in|resulted\s+in|resulting\s+in)\b',
            r'\b(due\s+to|because\s+of|owing\s+to)\b',
            r'\b(triggers?|triggered|triggering)\b',
            r'\b(produces?|produced|producing)\b',
            r'\b(creates?|created|creating)\b',
            r'\b(influences?|influenced|influencing)\b',
            r'\b(affects?|affected|affecting)\b',
            r'\b(contributes?\s+to|contributed\s+to)\b',
        ]
        
        for pattern in causal_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        return False

    def _has_comparative_claim(self, sentence: str, doc) -> bool:
        """NEW: Detect comparative claims (X is more/less than Y)"""
        comparative_patterns = [
            r'\b(larger|smaller|bigger|taller|shorter|faster|slower|heavier|lighter)\s+than\b',
            r'\b(more|less|fewer)\s+\w+\s+than\b',
            r'\b(most|least|greatest|smallest)\b',
            r'\b(exceeds?|exceeded|exceeding|surpasses?|surpassed)\b',
            r'\b(compared\s+to|in\s+comparison\s+to|relative\s+to)\b',
            r'\b(higher|lower)\s+than\b',
            r'\b(better|worse)\s+than\b',
        ]
        
        for pattern in comparative_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Should have entities or numbers to be factual
                has_entities = len(doc.ents) > 0
                has_numbers = bool(re.search(r'\d+', sentence))
                return has_entities or has_numbers
        
        return False

    def _get_verb_tense(self, doc) -> Optional[str]:
        """Determine the primary verb tense in the sentence"""
        for token in doc:
            if token.pos_ == 'VERB':
                if token.tag_ in ['VBD', 'VBN']:
                    return 'past'
                elif token.tag_ in ['VBP', 'VBZ']:
                    return 'present'
                elif token.tag_ in ['VBG']:
                    return 'progressive'
        return None

    def _has_strong_verbs(self, doc) -> bool:
        """Check for strong, factual verbs (not weak verbs like 'is', 'seems')"""
        weak_verbs = {'be', 'is', 'are', 'was', 'were', 'seem', 'appear', 'feel', 'think', 'believe'}
        strong_verbs = {
            'discovered', 'invented', 'founded', 'built', 'created', 'won',
            'defeated', 'achieved', 'completed', 'produced', 'developed',
            'established', 'launched', 'signed', 'declared', 'announced'
        }
        
        for token in doc:
            if token.pos_ == 'VERB':
                lemma = token.lemma_.lower()
                if lemma in strong_verbs:
                    return True
                if lemma not in weak_verbs and token.tag_ in ['VBD', 'VBN']:
                    return True
        
        return False

    def _is_non_claim(self, sentence: str, doc) -> bool:
        """Enhanced filtering of non-claims"""
        # Questions
        if sentence.strip().endswith('?'):
            return True
        
        # Opinion markers
        opinion_markers = [
            'i think', 'i believe', 'in my opinion', 'i feel', 'i would say',
            'might', 'maybe', 'possibly', 'probably', 'could be', 'may be',
            'seems', 'appears', 'suggests', 'arguably', 'perhaps',
            'supposedly', 'allegedly', 'reportedly',  # Hedging words
            'i guess', 'i suppose', 'personally'
        ]
        sentence_lower = sentence.lower()
        if any(marker in sentence_lower for marker in opinion_markers):
            return True
        
        # Subjective adjectives (expanded list)
        subjective_words = [
            'beautiful', 'ugly', 'best', 'worst', 'amazing', 'terrible', 
            'good', 'bad', 'great', 'awful', 'wonderful', 'horrible',
            'delicious', 'disgusting', 'boring', 'exciting', 'fun',
            'lovely', 'nasty', 'pleasant', 'unpleasant'
        ]
        for token in doc:
            if token.text.lower() in subjective_words and token.dep_ in ['acomp', 'attr']:
                return True
        
        # Commands/imperatives
        if doc[0].tag_ == 'VB' and doc[0].dep_ == 'ROOT':
            return True
        
        # Very short sentences (likely not factual claims)
        if len(doc) < 4:
            return True
        
        return False


# ==================== CONTEXT-AWARE CLAIM DETECTION ====================

class ContextAwareClaimDetector(ImprovedClaimDetector):
    """
    Extended detector that considers context from surrounding sentences
    for better multi-sentence claim detection.
    """
    
    def detect_claims_with_context(self, sentences: List[str]) -> List[Dict]:
        """
        Detect claims considering context from surrounding sentences.
        
        Returns list of dicts with: {sentence, is_claim, confidence, type, context}
        """
        results = []
        
        for i, sentence in enumerate(sentences):
            # Get surrounding context
            prev_sentence = sentences[i-1] if i > 0 else None
            next_sentence = sentences[i+1] if i < len(sentences) - 1 else None
            
            # Standard detection
            is_claim, confidence, claim_type = self.is_factual_claim(sentence)
            
            # Adjust confidence based on context
            adjusted_confidence = self._adjust_confidence_with_context(
                sentence, confidence, prev_sentence, next_sentence
            )
            
            results.append({
                'sentence': sentence,
                'is_claim': adjusted_confidence >= self.claim_threshold,
                'confidence': adjusted_confidence,
                'type': claim_type,
                'has_context': prev_sentence is not None or next_sentence is not None
            })
        
        return results
    
    def _adjust_confidence_with_context(
        self, 
        sentence: str, 
        base_confidence: float,
        prev_sentence: Optional[str],
        next_sentence: Optional[str]
    ) -> float:
        """Adjust confidence based on surrounding sentences"""
        confidence = base_confidence
        
        # Boost if previous sentence introduces the topic with strong entities
        if prev_sentence:
            prev_doc = self.nlp(prev_sentence)
            if len(prev_doc.ents) > 0:
                confidence += 0.05
        
        # Boost if next sentence continues with related claim
        if next_sentence:
            next_doc = self.nlp(next_sentence)
            curr_doc = self.nlp(sentence)
            
            # Check for entity overlap
            curr_entities = {ent.text.lower() for ent in curr_doc.ents}
            next_entities = {ent.text.lower() for ent in next_doc.ents}
            
            if curr_entities & next_entities:  # Intersection
                confidence += 0.05
        
        return min(confidence, 1.0)