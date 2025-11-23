"""
Improved ClaimDetector with better pattern matching and scoring.
"""

import re
import spacy
from app.core.config import settings


class ClaimDetector:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)
        self.claim_threshold = settings.CLAIM_CONFIDENCE_THRESHOLD

    def is_factual_claim(self, sentence):
        doc = self.nlp(sentence)

        # CRITICAL FIX: Check non-claims FIRST before pattern matching
        if self._is_non_claim(sentence, doc):
            return False, 0.0, None

        # Check all pattern types
        patterns = {
            'temporal': self._has_temporal_claim(sentence, doc),
            'numerical': self._has_numerical_claim(sentence, doc),
            'relational': self._has_relational_claim(sentence, doc),
            'definitional': self._has_definitional_claim(sentence, doc),
            'event': self._has_event_claim(sentence, doc),
            'basic_fact': self._has_basic_fact_claim(sentence, doc)
        }

        detected_patterns = [k for k, v in patterns.items() if v]

        # Improved confidence scoring
        if detected_patterns:
            # Base confidence on number of patterns + type of patterns
            base_confidence = len(detected_patterns) * 0.25
            
            # Boost confidence for strong patterns
            strong_patterns = {'temporal', 'numerical', 'event'}
            if any(p in strong_patterns for p in detected_patterns):
                base_confidence += 0.2
            
            # Check if sentence has proper structure (subj-verb-obj)
            if self._has_proper_structure(doc):
                base_confidence += 0.15
            
            confidence = min(base_confidence, 1.0)
            
            # Only consider it a claim if confidence exceeds threshold
            if confidence >= self.claim_threshold:
                return True, confidence, detected_patterns[0]
        
        return False, 0.0, None

    def _has_temporal_claim(self, sentence, doc):
        temporal_patterns = [
            r'\b(in|on|during|since|until|by|before|after)\s+\d{4}\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(was|were|is|are|became|happened)\s+.*\s+(in|on|during)\s+\d{4}\b',
            r'\b(born|died|founded|established|created|built|opened|launched|released|published)\s+(in|on)\s+\d{4}\b',
            r'\bfrom\s+\d{4}\s+to\s+\d{4}\b',
            r'\b(early|late|mid)[-\s]\d{4}s?\b',
            r'\b\d{1,2}(st|nd|rd|th)?\s+century\b'
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for time-related entities
        has_date = any(ent.label_ == 'DATE' for ent in doc.ents)
        has_past_tense = any(token.tag_ in ['VBD', 'VBN'] for token in doc)
        
        return has_date and has_past_tense

    def _has_numerical_claim(self, sentence, doc):
        numerical_patterns = [
            r'\b\d+[\d,]*\.?\d*\s*(meters?|feet|ft|miles?|mi|kilometers?|km|kgs?|kilograms?|pounds?|lbs?|tons?|tonnes?|degrees?|celsius|fahrenheit)\b',
            r'\b\d+[\d,]*\.?\d*\s*(percent|%)\b',
            r'\b(population|area|height|width|weight|length|distance|speed|temperature|cost|price|revenue|profit)\s+(of|is|was|are|were|reached|exceeded|totals?)\s+[\$€£¥]?\d+[\d,]*\.?\d*',
            r'\bmeasures?\s+\d+[\d,]*\.?\d*',
            r'\bstands?\s+(at\s+)?\d+[\d,]*\.?\d*',
            r'\bweighs?\s+\d+[\d,]*\.?\d*',
            r'\b\d+[\d,]*\.?\d*\s+(million|billion|trillion|thousand)',
            r'\bover\s+\d+[\d,]*',
            r'\bmore than\s+\d+[\d,]*',
            r'\bless than\s+\d+[\d,]*',
            r'\bapproximately\s+\d+[\d,]*',
            r'\babout\s+\d+[\d,]*'
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for quantity entities
        return any(ent.label_ in ['QUANTITY', 'PERCENT', 'MONEY', 'CARDINAL'] 
                  for ent in doc.ents)

    def _has_relational_claim(self, sentence, doc):
        relational_patterns = [
            r'\b(located|situated|found|positioned|placed|based)\s+(in|at|near|on)\b',
            r'\b(capital|president|king|queen|leader|founder|CEO|director|chairman|head|chief|minister|governor)\s+of\b',
            r'\b(married|divorced|related|connected|linked|associated)\s+to\b',
            r'\bis\s+(the|a)\s+\w+\s+of\b',
            r'\b(owns|controls|manages|operates|runs)\s+\w+',
            r'\b(part|member|subsidiary)\s+of\b',
            r'\b(north|south|east|west|northwest|northeast|southwest|southeast)\s+of\b',
            r'\b(borders?|adjacent|neighboring)\b'
        ]
        
        for pattern in relational_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for location relationships
        has_location = any(ent.label_ in ['GPE', 'LOC', 'FAC'] for ent in doc.ents)
        has_preposition = any(token.dep_ == 'prep' and token.text.lower() in ['in', 'at', 'on', 'near'] 
                             for token in doc)
        
        return has_location and has_preposition

    def _has_definitional_claim(self, sentence, doc):
        definitional_patterns = [
            r'\b(is|are|was|were)\s+(a|an|the)\s+[\w\s]+\b',
            r'\b(defined|known|called|termed|referred|regarded|considered)\s+(as|to be)\b',
            r'\bknown for\b',
            r'\bfamous for\b',
            r'\btype of\b',
            r'\bkind of\b',
            r'\bform of\b'
        ]
        
        for pattern in definitional_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Must have named entities OR be defining something specific
                has_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART', 'EVENT'] 
                               for ent in doc.ents)
                has_proper_noun = any(token.pos_ == 'PROPN' for token in doc)
                
                if has_entity or has_proper_noun:
                    return True
        
        return False

    def _has_event_claim(self, sentence, doc):
        # Check for past tense verbs
        has_past_verb = any(token.tag_ in ['VBD', 'VBN'] for token in doc)
        
        # Check for relevant entities
        has_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'PRODUCT'] 
                        for ent in doc.ents)
        
        # Expanded event keywords
        event_keywords = [
            'won', 'defeated', 'conquered', 'discovered', 'invented', 'wrote', 'painted',
            'directed', 'signed', 'originated', 'started', 'formed', 'founded', 'created',
            'built', 'launched', 'released', 'published', 'announced', 'declared',
            'established', 'developed', 'introduced', 'achieved', 'accomplished',
            'became', 'served', 'led', 'commanded', 'ruled', 'governed', 'elected'
        ]
        
        has_event_keyword = any(keyword in sentence.lower() for keyword in event_keywords)
        
        # Event claim if it has past tense AND (entity + keyword OR event entity)
        has_event_entity = any(ent.label_ == 'EVENT' for ent in doc.ents)
        
        return has_past_verb and (
            (has_entity and has_event_keyword) or 
            has_event_entity
        )

    def _is_non_claim(self, sentence, doc):
        if not sentence.strip() or len(doc) == 0:
            return True
        
        sentence_lower = sentence.lower().strip()
        
        # Questions
        if sentence.strip().endswith('?'):
            return True
        
        # Check for question words at the start
        question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom']
        if any(sentence_lower.startswith(qw) for qw in question_words):
            return True
        
        # Opinion markers - be more specific to avoid false positives
        strong_opinion_markers = [
            'i think', 'i believe', 'in my opinion', 'i feel', 'my view',
            'i would say', 'personally', 'i suppose', 'i guess'
        ]
        
        if any(marker in sentence_lower for marker in strong_opinion_markers):
            return True
        
        # Uncertainty markers - only reject if they're the main claim
        uncertainty_markers = ['might', 'maybe', 'possibly', 'probably', 'perhaps', 'could be']
        # Only reject if uncertainty is about the main verb
        main_verb_uncertain = False
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp']:
                # Check if verb is modified by uncertainty marker
                for child in token.children:
                    if child.text.lower() in uncertainty_markers:
                        main_verb_uncertain = True
                        break
        
        if main_verb_uncertain:
            return True
        
        # Subjective adjectives - only reject if they're the main predicate
        subjective_words = ['beautiful', 'ugly', 'best', 'worst', 'amazing', 'terrible', 
                          'good', 'bad', 'great', 'awful', 'wonderful', 'horrible']
        
        for token in doc:
            # Only reject if subjective word is the main attribute or complement
            if (token.text.lower() in subjective_words and 
                token.dep_ in ['acomp', 'attr'] and
                token.head.dep_ == 'ROOT'):
                return True
        
        # Imperatives (commands)
        if doc[0].tag_ == 'VB' and doc[0].dep_ == 'ROOT':
            return True
        
        return False

    def _has_basic_fact_claim(self, sentence, doc):
        has_subj = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
        has_verb = any(token.pos_ == 'VERB' for token in doc)
        has_obj = any(token.dep_ in ['dobj', 'attr', 'pobj', 'acomp'] for token in doc)
        has_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 
                                        'EVENT', 'WORK_OF_ART', 'FAC'] 
                        for ent in doc.ents)
        
        # Also check for proper nouns as subjects
        has_proper_subj = any(token.pos_ == 'PROPN' and token.dep_ in ['nsubj', 'nsubjpass'] 
                             for token in doc)
        
        return (has_subj or has_proper_subj) and has_verb and (has_obj or has_entity)
    
    def _has_proper_structure(self, doc):
        # Must have a root verb
        has_root_verb = any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' for token in doc)
        
        # Must have a subject
        has_subject = any(token.dep_ in ['nsubj', 'nsubjpass'] for token in doc)
        
        # Sentence should be declarative (not too short, not a fragment)
        is_reasonable_length = len(doc) >= 3
        
        return has_root_verb and has_subject and is_reasonable_length