""""
This is the class that detects claims and classifies based on
cleaned text from the TextPreprocessor class.
"""

import re
import spacy


class ClaimDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.claim_threshold = 0.3  # the confidence threshold to consider a sentence a claim

    # detect if a sentence contains a factual claim
    # returns: (is_claim: bool, confidence: float, claim_type: str)
    def is_factual_claim(self, sentence):
        doc = self.nlp(sentence)

        # create rules based on sentence structures and keywords
        patterns = {
            'temporal': self._has_temporal_claim(sentence, doc),
            'numerical': self._has_numerical_claim(sentence, doc),
            'relational': self._has_relational_claim(sentence, doc),
            'definitional': self._has_definitional_claim(sentence, doc),
            'event': self._has_event_claim(sentence, doc)
        }

        # edge case where sentence isn't a staement (questio, exclaimation, ..)
        if self._is_non_claim(sentence, doc):
            return False, 0.0, None

        # group patterns
        detected_patterns = [k for k, v in patterns.items() if v]

        # create simple confidence scoring process based on number of
        # detected patterns
        if detected_patterns:
            confidence = len(detected_patterns) * self.claim_threshold
            confidence = min(confidence, 1.0)
            return True, confidence, detected_patterns[0]
        return False, 0.0, None

    # detect time dependent claims based on common temporal phrases
    def _has_temporal_claim(self, sentence, doc):
        temporal_patterns = [
            r'\b(in|on|during|since|until|by)\s+\d{4}\b', 
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(was|were|is|are)\s+\w+\s+(in|on|during)\s+\d{4}\b',
            r'\b(born|died|founded|established|created|built|opened)\s+(in|on)\s+\d{4}\b'
        ]
        # use regex to match patterns
        for pattern in temporal_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    # detect numerical claims based on common phrases
    # this can be beefed up a lot with more patterns
    def _has_numerical_claim(self, sentence, doc):
        numerical_patterns = [
            r'\b\d+\s*(meters?|feet|miles?|kilometers?|kgs?|pounds?|tons?|degrees?)\b',
            r'\b\d+(\.\d+)?\s*percent|%\b',
            r'\b(population|area|height|width|weight|length|distance)\s+(of|is|was)\s+\d+',
            r'\bmeasures?\s+\d+',
            r'\bstands?\s+\d+',
            r'\bweighs?\s+\d+'
        ]
        for pattern in numerical_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    # detect relational claims
    def _has_relational_claim(self, sentence, doc):
        """Detects claims about relationships or locations"""
        relational_patterns = [
            r'\b(located|situated|found|positioned)\s+in\b',
            r'\b(capital|president|king|queen|leader|founder|CEO|director)\s+of\b',
            r'\b(married|divorced|related)\s+to\b',
            r'\bis\s+(the|a)\s+\w+\s+of\b'
        ]
        for pattern in relational_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    # detect definitional claims
    def _has_definitional_claim(self, sentence, doc):
        # "x is a y" or "x are y" patterns
        definitional_patterns = [
            r'\b(is|are|was|were)\s+(a|an|the)\s+\w+',
            r'\b(defined|known|called|termed|referred)\s+as\b'
        ]
        for pattern in definitional_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Must have named entities to be factual
                if any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT'] 
                       for ent in doc.ents):
                    return True
        return False

    # detect event claims
    def _has_event_claim(self, sentence, doc):
        # looks for past tense verbs with named entities
        has_past_verb = any(token.tag_ in ['VBD', 'VBN'] for token in doc)
        has_entity = any(ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT'] 
                        for ent in doc.ents)

        event_keywords = ['won', 'defeated', 'conquered', 'discovered', 'invented', 'wrote', 'painted', 'directed', 'signed']
        has_event_keyword = any(keyword in sentence.lower() 
                               for keyword in event_keywords)

        return has_past_verb and has_entity and has_event_keyword

    # filter out non-claims like questions and opinions
    def _is_non_claim(self, sentence, doc):
        # quetions end with ?, remove those
        if sentence.strip().endswith('?'):
            return True
        # if opinion keywords hit, remove those
        opinion_markers = [
            'i think', 'i believe', 'in my opinion', 'i feel',
            'might', 'maybe', 'possibly', 'probably', 'could be',
            'seems', 'appears', 'suggests', 'arguably'
        ]
        if any(marker in sentence.lower() for marker in opinion_markers):
            return True
        # if using an adjective to make a claim - she is the funniest.
        subjective_words = ['beautiful', 'ugly', 'best', 'worst', 'amazing', 'terrible', 'good', 'bad']        
        for token in doc:
            if token.text.lower() in subjective_words and token.dep_ in ['acomp', 'attr']:
                return True
        return False