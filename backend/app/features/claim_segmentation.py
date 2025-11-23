""""
This is the class that pulls the claim components from the text
"""

import re
import spacy
from app.core.config import settings
from app.models.claim import Claim, ClaimComponent
from app.features.claim_detection import ClaimDetector


class ClaimSegmentator:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)
        self.detector = ClaimDetector(cfg=self.cfg)
        self.segments = []

    def segment(self, sentences):
        claims = []
        clause_idx = 0
        for i, s in enumerate(sentences):
            doc = self.nlp(s)
            clauses = self.split_into_clauses(doc)
            for clause in clauses:
                if self.is_clause_claim(clause):
                    claims.append({'text': clause.text,
                                   'start_sentence_idx': i,
                                   'end_sentence_idx': i,
                                   'original_text': s,
                                   'clause_index': clause_idx
                                   })
                    clause_idx += 1
        claims = self.merge_cross_sentence_claims(claims, self.nlp)
        self.segments = claims
        return self.segments

    def split_into_clauses(self, doc):
        clauses = []
        clause_start = 0
        for token in doc:
            if token.text in ['.', ';', ':', ','] or token.dep_ == 'cc':
                span = doc[clause_start: token.i+1]
                clauses.append(span)
                clause_start = token.i + 1
        if clause_start < len(doc):
            clauses.append(doc[clause_start:])
        return clauses

    def is_clause_claim(self, clause):
        is_claim, confidence, _ = self.detector.is_factual_claim(clause.text)
        return is_claim and confidence > self.detector.claim_threshold

    def merge_cross_sentence_claims(self, claim_units, nlp):
        if not claim_units:
            return []

        merged_claims = []
        buffer = [claim_units[0]]
        for prev, curr in zip(claim_units, claim_units[1:]):
            prev_doc = nlp(prev['text'])
            curr_doc = nlp(curr['text'])

            # heuristic 1: claim start with a pronoun
            curr_starts_with_pronoun = curr_doc[0].pos_ == "PRON"
            # heuristic 2: shared named entities
            prev_entities = {ent.text.lower() for ent in prev_doc.ents}
            curr_entities = {ent.text.lower() for ent in curr_doc.ents}
            shared_entities = prev_entities & curr_entities
            # merge?
            if curr_starts_with_pronoun or shared_entities:
                # merge current claim into buffer
                buffer.append(curr)
            else:
                # flush buffer as a merged claim
                merged_claims.append({
                    "text": " ".join([c['text'] for c in buffer]),
                    "start_sentence_idx": buffer[0]['start_sentence_idx'],
                    "end_sentence_idx": buffer[-1]['end_sentence_idx'],
                    "original_text": " ".join([c['original_text'] for c in buffer]),
                    "clause_index": buffer[0]['clause_index']
                })
                buffer = [curr]
        # flush the final buffer
        if buffer:
            merged_claims.append({
                "text": " ".join([c['text'] for c in buffer]),
                "start_sentence_idx": buffer[0]['start_sentence_idx'],
                "end_sentence_idx": buffer[-1]['end_sentence_idx'],
                "original_text": " ".join([c['original_text'] for c in buffer]),
                "clause_index": buffer[0]['clause_index']
            })

        return merged_claims
