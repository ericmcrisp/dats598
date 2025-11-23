import spacy
import torch
import difflib
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from app.core.config import settings
from app.models.claim import Claim, ClaimComponent
from fastcoref import FCoref


class CoreferenceResolver:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.model = FCoref(device="cpu")   # or "cuda" if available
        self.nlp = nlp
        self.original_text = None
        self.resolved_text = None

    def resolve(self, text: str) -> str:
        self.original_text = text

        try:
            clusters = self.model.predict(texts=[text])
            # fastcoref returns a list of predictions, take first
            resolved = clusters[0].get_resolved_utterance()
            self.resolved_text = resolved if resolved else text
        except Exception as e:
            print(f"fastcoref resolution failed: {e}")
            self.resolved_text = text

        return self.resolved_text

    def resolve_span(self, span_text: str, search_window: int = 50) -> str:
        if not self.resolved_text:
            return span_text

        if span_text in self.resolved_text:
            return span_text

        matches = difflib.get_close_matches(span_text, [self.resolved_text], n=1, cutoff=0.3)
        return matches[0] if matches else span_text
