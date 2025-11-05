""""
This is the class that defines the text processing steps from raw AI
output to claim detecting input.
"""

import re
import spacy
from app.core.config import settings

# create cache
_SPACY_MODELS = {}


class TextPreprocessor:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        model = self.cfg.SPACY_MODEL
        if model not in _SPACY_MODELS:
            _SPACY_MODELS[model] = spacy.load(model)
        self.nlp = _SPACY_MODELS[model]

    # clean and normalize the text
    def clean_text(self, text):
        # remove whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # drop URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # drop special characters
        text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
        return text

    # segment text into sentences
    def segment_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]