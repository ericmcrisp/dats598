""""
This is the class that defines the text processing steps from raw AI
output to claim detecting input.
"""

import re
import spacy


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

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