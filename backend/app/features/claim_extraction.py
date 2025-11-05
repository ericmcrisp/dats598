""""
This is the class that pulls the claim components from a claim sentence.
"""

import re
import spacy
from app.core.config import settings


class ClaimExtractor:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)

    # extract structured components from a claim
    # return: d[sentence] = {subject, predicate, object, entities, ...}
    def extract_claim_components(self, sentence):
        doc = self.nlp(sentence)
        # determine which components to extract from sentence
        components = {
            'original_text': sentence,
            'subject': None,
            'predicate': None,
            'object': None,
            'entities': [],
            'dates': [],
            'numbers': [],
            'locations': []
        }
        # get named entities
        for ent in doc.ents:
            components['entities'].append({
                'text': ent.text,
                'type': ent.label_
            })

            if ent.label_ == 'DATE':
                components['dates'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                components['locations'].append(ent.text)
        # get numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
        components['numbers'] = numbers
        # get subject-verb-object using dependency parsing
        for token in doc:
            if token.dep_ == 'nsubj':
                components['subject'] = token.text
            if token.pos_ == 'VERB':
                components['predicate'] = token.lemma_
            if token.dep_ in ['dobj', 'attr', 'pobj']:
                components['object'] = token.text

        return components

    # generate search queries from for ER
    # return: list of search query strings
    def generate_search_queries(self, claim_components):
        queries = []
        # the original claim
        queries.append(claim_components['original_text'])
        
        # subject and key entities
        if claim_components['subject']:
            entity_texts = [e['text'] for e in claim_components['entities']]
            query = f"{claim_components['subject']} {' '.join(entity_texts[:2])}"
            queries.append(query.strip())

        # main entities only (essentially keywords)
        main_entities = [e['text'] for e in claim_components['entities'] 
                        if e['type'] in ['PERSON', 'ORG', 'GPE', 'EVENT']]
        if main_entities:
            queries.append(' '.join(main_entities[:3]))

        # subject + predicate 
        if claim_components['subject'] and claim_components['predicate']:
            queries.append(f"{claim_components['subject']} {claim_components['predicate']}")

        # what else would be a good query to include....

        # strip duplicates and empty queries
        queries = list(set([q for q in queries if q and len(q.strip()) > 3]))

        return queries
