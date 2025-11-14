""""
Improved ClaimExtractor with better component extraction and
smarter search query generation.
"""

import re
import spacy
from typing import List, Dict, Set
from app.core.config import settings
from app.models.claim import ClaimComponent


class ImprovedClaimExtractor:
    def __init__(self, cfg=None, nlp=None):
        self.cfg = cfg or settings
        self.nlp = nlp or spacy.load(self.cfg.SPACY_MODEL)

    def extract_claim_components(self, sentence: str) -> ClaimComponent:
        """
        Enhanced extraction of structured components from a claim.
        
        Returns a ClaimComponent with improved entity and relationship extraction.
        """
        doc = self.nlp(sentence)
        
        components = {
            'original_text': sentence,
            'subject': None,
            'predicate': None,
            'object': None,
            'entities': [],
            'dates': [],
            'numbers': [],
            'locations': [],
            'organizations': [],  # NEW
            'persons': [],  # NEW
            'modifiers': [],  # NEW (adjectives, adverbs)
            'relations': []  # NEW (prepositions showing relationships)
        }
        
        # Extract named entities with enhanced categorization
        self._extract_entities(doc, components)
        
        # Extract numbers with context
        self._extract_numbers_with_context(sentence, doc, components)
        
        # Extract subject-verb-object with improved dependency parsing
        self._extract_svo(doc, components)
        
        # Extract modifiers and qualifiers
        self._extract_modifiers(doc, components)
        
        # Extract relational phrases
        self._extract_relations(doc, components)
        
        return ClaimComponent(**components)

    def _extract_entities(self, doc, components: Dict):
        """Enhanced entity extraction with categorization"""
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            components['entities'].append(entity_info)
            
            # Categorize by type
            if ent.label_ == 'DATE':
                components['dates'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC', 'FAC']:
                components['locations'].append(ent.text)
            elif ent.label_ == 'ORG':
                components['organizations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                components['persons'].append(ent.text)

    def _extract_numbers_with_context(self, sentence: str, doc, components: Dict):
        """Extract numbers along with their units and context"""
        # Pattern to match numbers with units
        number_patterns = [
            (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(meters?|feet|ft|miles?|mi|kilometers?|km)', 'distance'),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(kgs?|kilograms?|pounds?|lbs?|tons?|tonnes?)', 'weight'),
            (r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:percent|%)', 'percentage'),
            (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|billion|trillion)?', 'money'),
            (r'(\d+(?:,\d{3})*)\s*(people|citizens|residents|inhabitants)', 'population'),
        ]
        
        for pattern, context_type in number_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                components['numbers'].append({
                    'value': match.group(1),
                    'unit': match.group(2) if match.lastindex > 1 else None,
                    'type': context_type,
                    'full_text': match.group(0)
                })
        
        # Also get standalone numbers
        standalone_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence)
        for num in standalone_numbers:
            if not any(num == n.get('value') or num in n.get('full_text', '') 
                      for n in components['numbers'] if isinstance(n, dict)):
                components['numbers'].append({
                    'value': num,
                    'unit': None,
                    'type': 'unknown',
                    'full_text': num
                })

    def _extract_svo(self, doc, components: Dict):
        """
        Improved subject-verb-object extraction using dependency parsing.
        Handles more complex sentence structures.
        """
        subjects = []
        predicates = []
        objects = []
        
        for token in doc:
            # Subject extraction (including compound subjects)
            if token.dep_ in ['nsubj', 'nsubjpass']:
                subject_phrase = self._get_full_phrase(token)
                subjects.append(subject_phrase)
            
            # Predicate/verb extraction
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Get the verb with any auxiliaries
                verb_phrase = self._get_verb_phrase(token)
                predicates.append(verb_phrase)
            
            # Object extraction (direct objects, prepositional objects, attributes)
            if token.dep_ in ['dobj', 'attr', 'pobj', 'dative']:
                object_phrase = self._get_full_phrase(token)
                objects.append(object_phrase)
        
        # Store the most complete versions
        components['subject'] = ', '.join(subjects) if subjects else None
        components['predicate'] = predicates[0] if predicates else None
        components['object'] = ', '.join(objects) if objects else None

    def _get_full_phrase(self, token) -> str:
        """Get the full noun phrase including compounds and modifiers"""
        # Get all children (compounds, adjectives, determiners)
        phrase_tokens = [token]
        
        for child in token.children:
            if child.dep_ in ['compound', 'amod', 'det', 'nummod', 'poss']:
                phrase_tokens.append(child)
        
        # Sort by position in sentence and join
        phrase_tokens.sort(key=lambda t: t.i)
        return ' '.join([t.text for t in phrase_tokens])

    def _get_verb_phrase(self, verb_token) -> str:
        """Get the verb with auxiliaries (e.g., 'was built', 'has been')"""
        verb_tokens = [verb_token]
        
        for child in verb_token.children:
            if child.dep_ in ['aux', 'auxpass', 'neg']:
                verb_tokens.append(child)
        
        verb_tokens.sort(key=lambda t: t.i)
        return ' '.join([t.text for t in verb_tokens])

    def _extract_modifiers(self, doc, components: Dict):
        """Extract important modifiers (adjectives, adverbs) that add context"""
        modifiers = []
        
        for token in doc:
            if token.pos_ in ['ADJ', 'ADV']:
                # Only include if modifying a noun or verb (not standalone)
                if token.dep_ in ['amod', 'advmod', 'acomp']:
                    modifiers.append({
                        'text': token.text,
                        'type': token.pos_,
                        'modifies': token.head.text
                    })
        
        components['modifiers'] = modifiers

    def _extract_relations(self, doc, components: Dict):
        """Extract relational phrases (prepositional phrases showing relationships)"""
        relations = []
        
        for token in doc:
            if token.pos_ == 'ADP':  # Preposition
                # Get the object of the preposition
                pobj = [child for child in token.children if child.dep_ == 'pobj']
                if pobj:
                    relations.append({
                        'preposition': token.text,
                        'object': self._get_full_phrase(pobj[0]),
                        'full_phrase': f"{token.text} {self._get_full_phrase(pobj[0])}"
                    })
        
        components['relations'] = relations

    def generate_search_queries(self, claim_components: ClaimComponent) -> List[str]:
        """
        Generate diverse, high-quality search queries from claim components.
        
        Strategy:
        1. Full claim (baseline)
        2. Key entities only
        3. Subject + predicate + key object
        4. Entities + numbers/dates (for factual verification)
        5. Specific entity combinations
        """
        queries = []
        seen_queries: Set[str] = set()
        
        # Helper to add unique queries
        def add_query(q: str):
            q = q.strip()
            if q and len(q) > 3 and q.lower() not in seen_queries:
                queries.append(q)
                seen_queries.add(q.lower())
        
        # 1. Original claim (slightly cleaned)
        cleaned_claim = self._clean_query(claim_components.original_text)
        add_query(cleaned_claim)
        
        # 2. Main entities only (most important for search)
        main_entity_types = ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']
        main_entities = [
            e['text'] for e in claim_components.entities 
            if e['type'] in main_entity_types
        ]
        if main_entities:
            add_query(' '.join(main_entities[:3]))
        
        # 3. Subject + predicate (core claim structure)
        if claim_components.subject and claim_components.predicate:
            core_query = f"{claim_components.subject} {claim_components.predicate}"
            add_query(core_query)
            
            # Add object if available
            if claim_components.object:
                add_query(f"{core_query} {claim_components.object}")
        
        # 4. Entities + temporal context (for time-based claims)
        if main_entities and claim_components.dates:
            temporal_query = f"{main_entities[0]} {claim_components.dates[0]}"
            add_query(temporal_query)
        
        # 5. Entities + numerical context (for quantitative claims)
        if main_entities and claim_components.numbers:
            # Get the first number value
            first_num = claim_components.numbers[0]
            num_value = first_num['value'] if isinstance(first_num, dict) else first_num
            numerical_query = f"{main_entities[0]} {num_value}"
            add_query(numerical_query)
        
        # 6. Organization + location (for organizational claims)
        if claim_components.organizations and claim_components.locations:
            add_query(f"{claim_components.organizations[0]} {claim_components.locations[0]}")
        
        # 7. Person + organization (for biographical claims)
        if claim_components.persons and claim_components.organizations:
            add_query(f"{claim_components.persons[0]} {claim_components.organizations[0]}")
        
        # 8. Subject + key relation (for relational claims)
        if claim_components.subject and claim_components.relations:
            relation = claim_components.relations[0]['full_phrase']
            add_query(f"{claim_components.subject} {relation}")
        
        # 9. Question form (sometimes better for fact-checking)
        if claim_components.subject and claim_components.predicate:
            # Convert to question form for better search results
            question = self._create_question_query(claim_components)
            if question:
                add_query(question)
        
        # 10. Fallback: just the most important words (nouns and verbs)
        if len(queries) < 3:
            doc = self.nlp(claim_components.original_text)
            important_words = [
                token.text for token in doc 
                if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and len(token.text) > 3
            ]
            if important_words:
                add_query(' '.join(important_words[:5]))
        
        return queries[:8]  # Return top 8 most diverse queries

    def _clean_query(self, query: str) -> str:
        """Clean a query string for better search results"""
        # Remove filler words
        filler_words = ['the', 'a', 'an', 'is', 'was', 'were', 'are', 'been', 'be']
        words = query.split()
        cleaned_words = [w for w in words if w.lower() not in filler_words]
        
        # Remove punctuation except hyphens
        cleaned = ' '.join(cleaned_words)
        cleaned = re.sub(r'[^\w\s\-]', '', cleaned)
        
        return cleaned.strip()

    def _create_question_query(self, components: ClaimComponent) -> str:
        """Convert claim components into a question form for search"""
        # Simple question formation based on predicate
        subject = components.subject
        predicate = components.predicate
        obj = components.object
        
        if not subject or not predicate:
            return None
        
        # Handle different verb types
        if predicate in ['is', 'are', 'was', 'were']:
            if obj:
                return f"what is {subject}"
            return f"who is {subject}"
        elif predicate in ['has', 'have', 'had']:
            return f"does {subject} have {obj}" if obj else f"what does {subject} have"
        elif predicate in ['located', 'situated', 'found']:
            return f"where is {subject} located"
        else:
            # Generic question form
            return f"did {subject} {predicate}" if obj is None else f"when did {subject} {predicate}"


class MultiSentenceClaimHandler:
    """
    Handles claims that span multiple sentences by grouping related
    sentences and treating them as a single claim unit.
    """
    
    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load(settings.SPACY_MODEL)
    
    def group_related_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Group sentences that likely form a single multi-sentence claim.
        
        Returns: List of sentence groups (each group is a list of sentences)
        """
        if not sentences:
            return []
        
        groups = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            prev_sentence = sentences[i - 1]
            curr_sentence = sentences[i]
            
            if self._are_related(prev_sentence, curr_sentence):
                current_group.append(curr_sentence)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [curr_sentence]
        
        # Add the last group
        groups.append(current_group)
        
        return groups
    
    def _are_related(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are related and should be grouped"""
        doc1 = self.nlp(sent1)
        doc2 = self.nlp(sent2)
        
        # Check entity overlap
        entities1 = {ent.text.lower() for ent in doc1.ents}
        entities2 = {ent.text.lower() for ent in doc2.ents}
        
        entity_overlap = len(entities1 & entities2) > 0
        
        # Check for pronouns at start of second sentence (indicates continuation)
        pronouns = {'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those'}
        starts_with_pronoun = doc2[0].text.lower() in pronouns
        
        # Check for continuation words
        continuation_words = {
            'additionally', 'furthermore', 'moreover', 'also', 
            'however', 'nevertheless', 'therefore', 'thus',
            'subsequently', 'consequently', 'meanwhile'
        }
        has_continuation = doc2[0].text.lower() in continuation_words
        
        return entity_overlap or starts_with_pronoun or has_continuation
    
    def merge_sentence_group(self, sentence_group: List[str]) -> str:
        """Merge a group of related sentences into a single claim text"""
        return ' '.join(sentence_group)