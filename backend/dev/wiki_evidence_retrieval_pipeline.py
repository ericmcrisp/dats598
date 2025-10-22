"""
Class for piping the process of evidence retrieval from wikipedia
"""

import requests
import wikipediaapi
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from claim_pipe import ClaimDetectionPipeline as pipe
from wiki_evidence_retrieval import WikipediaRetriever
from semantic_retrieval import SemanticRetriever
from claim_detection import ClaimDetector
from evidence_model import Evidence, EvidenceSource


class WikiEvidenceRetrievalPipeline:
    def __init__(self):
        self.wiki_retriever = WikipediaRetriever()
        self.semantic_retriever = SemanticRetriever()
        self.cache = {}
        self.top_k_evidence = 10

    def retrieve_evidence(self, claim: ClaimDetector, max_articles: int = 3, max_passages_per_article: int = 3) -> List[Evidence]:
        all_evidence = []
        # Use multiple search queries from claim extraction
        for query in claim.search_queries[:2]:  # Use top 2 queries
            # Search Wikipedia for articles
            article_titles = self.wiki_retriever.search_articles(query, limit=max_articles)

            for title in article_titles:
                # Check cache
                cache_key = f"{title}:{query}"
                if cache_key in self.cache:
                    all_evidence.extend(self.cache[cache_key])
                    continue

                # Get article content
                content = self.wiki_retriever.get_article_content(title)
                if not content:
                    continue

                # get the article chunks
                passages = self.semantic_retriever.chunk_text(content)

                # perform semantic search for relevant passages
                relevant_passages = self.semantic_retriever.semantic_search(
                    claim.text,
                    passages,
                    top_k=max_passages_per_article
                )

                # convert to Evidence objects
                evidence_list = []
                for passage_data in relevant_passages:
                    evidence = Evidence(
                        text=passage_data['text'],
                        source=EvidenceSource(
                            type='wikipedia',
                            title=title,
                            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        ),
                        relevance_score=passage_data['similarity'],
                        claim_text=claim.text
                    )
                    evidence_list.append(evidence)

                # save the results
                self.cache[cache_key] = evidence_list
                all_evidence.extend(evidence_list)

        # sort by score
        all_evidence = self._deduplicate_evidence(all_evidence)
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)

        n = min(len(all_evidence), self.top_k_evidence)
        return all_evidence[:n]

    def _deduplicate_evidence(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence passages."""
        seen_texts = set()
        unique_evidence = []

        for evidence in evidence_list:
            # Use first 100 chars as key for deduplication
            key = evidence.text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_evidence.append(evidence)

        return unique_evidence

    def batch_retrieve_evidence(self, claims: List[ClaimDetector]) -> Dict[str, List[Evidence]]:
        results = {}
        for claim in claims:
            evidence = self.retrieve_evidence(claim)
            results[claim.text] = evidence
        return results