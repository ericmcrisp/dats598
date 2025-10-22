"""
IR for evidence gathering from Wikipedia.
"""
import requests
import wikipediaapi
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from evidence_model import Evidence, EvidenceSource


class WikipediaRetriever:
    """Retrieves evidence from Wikipedia using search and semantic matching."""

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='FactChecker/1.0 (your-email@example.com)'
        )
        self.api_url = "https://en.wikipedia.org/api/rest_v1"

    def search_articles(self, query: str, limit: int = 5) -> List[str]:
        """
        Search Wikipedia for relevant articles.

        Args:
            query: Search query string
            limit: Maximum number of article titles to return

        Returns:
            List of article titles
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json"
        }

        try:
            # set timeout to like 5 seconds because this is going to go live
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            results = response.json()
            # get article titles
            return results[1]
        # catch if this times out or anything goes wrong and print the error
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

    # parse the article contents
    def get_article_content(self, title: str) -> Optional[str]:
        page = self.wiki.page(title)
        if page.exists():
            return page.text
        return None

    # get summary of article
    def get_article_summary(self, title: str) -> Optional[str]:
        page = self.wiki.page(title)
        if page.exists():
            return page.summary
        return None
