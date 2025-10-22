"""
Class for determining semantic similarity between a claim and evidence passages
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticRetriever:

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.min_similarity_threshold = 0.25

    # create chunks of overlappign sentences from text
    def chunk_text(self, text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
        sentences = text.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def semantic_search(self, query: str, passages: List[str], top_k: int = 5) -> List[Dict[str, float]]:
        if not passages:
            return []
        # encode
        query_embedding = self.encoder.encode([query])
        passage_embeddings = self.encoder.encode(passages)
        # get similarity
        similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
        # get knn indicies
        knn_idx = np.argsort(similarities)[-top_k:][::-1]
        # parse and gather results
        results = []
        for i in knn_idx:
            if similarities[i] > self.min_similarity_threshold:
                results.append({
                    'index': int(i),
                    'similarity': float(similarities[i]),
                    'text': passages[i]
                })

        return results
