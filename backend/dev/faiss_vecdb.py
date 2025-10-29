"""
create the faiss vector database
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import Config

class FaissVecDB:
    def __init__(self):
        self.index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        self.documents = []
        self.metadata = []
        self.top_k = Config.EVIDENCE_TOP_K

    def add_documents(self, documents: list, metadata: list):
        # encode doc
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend(metadata)

    # find the top k documents
    def search(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k
        # encode and normalize query for search
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding, top_k)
        # parse results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'text': self.documents[idx],
                    'similarity': float(similarity),
                    'metadata': self.metadata[idx]
                })
        return results

    def save(self, path: str = 'faiss_index'):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}_docs.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'metadata': self.metadata}, f)

    def load(self, path: str = 'faiss_index'):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}_docs.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']