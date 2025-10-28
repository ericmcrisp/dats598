""" 
Exposing a configuration class so things like embedding models, thresholds, etc. can be easily modified in one place.

Going to use this as drop down list in the application too.

"""

class Config:

    # vector db settings
    FAISS_INDEX_PATH = '../data/wikipedia/faiss_index'
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
    # EMBEDDING_MODEL_NAME = 'nvidia/llama-embed-nemotron-8b'
    # EMBEDDING_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
    
    EMBEDDING_DIM = 384

    # claim_detection.py settings
    # threholds for determining whether a statement contains a claim
    CLAIM_CONFIDENCE_THRESHOLD = 0.6

    # evidence retrieval settings
    EVIDENCE_TOP_K = 5
    EVIDENCE_MIN_SIMILARITY = 0.3

    # Fact verification thresholds
    SUPPORTS_THRESHOLD = 0.50