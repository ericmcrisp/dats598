from pydantic import BaseModel

class PublicConfig(BaseModel):
    embedding_model_name: str
    claim_confidence_threshold: float
    evidence_top_k: int
    evidence_min_similarity: float
    supports_threshold: float