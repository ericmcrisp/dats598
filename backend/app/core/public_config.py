from pydantic import BaseModel

class PublicConfig(BaseModel):
    embedding_model_common_name: str = "mini_L6"
    claim_confidence_threshold: float = 0.6
    evidence_top_k: int = 5
    evidence_min_similarity: float = 0.3
    supports_threshold: float = 0.5
    mode: str = 'rules'
    claim_mode: str = 'advanced'
