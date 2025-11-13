from app.core.config import settings
from app.core.public_config import PublicConfig


def sync(cfg: PublicConfig):
    settings.EMBEDDING_MODEL_NAME = cfg.embedding_model_name
    settings.CLAIM_CONFIDENCE_THRESHOLD = cfg.claim_confidence_threshold
    settings.EVIDENCE_TOP_K = cfg.evidence_top_k
    settings.EVIDENCE_MIN_SIMILARITY = cfg.evidence_min_similarity
    settings.SUPPORTS_THRESHOLD = cfg.supports_threshold