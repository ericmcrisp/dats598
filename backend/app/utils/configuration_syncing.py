from app.core.config import settings
from app.core.config import get_faiss_index_path
from app.core.public_config import PublicConfig


def sync(cfg: PublicConfig):
    settings.EMBEDDING_MODEL_COMMON_NAME = cfg.embedding_model_common_name
    settings.CLAIM_CONFIDENCE_THRESHOLD = cfg.claim_confidence_threshold
    settings.EVIDENCE_TOP_K = cfg.evidence_top_k
    settings.EVIDENCE_MIN_SIMILARITY = cfg.evidence_min_similarity
    settings.SUPPORTS_THRESHOLD = cfg.supports_threshold
    settings.MODE = cfg.mode
    settings.CLAIM_MODE = cfg.claim_mode

    try:
        faiss_path, model_name = get_faiss_index_path(cfg.embedding_model_common_name)
        settings.FAISS_INDEX_PATH = faiss_path
        settings.EMBEDDING_MODEL_NAME = model_name
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot sync config: {e}")