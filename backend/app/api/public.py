from fastapi import APIRouter
from app.core.config import settings
from app.core.public_config import PublicConfig

router = APIRouter()

@router.get("/config", response_model=PublicConfig)
async def get_config():
    return PublicConfig(
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
        claim_confidence_threshold=settings.CLAIM_CONFIDENCE_THRESHOLD,
        evidence_top_k=settings.EVIDENCE_TOP_K,
        evidence_min_similarity=settings.EVIDENCE_MIN_SIMILARITY,
        supports_threshold=settings.SUPPORTS_THRESHOLD,
    )

@router.post("/config", response_model=PublicConfig)
async def update_config(new_config: PublicConfig):
    settings.EMBEDDING_MODEL_NAME = new_config.embedding_model_name
    settings.CLAIM_CONFIDENCE_THRESHOLD = new_config.claim_confidence_threshold
    settings.EVIDENCE_TOP_K = new_config.evidence_top_k
    settings.EVIDENCE_MIN_SIMILARITY = new_config.evidence_min_similarity
    settings.SUPPORTS_THRESHOLD = new_config.supports_threshold
    return new_config
