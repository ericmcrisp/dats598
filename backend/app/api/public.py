from fastapi import APIRouter
from app.core.config import settings
from app.core.public_config import PublicConfig
from app.utils.configuration_syncing import sync

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


@router.put("/config", response_model=PublicConfig)
def update_public_config(new_cfg: PublicConfig):
    sync(new_cfg)
    return new_cfg
