from fastapi import APIRouter

from app.features.factcheck_pipe import FactCheckPipe as fcp
from app.models.prompt import PromptRequest
from app.models.factcheck import FactCheckResponse

router = APIRouter()

# define the way the endpoint is handled
@router.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(payload: PromptRequest):
    fcp(payload.text)
    return fcp.process()
