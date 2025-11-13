from fastapi import APIRouter, HTTPException
from app.features.factcheck_pipe import FactCheckPipe as fcp
from app.models.prompt import PromptRequest
from app.models.factcheck import FactCheckResponse

# from app.core.config import settings

router = APIRouter()

# define the way the endpoint is handled
@router.post("/factcheck", response_model=FactCheckResponse)
async def factcheck(request: PromptRequest):
    try:
        pipe = fcp()
        print("Using this config:", pipe.cfg)
        response = pipe.process(request.text)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
