from fastapi import APIRouter
from models.prompt import PromptRequest, PromptResponse
from features.analyze_prompt import analyze_prompt

router = APIRouter()

# define the way the endpoint is handled
@router.post("/prompt", response_model=PromptResponse)
async def handle_prompt(payload: PromptRequest):
    result = analyze_prompt(payload.text)
    return result
