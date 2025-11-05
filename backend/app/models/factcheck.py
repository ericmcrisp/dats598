from pydantic import BaseModel
from typing import List, Dict
from app.models.claim import Claim


class FactCheckResponse(BaseModel):
    claims: List[Claim]
    summary: Dict
