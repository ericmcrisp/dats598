from pydantic import BaseModel
from typing import List, Dict
from app.models.verification import Verification


class FactCheckResponse(BaseModel):
    claims: List[Verification]
    summary: Dict
    config: Dict
