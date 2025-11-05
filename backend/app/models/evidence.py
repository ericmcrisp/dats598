from pydantic import BaseModel
from typing import List, Dict, Optional


class EvidenceSource(BaseModel):
    type: str
    title: str
    url: Optional[str] = None

class Evidence(BaseModel):
    text: str
    source: EvidenceSource
    relevance_score: float
    claim_text: str
