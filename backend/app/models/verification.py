from pydantic import BaseModel
from typing import List, Dict, Optional
from app.models.claim import Claim
from app.models.evidence import Evidence

# this ties into claim extraction
class Verification(BaseModel):
    claim: Claim
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    evidence_count: Optional[int] = None
    max_similarity: Optional[float] = None
    avg_similarity: Optional[float] = None
    best_evidence: Optional[Dict] = None
    all_evidence: Optional[List[Evidence]] = None
    explanation: Optional[str] = None
