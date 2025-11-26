from pydantic import BaseModel
from typing import List, Dict, Optional
from app.models.evidence import Evidence

# this ties into claim extraction
class ClaimComponent(BaseModel):
    original_text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    entities: List[Dict[str, str]] = []
    dates: List[str] = []
    numbers: List[str] = []
    locations: List[str] = []

# this ties into claim detection
class Claim(BaseModel):
    text: str
    type: str
    confidence: float
    components: ClaimComponent
    queries: List[str]
    start_sentence_idx: Optional[int] = None
    end_sentence_idx: Optional[int] = None
    clause_index: Optional[int] = None
    context_text: Optional[str] = None
    evidence: Optional[List[Evidence]] = None
    verdict: Optional[str] = None
    explanation: Optional[str] = None
    evidence_used: Optional[List[Evidence]] = None
