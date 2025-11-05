from pydantic import BaseModel
from typing import List, Dict, Optional


class ClaimComponent(BaseModel):
    original_text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    entities: List[str] = []
    dates: List[str] = []
    numbers: List[str] = []
    locations: List[str] = []

class Claim(BaseModel):
    text: str
    type: str
    confidence: float
    components: ClaimComponent
    searches: List[str]
