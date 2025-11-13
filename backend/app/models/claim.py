from pydantic import BaseModel
from typing import List, Dict, Optional

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
    searches: List[str]
