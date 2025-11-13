from pydantic import BaseModel
from typing import Optional, List


class PromptRequest(BaseModel):
    text: str
    cfg: Optional[dict] = None

# define what a cleaned response looks like
class PromptResponse(BaseModel):
    sentence_type: Optional[str] = None
    is_factual: Optional[str] = None
    subject: Optional[str] = None
    entities: Optional[List] = None
    main_verb: Optional[str] = None
    cleaned: Optional[str] = None
