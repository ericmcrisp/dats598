"""
Class definitions for tracking evidence to claim for retrieval and verification.
"""
from typing import Optional
from pydantic import BaseModel, HttpUrl


# source information for a piece of evidence
class EvidenceSource(BaseModel):
    type: str  # 'wikipedia', 'fever', 'news', etc.
    title: str
    url: Optional[HttpUrl] = None
    author: Optional[str] = None
    date_published: Optional[str] = None


# piece of evidence retrieved for claim verification
class Evidence(BaseModel):
    text: str
    source: EvidenceSource
    relevance_score: float
    claim_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The Eiffel Tower was completed in 1889 as the entrance arch for the World's Fair.",
                "source": {
                    "type": "wikipedia",
                    "title": "Eiffel Tower",
                    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower"
                },
                "relevance_score": 0.89,
                "claim_text": "The Eiffel Tower was built in 1889."
            }
        }