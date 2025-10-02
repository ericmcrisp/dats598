from pydantic import BaseModel


class PromptRequest(BaseModel):
    text: str


class PromptResponse(BaseModel):
    sentence_type: str
    is_factual: bool
    subjectivity: float
    entities: list[str]
    main_verb: str
