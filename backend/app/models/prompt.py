from pydantic import BaseModel


class PromptRequest(BaseModel):
    text: str


# placeholder response
class PromptResponse(BaseModel):
    text: str

# class PromptResponse(BaseModel):
#     sentence_type: str
#     is_factual: bool
#     subjectivity: float
#     entities: list[str]
#     main_verb: str
