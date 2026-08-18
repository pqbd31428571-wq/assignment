from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    Request body for asking a question.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the documents."
    )


class SourceResponse(BaseModel):
    """
    Information about a retrieved source.
    """

    file_name: str
    chunk_id: str
    score: float


class AnswerResponse(BaseModel):
    """
    Response returned by the RAG API.
    """

    question: str
    answer: str
    sources: list[SourceResponse]