from pydantic import BaseModel, Field


class RagSummarizeRequest(BaseModel):
    query: str = Field(min_length=1, description="检索/总结问题")


class RagSummarizeResponse(BaseModel):
    answer: str

