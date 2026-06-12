from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    deep_thought: bool = Field(default=False, description="是否开启深度思考（目前只透传，便于扩展）")
    user_location: str | None = Field(default=None, description="可选：用户位置（如 深圳(440300)）")


class ChatResponse(BaseModel):
    answer: str

