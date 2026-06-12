from __future__ import annotations

from typing import Iterable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from backend.core.auth import require_api_key
from backend.core.rate_limit import limiter
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.agent_service import answer, stream_answer


router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _build_prompt(req: ChatRequest) -> str:
    # 与现有 Streamlit 的 enriched_prompt 逻辑对齐，方便复用同一套智能体行为
    mode_str = "深度思考" if req.deep_thought else "标准模式"
    loc_str = req.user_location or "未知/未提供"
    return (
        f"【系统环境信息】\n"
        f"- 当前模式：{mode_str}\n"
        f"- 用户定位：{loc_str}\n"
        f"请结合以上环境信息回答用户的问题。\n\n"
        f"用户问题：{req.question}"
    )


def _to_sse(stream: Iterable[str]):
    # 先实现事件流式输出，后续可平滑升级为 token 级别流式
    yield "event: start\ndata: [START]\n\n"
    for chunk in stream:
        if chunk.startswith("[event]"):
            continue
        yield f"data: {chunk.rstrip()}\n\n"
    yield "event: done\ndata: [DONE]\n\n"


@router.post("", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest):
    prompt = _build_prompt(req)
    return ChatResponse(answer=answer(prompt))


@router.post("/stream", dependencies=[Depends(require_api_key)])
@limiter.limit("10/minute")
def chat_stream(request: Request, req: ChatRequest):
    prompt = _build_prompt(req)
    stream = stream_answer(prompt)
    return StreamingResponse(_to_sse(stream), media_type="text/event-stream")

