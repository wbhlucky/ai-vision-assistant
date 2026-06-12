from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect

from backend.core.auth import require_api_key
from backend.core.errors import AppError, unauthorized
from backend.core.logging import get_request_id
from backend.core.metrics import (
    speech_chunks_processed,
    vad_silence_skipped,
    vision_api_errors,
    vision_frames_processed,
    websocket_connections,
)
from backend.core.rate_limit import limiter
from backend.core.settings import settings
from backend.schemas.vision import (
    AudioTranscribeRequest,
    AudioTranscribeResponse,
    DialogueMessageIn,
    DialogueMessageOut,
    OcrRequest,
    OcrResponse,
    SceneMemoryQuery,
    SceneMemoryQueryResponse,
    VisionFrameRequest,
    VisionFrameResponse,
)
from backend.services.agent_service import answer
from backend.services.cost_controller import ApiCallBudget, FrameRateGate
from backend.services.ocr_service import extract_text, ocr_and_summarize, ocr_and_translate
from backend.services.predictive_agent import PredictiveAgent
from backend.services.proactive_agent import ProactiveAgent, generate_event_description
from backend.services.scene_memory import SceneMemory
from backend.services.session_manager import get_or_create_session
from backend.services.speech_service import detect_speech, transcribe
from backend.services.trace_service import TraceLog, get_trace, new_trace
from backend.services.vision_service import analyze_frame

router = APIRouter(prefix="/v1/vision", tags=["vision"])


# ── 健康检查 ──────────────────────────────────────────────

@router.get("/health")
def vision_health():
    """返回视觉对话模块的服务状态。"""
    return {
        "ok": True,
        "vision_model": settings.vision_model,
        "speech_model": settings.speech_model,
        "vad_enabled": settings.vad_enabled,
        "frame_interval_ms": settings.vision_frame_interval_ms,
        "features": {
            "scene_memory": True,
            "proactive_agent": True,
            "ocr": True,
        },
    }


# ── HTTP 单帧分析 ─────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=VisionFrameResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
def vision_analyze(request: Request, req: VisionFrameRequest):
    """对单帧图像进行视觉分析。"""
    result = analyze_frame(req.image_base64, req.prompt)
    vision_frames_processed.labels(result["model"]).inc()
    return VisionFrameResponse(**result)


# ── HTTP 语音转录 ─────────────────────────────────────────

@router.post(
    "/transcribe",
    response_model=AudioTranscribeResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("20/minute")
def vision_transcribe(request: Request, req: AudioTranscribeRequest):
    """对音频片段进行语音转录。"""
    result = transcribe(req.audio_base64, req.sample_rate)
    speech_chunks_processed.inc()
    return AudioTranscribeResponse(**result)


# ── HTTP OCR ──────────────────────────────────────────────

@router.post(
    "/ocr",
    response_model=OcrResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("10/minute")
def vision_ocr(request: Request, req: OcrRequest):
    """OCR 文字提取，支持 extract / summarize / translate 三种模式。"""
    if req.mode == "summarize":
        result = ocr_and_summarize(req.image_base64)
        return OcrResponse(
            text=result["ocr_text"],
            summary=result["summary"],
            model=settings.vision_model,
            latency_ms=result["latency_ms"],
        )
    elif req.mode == "translate":
        result = ocr_and_translate(req.image_base64, target_lang=req.target_lang)
        return OcrResponse(
            text=result["ocr_text"],
            translated=result["translated"],
            model=settings.vision_model,
            latency_ms=result["latency_ms"],
        )
    else:
        result = extract_text(req.image_base64)
        return OcrResponse(
            text=result["text"],
            model=result["model"],
            latency_ms=result["latency_ms"],
        )


# ── 场景时间线 API ──────────────────────────────────────

@router.get("/session/{session_id}/timeline")
def scene_timeline(session_id: str):
    """返回指定会话的场景变化时间线，供前端可视化展示。

    Demo 视频中可展示场景变化时序图/柱状图。
    """
    session = get_or_create_session(session_id)
    mem = session.scene_memory

    events = []
    for i, snap in enumerate(mem.snapshots):
        entry: dict = {
            "index": i,
            "time": snap.time_str,
            "description": snap.description[:120],
            "objects": snap.objects,
            "object_count": len(snap.objects),
        }
        if i > 0:
            diff = mem._diff(mem.snapshots[i - 1].objects, snap.objects)
            entry["added"] = diff["added"]
            entry["removed"] = diff["removed"]
            entry["change_count"] = len(diff["added"]) + len(diff["removed"])
        else:
            entry["added"] = []
            entry["removed"] = []
            entry["change_count"] = 0
        events.append(entry)

    return {
        "session_id": session_id,
        "total_snapshots": len(mem.snapshots),
        "total_events": len(mem.event_log),
        "event_log": mem.event_log[-20:],
        "timeline": events,
    }


# ── 多模态记忆检索 API ────────────────────────────────

@router.get("/session/{session_id}/memory")
def multimodal_memory_query(session_id: str, q: str = "最近发生了什么？", minutes: float = 5):
    """跨模态检索：统一搜索视觉+OCR+语音历史。

    Demo 视频中可展示"半小时前桌上有什么"这类跨模态回溯能力。
    """
    session = get_or_create_session(session_id)
    mem = session.multimodal_memory

    context = mem.get_context_window(minutes=minutes)
    answer = mem.query_answer(q)

    return {
        "session_id": session_id,
        "question": q,
        "answer": answer,
        "context_entries": [e.to_dict() for e in context[-10:]],
        "summary": mem.summary(),
    }


# ── 推理链路 Trace API ─────────────────────────────────

@router.get("/session/{session_id}/trace")
def reasoning_trace(session_id: str):
    """返回 Agent 推理链路 trace，展示可视化决策过程。

    包含四层的完整链路：Perception → Memory → Reasoning → Action。
    评委可通过此端点直观理解系统如何从视觉输入到最终回答。
    """
    trace = get_trace(session_id)
    if trace is None:
        return {"session_id": session_id, "error": "No trace found for this session"}

    return {
        "trace": trace.to_dict(),
        "ascii_diagram": trace.ascii_diagram(),
    }


# ── 仪表板统计 ──────────────────────────────────────────

def _build_dashboard(budget: ApiCallBudget, frame_gate: FrameRateGate,
                     session: "DialogueSession") -> dict:
    """构造端云协同实时仪表板数据。"""
    mem = session.scene_memory
    return {
        "budget": budget.summary(),
        "frame_gate": {
            "interval_ms": frame_gate.interval_ms,
            "seconds_since_last": round(frame_gate.seconds_since_last, 1),
        },
        "scene": {
            "snapshots": len(mem.snapshots),
            "events": len(mem.event_log),
            "last_event": mem.event_log[-1] if mem.event_log else None,
        },
        "cost_estimation": {
            "vision_cost_rmb": round(budget.vision_calls * 0.008, 3),
            "stt_cost_rmb": round(budget.stt_calls * 0.002, 3),
            "total_cost_rmb": round(budget.vision_calls * 0.008 + budget.stt_calls * 0.002, 3),
            "frames_skipped_estimated": max(0, (budget.vision_calls * settings.vision_frame_interval_ms // 1000) - budget.vision_calls),
        },
    }


# ── WebSocket 辅助 ──────────────────────────────────────────

async def _ws_send(websocket: WebSocket, msg_type: str, content: str, metadata: dict | None = None) -> None:
    """发送一条结构化消息到 WebSocket 客户端。"""
    msg = DialogueMessageOut(
        type=msg_type,
        content=content,
        timestamp_ms=int(time.time() * 1000),
        metadata=metadata or {},
    )
    await websocket.send_text(msg.model_dump_json())


def _ws_auth(websocket: WebSocket) -> None:
    """WebSocket 握手阶段的 API Key 校验。"""
    if not settings.api_key:
        return
    key = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-Key")
    if key != settings.api_key:
        raise unauthorized(code="UNAUTHORIZED", message="Invalid or missing X-API-Key")


# ── WebSocket 实时对话端点（含视觉记忆 + 主动观察 + OCR）───

@router.websocket("/dialogue")
async def vision_dialogue(websocket: WebSocket):
    """WebSocket 实时视觉对话端点。

    协议（客户端 → 服务端）：
        {"type": "frame", "data": "<base64-jpeg>"}
        {"type": "audio", "data": "<base64-pcm16>", "metadata": {"sample_rate": 16000}}
        {"type": "ocr",   "data": "<base64-jpeg>", "metadata": {"mode": "extract|summarize|translate"}}
        {"type": "ping"}
        {"type": "end"}

    协议（服务端 → 客户端）：
        {"type": "vision_description",    "content": "...", ...}
        {"type": "scene_event",           "content": "新增: 水杯；移除: 手机", ...}
        {"type": "proactive_suggestion",  "content": "我看到你拿起了一本书...", ...}
        {"type": "transcript",            "content": "...", ...}
        {"type": "ocr_text",              "content": "...", ...}
        {"type": "agent_answer",          "content": "...", ...}
        {"type": "error",                 "content": "...", ...}
        {"type": "pong",                  "content": "alive", ...}
        {"type": "cost_summary",          "content": "{...}", ...}
    """
    await websocket.accept()
    websocket_connections.labels("connected").inc()

    # 鉴权
    try:
        _ws_auth(websocket)
    except AppError as exc:
        websocket_connections.labels("auth_failed").inc()
        await _ws_send(websocket, "error", f"Auth failed: {exc.detail}")
        await websocket.close(code=4001)
        return

    # 初始化会话、成本控制和主动观察引擎
    session_id = get_request_id() or "unknown"
    session = get_or_create_session(session_id)
    frame_gate = FrameRateGate()
    budget = ApiCallBudget()
    proactive = ProactiveAgent()
    predictive = PredictiveAgent()
    trace = new_trace(session_id)
    _last_dashboard_time = 0.0  # 仪表板推送节流

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = DialogueMessageIn.model_validate_json(raw)
            except Exception:
                await _ws_send(websocket, "error", "Invalid message format")
                continue

            # ── ping ──────────────────────────────────────────
            if msg.type == "ping":
                await _ws_send(websocket, "pong", "alive")

            # ── end ───────────────────────────────────────────
            elif msg.type == "end":
                summary = json.dumps(budget.summary(), ensure_ascii=False)
                # 附加场景记忆统计
                mem = session.scene_memory
                summary_data = budget.summary()
                summary_data["scene_snapshots"] = len(mem.snapshots)
                summary_data["scene_events"] = len(mem.event_log)
                summary = json.dumps(summary_data, ensure_ascii=False)
                await _ws_send(websocket, "cost_summary", summary)
                websocket_connections.labels("disconnected").inc()
                break

            # ── frame ─────────────────────────────────────────
            elif msg.type == "frame":
                if not frame_gate.should_capture():
                    continue
                if not budget.allow_vision():
                    await _ws_send(websocket, "error", "Vision API budget exhausted")
                    continue
                frame_gate.mark_captured()

                # 视觉分析
                t_frame_start = time.monotonic()
                result = analyze_frame(msg.data)
                desc = result["description"]
                vision_frames_processed.labels(result["model"]).inc()
                t_frame_ms = (time.monotonic() - t_frame_start) * 1000

                # ★ 场景记忆：添加快照，检测物体变化
                snapshot = session.scene_memory.add_snapshot(desc)
                session.add_frame_description(desc)

                # ★ 多模态记忆：统一记录
                mem_entry = session.multimodal_memory.record_vision(
                    desc, snapshot.objects,
                    metadata={"model": result["model"], "latency_ms": round(t_frame_ms, 2)},
                )

                # ★ Trace: Perception 层
                trace.add("perception", "视觉帧分析",
                          f"base64 图像 ({len(msg.data)} chars)",
                          f"{desc[:100]} ({snapshot.objects})",
                          t_frame_ms)

                # 发送视觉描述
                await _ws_send(
                    websocket, "vision_description", desc,
                    metadata={"model": result["model"], "latency_ms": result["latency_ms"]},
                )

                # ★ 场景变化事件
                if len(session.scene_memory.snapshots) >= 2:
                    diff = session.scene_memory.compare_with_previous()
                    if diff["added"] or diff["removed"]:
                        change_parts: list[str] = []
                        if diff["added"]:
                            change_parts.append(f"新增: {', '.join(diff['added'])}")
                        if diff["removed"]:
                            change_parts.append(f"移除: {', '.join(diff['removed'])}")
                        await _ws_send(
                            websocket, "scene_event",
                            "；".join(change_parts),
                            metadata={"added": diff["added"], "removed": diff["removed"]},
                        )

                # ★ 主动观察：检测触发条件，生成主动建议
                suggestion = proactive.observe(
                    desc, snapshot.objects, session.scene_memory
                )
                if suggestion:
                    await _ws_send(
                        websocket, "proactive_suggestion", suggestion,
                        metadata={"trigger": "auto"},
                    )

                # ★ 预测提醒：基于行为模式的智能预测
                prediction = predictive.feed_and_predict(
                    "vision", desc, snapshot.objects, session.multimodal_memory
                )
                if prediction:
                    await _ws_send(
                        websocket, "predictive_suggestion", prediction,
                        metadata={"trigger": "predictive"},
                    )

                # ★ 端云协同仪表板（定时推送，避免每帧都发）
                now_ts = time.monotonic()
                if now_ts - _last_dashboard_time >= settings.dashboard_report_interval_sec:
                    _last_dashboard_time = now_ts
                    dashboard_data = _build_dashboard(budget, frame_gate, session)
                    await _ws_send(
                        websocket, "dashboard",
                        json.dumps(dashboard_data, ensure_ascii=False),
                    )

            # ── ocr ───────────────────────────────────────────
            elif msg.type == "ocr":
                t_ocr_start = time.monotonic()
                mode = (msg.metadata or {}).get("mode", "extract")
                if mode == "summarize":
                    ocr_result = ocr_and_summarize(msg.data)
                    await _ws_send(
                        websocket, "ocr_text",
                        ocr_result["ocr_text"],
                        metadata={"mode": "summarize", "summary": ocr_result["summary"]},
                    )
                elif mode == "translate":
                    target = (msg.metadata or {}).get("target_lang", "中文")
                    ocr_result = ocr_and_translate(msg.data, target_lang=target)
                    await _ws_send(
                        websocket, "ocr_text",
                        ocr_result["ocr_text"],
                        metadata={"mode": "translate", "translated": ocr_result["translated"]},
                    )
                else:
                    ocr_result = extract_text(msg.data)
                    await _ws_send(
                        websocket, "ocr_text", ocr_result["text"],
                        metadata={"mode": "extract"},
                    )

                # ★ 多模态记忆 + Trace: OCR 层
                t_ocr_ms = (time.monotonic() - t_ocr_start) * 1000
                ocr_text = ocr_result.get("text") or ocr_result.get("ocr_text", "")
                if ocr_text and ocr_text != "[无文字]" and not ocr_text.startswith("[OCR"):
                    session.multimodal_memory.record_ocr(ocr_text)
                    trace.add("perception", f"OCR ({mode})",
                              f"图像 ({len(msg.data)} chars)",
                              ocr_text[:100],
                              t_ocr_ms)
                    # 预测
                    prediction = predictive.feed_and_predict(
                        "ocr", ocr_text, [], session.multimodal_memory
                    )
                    if prediction:
                        await _ws_send(
                            websocket, "predictive_suggestion", prediction,
                            metadata={"trigger": "predictive"},
                        )

            # ── audio ─────────────────────────────────────────
            elif msg.type == "audio":
                sample_rate = (msg.metadata or {}).get("sample_rate", 16000)

                # VAD 过滤
                is_speech = detect_speech(msg.data)
                if not is_speech:
                    vad_silence_skipped.inc()
                    continue

                if not budget.allow_stt():
                    await _ws_send(websocket, "error", "STT API budget exhausted")
                    continue

                stt_result = transcribe(msg.data, sample_rate)
                transcript_text = stt_result.get("text", "")
                if transcript_text:
                    session.add_transcript(transcript_text)
                    speech_chunks_processed.inc()
                    await _ws_send(
                        websocket, "transcript", transcript_text,
                        metadata={"confidence": stt_result.get("confidence", 0.0)},
                    )

                    # ★ 多模态记忆: 记录语音
                    session.multimodal_memory.record_speech(transcript_text)

                    # ★ 预测: 基于语音内容的预测
                    prediction = predictive.feed_and_predict(
                        "speech", transcript_text, [], session.multimodal_memory
                    )
                    if prediction:
                        await _ws_send(
                            websocket, "predictive_suggestion", prediction,
                            metadata={"trigger": "predictive"},
                        )

                # ★ Agent 推理（含场景记忆上下文）
                if transcript_text:
                    prompt = session.build_prompt(transcript_text)

                    # ★ 如果用户问"发生了什么"，附加事件摘要
                    if any(w in transcript_text for w in ["发生", "变化", "刚才", "之前", "原来"]):
                        event_desc = generate_event_description(session.scene_memory)
                        if event_desc:
                            prompt += f"\n\n【补充事件信息】\n{event_desc}"

                    try:
                        # ★ Trace: Memory 层 (跨模态检索)
                        t_mem_start = time.monotonic()
                        mem_ctx = session.multimodal_memory.query_answer(transcript_text) if transcript_text else ""
                        t_mem_ms = (time.monotonic() - t_mem_start) * 1000
                        if mem_ctx and "没有找到" not in mem_ctx:
                            trace.add("memory", "跨模态记忆检索",
                                      transcript_text[:80],
                                      mem_ctx[:120],
                                      t_mem_ms)

                        # ★ Agent 推理 (Reasoning 层)
                        t_agent_start = time.monotonic()
                        agent_answer = answer(prompt)
                        t_agent_ms = (time.monotonic() - t_agent_start) * 1000
                        trace.add("reasoning", "LangGraph Agent 推理",
                                  f"prompt ({len(prompt)} chars)",
                                  agent_answer[:120],
                                  t_agent_ms)
                        await _ws_send(websocket, "agent_answer", agent_answer)
                    except Exception as exc:
                        await _ws_send(websocket, "error", f"Agent error: {exc}")

            else:
                await _ws_send(websocket, "error", f"Unknown message type: {msg.type}")

    except WebSocketDisconnect:
        websocket_connections.labels("disconnected").inc()
    except Exception as exc:
        websocket_connections.labels("error").inc()
        try:
            await _ws_send(websocket, "error", f"Internal error: {exc}")
        except Exception:
            pass
