from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.core.settings import settings
from backend.services.multimodal_memory import MultimodalMemory
from backend.services.scene_memory import SceneMemory


@dataclass
class DialogueSession:
    """单个对话会话的状态，包含视觉描述历史、场景记忆和对话转录历史。

    职责：
      - 缓存最近的视觉描述和用户语音转录
      - 维护场景记忆引擎（快照历史 + 物体变更检测 + 事件日志）
      - 维护多模态统一记忆（视觉+OCR+语音跨模态检索）
      - 构建多模态 prompt 供 Agent 推理
      - 自动过期清理（基于 max_session_duration_sec）
    """

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    frame_descriptions: list[str] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    scene_memory: SceneMemory = field(default_factory=SceneMemory)
    multimodal_memory: MultimodalMemory = field(default_factory=MultimodalMemory)
    _max_history: int = 10

    # ── 添加记录 ──────────────────────────────────────────────

    def add_frame_description(self, desc: str) -> None:
        """添加一次视觉描述（保留最近 N 条）。同时同步到场景记忆。"""
        self.frame_descriptions.append(desc)
        if len(self.frame_descriptions) > self._max_history:
            self.frame_descriptions = self.frame_descriptions[-self._max_history:]

    def add_transcript(self, text: str) -> None:
        """添加一句用户语音转录（保留最近 N 条）。"""
        text = text.strip()
        if text:
            self.transcripts.append(text)
            if len(self.transcripts) > self._max_history:
                self.transcripts = self.transcripts[-self._max_history:]

    # ── 构建多模态 Prompt（含场景记忆）────────────────────────

    def build_prompt(self, latest_transcript: str = "") -> str:
        """构造传给 Agent 的多模态 prompt，融合视觉场景、场景记忆和对话历史。

        Args:
            latest_transcript: 最新一句用户转录（本轮触发 Agent 推理的输入）。

        Returns:
            结构化的中文 prompt 字符串。
        """
        parts: list[str] = []

        # 1) 视觉上下文
        if self.frame_descriptions:
            recent_frames = self.frame_descriptions[-3:]
            parts.append(
                "【当前视觉场景】\n" + "\n".join(f"- {d}" for d in recent_frames)
            )

        # 2) 场景记忆（★ 新增：物体变化 + 事件日志）
        memory_ctx = self.scene_memory.build_memory_context()
        if memory_ctx and memory_ctx != "（暂无场景记忆）":
            parts.append(memory_ctx)

        # 3) 场景记忆查询：如果用户问题涉及变化/历史，附加记忆检索结果
        if latest_transcript:
            memory_answer = self.scene_memory.query(latest_transcript)
            if memory_answer and "暂无" not in memory_answer:
                parts.append(f"【场景记忆检索】\n{memory_answer}")

        # ★ 4) 多模态跨模态检索：统一搜索视觉+OCR+语音历史
        if latest_transcript:
            cross_modal_answer = self.multimodal_memory.query_answer(latest_transcript)
            if cross_modal_answer and "没有找到" not in cross_modal_answer:
                parts.append(f"【跨模态记忆检索】\n{cross_modal_answer}")

        # 5) 对话历史
        if self.transcripts:
            recent_text = self.transcripts[-5:]
            parts.append(
                "【对话历史】\n" + "\n".join(f"- {t}" for t in recent_text)
            )

        # 6) 当前用户输入
        if latest_transcript:
            parts.append(f"用户刚刚说：{latest_transcript}")

        # 7) 指令（含跨模态引导）
        parts.append(
            "请根据以上视觉场景、场景记忆、跨模态检索结果和对话内容，"
            "用中文给出自然、有帮助的回复。"
            "如果跨模态记忆中有相关信息，请优先引用。"
        )

        return "\n\n".join(parts)

    # ── 过期 ─────────────────────────────────────────────────

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > settings.max_session_duration_sec


# ── 全局会话存储（内存）─────────────────────────────────────

_sessions: dict[str, DialogueSession] = {}


def get_or_create_session(session_id: str) -> DialogueSession:
    """获取或创建对话会话。自动清理已过期的旧会话。"""
    existing = _sessions.get(session_id)
    if existing is None or existing.expired:
        _sessions[session_id] = DialogueSession(session_id=session_id)
    return _sessions[session_id]


def remove_session(session_id: str) -> None:
    """显式移除会话（可选清理入口）。"""
    _sessions.pop(session_id, None)
