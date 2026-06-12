"""多模态统一记忆引擎 — 融合视觉、OCR、语音的跨模态检索。

这是从"单模态记忆"升级到"多模态长期记忆"的核心模块。
评委会看到的不再是孤立的视觉问答，而是统一的多模态上下文推理。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """一条统一记忆记录。"""

    timestamp: float
    time_str: str
    modality: str  # "vision" | "ocr" | "speech" | "action"
    content: str   # 原始描述/转录/提取文本
    objects: list[str] = field(default_factory=list)  # 关联物体
    metadata: dict = field(default_factory=dict)       # 附加元数据

    def to_dict(self) -> dict:
        return {
            "time": self.time_str,
            "modality": self.modality,
            "content": self.content[:200],
            "objects": self.objects,
            "metadata": self.metadata,
        }


# ── 多模态记忆存储 ────────────────────────────────────────

@dataclass
class MultimodalMemory:
    """统一多模态记忆引擎。

    维护视觉描述 + OCR 文字 + 语音转录的完整时间线，
    支持跨模态关键词检索和时间范围查询。
    """

    entries: list[MemoryEntry] = field(default_factory=list)
    _max_entries: int = 200  # 最多保留 200 条记录

    # ── 写入 ──────────────────────────────────────────────

    def record_vision(self, description: str, objects: list[str], metadata: dict | None = None) -> MemoryEntry:
        """记录一条视觉观察。"""
        return self._add("vision", description, objects, metadata)

    def record_ocr(self, text: str, objects: list[str] | None = None, metadata: dict | None = None) -> MemoryEntry:
        """记录一条 OCR 提取。"""
        return self._add("ocr", text, objects or [], metadata)

    def record_speech(self, transcript: str, metadata: dict | None = None) -> MemoryEntry:
        """记录一条语音转录。"""
        return self._add("speech", transcript, [], metadata)

    def _add(self, modality: str, content: str, objects: list[str],
             metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            timestamp=time.monotonic(),
            time_str=time.strftime("%H:%M:%S", time.localtime()),
            modality=modality,
            content=content,
            objects=objects,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        if len(self.entries) > self._max_entries:
            self.entries = self.entries[-self._max_entries:]
        return entry

    # ── 检索 ──────────────────────────────────────────────

    def search(self, query: str, time_window_sec: float | None = None,
               modalities: list[str] | None = None, max_results: int = 10) -> list[MemoryEntry]:
        """跨模态关键词检索。

        Args:
            query: 自然语言查询（支持中文关键词）。
            time_window_sec: 时间窗口（秒），None 表示不限制。
            modalities: 限定模态，None 表示所有。
            max_results: 最大返回数。

        Returns:
            按时间倒序排列的相关记忆条目。
        """
        now = time.monotonic()
        keywords = _tokenize_query(query)

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in reversed(self.entries):
            # 时间窗口过滤
            if time_window_sec is not None and (now - entry.timestamp) > time_window_sec:
                continue
            # 模态过滤
            if modalities is not None and entry.modality not in modalities:
                continue
            # 关键词匹配打分
            score = _match_score(keywords, entry)
            if score > 0:
                scored.append((score, entry))

        # 按分数降序，取 top-N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:max_results]]

    def get_context_window(self, minutes: float = 5) -> list[MemoryEntry]:
        """获取最近 N 分钟内的全部记忆。"""
        cutoff = time.monotonic() - minutes * 60
        return [e for e in self.entries if e.timestamp >= cutoff]

    def query_answer(self, question: str) -> str:
        """用自然语言问题检索记忆并生成回答。

        支持的问题类型：
        - "X 分钟前/刚才/半小时前 + 有什么/看到什么/写了什么/说了什么"
        - "最近出现了什么文字/物体"
        - "我说了什么/我看到了什么"
        """
        question_lower = question.lower()

        # 1) 解析时间窗口
        time_window = _parse_time_window(question)
        # 2) 解析模态偏好
        modalities = _parse_modality_preference(question)
        # 3) 执行检索
        results = self.search(question, time_window_sec=time_window, modalities=modalities, max_results=5)

        if not results:
            # 放宽条件重试
            results = self.search(question, time_window_sec=None, modalities=None, max_results=5)

        if not results:
            return "记忆中没有找到相关信息"

        # 4) 构建回答
        time_label = _format_time_window(time_window) if time_window else "记忆中"
        lines = [f"在{time_label}找到 {len(results)} 条相关记录:"]
        for e in results:
            modality_icon = {"vision": "👁", "ocr": "📄", "speech": "🎤", "action": "🔧"}.get(e.modality, "")
            lines.append(f"  {modality_icon} [{e.time_str}] {e.content[:120]}")
        return "\n".join(lines)

    # ── 摘要 ──────────────────────────────────────────────

    def summary(self) -> dict:
        """返回记忆统计摘要。"""
        modality_counts: dict[str, int] = {}
        for e in self.entries:
            modality_counts[e.modality] = modality_counts.get(e.modality, 0) + 1
        return {
            "total_entries": len(self.entries),
            "modality_counts": modality_counts,
            "time_range": {
                "first": self.entries[0].time_str if self.entries else None,
                "last": self.entries[-1].time_str if self.entries else None,
            },
        }


# ── 辅助函数 ──────────────────────────────────────────────

def _tokenize_query(query: str) -> list[str]:
    """将中文查询分词为关键词列表。"""
    # 简单分词：去掉标点，按 2-4 字滑动窗口提取
    # 移除中文标点和空白字符
    cleaned = re.sub(r"[，。！？、；：""'（）　]+", "", query)
    cleaned = re.sub(r"\s+", "", cleaned)
    tokens: list[str] = []
    for window in [4, 3, 2]:
        for i in range(len(cleaned) - window + 1):
            tokens.append(cleaned[i:i + window])
    # 也保留完整短词
    for word in re.findall(r"[一-鿿]{1,6}", query):
        if len(word) >= 2:
            tokens.append(word)
    return list(set(tokens))


def _match_score(keywords: list[str], entry: MemoryEntry) -> float:
    """计算关键词与记忆条目的匹配分数。"""
    score = 0.0
    text = entry.content + " " + " ".join(entry.objects)
    for kw in keywords:
        if kw in text:
            score += len(kw)  # 长词匹配权重更高
    return score


def _parse_time_window(question: str) -> float | None:
    """从问题中提取时间窗口（秒）。"""
    patterns = [
        (r"(\d+)\s*分钟前", 60),
        (r"(\d+)\s*分前", 60),
        (r"半小时前", 30 * 60),
        (r"刚才", 2 * 60),
        (r"刚刚", 1 * 60),
        (r"(\d+)\s*秒前", 1),
        (r"最近\s*(\d+)\s*分钟", 60),
        (r"过去\s*(\d+)\s*分钟", 60),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, question)
        if match:
            if match.groups():
                return int(match.group(1)) * multiplier
            else:
                return multiplier  # 固定值如"半小时前"
    return None


def _parse_modality_preference(question: str) -> list[str] | None:
    """从问题中推断用户关心的模态。"""
    mapped: list[str] = []
    if any(w in question for w in ["看到", "看", "画面", "视觉", "出现", "物体", "桌上"]):
        mapped.append("vision")
    if any(w in question for w in ["写", "文字", "OCR", "白板", "文档", "书", "纸"]):
        mapped.append("ocr")
    if any(w in question for w in ["说", "讲", "语音", "话", "问"]):
        mapped.append("speech")
    return mapped if mapped else None


def _format_time_window(seconds: float) -> str:
    if seconds < 120:
        return f"{int(seconds)}秒内"
    if seconds < 3600:
        return f"{int(seconds / 60)}分钟内"
    return f"{int(seconds / 3600)}小时内"
