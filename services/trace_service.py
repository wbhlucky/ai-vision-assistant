"""推理链路追踪服务 — 可视化展示 Agent 决策过程。

将黑箱的 Agent 推理转化为可审计的决策链，
方便评委在 Demo 视频中直观理解系统如何工作。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TraceNode:
    """推理链中的一个节点。"""
    step: int
    layer: str          # "perception" | "memory" | "reasoning" | "action"
    action: str         # human-readable action name
    input_summary: str  # 输入摘要（截断）
    output_summary: str # 输出摘要（截断）
    latency_ms: float
    timestamp: str = field(default_factory=lambda: time.strftime("%H:%M:%S", time.localtime()))

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "layer": self.layer,
            "action": self.action,
            "input": self.input_summary,
            "output": self.output_summary,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class TraceLog:
    """一次推理的完整 trace 日志。"""

    session_id: str
    nodes: list[TraceNode] = field(default_factory=list)
    _step_counter: int = 0

    def add(self, layer: str, action: str, input_summary: str, output_summary: str,
            latency_ms: float) -> TraceNode:
        self._step_counter += 1
        node = TraceNode(
            step=self._step_counter,
            layer=layer,
            action=action,
            input_summary=input_summary[:200],
            output_summary=output_summary[:200],
            latency_ms=round(latency_ms, 2),
        )
        self.nodes.append(node)
        return node

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_steps": len(self.nodes),
            "total_latency_ms": round(sum(n.latency_ms for n in self.nodes), 2),
            "layers_visited": list(set(n.layer for n in self.nodes)),
            "trace": [n.to_dict() for n in self.nodes],
        }

    def ascii_diagram(self) -> str:
        """生成 ASCII 流程图，适合在日志/文档中展示。"""
        lines = [f"Trace for session {self.session_id[:8]}...", "─" * 60]
        for node in self.nodes:
            icon = {"perception": "👁", "memory": "🧠", "reasoning": "🤖", "action": "🔧"}.get(node.layer, "•")
            lines.append(f"  {icon} [{node.layer}] {node.action}")
            lines.append(f"     in:  {node.input_summary[:80]}")
            lines.append(f"     out: {node.output_summary[:80]}  ({node.latency_ms}ms)")
            lines.append("")
        lines.append(f"Total: {len(self.nodes)} steps, "
                      f"{round(sum(n.latency_ms for n in self.nodes))}ms")
        return "\n".join(lines)


# ── 全局 trace 存储 ───────────────────────────────────────

_traces: dict[str, TraceLog] = {}


def new_trace(session_id: str) -> TraceLog:
    trace = TraceLog(session_id=session_id)
    _traces[session_id] = trace
    # 限制内存：只保留最近 100 个
    if len(_traces) > 100:
        oldest = next(iter(_traces))
        del _traces[oldest]
    return trace


def get_trace(session_id: str) -> TraceLog | None:
    return _traces.get(session_id)
