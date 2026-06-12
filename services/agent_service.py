from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from agentic_rag.graph.graph import app as rag_app


@lru_cache(maxsize=1)
def get_agent() -> Any:
    """Unified entry point — currently backed by the agentic_rag LangGraph workflow."""
    return rag_app


def _build_graph_state(prompt: str) -> dict[str, Any]:
    return {
        "question": prompt,
        "generation": "",
        "web_search": False,
        "documents": [],
        "task_type": "chat",
        "user_context": {},
        "ragas_context": [],
    }


def _extract_answer(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("generation", "answer", "final_answer", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(result, str):
        return result.strip()
    return str(result).strip()


# ── public API ────────────────────────────────────────────────────────────────


def answer(prompt: str) -> str:
    """Run the agent graph and return the final answer string."""
    agent = get_agent()
    state = _build_graph_state(prompt)
    result = agent.invoke(state)
    return _extract_answer(result)


def stream_answer(prompt: str) -> Iterable[str]:
    """Streaming wrapper — currently event-level; ready for token-level upgrade."""
    yield "[event] start"
    yield answer(prompt)
    yield "[event] done"
