from __future__ import annotations

import os

from backend.core.settings import settings


def apply_runtime_mode() -> dict:
    """
    将 RUN_MODE 映射为可执行环境变量策略。

    learn: 稳定与可学习优先，默认关闭 HF 精排
    prod : 效果优先，默认开启 HF 精排（可被显式环境变量覆盖）
    """
    mode = settings.run_mode

    if mode == "prod":
        os.environ.setdefault("ENABLE_HF_RERANK", "1")
    else:
        os.environ.setdefault("ENABLE_HF_RERANK", "0")

    return {
        "run_mode": mode,
        "ENABLE_HF_RERANK": os.getenv("ENABLE_HF_RERANK", "0"),
    }
