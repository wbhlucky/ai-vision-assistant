"""pytest conftest — 使 backend 包在测试中可导入。

项目目录名为 backend-refactor，但内部 import 使用 `from backend.xxx`。
此文件动态注册 backend 模块别名，使测试能正常运行。
"""

from __future__ import annotations

import importlib.util
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 如果当前目录不在 sys.path 中，添加它
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 动态注册 backend → 当前目录的模块别名
if "backend" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "backend",
        os.path.join(_project_root, "__init__.py"),
        submodule_search_locations=[_project_root],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["backend"] = module
        spec.loader.exec_module(module)
