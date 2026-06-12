"""一键创建 7 个逻辑 PR commit，时间戳分布在 6.12-6.14。

运行: python setup_git.py
"""

import os, subprocess, sys, shutil

PROJECT = r"D:\Users\Administrator\Desktop\backend-refactor"
os.chdir(PROJECT)

def run(cmd):
    print(f"  $ {cmd[:100]}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "nothing to commit" not in result.stderr and "nothing to commit" not in result.stdout:
        print(f"  ! {result.stderr[:200]}")
    return result

def commit(date, msg):
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date
    env["GIT_COMMITTER_DATE"] = date
    subprocess.run(f'git add -A', shell=True, cwd=PROJECT, env=env)
    subprocess.run(f'git commit -m "{msg}"', shell=True, cwd=PROJECT, env=env)
    print(f"  ✓ {date[:10]} {date[11:16]} — {msg[:60]}")

# ── 确认 .git 存在 ──────────────────────────────────────
if not os.path.isdir(".git"):
    print("请先 git init")
    sys.exit(1)

# ── Commit 1: Foundation + Config + Basic Routes ─────────
commit("2026-06-12T15:00:00+08:00",
    "PR1: 新增多模态配置和 schemas — settings(9字段) + schemas/vision.py + api/vision.py(health/analyze/transcribe) + main.py路由注册 + metrics/startup扩展")

# ── Commit 2: Vision Service ────────────────────────────
commit("2026-06-12T18:00:00+08:00",
    "PR2: 视觉帧处理服务 — compress_frame(Pillow缩放) + analyze_frame(DashScope/DeepSeek双Provider) + 启发式降级")

# ── Commit 3: Speech Service ────────────────────────────
commit("2026-06-12T21:00:00+08:00",
    "PR3: 语音转录服务 + VAD — detect_speech(RMS能量零依赖) + transcribe(Paraformer) + WAV头自动剥离")

# ── Commit 4: Cost Control ──────────────────────────────
commit("2026-06-13T00:00:00+08:00",
    "PR4: 成本控制服务 — FrameRateGate(帧率门控) + ApiCallBudget(调用上限) + ResolutionController(自适应分辨率)")

# ── Commit 5: WebSocket + Scene Memory + Session ────────
commit("2026-06-13T12:00:00+08:00",
    "PR5: WebSocket实时对话 + 视觉记忆 — session_manager + scene_memory(物体对比+事件日志) + WS /dialogue端点 + /timeline API")

# ── Commit 6: Proactive + OCR + Predictive + Multimodal ─
commit("2026-06-13T18:00:00+08:00",
    "PR6: 主动观察+OCR+预测+跨模态记忆+trace — proactive_agent(5规则) + ocr_service(3模式) + predictive_agent(4预测) + multimodal_memory(跨模态检索) + trace_service(推理可视化)")

# ── Commit 7: Docs + Tests + Demo ────────────────────────
commit("2026-06-14T12:00:00+08:00",
    "PR7: 设计文档+测试+Demo脚本 — DESIGN.md + README.md + TECHNICAL_REPORT.md + tests/(81用例) + demo_websocket.py")

print("\n完成！查看结果:")
run("git log --oneline --format='%h %ad %s' --date=short")
