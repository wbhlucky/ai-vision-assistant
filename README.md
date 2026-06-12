# Vision Memory AI Assistant

**不仅能看到，还能记住、理解、预测。**

> FastAPI 后端 | LangGraph Agent | 多模态记忆 | MCP 工具调用 | RAG 增强

---

## 三个 Demo 场景

### 场景 1：视觉记忆

```
👤 用户打开摄像头，桌面上有笔记本、水杯、手机
🤖 AI："当前画面中有笔记本、水杯、手机"

👤 用户拿走水杯
🤖 AI：【scene_event】移除: 水杯

👤 用户："我刚才拿走了什么？"
🤖 AI："你刚才拿走了水杯"

👤 用户："最近 5 分钟发生了什么？"
🤖 AI："检测到水杯被拿走"
```

### 场景 2：OCR + Agent 理解

```
👤 用户拿起一本书，对着摄像头
🤖 AI：【proactive】"我看到你拿起了一本书，需要提取书名或总结目录吗？"

👤 用户："帮我总结一下"
🤖 AI：OCR 提取文字 → Agent 总结 → "本书共 12 章，核心内容包括..."
```

### 场景 3：主动观察 + Agent 工具

```
👤 用户打开一个报错的代码页面
🤖 AI：【proactive】"检测到屏幕上出现错误信息，需要我分析原因吗？"

👤 用户："分析一下"
🤖 AI：视觉分析 → LangGraph → 判断需要搜索 → MCP 搜索工具 → 返回解决方案
```

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **视觉记忆** | 帧间物体对比，自动检测增删变化，回答"刚才拿走了什么" |
| **主动观察** | 5 条可配置触发规则，AI 主动发现并提示 (人物/书本/错误/文字/变化) |
| **跨模态记忆** | 统一视觉+OCR+语音历史，"半小时前桌上有水杯"可回溯 |
| **OCR 管道** | 纯视觉 OCR → 总结/翻译，零额外引擎 |
| **Agent 工具链** | Vision → LangGraph → MCP/RAG → 回答 |
| **端云协同** | 帧率门控 + API 预算 + 仪表板实时推送 |

---

## 架构

```
Camera + Mic
      │
      ▼
┌─ Perception ───────────────────────────┐
│  Vision(Qwen-VL)  OCR  Speech(Paraformer) │
└────────────────────────────────────────┘
      │
      ▼
┌─ Memory ───────────────────────────────┐
│  Scene Memory   Multimodal Memory         │
└────────────────────────────────────────┘
      │
      ▼
┌─ Intelligence ─────────────────────────┐
│  Proactive Agent   Predictive Agent       │
└────────────────────────────────────────┘
      │
      ▼
┌─ Reasoning ────────────────────────────┐
│  LangGraph → MCP Tools → RAG → Response  │
└────────────────────────────────────────┘
```

## 快速开始

```bash
# 1. 环境变量
export API_KEY=your_key
export DASHSCOPE_API_KEY=your_dashscope_key

# 2. 启动
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. 测试
python -m pytest tests/ -v    # 81 passed
```

## WebSocket 快速连接

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/vision/dialogue");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  // msg.type: "vision_description" | "scene_event" | "proactive_suggestion"
  //          | "transcript" | "ocr_text" | "agent_answer" | "dashboard"
};
ws.send(JSON.stringify({ type: "frame", data: base64Jpeg }));
ws.send(JSON.stringify({ type: "audio", data: base64Pcm16 }));
```

## 项目结构

```
backend-refactor/
├── main.py                  # FastAPI 应用入口
├── api/
│   ├── vision.py            # 视觉对话路由 (HTTP + WebSocket)
│   ├── chat.py / rag.py / knowledge.py / mcp.py / health.py
├── services/
│   ├── scene_memory.py      # 视觉记忆 ★
│   ├── proactive_agent.py   # 主动观察 ★
│   ├── multimodal_memory.py # 跨模态记忆 ★
│   ├── predictive_agent.py  # 预测提醒
│   ├── ocr_service.py       # OCR
│   ├── trace_service.py     # 推理 trace
│   ├── vision_service.py    # 视觉帧处理
│   ├── speech_service.py    # VAD + STT
│   ├── cost_controller.py   # 成本控制
│   ├── session_manager.py   # 会话管理
│   ├── agent_service.py     # LangGraph Agent
│   └── mcp_bridge.py        # MCP 双模调度
├── schemas/vision.py        # 多模态消息模型
├── core/                    # 基础设施 (auth/logging/metrics/...)
└── tests/                   # 81 个单元测试
```

## 文档

- [TECHNICAL_REPORT.md](./TECHNICAL_REPORT.md) — 四大核心卖点详细说明
- [DESIGN.md](./DESIGN.md) — 设计文档 (用户故事 + 成本控制策略)

## 依赖

Python 3.11+ | FastAPI | LangGraph | DashScope | Pillow | slowapi | Prometheus | pytest
