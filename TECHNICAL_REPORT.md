# AI 视觉对话助手 — 技术报告

## 项目定位

不仅能看到，还能**记住、理解、预测**的 AI Agent 视觉助手。

```
Camera + Mic
      │
      ▼
┌─ Perception ───────────────────────────────┐
│  Vision (Qwen-VL)    OCR    Speech (Paraformer) │
└─────────────────────────────────────────────┘
      │
      ▼
┌─ Memory ───────────────────────────────────┐
│  Scene Memory (物体变化)  Multimodal (跨模态)    │
└─────────────────────────────────────────────┘
      │
      ▼
┌─ Intelligence ─────────────────────────────┐
│  Proactive Agent   Predictive Agent             │
└─────────────────────────────────────────────┘
      │
      ▼
┌─ Reasoning ────────────────────────────────┐
│  LangGraph → MCP Tools → RAG → Response        │
└─────────────────────────────────────────────┘
```

## 四大核心卖点

### 1. 视觉记忆 (Scene Memory)

普通方案只能回答"当前画面有什么"，无法回答"刚才发生了什么变化"。

```
用户："我桌子上有什么？"    → AI："有笔记本、水杯、手机"
用户："我刚才拿走了什么？"   → AI："你刚才拿走了水杯"
用户："最近5分钟发生了什么？" → AI："水杯被拿走，新增一本书"
```

**实现**: 每帧解析物体列表 → 帧间 diff 对比 → 自动写入事件日志 → 注入 Agent prompt。

### 2. 主动观察 (Proactive Agent)

不是"用户问→AI答"的被动模式，而是"AI 发现→主动提示"。

```
[用户举起一本书]
AI："我看到你拿起了一本书，需要我帮你提取书名或总结目录吗？"

[屏幕上出现代码报错]
AI："检测到错误信息，需要我帮你分析原因吗？"

[画面中出现文字]
AI："画面中有文字内容，需要我帮你识别或翻译吗？"
```

**实现**: 5 条可配置触发器规则 + 15s 冷却机制 + WebSocket `proactive_suggestion` 推送。

### 3. 跨模态长期记忆 (Multimodal Memory)

统一存储视觉描述、OCR 文本、语音转录，支持跨模态时间回溯检索。

```
用户："半小时前桌上有什么？"
→ 自动解析时间窗口 (1800s) + 模态偏好 (vision)
→ 返回: "30分钟内找到 3 条记录: 👁 [10:20] 桌上有水杯、笔记本、手机..."

用户："刚才白板上写了什么公式？"
→ 自动解析时间窗口 + 模态偏好 (ocr)
→ 返回 OCR 提取的公式文本
```

**实现**: 统一 `MemoryEntry` 存储 → 智能分词 + 关键词打分 → 时间窗口 + 模态偏好自动推断。

### 4. Agent 工具链

多模态输入 → LangGraph 推理 → MCP 工具调用 → RAG 增强 → 结构化回答。

```
Vision 描述
    │
OCR 文本 ──→ LangGraph Agent ──→ 需要天气? → MCP 天气工具
    │                │
Speech 转录          │
    │                └──→ 需要总结? → RAG 检索
    ▼
构建多模态 Prompt (含场景记忆 + 跨模态检索)
    │
    ▼
Agent 回答
```

**区别于普通方案**: 不是单纯 VLM 问答，而是 Agent 自主判断是否需要调用工具。

---

## 技术选型

| 能力 | 主方案 | 降级 (无 API 时) |
|------|--------|-----------------|
| 视觉理解 | DashScope Qwen-VL Plus | Pillow 启发式 (分辨率+主色调) |
| 语音识别 | DashScope Paraformer v2 | 占位文本 |
| VAD | stdlib `math` + `struct` (零依赖) | 始终当作有语音 |
| Agent | LangGraph + agentic_rag | 同 |
| OCR | Qwen-VL 多模态 (零额外引擎) | 同 |
| TTS | 浏览器 SpeechSynthesis | 前端负责 |

## 与现有系统集成

| 系统 | 集成方式 |
|------|---------|
| LangGraph Agent | 复用 `answer()`，通过结构化 prompt 注入视觉上下文 + 记忆 |
| MCP 工具 | 保留双模调度 (local/http)，Agent 可自主调用天气/时间/位置等 |
| RAG | 知识库检索增强 Agent 回答 |
| Auth | HTTP 端点使用 `Depends(require_api_key)`，WebSocket 手动校验 |
| Rate Limit | HTTP 使用 `slowapi`，WebSocket 使用内置 `ApiCallBudget` |
| Metrics | 12 个 Prometheus 指标，统一 `/metrics` 端点 |
| SSE Streaming | 保留 `/v1/chat/stream`，视觉对话走 WebSocket |

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/vision/health` | GET | 模型配置 + feature flags |
| `/v1/vision/analyze` | POST | 单帧视觉分析 |
| `/v1/vision/transcribe` | POST | 音频转录 |
| `/v1/vision/ocr` | POST | OCR 提取/总结/翻译 |
| `/v1/vision/dialogue` | **WS** | 实时视觉对话 |
| `/v1/vision/session/{id}/timeline` | GET | 场景变化时间线 |
| `/v1/vision/session/{id}/memory` | GET | 跨模态记忆检索 |
| `/v1/vision/session/{id}/trace` | GET | Agent 推理链路 |

## 测试

```bash
python -m pytest tests/ -v    # 81 passed
```

覆盖: CostController, SceneMemory, ProactiveAgent, MultimodalMemory, PredictiveAgent — 纯数据逻辑层 100% 覆盖。
