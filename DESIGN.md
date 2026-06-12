# AI 视觉对话助手 — 设计文档

## 1. 用户故事

| # | 计划实现 | 最终实现 | 说明 |
|---|---------|---------|------|
| 1 | 用户打开应用，AI 能通过摄像头看到画面并描述 | ✅ 已实现 | 支持单帧分析（`POST /v1/vision/analyze`）和 WebSocket 实时帧推送 |
| 2 | 用户说话，AI 能转录并理解语义 | ✅ 已实现 | 支持音频转录（`POST /v1/vision/transcribe`）和 WebSocket 实时音频推送，含 VAD 静音检测 |
| 3 | AI 结合视觉场景和语音输入给出智能回复 | ✅ 已实现 | 通过 LangGraph Agent 融合视觉描述 + 对话历史生成回复 |
| 4 | 实时双向交互（WebSocket）| ✅ 已实现 | `WS /v1/vision/dialogue` 支持帧/音频推送和 AI 回答回流 |
| 5 | API 不可用时的优雅降级 | ✅ 已实现 | 视觉降级为启发式描述（分辨率+主色调），语音降级为占位文本 |
| 6 | 前端 TTS 语音播报 | ⚠️ 前端负责 | 后端返回文本，由浏览器 `SpeechSynthesis` API 朗读 |

## 2. 成本控制策略

| # | 计划方案 | 实际采用 | 说明 |
|---|---------|---------|------|
| 1 | 帧率门控（最小间隔）| ✅ `FrameRateGate` | 默认 2000ms/帧，可配置 `VISION_FRAME_INTERVAL_MS` |
| 2 | VAD 静音过滤 | ✅ `detect_speech()` | 基于 RMS 能量阈值（默认 300），过滤无效音频段 |
| 3 | 图像分辨率压缩 | ✅ `compress_frame()` | 默认上限 640×480，JPEG quality 60，可配置 |
| 4 | API 调用次数上限 | ✅ `ApiCallBudget` | 每会话限 50 次视觉 + 100 次语音调用 |
| 5 | 自适应分辨率 | ✅ `ResolutionController` | 预算使用 > 80% 降到 320×240，> 50% 降到 480×360 |
| 6 | 会话自动过期 | ✅ 5 分钟 TTL | 可配置 `MAX_SESSION_DURATION_SEC` |
| 7 | 模型选型控制成本 | ✅ Qwen-VL + Paraformer | 性价比高的 DashScope 模型，通过 `VISION_MODEL` / `SPEECH_MODEL` 可切换 |

### 成本估算示例（单次 5 分钟会话）

| 组件 | 调用次数 | 单价（参考）| 费用 |
|------|---------|------------|------|
| Qwen-VL Plus | ~150 帧 (2fps × 5min) 但门控后 ~15 帧 | ~0.008 元/次 | ~0.12 元 |
| Paraformer v2 | ~10 次 (VAD 过滤后) | ~0.002 元/次 | ~0.02 元 |
| **合计** | | | **~0.14 元 / 会话** |

## 3. 架构

```
前端 (React + WebRTC/MediaDevices)
    │
    ├── POST /v1/vision/analyze     (HTTP 单帧分析)
    ├── POST /v1/vision/transcribe  (HTTP 单次转录)
    │
    └── WS  /v1/vision/dialogue     (WebSocket 实时对话)
         │
         ├── FrameRateGate  →  帧率门控
         ├── VAD            →  静音过滤
         ├── compress_frame →  图像压缩
         ├── analyze_frame  →  视觉理解 (DashScope / fallback)
         ├── transcribe     →  语音转录 (DashScope / fallback)
         ├── ApiCallBudget  →  调用预算管理
         │
         ├── DialogueSession → 会话上下文（视觉历史 + 对话历史）
         │
         └── agent_service.answer() → LangGraph Agent 推理
```

## 4. 模型选型

| 能力 | 模型 | 选型理由 |
|------|------|---------|
| 视觉理解 | DashScope Qwen-VL Plus | 已配置 `DASHSCOPE_API_KEY`；中文理解优秀；多模态原生支持 |
| 语音识别 | DashScope Paraformer v2 | 同账号体系；中文识别准确率高；流式支持 |
| VAD | stdlib RMS 能量检测 | 零依赖、零延迟、零成本；对演示场景足够 |
| Agent 推理 | agentic_rag LangGraph | 复用现有能力，无需额外集成 |
| TTS | 浏览器 SpeechSynthesis | 避免后端音频生成开销；前端原生支持 |
| 降级视觉 | Pillow 启发式 | 无需 API Key；作为 DashScope 不可用时的兜底 |
| 降级语音 | 占位文本 | 不阻塞交互流程；提示用户当前 STT 不可用 |

## 5. WebSocket 协议

### 客户端 → 服务端

```json
{"type": "frame", "data": "<base64-jpeg>"}
{"type": "audio", "data": "<base64-pcm16-wav>", "metadata": {"sample_rate": 16000}}
{"type": "ping"}
{"type": "end"}
```

### 服务端 → 客户端

```json
{"type": "vision_description", "content": "...", "timestamp_ms": 1718123456789, "metadata": {"model": "qwen-vl-plus", "latency_ms": 450}}
{"type": "transcript", "content": "...", "timestamp_ms": 1718123457000, "metadata": {"confidence": 0.95}}
{"type": "agent_answer", "content": "...", "timestamp_ms": 1718123458000, "metadata": {}}
{"type": "error", "content": "...", "timestamp_ms": 1718123459000, "metadata": {}}
{"type": "pong", "content": "alive", "timestamp_ms": 1718123450000, "metadata": {}}
{"type": "cost_summary", "content": "{\"vision_calls\":12,\"stt_calls\":25,...}", "timestamp_ms": 1718123460000, "metadata": {}}
```

## 6. 环境变量参考

```bash
# ── 视觉对话 ──────────────────────────────
VISION_MODEL=qwen-vl-plus          # 视觉模型
SPEECH_MODEL=paraformer-v2         # 语音模型
VISION_FRAME_INTERVAL_MS=2000      # 帧间隔 (ms)
VISION_MAX_WIDTH=640               # 最大宽度
VISION_MAX_HEIGHT=480              # 最大高度
VISION_JPEG_QUALITY=60             # JPEG 压缩质量
VAD_ENABLED=1                      # 启用 VAD
VAD_ENERGY_THRESHOLD=300           # VAD 能量阈值
MAX_SESSION_DURATION_SEC=300       # 会话最大时长 (s)
```
