"""WebSocket 实时对话 + 场景记忆 体验脚本。

用法：python demo_websocket.py
"""

import asyncio
import base64
import json
import urllib.request

API = "http://localhost:8000"
WS = "ws://localhost:8000/v1/vision/dialogue"
HEADERS = {"X-API-Key": "my-secret-key"}

# ── 准备测试图 ────────────────────────────────────────

def load_test_image():
    """从 test.jpg 读图转 base64，没有就现画一张。"""
    import io, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    # 没图就画一张
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 80, 200, 200], fill=(50, 50, 50))
    draw.ellipse([210, 100, 250, 180], fill=(200, 200, 255))
    draw.text((70, 205), "Laptop", fill=(255, 255, 255))
    draw.text((215, 190), "Cup", fill=(0, 0, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


async def main():
    frame_b64 = load_test_image()
    session_id = None

    import websockets
    print("=" * 60)
    print("  连接 WebSocket...")
    print("=" * 60)

    async with websockets.connect(WS, additional_headers=HEADERS) as ws:

        # ── 步骤 1: 发第一帧 ──────────────────────────
        print("\n[步骤 1] 发送第一帧图像...")
        await ws.send(json.dumps({"type": "frame", "data": frame_b64}))

        # 收回复，直到收到 vision_description
        while True:
            msg = json.loads(await ws.recv())
            t = msg["type"]
            content = msg["content"]
            print(f"  ← [{t}] {content[:120]}")
            if t == "vision_description":
                break

        # 继续收 scene_event / proactive_suggestion
        for _ in range(3):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3))
                print(f"  ← [{msg['type']}] {msg['content'][:120]}")
            except asyncio.TimeoutError:
                break

        # ── 步骤 2: 发第二帧（模拟场景变化——换一张不同的图）──
        print("\n[步骤 2] 发送第二帧（场景变化）...")
        from PIL import Image, ImageDraw
        import io as _io
        img2 = Image.new("RGB", (320, 240), color=(200, 200, 200))
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([50, 80, 200, 200], fill=(50, 50, 50))  # 笔记本还在
        # 杯子没了，换成手机
        draw2.rectangle([220, 100, 260, 160], fill=(0, 0, 0))
        draw2.text((70, 205), "Laptop", fill=(255, 255, 255))
        draw2.text((225, 165), "Phone", fill=(255, 255, 255))
        buf2 = _io.BytesIO()
        img2.save(buf2, "JPEG", quality=70)
        frame2 = base64.b64encode(buf2.getvalue()).decode()

        await ws.send(json.dumps({"type": "frame", "data": frame2}))

        # 收回复，应该看到 scene_event (移除: Cup, 新增: Phone)
        for _ in range(5):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                print(f"  ← [{msg['type']}] {msg['content'][:120]}")
                if msg['type'] == 'dashboard':
                    continue  # 跳过仪表板
            except asyncio.TimeoutError:
                break

        # ── 步骤 3: 语音模拟 → Agent 回答 ──────────────
        print("\n[步骤 3] 模拟语音转录 → Agent 推理...")
        # 构造一个假的 PCM 音频 chunk（模拟"刚才拿走了什么"）
        # 实际场景下前端会传真实音频
        import struct, math
        sample_rate = 16000
        duration = 0.5  # 半秒
        samples = []
        for i in range(int(sample_rate * duration)):
            samples.append(int(15000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        audio_b64 = base64.b64encode(pcm).decode()

        await ws.send(json.dumps({
            "type": "audio",
            "data": audio_b64,
            "metadata": {"sample_rate": sample_rate},
        }))

        # 收回复
        for _ in range(5):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                print(f"  ← [{msg['type']}] {msg['content'][:150]}")
                if msg['type'] == 'agent_answer':
                    break
            except asyncio.TimeoutError:
                break

        # ── 步骤 4: OCR ────────────────────────────────
        print("\n[步骤 4] OCR 文字提取...")
        await ws.send(json.dumps({
            "type": "ocr",
            "data": frame_b64,
            "metadata": {"mode": "extract"},
        }))
        for _ in range(3):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                print(f"  ← [{msg['type']}] {msg['content'][:120]}")
            except asyncio.TimeoutError:
                break

        # ── 步骤 5: 结束会话 ──────────────────────────
        print("\n[步骤 5] 结束会话...")
        await ws.send(json.dumps({"type": "end"}))
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            print(f"  ← [{msg['type']}]")
            data = json.loads(msg['content']) if isinstance(msg['content'], str) else msg['content']
            print(f"     视觉调用: {data.get('vision_calls', '?')} 次")
            print(f"     语音调用: {data.get('stt_calls', '?')} 次")
            print(f"     场景快照: {data.get('scene_snapshots', '?')} 帧")
            print(f"     场景事件: {data.get('scene_events', '?')} 条")
        except asyncio.TimeoutError:
            pass

    # ── 步骤 6: 场景时间线 ──────────────────────────────
    print("\n" + "=" * 60)
    print("  场景时间线 (HTTP)")
    print("=" * 60)

    # 获取最近的 session（用 health 的 request_id 推测）
    # 实际生产环境应从 WebSocket 协议中获取 session_id
    def http_get(path):
        req = urllib.request.Request(f"{API}{path}", headers=HEADERS)
        return json.loads(urllib.request.urlopen(req, timeout=10).read())

    print("\n  试试直接查视觉分析历史:")
    print("  (session 存储在服务端内存中，session_id = WebSocket 连接时的 request_id)")
    print("  在 Swagger 中手动传入 session_id 即可体验")
    print("  或运行: curl http://localhost:8000/v1/vision/session/{id}/timeline")

    print("\n" + "=" * 60)
    print("  体验完成！")
    print("=" * 60)
    print()
    print("  你刚才看到了:")
    print("  1. vision_description   — AI 描述画面内容")
    print("  2. scene_event          — 物体变化（新增/移除）")
    print("  3. agent_answer         — Agent 综合推理回答")
    print("  4. ocr_text             — OCR 文字提取")
    print("  5. cost_summary         — 会话成本统计")
    print()
    print("  接下来去 http://localhost:8000/docs 手动体验:")

    # 试试直接拿一个 session_id
    try:
        health = http_get("/v1/vision/health")
        print(f"  - 健康检查: OK (model={health['vision_model']})")
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
