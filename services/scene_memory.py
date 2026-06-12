from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


@dataclass
class SceneSnapshot:
    """单帧场景快照，包含时间、描述和检测到的物体列表。"""

    timestamp: float
    time_str: str
    description: str
    objects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "time": self.time_str,
            "description": self.description,
            "objects": self.objects,
        }


# ── 物体名称提取（轻量解析，无需额外 API 调用）──────────────

# 常见中文量词/前缀，用于清洗 Qwen-VL 返回的物体描述
_CLEAN_PATTERNS = [
    re.compile(r"[，，。；！？、\s]+"),
    re.compile(r"(一个|一张|一把|一台|一部|一本|一支|一条|一件|一份|一款|一种)"),
    re.compile(r"[的之]?(左边|右边|上面|下面|前面|后面|中间|旁边|附近|远处|近处)"),
]


def _parse_objects_from_text(description: str) -> list[str]:
    """从自然语言描述中粗提取物体名词列表。

    启发式规则：
    1. 按中文标点断句
    2. 提取被量词修饰的名词
    3. 提取「有 X」「看到 X」「包含 X」等模式中的名词
    4. 去重、去空、截断（最多 15 个）
    """
    candidates: set[str] = set()

    # 按常见分隔符断句
    segments = re.split(r"[，。；！？\n,;!?]+", description)

    for seg in segments:
        seg = seg.strip()
        if len(seg) < 2:
            continue

        # 模式 1：量词 + 疑似名词（如"一个水杯"）
        quant_matches = re.findall(r"(?:一个|一张|一把|一台|一部|一本|一支|一条)[一-鿿\w]{1,8}", seg)
        for m in quant_matches:
            # 去掉量词前缀
            cleaned = re.sub(r"^(?:一个|一张|一把|一台|一部|一本|一支|一条)", "", m)
            if cleaned and len(cleaned) >= 2:
                candidates.add(cleaned)

        # 模式 2：「看到/发现/包含/有 + 名词」（简化的 NER 近似）
        entity_matches = re.findall(
            r"(?:看到|发现|包含|有|放着|摆着|放着|出现|存在)\s*[一-鿿\w]{2,10}",
            seg,
        )
        for m in entity_matches:
            cleaned = re.sub(r"^(?:看到|发现|包含|有|放着|摆着|出现|存在)\s*", "", m)
            if cleaned and len(cleaned) >= 2:
                candidates.add(cleaned)

    return list(candidates)[:15]


# ── 场景记忆引擎 ────────────────────────────────────────────

@dataclass
class SceneMemory:
    """视觉记忆系统：维护场景快照历史 + 事件日志 + 物体变更检测。

    这是让产品从"视觉问答"升级到"视觉记忆 Agent"的核心组件。
    """

    snapshots: list[SceneSnapshot] = field(default_factory=list)
    event_log: list[str] = field(default_factory=list)
    _max_snapshots: int = 30
    _max_events: int = 50

    # ── 添加快照 ────────────────────────────────────────────

    def add_snapshot(self, description: str, objects: list[str] | None = None) -> SceneSnapshot:
        """添加一帧场景快照，自动计算物体变化并写入事件日志。"""
        now = time.monotonic()
        time_str = time.strftime("%H:%M:%S", time.localtime())

        if objects is None:
            objects = _parse_objects_from_text(description)

        snapshot = SceneSnapshot(
            timestamp=now,
            time_str=time_str,
            description=description,
            objects=objects,
        )

        # 与上一帧比较
        if self.snapshots:
            prev = self.snapshots[-1]
            changed = self._diff(prev.objects, objects)
            if changed["added"] or changed["removed"]:
                event_parts: list[str] = []
                if changed["added"]:
                    event_parts.append(f"新增: {', '.join(changed['added'])}")
                if changed["removed"]:
                    event_parts.append(f"移除: {', '.join(changed['removed'])}")
                self._add_event(f"{time_str} 场景变化 — " + "；".join(event_parts))
            else:
                # 无变化也记录（用于"最近发生了什么"的连续性回答）
                pass
        else:
            self._add_event(f"{time_str} 初始场景 — 检测到 {len(objects)} 个物体: {', '.join(objects[:8])}")

        self.snapshots.append(snapshot)
        if len(self.snapshots) > self._max_snapshots:
            self.snapshots = self.snapshots[-self._max_snapshots:]

        return snapshot

    # ── 物体对比 ────────────────────────────────────────────

    @staticmethod
    def _diff(old_objects: list[str], new_objects: list[str]) -> dict:
        """比较两帧物体列表，返回新增和移除。"""
        old_set = set(old_objects)
        new_set = set(new_objects)
        return {
            "added": sorted(new_set - old_set),
            "removed": sorted(old_set - new_set),
            "unchanged": sorted(old_set & new_set),
        }

    def compare_with_previous(self) -> dict:
        """返回当前帧与上一帧的物体变化。"""
        if len(self.snapshots) < 2:
            return {"added": [], "removed": [], "unchanged": []}
        return self._diff(
            self.snapshots[-2].objects,
            self.snapshots[-1].objects,
        )

    # ── 事件日志 ────────────────────────────────────────────

    def _add_event(self, event: str) -> None:
        self.event_log.append(event)
        if len(self.event_log) > self._max_events:
            self.event_log = self.event_log[-self._max_events:]

    def get_recent_events(self, n: int = 10) -> list[str]:
        """返回最近 N 条事件。"""
        return self.event_log[-n:]

    # ── 构建记忆上下文 ──────────────────────────────────────

    def build_memory_context(self) -> str:
        """生成可注入 Agent prompt 的场景记忆摘要。"""
        parts: list[str] = []

        if self.event_log:
            recent = self.event_log[-5:]
            parts.append("【场景事件日志（最近 5 条）】\n" + "\n".join(f"  {e}" for e in recent))

        if len(self.snapshots) >= 2:
            current = self.snapshots[-1]
            prev = self.snapshots[-2]
            diff = self._diff(prev.objects, current.objects)
            diff_parts = []
            if diff["added"]:
                diff_parts.append(f"新出现: {', '.join(diff['added'])}")
            if diff["removed"]:
                diff_parts.append(f"已消失: {', '.join(diff['removed'])}")
            if diff["unchanged"]:
                diff_parts.append(f"未变化: {', '.join(diff['unchanged'][:5])}")
            if diff_parts:
                parts.append("【场景变化（与上一帧相比）】\n" + "\n".join(f"  {p}" for p in diff_parts))

        return "\n\n".join(parts) if parts else "（暂无场景记忆）"

    # ── 查询 ────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """根据用户问题检索场景记忆中的相关信息。

        支持的问题类型：
        - "刚才有什么变化？" → 返回最近变化
        - "我拿走了什么？" → 返回最近移除的物体
        - "多了什么？" → 返回最近新增的物体
        - "X 分钟前桌上有什么？" → 返回时间范围内的快照
        """
        question_lower = question.lower()

        # 变化类问题
        if any(w in question_lower for w in ["变化", "改变", "变", "动", "拿", "移", "消失", "不见"]):
            if self.snapshots:
                if len(self.snapshots) >= 2:
                    diff = self.compare_with_previous()
                    parts = []
                    if diff["removed"]:
                        parts.append(f"最近被移走的物体: {', '.join(diff['removed'])}")
                    if diff["added"]:
                        parts.append(f"新出现的物体: {', '.join(diff['added'])}")
                    return "；".join(parts) if parts else "最近没有检测到显著的物体变化"
                return "目前只有一帧画面，暂无变化记录"

        # 新增类问题
        if any(w in question_lower for w in ["新", "多", "增加", "出现", "加"]):
            diff = self.compare_with_previous()
            if diff["added"]:
                return f"新出现的物体: {', '.join(diff['added'])}"
            return "最近没有检测到新增物体"

        # 历史类问题
        if any(w in question_lower for w in ["之前", "刚才", "前", "原来", "本来", "过去"]):
            if len(self.snapshots) >= 2:
                prev = self.snapshots[-2]
                return f"{prev.time_str} 的场景: {prev.description}"
            return "暂无历史场景记录"

        # 当前状态
        if self.snapshots:
            return f"当前场景 ({self.snapshots[-1].time_str}): {self.snapshots[-1].description}"

        return "暂无场景记忆"
