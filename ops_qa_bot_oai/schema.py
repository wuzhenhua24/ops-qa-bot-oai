"""结构化输出契约（差异化 #1 的原型）。

ops-qa-bot（Claude Agent SDK 版）靠在自由文本答案里塞 `<<ESCALATE>>`/`<<CLARIFY>>`/
`<<FOLLOWUPS>>` 标记 + 正则解析来传递「路由决策」，模型偶尔写错格式就丢信息。

OpenAI Agents SDK 支持 `output_type=`：把一轮回答强制成**带 schema 校验的类型对象**，
模型必须按字段填、不合法 SDK 会重试。这是 Claude Agent SDK 给不了的一等公民能力
（那边只能自己 parse 文本）。本模块定义这个契约，外加一个「每条来源是否真实存在」
的校验函数（把"答案必须引用真实文档"从 prompt 自律升级成代码强校验）。

为可跨 provider（Claude / 智谱 / 火山等不一定支持 OpenAI strict 结构化输出），契约
设计成 strict 友好但实际用**非严格** schema 下发（见 bot.py 的 AgentOutputSchema）。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from agents import AgentOutputSchema
from pydantic import BaseModel, Field

from .tools import _resolve_within

# 有些 provider 的 json_schema 实现会把 JSON 裹在 ```json ... ``` 代码围栏里返回
# （实测智谱 GLM 如此，有时前面还带一段前言文字），SDK 直接解析会 ModelBehaviorError。
# 用 search（非锚定）抓第一个围栏块，兼容围栏前的前言。
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE)


class FenceTolerantOutputSchema(AgentOutputSchema):
    """容忍非 OpenAI provider 结构化输出的两类不规范，尽量把契约解析出来：

    1. **代码围栏**：把 JSON 裹在 ```json ... ``` 里（GLM 常见）——校验前剥掉。
    2. **字符串里的裸控制字符**：把 markdown 正文的换行等直接塞进 JSON 字符串而不转义成
       `\\n`（strict=False 的端点常见），标准 JSON 视作非法——严格路径失败时用
       `json.loads(strict=False)` 宽松解析后再按 schema 校验。

    正常合法 JSON 一律走 SDK 原生严格路径；宽松兜底只在严格失败时才启用，不放宽 schema 本身。
    """

    def validate_json(self, json_str: str):
        # 先按 SDK 原生严格路径试原始串：合法 JSON 直接过，也不会误伤 answer 里的内部 ``` 代码块。
        try:
            return super().validate_json(json_str)
        except Exception as original:
            # 失败了再容错：抓（可能带前言的）围栏块 → 严格试 → 宽松试（裸控制字符 + 补反斜杠）。
            m = _JSON_FENCE_RE.search(json_str)
            text = m.group(1) if m else json_str
            if text != json_str:
                try:
                    return super().validate_json(text)
                except Exception:
                    pass
            try:
                repaired = re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", text)
                return self._type_adapter.validate_python(json.loads(repaired, strict=False))
            except Exception:
                # 兜底也救不了 → 抛回 SDK 原始异常（通常是 ModelBehaviorError），让 SDK 按既有
                # 逻辑重试；别抛 raw JSONDecodeError 破坏重试契约。
                raise original from None


class Decision(str, Enum):
    """这一轮答题的路由决策——取代旧的 marker。"""

    answer = "answer"  # 基于文档正常作答
    clarify = "clarify"  # 信息不足，先反问（对应旧 <<CLARIFY>>）
    escalate = "escalate"  # 文档确实查不到，升级给负责人（对应旧 <<ESCALATE>>）
    reject = "reject"  # 运维范围外，友好拒绝


class Followup(str, Enum):
    """建议的快捷追问类型（对应旧 <<FOLLOWUPS:...>> 的白名单）。"""

    troubleshoot = "troubleshoot"
    risks = "risks"
    rollback = "rollback"
    checklist = "checklist"
    commands = "commands"
    related = "related"


class AnswerContract(BaseModel):
    """一轮问答的结构化结果。模型必须按此 schema 产出。"""

    decision: Decision = Field(description="这一轮的路由决策")
    answer: str = Field(
        description="给用户看的最终回答（中文 markdown）。反问/拒绝时这里就是反问语/拒绝语。"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="答案所依据的文档相对路径（相对文档根目录，如 'redis/troubleshooting.md'）。"
        "decision=answer 时应至少有一条；reject/clarify 时通常为空。",
    )
    escalate_to: str = Field(
        default="",
        description="decision=escalate 时，INDEX.md 里该组件负责人的 open_id（ou_xxx）；否则留空。",
    )
    escalate_dir: str = Field(
        default="",
        description="decision=escalate 时，该组件的目录名（如 'redis'，不带斜杠）；否则留空。",
    )
    followups: list[Followup] = Field(
        default_factory=list, description="建议的快捷追问，最多 3 个；不需要时留空。"
    )
    confidence: float = Field(
        default=0.0, description="对本答案的可信度自评，0~1。文档命中充分则高，靠推断则低。"
    )


def validate_citations(docs_root: Path, citations: list[str]) -> list[str]:
    """返回 citations 里**不存在 / 越界**的路径列表（空列表表示全部真实存在）。

    把"答案必须引用真实文档"从 prompt 自律变成可执行的代码校验——这是结构化契约
    带来的直接好处：拿到强类型 citations 后能在返回给用户前逐条核对。
    """
    invalid: list[str] = []
    for rel in citations:
        try:
            target = _resolve_within(docs_root, rel)
        except ValueError:
            invalid.append(rel)  # 越界路径
            continue
        if not target.is_file():
            invalid.append(rel)
    return invalid
