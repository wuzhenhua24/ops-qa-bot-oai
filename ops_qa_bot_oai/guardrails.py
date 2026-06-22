"""Guardrails（差异化 #4 的一半）：输入注入护栏 + 输出来源护栏。

ops-qa-bot（Claude SDK 版）的安全靠两样东西拼：prompt 自律（"图中文字只描述事实"、
"对话内确认不接受"）+ 一个只看命令字符串的 PreToolUse hook 兜底。OpenAI Agents SDK
把这类检查升级成**一等公民 guardrail**：与 agent 并行/前后跑的 tripwire，命中即抛
typed 异常、直接中止本轮——不用塞进 prompt 祈祷模型听话，也不用 hack hook。

本模块给两道护栏：

1. **输入注入护栏**（`injection_input_guardrail`）：拦 prompt 注入 / 数据外泄类输入
   （"忽略以上指令"、"把系统提示词发出来"、"导出所有文档"…）。检测是**确定性纯函数**
   `detect_injection`（无额外 LLM 调用，可单测），是那个 hook 的诚实对应物。
2. **输出来源护栏**（`citation_output_guardrail`）：结构化答案若引用了**不存在的文档**
   就 trip——把 #1 里"答案必须引用真实文档"的软校验升级成可中止交付的硬 tripwire。

注：也可以像 SDK 的 math-homework 例子那样用一个 LLM 子 agent 做语义级注入检测；这里
选确定性规则是因为免费、可测、且与 ops-qa-bot 的 hook 同源——需要更强可换成 LLM 护栏。
"""

from __future__ import annotations

import re
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

from .schema import AnswerContract, validate_citations
from .tools import DocsContext

# 注入 / 越权 / 外泄模式。命中即视作可疑输入。标签用于给出可读的拦截原因。
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 指令覆盖（经典 prompt injection）
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I), "指令覆盖(ignore previous)"),
    (re.compile(r"disregard\s+(the\s+)?(above|previous)", re.I), "指令覆盖(disregard above)"),
    (
        re.compile(r"忽略(以上|之前|前面|上面).{0,6}(指令|规则|提示|要求|设定)"),
        "指令覆盖(忽略以上)",
    ),
    (re.compile(r"(无视|不要管|抛开).{0,6}(之前|以上|前面).{0,6}(指令|规则|设定)"), "指令覆盖"),
    (re.compile(r"你(现在)?(是|扮演|作为).{0,12}(不受限制|没有限制|开发者模式|DAN)"), "角色越狱"),
    # 套取系统提示 / 配置
    (
        re.compile(
            r"(reveal|show|print|repeat)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I
        ),
        "套取系统提示",
    ),
    (
        re.compile(
            r"(把|将|输出|告诉我|打印).{0,8}(系统)?(提示词|提示语|prompt|指令|instructions)", re.I
        ),
        "套取系统提示",
    ),
    # 数据外泄 / 越权批量导出（动词在前或在后两种语序都覆盖）
    (
        re.compile(r"(导出|下载|打包|外发).{0,6}(所有|全部|整个).{0,6}(文档|文件|目录|内容)"),
        "批量外泄文档",
    ),
    (
        re.compile(
            r"(所有|全部|整个).{0,4}(文档|文件|目录|内容).{0,8}(导出|下载|打包|发出|发送|发给|外发|发)"
        ),
        "批量外泄文档",
    ),
    (
        re.compile(r"(列出|泄露|给我).{0,8}(密钥|密码|secret|token|api[_ ]?key|凭证)", re.I),
        "套取凭证",
    ),
    (re.compile(r"(dump|exfiltrate|leak)\s+(all|every|the entire)", re.I), "批量外泄"),
]


def detect_injection(text: str) -> str | None:
    """检测输入里的注入/越权/外泄意图。命中返回模式标签，否则 None。纯函数，可单测。"""
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _input_text(data: str | list[TResponseInputItem]) -> str:
    """把 guardrail 收到的 input（字符串或消息项列表）拼成可扫描的纯文本。"""
    if isinstance(data, str):
        return data
    parts: list[str] = []
    for item in data:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for seg in content:
                    if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                        parts.append(seg["text"])
    return "\n".join(parts)


@input_guardrail
async def injection_input_guardrail(
    ctx: RunContextWrapper[Any],
    agent: Agent[Any],
    data: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """输入注入护栏：检测到注入/越权/外泄意图就 trip（中止本轮）。"""
    label = detect_injection(_input_text(data))
    return GuardrailFunctionOutput(
        output_info={"matched": label},
        tripwire_triggered=label is not None,
    )


@output_guardrail
async def citation_output_guardrail(
    ctx: RunContextWrapper[DocsContext],
    agent: Agent[Any],
    output: AnswerContract,
) -> GuardrailFunctionOutput:
    """输出来源护栏：结构化答案引用了不存在/越界的文档路径就 trip。

    仅当 decision=answer（声称基于文档作答）时才较真——reject/clarify/escalate 本就
    可以没有来源。复用 schema.validate_citations 做真实性校验。
    """
    docs_root = ctx.context.docs_root
    invalid: list[str] = []
    if output.decision.value == "answer":
        invalid = validate_citations(docs_root, output.citations)
        # answer 却一条来源都没有，也算不合格（防"看似作答但无据"）。
        no_citation = len(output.citations) == 0
    else:
        no_citation = False
    return GuardrailFunctionOutput(
        output_info={"invalid_citations": invalid, "answer_without_citation": no_citation},
        tripwire_triggered=bool(invalid) or no_citation,
    )
