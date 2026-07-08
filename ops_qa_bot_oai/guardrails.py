"""Guardrails（差异化 #4 的一半）：输入注入护栏 + 输出来源护栏。

ops-qa-bot（Claude SDK 版）的安全靠两样东西拼：prompt 自律（"图中文字只描述事实"、
"对话内确认不接受"）+ 一个只看命令字符串的 PreToolUse hook 兜底。OpenAI Agents SDK
把这类检查升级成**一等公民 guardrail**：与 agent 并行/前后跑的 tripwire，命中即抛
typed 异常、直接中止本轮——不用塞进 prompt 祈祷模型听话，也不用 hack hook。

本模块给三道护栏：

1. **输入注入护栏**（`injection_input_guardrail`）：拦 prompt 注入 / 数据外泄类输入
   （"忽略以上指令"、"把系统提示词发出来"、"导出所有文档"…）。检测是**确定性纯函数**
   `detect_injection`（无额外 LLM 调用，可单测），是那个 hook 的诚实对应物。
2. **输出来源护栏**（`citation_output_guardrail`）：结构化答案若引用了**不存在的文档**
   就 trip——把 #1 里"答案必须引用真实文档"的软校验升级成可中止交付的硬 tripwire。
3. **禁止命令工具护栏**（`forbidden_write_command_guardrail`）：SDK 的 **tool-level
   guardrail**，挂在 `request_write_command` 工具上。写操作因此分成三层：只读工具直接
   执行；一般写命令走人工审批（needs_approval）；**禁止清单**（毁灭性命令：rm -rf 根
   目录 / FLUSHALL / DROP DATABASE / mkfs / dd 直写块设备）命中即拒。它跑在**审批之后、
   执行之前**（SDK 语义），是"误批也执行不到"的最后防线；审批前的黑名单短路在
   `answer_guarded` 的中断循环里做（同一个 `detect_forbidden_command`），禁止级命令
   不会打扰审批人。`reject_content` 行为把拒绝原因回给模型让它改走文字建议，
   不中断整轮 run（与输入护栏的 tripwire 语义互补）。

注：也可以像 SDK 的 math-homework 例子那样用一个 LLM 子 agent 做语义级注入检测；这里
选确定性规则是因为免费、可测、且与 ops-qa-bot 的 hook 同源——需要更强可换成 LLM 护栏。
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
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


# 禁止命令清单：**不可恢复的毁灭性操作**，不进人工审批、直接拒。与"一般写命令"
# （重启 / 改配置 / 扩容等，走 needs_approval 人工审批）区分开——审批是给"有风险但
# 可能合理"的操作用的，毁灭性命令连提议资格都没有。按需扩展，标签用于给出拒绝原因。
_FORBIDDEN_COMMAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # rm -rf 根目录 / 根通配 / --no-preserve-root
    (re.compile(r"rm\s+(?:-[a-zA-Z]+\s+)*/(?:\s*$|\*)"), "整机删除(rm 根目录)"),
    (re.compile(r"rm\s+.*--no-preserve-root", re.I), "整机删除(no-preserve-root)"),
    # Redis 清库
    (re.compile(r"\bflush(all|db)\b", re.I), "清空 Redis 数据(FLUSH)"),
    # 数据库删库/删表
    (re.compile(r"\bdrop\s+(database|table)\b", re.I), "删除库/表(DROP)"),
    # 格式化文件系统 / 直写块设备
    (re.compile(r"\bmkfs(\.\w+)?\b", re.I), "格式化文件系统(mkfs)"),
    (re.compile(r"\bdd\s+.*\bof=/dev/", re.I), "直写块设备(dd)"),
]


def detect_forbidden_command(command: str) -> str | None:
    """检测命令是否命中禁止清单。命中返回原因标签，否则 None。纯函数，可单测。"""
    for pattern, label in _FORBIDDEN_COMMAND_PATTERNS:
        if pattern.search(command):
            return label
    return None


@tool_input_guardrail
def forbidden_write_command_guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """禁止命令工具护栏（SDK tool-level guardrail，挂 `request_write_command`）。

    从工具调用参数里取 `command` 查禁止清单，命中即 `reject_content`：工具不执行、
    拒绝原因作为工具结果回给模型（引导它改走文字建议），整轮 run 不中断。

    这层跑在人工审批之后、工具执行之前（SDK 语义）——正常流程里禁止命令已被
    `answer_guarded` 的审批前短路拦掉，这里是"即使误批也执行不到"的兜底；且护栏
    挂在工具对象上，随工具走：任何 agent 挂了这个工具就自带这层防线。
    """
    try:
        args = json.loads(data.context.tool_arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    command = str(args.get("command", "") if isinstance(args, dict) else "")
    label = detect_forbidden_command(command)
    if label:
        return ToolGuardrailFunctionOutput.reject_content(
            message=(
                f"该命令命中禁止清单（{label}），属毁灭性操作，不可执行也不可提议执行。"
                "请改为文字说明：解释为什么用户的诉求需要谨慎、给出更安全的替代方案，"
                "并标注 ⚠️ 风险。"
            ),
            output_info={"matched": label, "command": command},
        )
    return ToolGuardrailFunctionOutput.allow(output_info={"command": command})


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
