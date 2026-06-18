"""OpsQABot：基于 OpenAI Agents SDK 的运维文档问答机器人。

对标 `ops-qa-bot`（Claude Agent SDK 版）的 bot.py，对外接口刻意保持一致，方便并排
对比：

- `ask(question)`：流式异步生成器，逐段吐 {type: tool|text|done} 事件（适合 CLI）。
- `answer(question)`：一次性返回完整答案 + 用量元数据（适合接入层）。
- 多轮对话：内部维护输入历史，`reset()` 清空开新会话。
- `max_turns`：单轮答题步数保险丝，撞上时 subtype 标 "error_max_turns"。
- 解析 `<<ESCALATE>>`/`<<CLARIFY>>`/`<<FOLLOWUPS>>` 标记（与 ops-qa-bot markers 对齐）。

与 Claude 版的结构差异：Claude SDK 用常驻 `ClaudeSDKClient` 子进程维护会话；这里用
无状态的 `Runner` + 自己持有的 `input` 历史列表实现多轮，更轻、也更贴近 OpenAI SDK
的惯用法。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import Agent, MaxTurnsExceeded, Runner
from openai.types.responses import ResponseTextDeltaEvent

from .model import ModelChoice, resolve_model
from .prompt import build_system_prompt
from .tools import DOC_TOOLS, DocsContext

logger = logging.getLogger(__name__)

# 单轮答题步数上限：防 agent 在文档里迷路 / 反复检索时无限烧 token。默认 30 与
# ops-qa-bot 对齐（典型问答 turns 在个位数）。<=0 视作不限。
DEFAULT_MAX_TURNS = 30

_ESCALATE_RE = re.compile(r"<<ESCALATE:([^>]*)>>")
_CLARIFY_RE = re.compile(r"<<CLARIFY>>")
_FOLLOWUPS_RE = re.compile(r"<<FOLLOWUPS:([^>]*)>>")
_VALID_FOLLOWUP_KEYS = {
    "troubleshoot",
    "risks",
    "rollback",
    "checklist",
    "commands",
    "related",
}


@dataclass
class Markers:
    """从答案文本里解析出的标记（解析后会从展示文本里剥离）。"""

    escalate: str | None = None  # ESCALATE 的负载，如 "ou_xxx:redis" 或 "none"
    clarify: bool = False
    followups: list[str] = field(default_factory=list)


def parse_markers(text: str) -> tuple[str, Markers]:
    """剥离标记并返回 (清理后的文本, Markers)。对齐 ops-qa-bot 的 markers 行为。"""
    markers = Markers()

    m = _ESCALATE_RE.search(text)
    if m:
        markers.escalate = m.group(1).strip()
    if _CLARIFY_RE.search(text):
        markers.clarify = True
    fm = _FOLLOWUPS_RE.search(text)
    if fm:
        keys = [k.strip() for k in fm.group(1).split("|") if k.strip()]
        # 白名单过滤 + 去重保序，最多 3 个（与参考项目一致）。
        seen: set[str] = set()
        for k in keys:
            if k in _VALID_FOLLOWUP_KEYS and k not in seen:
                seen.add(k)
                markers.followups.append(k)
            if len(markers.followups) >= 3:
                break

    cleaned = _ESCALATE_RE.sub("", text)
    cleaned = _CLARIFY_RE.sub("", cleaned)
    cleaned = _FOLLOWUPS_RE.sub("", cleaned)
    # 标记常独占一行，剥离后会留下多余空行，收一收。
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, markers


@dataclass
class AnswerResult:
    """`OpsQABot.answer()` 的返回值：答案 + 用量 + 解析出的标记。"""

    text: str
    markers: Markers = field(default_factory=Markers)
    usage: dict[str, Any] | None = None  # input/output/total tokens, requests, cached 等
    num_turns: int | None = None
    # "success" 正常；"error_max_turns" 撞了 max_turns 保险丝（答案可能不完整）。
    subtype: str = "success"


def format_tool_call(name: str, args: dict) -> str:
    """紧凑展示工具调用，用于日志和 CLI。"""
    if name == "read_doc":
        return f"read_doc {args.get('path', '?')}"
    if name == "glob_docs":
        return f"glob_docs {args.get('pattern', '?')}"
    if name == "grep_docs":
        pattern = args.get("pattern", "?")
        path = args.get("path") or ""
        return f"grep_docs '{pattern}'" + (f" in {path}" if path else "")
    return f"{name}({args})"


class OpsQABot:
    """运维文档问答机器人（OpenAI Agents SDK 实现）。

    用法（流式，适合 CLI）：
        bot = OpsQABot(docs_root="./docs")
        async for event in bot.ask("Redis 内存告警怎么处理？"):
            ...

    用法（一次性拿完整答案，适合接入层）：
        bot = OpsQABot(docs_root="./docs")
        result = await bot.answer("Redis 内存告警怎么处理？")
    """

    def __init__(
        self,
        docs_root: str | Path,
        *,
        max_turns: int | None = DEFAULT_MAX_TURNS,
        model_choice: ModelChoice | None = None,
    ):
        self.docs_root = Path(docs_root).resolve()
        if not self.docs_root.is_dir():
            raise ValueError(f"docs_root 不存在或不是目录: {self.docs_root}")
        if not (self.docs_root / "INDEX.md").is_file():
            raise ValueError(f"docs_root 下缺少 INDEX.md 路由表: {self.docs_root / 'INDEX.md'}")

        self.model_choice = model_choice or resolve_model()
        if max_turns is not None and max_turns <= 0:
            max_turns = None
        self.max_turns = max_turns

        self._agent: Agent[DocsContext] = Agent(
            name="ops-qa-bot",
            instructions=build_system_prompt(self.docs_root),
            tools=list(DOC_TOOLS),
            model=self.model_choice.model,
        )
        self._context = DocsContext(docs_root=self.docs_root)
        # 多轮对话历史（OpenAI SDK 的 input 列表形态）；reset() 清空。
        self._history: list[Any] = []

    def reset(self) -> None:
        """清空会话上下文，开始新对话。"""
        self._history = []

    def _run_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"context": self._context}
        if self.max_turns is not None:
            kwargs["max_turns"] = self.max_turns
        return kwargs

    async def ask(self, question: str) -> AsyncIterator[dict]:
        """向 bot 提问，流式返回事件字典：

        - {"type": "tool", "name": str, "input": dict}  —— agent 调用的工具
        - {"type": "text", "text": str}                 —— 回答文本片段
        - {"type": "done", "usage": dict | None,
           "num_turns": int | None, "subtype": str}     —— 本轮结束
        """
        input_items = self._history + [{"role": "user", "content": question}]
        result = Runner.run_streamed(self._agent, input=input_items, **self._run_kwargs())

        subtype = "success"
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield {"type": "text", "text": event.data.delta}
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        name, args = _extract_tool_call(event.item.raw_item)
                        yield {"type": "tool", "name": name, "input": args}
        except MaxTurnsExceeded:
            subtype = "error_max_turns"

        if subtype == "success":
            # 正常收尾时把完整历史接上，供下一轮追问；撞 max_turns 时历史可能不完整，
            # 保守起见也接上已产出的部分（与 ops-qa-bot 一样保留上下文）。
            try:
                self._history = result.to_input_list()
            except Exception:  # noqa: BLE001 —— 极端情况下拿不到就保留旧历史
                logger.debug("to_input_list() 失败，保留旧历史", exc_info=True)

        yield {
            "type": "done",
            "usage": _usage_dict(result),
            "num_turns": _num_turns(result),
            "subtype": subtype,
        }

    async def answer(self, question: str) -> AnswerResult:
        """一次性返回完整答案 + 用量 + 解析出的标记。"""
        logger.info("question: %s", question)
        chunks: list[str] = []
        usage: dict | None = None
        num_turns: int | None = None
        subtype = "success"
        async for event in self.ask(question):
            if event["type"] == "tool":
                logger.info("  tool: %s", format_tool_call(event["name"], event["input"]))
            elif event["type"] == "text":
                chunks.append(event["text"])
            elif event["type"] == "done":
                usage = event.get("usage")
                num_turns = event.get("num_turns")
                subtype = event.get("subtype", "success")
        cleaned, markers = parse_markers("".join(chunks))
        return AnswerResult(
            text=cleaned,
            markers=markers,
            usage=usage,
            num_turns=num_turns,
            subtype=subtype,
        )


def _extract_tool_call(raw_item: Any) -> tuple[str, dict]:
    """从流式 tool_call_item 的 raw_item 里抠出工具名 + 参数 dict。

    函数工具调用的 raw_item 通常是 ResponseFunctionToolCall（.name + .arguments JSON 串）。
    解析失败时退化成原始串，保证 CLI 不崩。
    """
    import json

    name = getattr(raw_item, "name", None) or "?"
    raw_args = getattr(raw_item, "arguments", None)
    args: dict = {}
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    return name, args


def _usage_dict(result: Any) -> dict[str, Any] | None:
    """从 run 结果里取 token 用量，整理成扁平 dict（对齐 feedback 日志习惯）。"""
    try:
        usage = result.context_wrapper.usage
    except AttributeError:
        return None
    if usage is None:
        return None
    cached = 0
    reasoning = 0
    try:
        cached = usage.input_tokens_details.cached_tokens or 0
    except AttributeError:
        pass
    try:
        reasoning = usage.output_tokens_details.reasoning_tokens or 0
    except AttributeError:
        pass
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "requests": usage.requests,
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def _num_turns(result: Any) -> int | None:
    """用 model 请求次数近似 num_turns（OpenAI SDK 没有直接的 num_turns 字段）。"""
    try:
        return result.context_wrapper.usage.requests
    except AttributeError:
        return None
