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

from agents import (
    Agent,
    AgentOutputSchema,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from openai.types.responses import ResponseTextDeltaEvent

from .actions import WriteCommandLog, make_write_command_tool
from .guardrails import (
    citation_output_guardrail,
    injection_input_guardrail,
)
from .model import MODES, ModelChoice, ModelRouter, build_model_router, resolve_model
from .orchestration import (
    Component,
    build_auto_agent,
    build_coordinator_agent,
    build_triage_agent,
)
from .prompt import build_structured_system_prompt, build_system_prompt
from .schema import AnswerContract, Decision, validate_citations
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
    # 最终处理该问题的落点 agent 名（最后一次 handoff 目标）；None 表示入口 agent 自答
    # （auto 模式下即分诊台自答）。用于评测路由准确率。
    route: str | None = None


@dataclass
class StructuredAnswer:
    """`OpsQABot.answer_structured()` 的返回值：强类型契约 + 来源校验 + 用量。"""

    contract: AnswerContract
    # 契约里 citations 中**不存在/越界**的路径（空表示全部真实存在）。
    invalid_citations: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    num_turns: int | None = None
    subtype: str = "success"
    # 命中输出来源护栏（#4）时为拦截原因；正常为 None。
    guardrail_blocked: str | None = None


@dataclass
class ApprovalRequest:
    """一次写操作审批请求（HITL interruption 的归一化视图）。"""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class GuardedAnswer:
    """`OpsQABot.answer_guarded()` 的返回值：答案 + 审批记录 + 护栏拦截信息。"""

    text: str
    # 本轮经过审批的请求及其结果（approved=True/False）。
    approvals: list[tuple[ApprovalRequest, bool]] = field(default_factory=list)
    # 被批准并登记执行的写命令（target/command/reason）。
    approved_writes: list[Any] = field(default_factory=list)
    # 命中输入注入护栏时为拦截原因；正常为 None。
    blocked: str | None = None
    usage: dict[str, Any] | None = None
    num_turns: int | None = None
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
    if name.startswith("ask_"):
        # 协调者调用组件专家（agents-as-tools）；子问题可能很长，截断展示。
        q = str(args.get("input") or args.get("question") or args.get("query") or args)
        if len(q) > 60:
            q = q[:60] + "…"
        return f"{name} ← {q}"
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
        mode: str = "single",
        model_router: ModelRouter | None = None,
        guardrails: bool = False,
    ):
        if mode not in MODES:
            raise ValueError(f"未知 mode={mode!r}，可选：{' / '.join(MODES)}")
        self.docs_root = Path(docs_root).resolve()
        if not self.docs_root.is_dir():
            raise ValueError(f"docs_root 不存在或不是目录: {self.docs_root}")
        if not (self.docs_root / "INDEX.md").is_file():
            raise ValueError(f"docs_root 下缺少 INDEX.md 路由表: {self.docs_root / 'INDEX.md'}")

        self.model_choice = model_choice or resolve_model()
        if max_turns is not None and max_turns <= 0:
            max_turns = None
        self.max_turns = max_turns
        self.mode = mode
        self.guardrails = guardrails

        # 差异化 #4：开启 guardrails 时挂输入注入护栏 + 写操作审批工具（HITL）。
        # write_log 记录被批准并登记执行的写命令。
        self.write_log = WriteCommandLog()
        self._input_guardrails = [injection_input_guardrail] if guardrails else []
        self._extra_tools = [make_write_command_tool(self.write_log)] if guardrails else []

        self._agent: Agent[DocsContext]
        self.components: list[Component]
        self.model_router: ModelRouter | None = None
        if mode == "coordinator":
            # 跨组件协作：协调者把各组件专家当工具（agents-as-tools）调用、综合根因。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_coordinator_agent(
                self.docs_root, self.model_router
            )
        elif mode == "auto":
            # 自适应默认：分诊台 handoff 给单专家（常见）或跨组件协调者（少数）。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_auto_agent(self.docs_root, self.model_router)
        elif mode == "multi":
            # 差异化 #3：入口换成分诊 agent，handoff 给从 INDEX.md 动态生成的组件专家。
            # 差异化 #2：用 ModelRouter 按角色/组件分配模型（分诊便宜、专家强）。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_triage_agent(self.docs_root, self.model_router)
        else:  # single
            self.components = []
            self._agent = Agent(
                name="ops-qa-bot",
                instructions=build_system_prompt(self.docs_root),
                tools=list(DOC_TOOLS) + self._extra_tools,
                model=self.model_choice.model,
                input_guardrails=self._input_guardrails,
            )
        self._context = DocsContext(docs_root=self.docs_root)
        # 多轮对话历史（OpenAI SDK 的 input 列表形态）；reset() 清空。
        self._history: list[Any] = []
        # 结构化输出 agent 懒构造（仅 answer_structured 用到时才建）。
        self._structured_agent: Agent[DocsContext] | None = None

    def reset(self) -> None:
        """清空会话上下文，开始新对话。"""
        self._history = []

    def _get_structured_agent(self) -> Agent[DocsContext]:
        """懒构造结构化输出 agent：同样的工具/模型，但 output_type=AnswerContract。

        用非严格 schema（strict_json_schema=False）下发，以兼容 Claude / 智谱 / 火山等
        不支持 OpenAI strict 结构化输出的 provider。
        """
        if self._structured_agent is None:
            self._structured_agent = Agent(
                name="ops-qa-bot-structured",
                instructions=build_structured_system_prompt(self.docs_root),
                tools=list(DOC_TOOLS),
                model=self.model_choice.model,
                output_type=AgentOutputSchema(AnswerContract, strict_json_schema=False),
                # 差异化 #4：结构化模式下加输入注入护栏 + 输出来源护栏（引用不实就 trip）。
                input_guardrails=self._input_guardrails,
                output_guardrails=[citation_output_guardrail] if self.guardrails else [],
            )
        return self._structured_agent

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
        seen_first_agent = False
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield {"type": "text", "text": event.data.delta}
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        name, args = _extract_tool_call(event.item.raw_item)
                        yield {"type": "tool", "name": name, "input": args}
                elif event.type == "agent_updated_stream_event":
                    # 首个 agent_updated 是入口 agent（非真 handoff），跳过；
                    # 之后的才是 handoff 切换（分诊 → 专家 / 专家间）。
                    if seen_first_agent:
                        yield {"type": "handoff", "agent": event.new_agent.name}
                    seen_first_agent = True
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
        route: str | None = None  # 最后一次 handoff 目标 = 最终落点 agent
        async for event in self.ask(question):
            if event["type"] == "tool":
                logger.info("  tool: %s", format_tool_call(event["name"], event["input"]))
            elif event["type"] == "handoff":
                route = event["agent"]
                logger.info("  → 转交给 %s", event["agent"])
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
            route=route,
        )

    async def answer_structured(self, question: str) -> StructuredAnswer:
        """结构化输出模式（差异化 #1）：返回强类型 AnswerContract + 来源真实性校验。

        与 `answer()` 不同，这里 agent 的 `output_type=AnswerContract`，SDK 强制模型
        按 schema 产出（不合法会重试），路由决策（answer/clarify/escalate/reject）、
        来源、追问都是类型字段而非文本标记。拿到契约后再用代码逐条核对 citations
        是否指向真实存在的文档——把"答案必须引用真实文档"从 prompt 自律变成硬校验。

        注意：结构化输出不便逐 token 流式，这里走非流式 `Runner.run`。多轮历史与
        流式 `ask()` 共享 `self._history`。
        """
        logger.info("question (structured): %s", question)
        agent = self._get_structured_agent()
        input_items = self._history + [{"role": "user", "content": question}]
        subtype = "success"
        try:
            result = await Runner.run(agent, input=input_items, **self._run_kwargs())
        except MaxTurnsExceeded:
            # 撞上限时没有合法契约可返回，退化成一个 reject 契约并标记 subtype。
            return StructuredAnswer(
                contract=AnswerContract(
                    decision=Decision.reject,
                    answer="本轮检索步数过多被中断（max_turns），结论可能不完整，请换个问法或缩小范围。",
                ),
                subtype="error_max_turns",
            )
        except InputGuardrailTripwireTriggered as e:
            # 差异化 #4：输入注入护栏拦截（结构化模式同样生效）。
            reason = _guardrail_reason(e)
            return StructuredAnswer(
                contract=AnswerContract(
                    decision=Decision.reject,
                    answer="这条输入触发了安全护栏（疑似提示注入/越权），已拦截，不予处理。",
                ),
                subtype="blocked_input",
                guardrail_blocked=reason,
            )
        except OutputGuardrailTripwireTriggered as e:
            # 输出来源护栏：答案引用了不存在的文档 / 声称作答却无来源 → 拦下不交付。
            info = getattr(getattr(e, "guardrail_result", None), "output", None)
            detail = getattr(info, "output_info", None)
            logger.warning("结构化答案命中输出来源护栏: %s", detail)
            return StructuredAnswer(
                contract=AnswerContract(
                    decision=Decision.escalate,
                    answer="答案未通过来源校验（引用了不存在的文档或缺少可靠来源），已拦截以防幻觉。建议人工复核或补充文档。",
                ),
                subtype="blocked_output",
                guardrail_blocked=str(detail),
            )

        contract = result.final_output  # 已被 SDK 按 schema 校验
        assert isinstance(contract, AnswerContract)
        try:
            self._history = result.to_input_list()
        except Exception:  # noqa: BLE001
            logger.debug("to_input_list() 失败，保留旧历史", exc_info=True)

        invalid = validate_citations(self.docs_root, contract.citations)
        if invalid:
            logger.warning("结构化答案引用了不存在/越界的来源: %s", invalid)
        return StructuredAnswer(
            contract=contract,
            invalid_citations=invalid,
            usage=_usage_dict(result),
            num_turns=_num_turns(result),
            subtype=subtype,
        )

    async def answer_guarded(self, question: str, approver: Any = None) -> GuardedAnswer:
        """带护栏 + 写操作审批（HITL）的问答（差异化 #4，需 guardrails=True）。

        - 输入注入护栏命中 → 直接返回拦截结果（blocked）。
        - agent 想调写操作工具 → run 暂停抛 interruption；用 `approver(req)->bool`
          决定批准/驳回，再 `Runner.run(agent, state)` 续跑。approver 为 None 时一律
          驳回（安全默认：无人值守不放行写操作）。

        这套 pause→approve→resume 走 SDK 的 RunState，是 ops-qa-bot 用 hook + 飞书
        回调拼出来那套 HITL 的一等公民替代。
        """
        if not self.guardrails:
            raise RuntimeError("answer_guarded 需要以 guardrails=True 构造 OpsQABot")
        logger.info("question (guarded): %s", question)
        input_items = self._history + [{"role": "user", "content": question}]
        approvals: list[tuple[ApprovalRequest, bool]] = []
        try:
            result = await Runner.run(self._agent, input=input_items, **self._run_kwargs())
            # 处理写操作审批中断，直到没有待批项。
            while result.interruptions:
                state = result.to_state()
                for itr in result.interruptions:
                    req = ApprovalRequest(
                        tool_name=getattr(itr, "name", "?"),
                        arguments=_interruption_args(itr),
                    )
                    approved = bool(approver(req)) if approver is not None else False
                    approvals.append((req, approved))
                    if approved:
                        state.approve(itr)
                    else:
                        state.reject(itr)
                result = await Runner.run(self._agent, state, max_turns=self.max_turns)
        except InputGuardrailTripwireTriggered as e:
            reason = _guardrail_reason(e)
            return GuardedAnswer(
                text="这条输入触发了安全护栏（疑似提示注入/越权），已拦截，不予处理。",
                blocked=reason,
                subtype="blocked_input",
            )
        except MaxTurnsExceeded:
            return GuardedAnswer(
                text="本轮检索步数过多被中断（max_turns），结论可能不完整。",
                approvals=approvals,
                subtype="error_max_turns",
            )

        try:
            self._history = result.to_input_list()
        except Exception:  # noqa: BLE001
            logger.debug("to_input_list() 失败，保留旧历史", exc_info=True)

        text = (
            result.final_output
            if isinstance(result.final_output, str)
            else str(result.final_output or "")
        )
        cleaned, _ = parse_markers(text)
        approved_writes = [r for r in self.write_log.requests if r.approved]
        return GuardedAnswer(
            text=cleaned,
            approvals=approvals,
            approved_writes=approved_writes,
            usage=_usage_dict(result),
            num_turns=_num_turns(result),
        )


def _interruption_args(itr: Any) -> dict[str, Any]:
    """从 interruption 取工具参数 dict（兼容 str/JSON/dict 几种形态）。"""
    import json

    raw = getattr(itr, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _guardrail_reason(exc: Any) -> str:
    """从 tripwire 异常里抠出可读的拦截原因（matched 标签）。"""
    info = getattr(getattr(exc, "guardrail_result", None), "output", None)
    detail = getattr(info, "output_info", None)
    if isinstance(detail, dict) and detail.get("matched"):
        return str(detail["matched"])
    return str(detail) if detail else "input guardrail"


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
