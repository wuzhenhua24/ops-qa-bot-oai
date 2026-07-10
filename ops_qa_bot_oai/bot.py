"""OpsQABot：基于 OpenAI Agents SDK 的运维文档问答机器人。

对标 `ops-qa-bot`（Claude Agent SDK 版）的 bot.py，对外接口刻意保持一致，方便并排
对比：

- `ask(question)`：流式异步生成器，逐段吐 {type: tool|text|done} 事件（适合 CLI）。
- `answer(question)`：一次性返回完整答案 + 用量元数据（适合接入层）。
- 多轮对话：交给 SDK 的 **Session**（`Runner.run(..., session=...)` 自动读写历史），
  `reset()` 清空开新会话。默认进程内 SQLiteSession；接入层可注入持久化 session
  （如飞书按 (chat,user) 一个 session_id + 落盘 db，重启不丢上下文）。
- `max_turns`：单轮答题步数保险丝，撞上时 subtype 标 "error_max_turns"。
- 解析 `<<ESCALATE>>`/`<<CLARIFY>>` 标记（与 ops-qa-bot markers 对齐）。

与 Claude 版的结构差异：Claude SDK 用常驻 `ClaudeSDKClient` 子进程维护会话；这里用
无状态的 `Runner` + SDK Session 实现多轮，更轻、也更贴近 OpenAI SDK 的惯用法。
"""

from __future__ import annotations

import base64
import inspect
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    Runner,
    SQLiteSession,
)
from agents.extensions.handoff_filters import remove_all_tools
from agents.memory import Session
from openai.types.responses import ResponseTextDeltaEvent

from .actions import WriteCommandLog, make_write_command_tool
from .db_query import (
    DB_CHANGE_TOOL_NAME,
    DB_TOOL_NAME,
    DatabaseClient,
    DbConfig,
    DbQueryLog,
    make_db_change_tool,
    make_query_database_tool,
    validate_change_args,
)
from .diagnostics import DiagConfig, DiagnosticLog, make_diagnostic_tool
from .doc_qa import DOC_QA_TOOL_NAME, DocQAConfig, DocQALog, make_feishu_doc_tool
from .followup import FOLLOWUP_TOOL_NAME, FollowupConfig, make_schedule_followup_tool
from .gateway_trace import (
    GW_TRACE_TOOL_NAME,
    GatewayTraceConfig,
    GatewayTraceLog,
    make_gateway_trace_tool,
)
from .guardrails import (
    citation_output_guardrail,
    detect_forbidden_command,
    injection_input_guardrail,
)
from .hooks import RunTelemetry
from .index import parse_index_components
from .model import (
    MODES,
    ModelChoice,
    ModelRouter,
    build_model_router,
    env_flag,
    resolve_model,
    role_model_settings,
)
from .orchestration import (
    Component,
    build_auto_agent,
    build_coordinator_agent,
    build_triage_agent,
)
from .prompt import build_structured_system_prompt, build_system_prompt
from .review import (
    REVIEWER_ROLE,
    Draft,
    ReviewConfig,
    ReviewOutcome,
    build_reviewer_agent,
    extract_citations,
    format_findings,
    gather_evidence,
    is_review_eligible,
    review_and_revise,
)
from .schema import AnswerContract, Decision, FenceTolerantOutputSchema, validate_citations
from .tools import DOC_TOOLS, DocsContext

logger = logging.getLogger(__name__)

# 单轮答题步数上限：防 agent 在文档里迷路 / 反复检索时无限烧 token。默认 30 与
# ops-qa-bot 对齐（典型问答 turns 在个位数）。<=0 视作不限。
DEFAULT_MAX_TURNS = 30

# answer_guarded 审批循环的轮次保险丝：写提议被驳回后模型可能反复重新提议同一命令
# （resume 不计入 max_turns，会无限循环——实测 GLM 撞上过），超过即终止本轮。
MAX_APPROVAL_ROUNDS = 5

# 驳回时回给模型的解释（rejection_message）。SDK 默认消息只说"被拒绝"不说原因，
# 模型容易换个姿势重试；说清"别再提交、改文字建议"才能让循环自然收敛。
_BLACKLIST_REJECTION_MSG = (
    "该命令命中禁止清单（{label}），属毁灭性操作，**永远不会被批准**。"
    "不要再次提交该命令或其变体；请在最终回答里说明为何拒绝、"
    "给出更安全的替代方案，并标注 ⚠️ 风险。"
)
_HUMAN_REJECTION_MSG = (
    "运维值班人已驳回该写操作提议。不要重复提交；"
    "请在最终回答里给出文字操作建议（含步骤与 ⚠️ 风险），由人工执行。"
)

_ESCALATE_RE = re.compile(r"<<ESCALATE:([^>]*)>>")
_CLARIFY_RE = re.compile(r"<<CLARIFY>>")


@dataclass(frozen=True)
class _EvidenceMark:
    """答题前各证据日志的长度快照。

    复核只该看**本轮**新增的证据：诊断日志、数据库查询、网关链路和飞书文档答案在多轮会话里
    是累积的，不切一刀，第二轮的 reviewer 会拿到第一轮的诊断输出当"本轮证据"，据此判定第二
    轮答案"与证据矛盾"。
    """

    diag: int
    doc_qa: int
    db: int
    gw_trace: int


@dataclass
class Markers:
    """从答案文本里解析出的标记（解析后会从展示文本里剥离）。"""

    escalate: str | None = None  # ESCALATE 的负载，如 "ou_xxx:redis" 或 "none"
    clarify: bool = False


def parse_markers(text: str) -> tuple[str, Markers]:
    """剥离标记并返回 (清理后的文本, Markers)。对齐 ops-qa-bot 的 markers 行为。"""
    markers = Markers()

    m = _ESCALATE_RE.search(text)
    if m:
        markers.escalate = m.group(1).strip()
    if _CLARIFY_RE.search(text):
        markers.clarify = True

    cleaned = _ESCALATE_RE.sub("", text)
    cleaned = _CLARIFY_RE.sub("", cleaned)
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
    # （auto 模式下即分诊台自答）。用于评测路由准确率。来自 lifecycle hooks 的精确转交链。
    route: str | None = None
    # 按 agent 名的 token 用量（lifecycle hooks 归账），如 {"triage": {...}, "redis_specialist":
    # {...}}。多模型路由（#2）下可拆分"分诊便宜模型 vs 专家强模型"各花多少。
    agent_usage: dict[str, dict[str, int]] | None = None
    # 二次复核（#7）元信息：是否复核过 / 是否触发了重答 / 是否标记需人工复核（B 兜底）。
    reviewed: bool = False
    revised: bool = False
    needs_human_review: bool = False


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
    # 最终落点 agent 名（handoff 目标）；None 表示入口 agent 自答。评测路由准确率用。
    # 来自 lifecycle hooks 的精确转交链（不再用 last_agent 推断）。
    route: str | None = None
    # 按 agent 名的 token 用量（lifecycle hooks 归账）。
    agent_usage: dict[str, dict[str, int]] | None = None
    # 二次复核（#7）元信息。
    reviewed: bool = False
    revised: bool = False
    needs_human_review: bool = False


@dataclass
class ApprovalRequest:
    """一次写操作审批请求（HITL interruption 的归一化视图）。"""

    tool_name: str
    arguments: dict[str, Any]


@dataclass
class GuardedAnswer:
    """`OpsQABot.answer_guarded()` 的返回值：答案 + 审批记录 + 护栏拦截信息。"""

    text: str
    # 从答案里解析出的标记（escalate/clarify），接入层渲染用（如飞书 @负责人）。
    markers: Markers = field(default_factory=Markers)
    # 本轮经过**人工**审批的请求及其结果（approved=True/False）。
    approvals: list[tuple[ApprovalRequest, bool]] = field(default_factory=list)
    # 命中禁止命令清单、审批前被自动驳回的请求及原因（没走到人工审批）。
    blacklist_rejections: list[tuple[ApprovalRequest, str]] = field(default_factory=list)
    # 被批准并登记执行的写命令（target/command/reason）。
    approved_writes: list[Any] = field(default_factory=list)
    # 命中输入注入护栏时为拦截原因；正常为 None。
    blocked: str | None = None
    usage: dict[str, Any] | None = None
    num_turns: int | None = None
    subtype: str = "success"
    # 按 agent 名的 token 用量（lifecycle hooks 归账）。
    agent_usage: dict[str, dict[str, int]] | None = None
    # 二次复核（#7）元信息。
    reviewed: bool = False
    revised: bool = False
    needs_human_review: bool = False


def build_user_input(question: str, images: list[tuple[str, bytes]] | None = None):
    """把 (问题, 图片) 组装成 `Runner.run(input=...)` 可接受的形态。

    没有图时返回原字符串（不改变历史行为）；有图时返回单条 user 消息的输入项列表，
    每张图是一个 `input_image` 块（data URI base64），文本放在图后——让模型先看到图、
    再读到引导问题，匹配"看图 → 答"的语序。与 ops-qa-bot（Claude 版）的 content
    blocks 组装一一对应，只是换成 Responses API 的块类型（要求底层模型支持视觉）。
    """
    if not images:
        return question
    content: list[dict[str, Any]] = [
        {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}",
            "detail": "auto",
        }
        for media_type, raw in images
    ]
    content.append({"type": "input_text", "text": question})
    return [{"role": "user", "content": content}]


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
    if name == "run_diagnostic":
        return f"run_diagnostic {args.get('host', '?')}: {args.get('command', '?')}"
    if name == "request_write_command":
        return f"request_write_command {args.get('target', '?')}: {args.get('command', '?')}"
    if name == DB_TOOL_NAME:
        sql = str(args.get("sql", "?"))
        if len(sql) > 60:
            sql = sql[:60] + "…"
        return f"query_database {args.get('db_type', '?')}@{args.get('host', '?')}: {sql}"
    if name == DB_CHANGE_TOOL_NAME:
        return (
            f"request_db_change {args.get('host', '?')}: "
            f"{args.get('param', '?')}={args.get('value', '?')}"
        )
    if name == GW_TRACE_TOOL_NAME:
        return f"query_gateway_trace {args.get('hi_trace_id', '?')}"
    if name == FOLLOWUP_TOOL_NAME:
        t = str(args.get("task", "?"))
        if len(t) > 60:
            t = t[:60] + "…"
        return f"schedule_followup +{args.get('delay_minutes', '?')}min: {t}"
    if name == DOC_QA_TOOL_NAME:
        q = str(args.get("question", "?"))
        if len(q) > 60:
            q = q[:60] + "…"
        return f"query_feishu_doc {args.get('component', '?')} ← {q}"
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
        session: Session | None = None,
        diag_config: DiagConfig | None = None,
        db_config: DbConfig | None = None,
        doc_qa_config: DocQAConfig | None = None,
        gw_trace_config: GatewayTraceConfig | None = None,
        review_config: ReviewConfig | None = None,
        followup_config: FollowupConfig | None = None,
        followup_submitter: Any = None,
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

        # 实时诊断（测试环境只读，OPS_QA_DIAG=1 开启）：缺省从环境读，None → from_env。
        # 开启时把 run_diagnostic 工具挂到答题 agent（single 的单 agent / 多 agent 的各专家），
        # 与写审批工具并列进 _extra_tools；关闭时零感知。写命令在诊断里被识别后会引导模型改走
        # request_write_command（需 guardrails 开），两者正交组合。见 diagnostics.py。
        self.diag_config = diag_config or DiagConfig.from_env()
        self.diagnostics = self.diag_config.enabled
        self.diag_log = DiagnosticLog()
        if self.diagnostics:
            self._extra_tools = self._extra_tools + [
                make_diagnostic_tool(self.diag_config, self.diag_log)
            ]

        # 数据库诊断（测试环境只读，OPS_QA_DB=1 开启）：query_database 用只读账号直连
        # 目标库跑诊断 SQL，与 run_diagnostic 并列进 _extra_tools（横切工具，挂到答题
        # agent / 各专家）。guardrails 开启时额外挂 request_db_change（needs_approval，
        # 参数变更走与 request_write_command 同一条审批闭环）。见 db_query.py。
        self.db_config = db_config or DbConfig.from_env()
        self.db = self.db_config.enabled
        self.db_log = DbQueryLog()
        self._has_db_change_tool = False
        if self.db:
            db_client = DatabaseClient(self.db_config)
            self._extra_tools = self._extra_tools + [
                make_query_database_tool(db_client, self.db_log)
            ]
            if guardrails:
                self._has_db_change_tool = True
                self._extra_tools = self._extra_tools + [
                    make_db_change_tool(db_client, self.write_log)
                ]

        # 定时跟进（OPS_QA_FOLLOWUP=1 且接入层注入了 submitter 才挂）：schedule_followup
        # 登记"过 N 分钟自动再查一次"，到点由接入层的定时器复用答题链路把结果推回群。
        # 横切工具（哪个组件的变更都可能要跟进），与写审批/诊断并列进 _extra_tools。
        # CLI 直用没有定时器（submitter=None），工具不挂、prompt 不加章节，零感知。
        # 见 followup.py / feishu/followup.py。
        self.followup_config = followup_config or FollowupConfig.from_env()
        self.followup = bool(self.followup_config.enabled and followup_submitter is not None)
        if self.followup:
            self._extra_tools = self._extra_tools + [
                make_schedule_followup_tool(self.followup_config, followup_submitter)
            ]

        # 飞书文档问答（OPS_QA_DOC_QA_BASE_URL 配了才开）：把「用飞书文档维护知识」的组件
        # 接进来。与 diagnostics/guardrails 的区别是它不是横切工具——它是某些组件的**唯一
        # 知识来源**，所以不进 _extra_tools，而是单独传给编排层：multi/auto/coordinator 下
        # 只挂到 feishu 组件的专家上（那些专家没有 read_doc/glob_docs/grep_docs），single 下
        # 与文档检索工具并列挂在同一个 agent 上、靠 prompt 区分来源。见 doc_qa.py。
        self.doc_qa_config = doc_qa_config or DocQAConfig.from_env()
        self.doc_qa = self.doc_qa_config.enabled
        self.doc_qa_log = DocQALog()
        self._feishu_tool = (
            make_feishu_doc_tool(self.doc_qa_config, self.doc_qa_log) if self.doc_qa else None
        )

        # 网关链路排查（OPS_QA_GW_TRACE=1 开启）：query_gateway_trace 按 Hi-Trace-Id 取一次
        # 请求的网关链路表。它既不是横切工具（只有网关专家用得上），也不是某组件的唯一知识
        # 来源（网关文档仍是本地 md）——所以走第三条路：**组件专属工具** scoped_tools，
        # 由编排层只挂到 INDEX.md 里那一个组件的专家上。见 gateway_trace.py / orchestration。
        self.gw_trace_config = gw_trace_config or GatewayTraceConfig.from_env()
        self.gw_trace = self.gw_trace_config.enabled
        self.gw_trace_log = GatewayTraceLog()
        self._scoped_tools: dict[str, list] = {}
        if self.gw_trace:
            self._scoped_tools[self.gw_trace_config.component] = [
                make_gateway_trace_tool(self.gw_trace_config, self.gw_trace_log)
            ]
            self._warn_if_scoped_component_missing()

        # 运行遥测（lifecycle hooks）：精确转交链 + 按 agent 的 token 归账。挂在 bot 上
        # 跨 run 复用（coordinator 的 as_tool 子 run 需要构建期注入同一实例），每次答题
        # reset_run() 清零。见 hooks.py。
        self._telemetry = RunTelemetry()

        # 转交剥噪音（run 级 handoff input_filter）：分诊 → 专家/协调者转交时，把对话里
        # 的工具调用/输出项（历史轮次专家 read_doc 的整篇文档 dump、分诊自己的检索）从
        # 新 agent 可见的输入里剥掉——多轮对话下这些是转交后最大的 token 负担；专家的
        # 正式来源在答案文本里保留，需要时它自己会重新 read_doc。只影响模型可见输入，
        # session 落盘的历史不动（SDK 语义）。所有 handoff 想要同一策略，故用 run 级
        # RunConfig.handoff_input_filter（未设 per-handoff filter 时全局生效）；
        # OPS_QA_HANDOFF_STRIP_TOOLS=0 可关（调试/评测 A/B 用）。
        self._run_config: RunConfig | None = (
            RunConfig(handoff_input_filter=remove_all_tools)
            if env_flag("OPS_QA_HANDOFF_STRIP_TOOLS", default=True)
            else None
        )

        self._agent: Agent[DocsContext]
        self.components: list[Component]
        self.model_router: ModelRouter | None = None
        # 护栏是横切关注点，与编排模式正交：输入注入护栏挂入口 agent，写审批工具挂各专家。
        # feishu_tool 由编排层按组件来源分配（只挂给 feishu 组件的专家）。
        gr = {
            "input_guardrails": self._input_guardrails,
            "specialist_extra_tools": self._extra_tools,
            "scoped_tools": self._scoped_tools,
            "feishu_tool": self._feishu_tool,
        }
        if mode == "coordinator":
            # 跨组件协作：协调者把各组件专家当工具（agents-as-tools）调用、综合根因。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_coordinator_agent(
                self.docs_root, self.model_router, agent_tool_hooks=self._telemetry, **gr
            )
        elif mode == "auto":
            # 自适应默认：分诊台 handoff 给单专家（常见）或跨组件协调者（少数）。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_auto_agent(
                self.docs_root, self.model_router, agent_tool_hooks=self._telemetry, **gr
            )
        elif mode == "multi":
            # 差异化 #3：入口换成分诊 agent，handoff 给从 INDEX.md 动态生成的组件专家。
            # 差异化 #2：用 ModelRouter 按角色/组件分配模型（分诊便宜、专家强）。
            self.model_router = model_router or build_model_router()
            self._agent, self.components = build_triage_agent(
                self.docs_root, self.model_router, **gr
            )
        else:  # single
            self.components = []
            self._agent = Agent(
                name="ops-qa-bot",
                instructions=build_system_prompt(
                    self.docs_root,
                    diagnostics=self.diagnostics,
                    has_write_tool=guardrails,
                    doc_qa=self.doc_qa,
                    db=self.db,
                    has_db_change_tool=self._has_db_change_tool,
                    gw_trace=self.gw_trace,
                    followup=self.followup,
                ),
                # single 模式只有一个 agent，"组件专属"无处可依附——链路工具与文档检索工具
                # 并列挂上，靠 prompt 章节区分何时用（同 query_feishu_doc 在 single 下的处境）。
                tools=(
                    list(DOC_TOOLS)
                    + self._extra_tools
                    + self._feishu_tools()
                    + self._all_scoped_tools()
                ),
                model=self.model_choice.model,
                model_settings=role_model_settings("single"),
                input_guardrails=self._input_guardrails,
            )
        self._context = DocsContext(docs_root=self.docs_root)
        # 多轮对话历史交给 SDK Session（Runner 自动读写）。默认进程内 SQLiteSession；
        # 接入层（如飞书）可注入带 session_id + 落盘 db 的持久化 session。
        # 流式 ask() 与结构化 answer_structured() 共享同一 session（同一段对话）。
        self._session: Session = session or SQLiteSession("default")
        # 结构化输出 agent 懒构造（仅 answer_structured 用到时才建）。
        self._structured_agent: Agent[DocsContext] | None = None

        # 二次复核（#7）：另一个模型对答案做证据核对，revise-once 后交付。缺省从环境读。
        # reviewer 用 model router 的 reviewer 角色（OPS_QA_REVIEWER_MODEL 可指到不同模型，
        # 降低同错同漏）；single 模式没有 router 时按需建一个（仅在开启复核时）。懒构造。
        self.review_config = review_config or ReviewConfig.from_env()
        self.review = self.review_config.enabled
        self._review_router: ModelRouter | None = (
            (self.model_router or build_model_router()) if self.review else None
        )
        self._reviewer_agent: Agent | None = None

    def _feishu_tools(self) -> list:
        """single 模式下 `query_feishu_doc` 与文档检索工具并列挂在同一个 agent 上（未开则空）。"""
        return [self._feishu_tool] if self._feishu_tool is not None else []

    def _all_scoped_tools(self) -> list:
        """把所有组件专属工具摊平（single 模式用：没有专家可挂，只能都挂在唯一的 agent 上）。"""
        return [t for tools in self._scoped_tools.values() for t in tools]

    def _warn_if_scoped_component_missing(self) -> None:
        """组件专属工具指向的目录不在 INDEX.md 里时告警。

        配错 `OPS_QA_GW_TRACE_COMPONENT` 的后果是**静默失效**：专家照常建出来，工具却没进去，
        问到网关时 agent 只会说"我查了文档没找到链路数据"——没有任何报错指向配置。真实部署里
        那一行未必叫 `gateway`，所以这条告警是必要的。single 模式不受影响（工具全挂在唯一
        agent 上），但配置写错仍值得提醒。
        """
        known = {c.dir for c in parse_index_components(self.docs_root)}
        for dir_ in self._scoped_tools:
            if dir_ not in known:
                logger.warning(
                    "组件专属工具指向的目录 %r 不在 INDEX.md 的组件表里（已登记：%s）；"
                    "%s 模式下该工具不会被任何专家挂载。请核对 OPS_QA_GW_TRACE_COMPONENT。",
                    dir_,
                    "、".join(sorted(known)) or "（无）",
                    self.mode,
                )

    async def reset(self) -> None:
        """清空会话上下文，开始新对话。"""
        await self._session.clear_session()

    # ------------------------------------------------------------------
    # 二次复核（#7）：另一个模型证据核对 + revise-once。见 review.py。
    # ------------------------------------------------------------------

    def _get_reviewer_agent(self) -> Agent:
        """懒构造复核者 agent（reviewer 角色的模型，无工具、typed ReviewVerdict）。"""
        if self._reviewer_agent is None:
            assert self._review_router is not None
            model = self._review_router.for_role(REVIEWER_ROLE)[1]
            self._reviewer_agent = build_reviewer_agent(model)
        return self._reviewer_agent

    def _evidence_mark(self) -> _EvidenceMark:
        """快照各证据日志的长度，用于把复核证据限定在"本轮之后新增的"。"""
        return _EvidenceMark(
            diag=len(self.diag_log.results),
            doc_qa=len(self.doc_qa_log.calls),
            db=len(self.db_log.results),
            gw_trace=len(self.gw_trace_log.records),
        )

    def _ran_realtime(self, mark: _EvidenceMark) -> bool:
        """本轮是否产生了实时证据（诊断命令 / 数据库查询 / 网关链路）——决定答案是否算 risky。

        链路查询是只读的，但它和 `run_diagnostic` 一样让答案的结论**建立在实时数据上**：
        复核两轮都不过时该转人工（兜底 B），而不是带个 ⚠️ 注解就发出去（兜底 A）。
        """
        return (
            len(self.diag_log.results) > mark.diag
            or len(self.db_log.results) > mark.db
            or len(self.gw_trace_log.records) > mark.gw_trace
        )

    async def _review(
        self, question: str, draft1: Draft, reanswer: Any, mark: _EvidenceMark
    ) -> ReviewOutcome:
        """对一版答案跑 revise-once 复核。不合格（反问/无可核对证据）时原样返回。

        证据 = 引用文档实际内容 + **本轮**（`mark` 之后）的诊断输出、数据库查询输出、网关
        链路表与飞书文档答案，reviewer 只在此范围内核对（见 review.gather_evidence）。飞书
        来源没有本地文件可读，必须把工具实际拿回的 markdown 一并喂进去，否则 reviewer 会对
        每条飞书引用读到 `[未找到]`。链路表同理：模型很容易扫一眼就下一个表里没有的结论
        （"网关没匹配到路由"），不把原始表喂给 reviewer 就核对不出来。
        """
        if not is_review_eligible(draft1.citations, draft1.risky, draft1.clarify):
            return ReviewOutcome(final=draft1)

        def evidence_fn(d: Draft) -> str:
            outs = [r.output for r in self.diag_log.results[mark.diag :] if r.output]
            outs += self.db_log.outputs(mark.db)
            outs += self.gw_trace_log.outputs(mark.gw_trace)
            return gather_evidence(
                self.docs_root,
                d.citations,
                outs,
                feishu_answers=self.doc_qa_log.answers(mark.doc_qa),
                max_chars=self.review_config.max_evidence_chars,
            )

        outcome = await review_and_revise(
            agent=self._get_reviewer_agent(),
            question=question,
            draft1=draft1,
            evidence_fn=evidence_fn,
            reanswer=reanswer,
        )
        if outcome.reviewed:
            logger.info(
                "review: verdict1=%s revised=%s verdict2=%s escalate=%s caveat=%s",
                getattr(outcome.verdict1, "verdict", None),
                outcome.revised,
                getattr(outcome.verdict2, "verdict", None),
                outcome.escalate,
                bool(outcome.caveat),
            )
        return outcome

    def _apply_review_outcome(self, outcome: ReviewOutcome) -> Any:
        """把复核结果落到最终结果对象上：附注解（A）/ 标记需人工复核（B）/ 记元信息。"""
        payload = outcome.final.payload
        if not outcome.reviewed:
            return payload
        payload.reviewed = True
        payload.revised = outcome.revised
        if outcome.caveat:  # A：低风险，带注解交付，把判断交回给人。
            self._append_answer_text(
                payload,
                "⚠️ **复核提示**：以下几点经二次复核后仍存疑，请留意核对：\n" + outcome.caveat,
            )
        if outcome.escalate:  # B：涉及诊断/写，复核仍不过 → 标记需人工复核。
            payload.needs_human_review = True
            v2 = outcome.verdict2
            findings = format_findings(v2.findings) if v2 else ""
            self._append_answer_text(
                payload,
                "⚠️ **二次复核未通过（涉及实时诊断/写操作）**：\n"
                + findings
                + "\n\n**建议人工复核确认后再执行相关操作。**",
            )
        return payload

    @staticmethod
    def _append_answer_text(payload: Any, note: str) -> None:
        """把复核注解追加到答案正文：结构化写进 contract.answer，其余写进 text。"""
        block = f"\n\n---\n{note}"
        if isinstance(payload, StructuredAnswer):
            payload.contract.answer = (payload.contract.answer or "") + block
        else:
            payload.text = (payload.text or "") + block

    def _get_structured_agent(self) -> Agent[DocsContext]:
        """懒构造结构化输出图：结构化输出与路由**正交**——按 self.mode 复用同一套编排图，
        只是终端 agent 的 `output_type=AnswerContract`（multi/auto 是各专家、coordinator 是
        协调者、single 是单 agent）。

        用非严格 schema（strict_json_schema=False）下发，以兼容 Claude / 智谱 / 火山等
        不支持 OpenAI strict 结构化输出的 provider。护栏：结构化走 `answer_structured`
        （非流式、无写审批中断循环），故只挂输入注入护栏 + 输出来源护栏，不挂写审批工具。

        `scoped_tools` **要挂**：链路排查是只读的、不触发审批中断，结构化路径正是评测
        （`evaluate.py` 走 `answer_structured`）跑的那条，不挂它就没法评测带 Hi-Trace-Id
        的题。这与"不挂写审批工具"不矛盾——排除的是会挂起 run 的 needs_approval 工具。
        """
        if self._structured_agent is not None:
            return self._structured_agent
        # 用围栏容忍版：GLM 等会把 JSON 裹在 ```json 围栏里 + 字符串里塞裸换行，剥围栏 +
        # 宽松解析后再校验（strict=False 的 provider 常见）。
        out = FenceTolerantOutputSchema(AnswerContract, strict_json_schema=False)
        og = [citation_output_guardrail] if self.guardrails else []
        gr: dict[str, Any] = {
            "input_guardrails": self._input_guardrails,
            "scoped_tools": self._scoped_tools,
            "feishu_tool": self._feishu_tool,
            "output_type": out,
            "output_guardrails": og,
        }
        if self.mode == "coordinator":
            agent, _ = build_coordinator_agent(
                self.docs_root,
                self.model_router or build_model_router(),
                agent_tool_hooks=self._telemetry,
                **gr,
            )
        elif self.mode == "auto":
            agent, _ = build_auto_agent(
                self.docs_root,
                self.model_router or build_model_router(),
                agent_tool_hooks=self._telemetry,
                **gr,
            )
        elif self.mode == "multi":
            agent, _ = build_triage_agent(
                self.docs_root, self.model_router or build_model_router(), **gr
            )
        else:  # single：单 agent + whole-docs 契约 prompt
            agent = Agent(
                name="ops-qa-bot-structured",
                instructions=build_structured_system_prompt(
                    self.docs_root, doc_qa=self.doc_qa, gw_trace=self.gw_trace
                ),
                tools=list(DOC_TOOLS) + self._feishu_tools() + self._all_scoped_tools(),
                model=self.model_choice.model,
                model_settings=role_model_settings("single"),
                output_type=out,
                input_guardrails=self._input_guardrails,
                output_guardrails=og,
            )
        self._structured_agent = agent
        return agent

    def _run_kwargs(self) -> dict[str, Any]:
        # session：SDK 自动做多轮历史的读取与落盘（run 前取历史拼 input、run 后存新 items）。
        # hooks：运行遥测（转交链 / 按 agent 用量），调用方在 run 前 reset_run()。
        # run_config：转交剥噪音的 handoff input_filter（见 __init__）。
        kwargs: dict[str, Any] = {
            "context": self._context,
            "session": self._session,
            "hooks": self._telemetry,
        }
        if self._run_config is not None:
            kwargs["run_config"] = self._run_config
        if self.max_turns is not None:
            kwargs["max_turns"] = self.max_turns
        return kwargs

    def _resume_kwargs(self) -> dict[str, Any]:
        """审批中断续跑（`Runner.run(agent, state)`）用的 kwargs：同 `_run_kwargs` 但
        **不传 context**。

        关键：RunState 自带 context（含 DocsContext 与 `state.approve()/reject()` 记的
        审批决定）。若 resume 再传 `context=`，SDK 的 `resolve_resumed_context` 会用一个
        新 RunContextWrapper 覆盖 `run_state._context`——审批决定被冲掉，工具永远得不到
        批准、模型反复重提议直到撞轮次保险丝。session/hooks/run_config 仍要保留。
        """
        kwargs = self._run_kwargs()
        kwargs.pop("context", None)
        return kwargs

    async def ask(
        self,
        question: str,
        images: list[tuple[str, bytes]] | None = None,
    ) -> AsyncIterator[dict]:
        """向 bot 提问，流式返回事件字典：

        - {"type": "tool", "name": str, "input": dict}  —— agent 调用的工具
        - {"type": "text", "text": str}                 —— 回答文本片段
        - {"type": "handoff", "agent": str}             —— 转交给某 agent（实时展示用）
        - {"type": "done", "usage": dict | None, "num_turns": int | None,
           "subtype": str, "route": str | None,
           "agent_usage": dict}                          —— 本轮结束（含遥测）

        `images` 给定时（list of (media_type, raw_bytes)），把每张图作为 input_image
        块 + 文本一起发给模型（要求底层模型/代理支持视觉）。没有 images 时走原
        string 路径，不改变历史行为。
        """
        self._telemetry.reset_run()
        result = Runner.run_streamed(
            self._agent, input=build_user_input(question, images), **self._run_kwargs()
        )

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
            # 历史落盘由 session 负责：已完成的 turns 在流式过程中逐轮持久化，撞上限
            # 也保留已产出的部分上下文（与 ops-qa-bot 保留上下文的做法一致）。
            subtype = "error_max_turns"
        finally:
            # run_streamed 的 run 在 SDK 内部的后台 task 里跑，消费方被取消/提前
            # 关闭（如 /cancel 取消在途提问）时它不会自动停——不停掉会白烧 token。
            # 正常收尾 is_complete=True，cancel 是 no-op。
            if not result.is_complete:
                result.cancel()

        yield {
            "type": "done",
            "usage": _usage_dict(result),
            "num_turns": _num_turns(result),
            "subtype": subtype,
            # 遥测（lifecycle hooks）：精确路由 + 按 agent 用量。
            "route": self._telemetry.route,
            "agent_usage": self._telemetry.agent_usage(),
        }

    async def answer(
        self,
        question: str,
        images: list[tuple[str, bytes]] | None = None,
    ) -> AnswerResult:
        """一次性返回完整答案 + 用量 + 解析出的标记。开启复核（#7）时叠加 revise-once 核对。

        `images` 透传给 `ask()`，开启视觉路径。复核重答（reanswer）保持纯文本——
        图已随首答进了 session 历史，重答的模型仍能看到。
        """
        if not self.review:
            return await self._answer_once(question, images)
        mark = self._evidence_mark()
        r1 = await self._answer_once(question, images)

        def to_draft(r: AnswerResult) -> Draft:
            return Draft(
                text=r.text,
                citations=extract_citations(r.text),
                risky=self._ran_realtime(mark),
                clarify=r.markers.clarify,
                payload=r,
            )

        async def reanswer(note: str) -> Draft:
            return to_draft(await self._answer_once(note))

        outcome = await self._review(question, to_draft(r1), reanswer, mark)
        return self._apply_review_outcome(outcome)

    async def _answer_once(
        self,
        question: str,
        images: list[tuple[str, bytes]] | None = None,
    ) -> AnswerResult:
        """产出一版自由文本答案（不含复核）。复核开启时被 answer() 调用两次（原答 + 重答）。"""
        if images:
            logger.info("question (with %d image(s)): %s", len(images), question)
        else:
            logger.info("question: %s", question)
        chunks: list[str] = []
        usage: dict | None = None
        num_turns: int | None = None
        subtype = "success"
        route: str | None = None
        agent_usage: dict[str, dict[str, int]] | None = None
        async for event in self.ask(question, images=images):
            if event["type"] == "tool":
                logger.info("  tool: %s", format_tool_call(event["name"], event["input"]))
            elif event["type"] == "handoff":
                logger.info("  → 转交给 %s", event["agent"])
            elif event["type"] == "text":
                chunks.append(event["text"])
            elif event["type"] == "done":
                usage = event.get("usage")
                num_turns = event.get("num_turns")
                subtype = event.get("subtype", "success")
                # 路由 / 按 agent 用量取自 done 事件里的遥测（lifecycle hooks，精确）。
                route = event.get("route")
                agent_usage = event.get("agent_usage") or None
        cleaned, markers = parse_markers("".join(chunks))
        return AnswerResult(
            text=cleaned,
            markers=markers,
            usage=usage,
            num_turns=num_turns,
            subtype=subtype,
            route=route,
            agent_usage=agent_usage,
        )

    async def answer_structured(self, question: str) -> StructuredAnswer:
        """结构化输出模式（差异化 #1）。开启复核（#7）时叠加 revise-once 核对。"""
        if not self.review:
            return await self._answer_structured_once(question)
        mark = self._evidence_mark()
        r1 = await self._answer_structured_once(question)

        def to_draft(r: StructuredAnswer) -> Draft:
            c = r.contract
            return Draft(
                text=c.answer,
                citations=list(c.citations),
                risky=self._ran_realtime(mark),
                # 只复核真答题（decision=answer）；反问/拒绝/升级不下事实结论，clarify=True 跳过。
                clarify=c.decision != Decision.answer,
                payload=r,
            )

        async def reanswer(note: str) -> Draft:
            return to_draft(await self._answer_structured_once(note))

        outcome = await self._review(question, to_draft(r1), reanswer, mark)
        return self._apply_review_outcome(outcome)

    async def _answer_structured_once(self, question: str) -> StructuredAnswer:
        """结构化输出模式（差异化 #1）：返回强类型 AnswerContract + 来源真实性校验。

        与 `answer()` 不同，这里 agent 的 `output_type=AnswerContract`，SDK 强制模型
        按 schema 产出（不合法会重试），路由决策（answer/clarify/escalate/reject）、
        来源、追问都是类型字段而非文本标记。拿到契约后再用代码逐条核对 citations
        是否指向真实存在的文档——把"答案必须引用真实文档"从 prompt 自律变成硬校验。

        注意：结构化输出不便逐 token 流式，这里走非流式 `Runner.run`。多轮历史与
        流式 `ask()` 共享同一 `self._session`。
        """
        logger.info("question (structured): %s", question)
        agent = self._get_structured_agent()
        subtype = "success"
        self._telemetry.reset_run()
        try:
            result = await Runner.run(agent, input=question, **self._run_kwargs())
        except MaxTurnsExceeded:
            # 撞上限时没有合法契约可返回，退化成一个 reject 契约并标记 subtype。
            return StructuredAnswer(
                contract=AnswerContract(
                    decision=Decision.reject,
                    answer="本轮检索步数过多被中断（max_turns），结论可能不完整，请换个问法或缩小范围。",
                ),
                subtype="error_max_turns",
            )
        except ModelBehaviorError as e:
            # 模型多次未能产出合法契约 JSON（部分 provider 的 json_schema 实现不规范，容错
            # 解析后仍失败）。退化成 reject 契约，别让单题掀翻批量评测。
            logger.warning("结构化输出解析失败（ModelBehaviorError）: %s", str(e)[:200])
            return StructuredAnswer(
                contract=AnswerContract(
                    decision=Decision.reject,
                    answer="模型未能产出合法的结构化契约（provider 的 json_schema 输出不规范），已跳过本题。",  # noqa: E501
                ),
                subtype="error_bad_output",
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

        invalid = validate_citations(self.docs_root, contract.citations)
        if invalid:
            logger.warning("结构化答案引用了不存在/越界的来源: %s", invalid)
        return StructuredAnswer(
            contract=contract,
            invalid_citations=invalid,
            usage=_usage_dict(result),
            num_turns=_num_turns(result),
            subtype=subtype,
            # 路由取自 lifecycle hooks 的精确转交链（此前非流式只能用 last_agent !=
            # 入口 agent 反推，hooks 让流式/非流式统一走同一来源）。
            route=self._telemetry.route,
            agent_usage=self._telemetry.agent_usage() or None,
        )

    async def answer_guarded(
        self,
        question: str,
        approver: Any = None,
        images: list[tuple[str, bytes]] | None = None,
    ) -> GuardedAnswer:
        """带护栏 + 写审批（#4）的问答；开启复核（#7）时叠加 revise-once 核对。

        `images` 开启视觉路径（同 `answer()`）；复核重答保持纯文本，图在 session 历史里。
        """
        if not self.guardrails:
            raise RuntimeError("answer_guarded 需要以 guardrails=True 构造 OpsQABot")
        if not self.review:
            return await self._answer_guarded_once(question, approver, images)
        mark = self._evidence_mark()
        r1 = await self._answer_guarded_once(question, approver, images)

        def to_draft(r: GuardedAnswer) -> Draft:
            # risky：跑了实时诊断，或提议过写命令（审批/黑名单驳回/已登记任一非空）。
            write_proposed = bool(r.approvals or r.blacklist_rejections or r.approved_writes)
            return Draft(
                text=r.text,
                citations=extract_citations(r.text),
                risky=self._ran_realtime(mark) or write_proposed,
                clarify=r.markers.clarify,
                payload=r,
            )

        async def reanswer(note: str) -> Draft:
            return to_draft(await self._answer_guarded_once(note, approver))

        outcome = await self._review(question, to_draft(r1), reanswer, mark)
        return self._apply_review_outcome(outcome)

    async def _answer_guarded_once(
        self,
        question: str,
        approver: Any = None,
        images: list[tuple[str, bytes]] | None = None,
    ) -> GuardedAnswer:
        """带护栏 + 写操作审批（HITL）的问答（差异化 #4，需 guardrails=True）。

        - 输入注入护栏命中 → 直接返回拦截结果（blocked）。
        - agent 想调写操作工具 → run 暂停抛 interruption；用 `approver(req)->bool`
          决定批准/驳回，再 `Runner.run(agent, state)` 续跑。approver 为 None 时一律
          驳回（安全默认：无人值守不放行写操作）。
        - **approver 可以是异步的**（返回 awaitable 即 await）：run 挂起在中断循环里等
          远端拍板——飞书接入用它实现"发审批卡片 → 等值班人点按钮 → 续跑"的 HITL
          闭环（超时/白名单在接入层的 approver 里处理，见 feishu/approvals.py）。
        - **禁止命令短路**：命令命中禁止清单（`detect_forbidden_command`）时不打扰
          审批人、直接驳回并记入 `blacklist_rejections`。工具自身还挂了 tool-level
          guardrail 兜底（即使误批也执行不到），见 guardrails.py 的三层分级。

        这套 pause→approve→resume 走 SDK 的 RunState，是 ops-qa-bot 用 hook + 飞书
        回调拼出来那套 HITL 的一等公民替代。
        """
        if not self.guardrails:
            raise RuntimeError("answer_guarded 需要以 guardrails=True 构造 OpsQABot")
        logger.info(
            "question (guarded%s): %s",
            f", with {len(images)} image(s)" if images else "",
            question,
        )
        approvals: list[tuple[ApprovalRequest, bool]] = []
        blacklist_rejections: list[tuple[ApprovalRequest, str]] = []
        # reset 只在整轮开始时做一次：审批后的续跑属于同一轮，遥测继续累计。
        self._telemetry.reset_run()
        try:
            result = await Runner.run(
                self._agent, input=build_user_input(question, images), **self._run_kwargs()
            )
            # 处理写操作审批中断，直到没有待批项（轮次保险丝防模型驳回后无限重提议）。
            rounds = 0
            while result.interruptions:
                if rounds >= MAX_APPROVAL_ROUNDS:
                    logger.warning("审批循环达 %d 轮仍有待批项，终止本轮", rounds)
                    return GuardedAnswer(
                        text=(
                            "写操作提议被驳回后模型反复重试，本轮已终止。"
                            "请把诉求改成咨询处理方案，或联系运维人工执行。"
                        ),
                        approvals=approvals,
                        blacklist_rejections=blacklist_rejections,
                        subtype="error_approval_loop",
                        agent_usage=self._telemetry.agent_usage() or None,
                    )
                rounds += 1
                state = result.to_state()
                for itr in result.interruptions:
                    req = ApprovalRequest(
                        tool_name=getattr(itr, "name", "?"),
                        arguments=_interruption_args(itr),
                    )
                    # 禁止清单短路：毁灭性命令不进人工审批，直接驳回并记录原因。
                    # rejection_message 明确告知"永不批准、别重试"，让模型收敛到文字建议。
                    forbidden = detect_forbidden_command(str(req.arguments.get("command", "")))
                    if forbidden:
                        logger.warning("写命令命中禁止清单（%s），审批前自动驳回", forbidden)
                        blacklist_rejections.append((req, forbidden))
                        state.reject(
                            itr, rejection_message=_BLACKLIST_REJECTION_MSG.format(label=forbidden)
                        )
                        continue
                    # 参数变更提议的发卡前校验（与禁止清单短路同一姿态）：参数/值/连接
                    # 信息确定性校验不过的提议直接驳回、不打扰审批人；带 config 时连
                    # host 白名单一起查。工具执行体内还会再校验一遍（纵深防御）。
                    if req.tool_name == DB_CHANGE_TOOL_NAME:
                        bad = validate_change_args(req.arguments, self.db_config)
                        if bad:
                            logger.warning("参数变更提议校验不过，审批前自动驳回：%s", bad)
                            blacklist_rejections.append((req, bad))
                            state.reject(
                                itr,
                                rejection_message=(
                                    f"该参数变更提议未通过校验（{bad}），已自动驳回、未进入"
                                    "人工审批。若能修正（如补齐/改正连接信息、参数名、目标值）"
                                    "请修正后重新提议；否则按「写操作建议输出格式」给文字建议"
                                    "由 DBA 人工执行。"
                                ),
                            )
                            continue
                    if approver is None:
                        approved = False
                    else:
                        decision = approver(req)
                        if inspect.isawaitable(decision):
                            decision = await decision  # 异步 approver：挂起等远端拍板
                        approved = bool(decision)
                    approvals.append((req, approved))
                    if approved:
                        state.approve(itr)
                    else:
                        state.reject(itr, rejection_message=_HUMAN_REJECTION_MSG)
                # 续跑：RunState 已携带 context（含审批决定），不能再传 context 覆盖它
                # （见 _resume_kwargs）；session/hooks/filter 保留，max_turns 以 state 为准。
                result = await Runner.run(self._agent, state, **self._resume_kwargs())
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
                blacklist_rejections=blacklist_rejections,
                subtype="error_max_turns",
            )

        text = (
            result.final_output
            if isinstance(result.final_output, str)
            else str(result.final_output or "")
        )
        cleaned, markers = parse_markers(text)
        approved_writes = [r for r in self.write_log.requests if r.approved]
        return GuardedAnswer(
            text=cleaned,
            markers=markers,
            approvals=approvals,
            blacklist_rejections=blacklist_rejections,
            approved_writes=approved_writes,
            usage=_usage_dict(result),
            num_turns=_num_turns(result),
            agent_usage=self._telemetry.agent_usage() or None,
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
