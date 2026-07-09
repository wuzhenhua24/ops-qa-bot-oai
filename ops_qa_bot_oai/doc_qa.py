"""飞书文档问答：`query_feishu_doc(component, question)` 工具。

部分组件负责人用飞书文档而不是本地 markdown 维护运维知识。另一个内部系统提供
`POST /doc_qa`：传一组飞书 doc token + 问题，返回 markdown 答案（它内部自己跑 agent
读文档和图）。本模块把它包成主 agent 的一个进程内工具，让 agent 像查本地文档一样查
飞书文档——升级 / 反问 / 复核 / 护栏那套现有管线一行不用改。

移植自 `ops-qa-bot`（Claude Agent SDK 版）的 doc_qa.py。两处结构差异：

1. **`@function_tool` 取代进程内 MCP server**：Claude SDK 要 `create_sdk_mcp_server` +
   `@tool` 把工具包成 MCP server（agent 侧全名 `mcp__docqa__query_feishu_doc`）；
   OpenAI Agents SDK 直接挂 python 函数，工具名就是函数名，返回值就是给模型的字符串。

2. **注册表不再自己解析**：doc token 从共享的 `index.feishu_registry` 取（见 index.py），
   与 orchestration 建专家、schema 校验来源用的是同一份解析结果。

从参考项目原样保留的两条设计（都是安全性的，不是风格问题）：

- **agent 只传组件名，doc token 由代码查注册表解析**。token 是登记在 INDEX.md 里的权威
  数据，agent 不碰原始 token——防写错，更防被文档内容注入诱导去拉任意飞书文档（和
  `tools._resolve_within` 不信任模型给的路径、`diagnostics.validate_host` 不信任模型给的
  机器名是同一套防御姿态：凡是模型能影响的标识，都不直接当资源定位符用）。
- **工具失败返回文字提示而不抛异常**。抛异常会打断 agent 这一轮；返回一段「取不到飞书
  文档，请按升级规则通知负责人」的提示，让 agent 自己决定 ESCALATE。

姿态与 diagnostics 一致：**未配 `OPS_QA_DOC_QA_BASE_URL` 时整个特性关**（工具不挂载、
prompt 不加章节、feishu 组件不建专家），零感知。但**没有 mock 模式**——理由见 `DocQAConfig`。
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from agents import RunContextWrapper, function_tool

from .index import Component, feishu_citation, feishu_registry, norm_key
from .tools import DocsContext

logger = logging.getLogger(__name__)

# 工具名常量：orchestration / bot 按名字判断专家挂了哪些横切工具（与 DIAG_TOOL_NAME 同理）。
DOC_QA_TOOL_NAME = "query_feishu_doc"

_DEFAULT_TIMEOUT = 60.0

# 给 agent 的兜底引导：让它按「升级规则」回"找不到 + 通知负责人"，而不是干等或编答案。
# 措辞是 agent-facing 的，不是给终端用户看的。
_ESCALATE_HINT = (
    "未能取得该组件的飞书文档内容。请按「升级规则」处理：回复文档中未找到相关内容，"
    "并通知该组件负责人（自由文本模式输出 <<ESCALATE:ou_xxx:目录>>；结构化模式填 "
    "decision=escalate + escalate_to/escalate_dir）。不要凭常识编答案。"
)


class DocQAError(Exception):
    """`/doc_qa` 调用失败。`agent_hint` 是给主 agent 看的引导文字。"""

    def __init__(self, log_detail: str, agent_hint: str = _ESCALATE_HINT):
        super().__init__(log_detail)
        self.agent_hint = agent_hint


# ---------------------------------------------------------------------------
# 配置（缺省关，从环境变量解析）
# ---------------------------------------------------------------------------


@dataclass
class DocQAConfig:
    """外部「飞书文档问答」服务（`POST {base_url}/doc_qa`）的接入配置。

    `base_url` 为空时整个特性关闭，让没有飞书文档需求的部署零感知。
    `token` 为空表示对端没开鉴权（仅可信内网）。`timeout` 必须 ≥ 对端最坏耗时——/doc_qa
    内部还要跑 agent + 拉文档图，比纯文本接口慢得多，留足余量，避免外层 agent 等不到
    结果误判失败。

    **刻意没有 mock 模式**（对比 `DiagConfig.mock`）：诊断的模拟数据是假的 `free -h` 输出，
    无害；而飞书文档问答的模拟数据是**假的知识库**——正是这个 bot 存在的意义所要防的东西。
    实测也印证了这点：任何如实标注"这不是真文档"的模拟内容，都会（正确地）触发 prompt 里的
    防幻觉条款让 agent 拒答并升级，链路照样跑不通；而不标注的模拟内容则会被当成真答案。
    没有上游服务时就让这个特性关着，别造一个假的知识来源。
    """

    base_url: str | None = None
    token: str | None = None
    timeout: float = _DEFAULT_TIMEOUT

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @classmethod
    def from_env(cls) -> DocQAConfig:
        """从 `OPS_QA_DOC_QA_*` 环境变量解析。缺省关。"""
        base_url = (os.environ.get("OPS_QA_DOC_QA_BASE_URL") or "").strip()
        raw_timeout = (os.environ.get("OPS_QA_DOC_QA_TIMEOUT") or "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else _DEFAULT_TIMEOUT
        except ValueError:
            timeout = _DEFAULT_TIMEOUT
        return cls(
            # 末尾斜杠归一化：拼 /doc_qa 时不想撞出 //doc_qa。
            base_url=base_url.rstrip("/") or None,
            token=(os.environ.get("OPS_QA_DOC_QA_TOKEN") or "").strip() or None,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


class DocQAClient:
    """`POST {base_url}/doc_qa` 的轻量 httpx 客户端。

    `transport` 仅供单测注入 `httpx.MockTransport`；生产留 None 走默认网络。
    """

    def __init__(self, config: DocQAConfig, transport: httpx.AsyncBaseTransport | None = None):
        if not config.base_url:
            raise ValueError("DocQAClient 需要非空 base_url")
        self._url = f"{config.base_url}/doc_qa"
        self._token = config.token
        self._timeout = config.timeout
        self._transport = transport

    async def ask(self, docs: list[str], question: str, req_id: str | None = None) -> str:
        """调 /doc_qa 拿 markdown 答案。失败抛 `DocQAError`（带 agent_hint）。"""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload: dict = {"docs": docs, "q": question}
        if req_id:
            payload["req_id"] = req_id
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                resp = await client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise DocQAError(f"doc_qa timeout: {e!r}") from e
        except httpx.HTTPError as e:
            raise DocQAError(f"doc_qa connect error: {e!r}") from e

        try:
            body: dict = resp.json() if resp.content else {}
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}

        if resp.status_code == 200 and body.get("ok"):
            answer = body.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
            raise DocQAError(f"doc_qa 200 but empty answer: req_id={body.get('req_id')}")

        # 失败分类。401/422 是配置错（鉴权 / docs 为空），打 error 提醒运维核对；
        # 5xx 及其它运行期错误打 warning。两类给 agent 的引导都是「按升级规则走」。
        detail = (
            f"doc_qa failed: status={resp.status_code} "
            f"req_id={body.get('req_id')} error={body.get('error')}"
        )
        if resp.status_code in (401, 422):
            logger.error("%s（检查 OPS_QA_DOC_QA_* 配置 / INDEX.md 的飞书文档登记）", detail)
        else:
            logger.warning(detail)
        raise DocQAError(detail)


# ---------------------------------------------------------------------------
# 调用日志（供二次复核取证据 / 观测）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocQACall:
    """一次成功的飞书文档问答。"""

    component: str  # 组件名（INDEX.md 的「组件」列）
    question: str
    answer: str


@dataclass
class DocQALog:
    """收集本会话拿到的飞书文档答案。

    存在的理由不只是观测：`review.gather_evidence` 复核答案时要读"引用来源的实际内容"，
    而飞书来源没有本地文件可读。把工具拿到的 markdown 记在这里当证据喂给复核者，复核才
    是真的在核对证据，而不是看到 `[未找到]` 就误判"引用不实"。
    """

    calls: list[DocQACall] = field(default_factory=list)

    def answers(self, since: int = 0) -> dict[str, str]:
        """`since` 之后按组件聚合的答案，喂给 `review.gather_evidence`。

        同一组件被问了多轮（反问回填后重问）时把各轮拼起来——复核者要看到答案实际依据过的
        全部内容，只留最后一轮会漏掉前几轮已经进了答案的事实。
        """
        merged: dict[str, str] = {}
        for c in self.calls[since:]:
            block = f"（问：{c.question}）\n{c.answer}"
            merged[c.component] = f"{merged[c.component]}\n\n{block}" if c.component in merged else block
        return merged


# ---------------------------------------------------------------------------
# 纯核心 + function_tool 包装
# ---------------------------------------------------------------------------

# 客户端协议：`async ask(docs, question, req_id) -> str`。单测注入自己的假实现。
_Client = DocQAClient


def render_answer(component: Component, answer: str) -> str:
    """把上游答案渲染成给模型的文本，抬头就是它该照抄的来源标识。

    与 `diagnostics.render_result` 用 `（实时数据：host）` 抬头同理：把"这段内容该怎么标
    来源"直接摆在模型眼前，比在 prompt 里讲一遍更不容易漂。
    """
    return f"（来源：{feishu_citation(component.name)}）\n\n{answer}"


def _unknown_component_hint(component: str, registry: dict[str, Component]) -> str:
    known = "、".join(sorted({c.name for c in registry.values()})) or "（无）"
    return (
        f"组件「{component}」未登记为飞书文档来源。当前飞书来源组件：{known}。"
        "请确认组件名；若该组件是本地文档来源，改用 read_doc / glob_docs / grep_docs。"
    )


async def query_feishu_doc_core(
    client: _Client,
    log: DocQALog,
    docs_root: Path,
    component: str,
    question: str,
) -> str:
    """工具的纯核心：查注册表 → 调上游 → 返回 markdown。脱离 SDK 可直接 await 单测。

    注册表每次调用都重新查（`parse_index_components` 有 mtime 缓存，开销很小），保证改了
    INDEX.md 的登记后无需重启进程。任何失败都**返回文字**而不抛——抛异常会打断 agent 这
    一轮，返回引导则让它自己按升级规则处理。
    """
    component = component.strip()
    question = question.strip()
    if not question:
        return "调用缺少 question 参数，无法查询。请给出一个自包含的完整问题。"

    registry = feishu_registry(docs_root)
    entry = registry.get(norm_key(component))
    if entry is None:
        return _unknown_component_hint(component, registry)

    req_id = uuid.uuid4().hex[:12]
    logger.info(
        "doc_qa call: component=%s req_id=%s docs=%d q_len=%d",
        entry.name,
        req_id,
        len(entry.docs),
        len(question),
    )
    try:
        answer = await client.ask(list(entry.docs), question, req_id=req_id)
    except DocQAError as e:
        logger.warning("doc_qa call failed: req_id=%s %s", req_id, e)
        return e.agent_hint
    logger.info("doc_qa ok: component=%s req_id=%s ans_len=%d", entry.name, req_id, len(answer))
    log.calls.append(DocQACall(component=entry.name, question=question, answer=answer))
    return render_answer(entry, answer)


def make_feishu_doc_tool(config: DocQAConfig, log: DocQALog, client: _Client | None = None):
    """构造 `query_feishu_doc(component, question)` 工具。`client` 仅供单测注入。"""
    resolved = client or DocQAClient(config)

    @function_tool
    async def query_feishu_doc(
        ctx: RunContextWrapper[DocsContext], component: str, question: str
    ) -> str:
        """查询「用飞书文档维护的组件」的运维知识（这类组件在本地没有 md 文件）。

        INDEX.md 里某组件的「来源」列是 `feishu` 时，用本工具代替 read_doc / glob_docs /
        grep_docs（对这类组件用文档检索工具查不到东西，白费一轮）。返回该组件飞书文档里
        的答案 markdown；取不到内容时返回升级提示。

        Args:
            component: INDEX.md 里该组件的「组件」列名（如 `Nginx`）。**不要传 doc token**
                ——token 由系统按组件名解析，你既不用也拿不到。
            question: 一个**自包含的完整问题**。本服务没有对话记忆，用户追问（"那它怎么
                回滚"）时要把前几轮的关键上下文（组件、版本、报错、已确认的环境）折进这
                一条 question，不能只发依赖上文的半句。
        """
        return await query_feishu_doc_core(
            resolved, log, ctx.context.docs_root, component, question
        )

    return query_feishu_doc
