"""二次复核（差异化 #7）：另一个模型对答案做**证据核对**，revise-once 后交付。

在文档问答 / 实时诊断的基础上，再配一个**复核者**（reviewer，另一个模型）对答案做一遍
二次核对。担心"复核没完没了、达不成共识"是设计问题、不是固有属性——只要不把它做成
"两个模型平等谈判"，就必然收敛。本模块用三条纪律根除发散：

1. **锚定不变的证据**：reviewer 判的是"答案里的结论，是否被它**引用文档的实际内容 /
   `run_diagnostic` 的实际输出**支持"。证据在两轮之间不变，第二轮要么补上了缺口、要么
   没补，球门不会移动——这跟"能不能更好"这种没有 ground truth 的主观标准有本质区别。
2. **结构化裁决，不是对话**：reviewer 只产出 typed `ReviewVerdict`（approve / revise +
   具体 findings），不写反驳段落；"是否通过"是字段、gate 逻辑是确定的代码。
3. **硬上限 + 明确兜底**：最多 **revise 一次**。重答后仍不满意时，**默认动作不是再来一轮**，
   而是：低风险 → 带 `⚠️ 复核提示` 交付（A）；涉及诊断/写操作 → 标记需人工复核（B）。
   出口是人或带注解交付，永远不是循环。

控制流（revise-once）：

    answer1 → 该复核？(有引用 / 跑了诊断 / 提议了写；非反问)
      ├─ 否 → 直接发 answer1
      └─ 是 → verdict1 = reviewer(answer1, 证据)
              ├─ approve → 发 answer1
              └─ revise  → answer2 = 重答(注入 findings)
                          verdict2 = reviewer(answer2, 证据)   # 只判收尾，绝不触发第三轮
                          ├─ approve → 发 answer2
                          └─ 仍 revise:
                             ├─ 涉及诊断/写 → B：标记需人工复核
                             └─ 否          → A：发 answer2 + ⚠️ 复核提示

关键纪律 **precision > recall**：收益在干净的本地文档上量不出（happy path 本来就不会错），
所以 reviewer 必须几乎不误伤——只在能**指出具体证据矛盾**时才 revise（引用文档没这句 /
结论和诊断输出打架 / 编造实时数据 / 写命令没走审批），主观表达一律放行。拿不准 → approve。

复核者失败（provider 异常等）时 **fail-open**（视作 approve）——一个坏掉的复核者不该
把好答案也挡下；它是安全网，不是必经闸。

实现拆成"纯函数 / 可注入"两层：`gather_evidence` / `is_review_eligible` / `format_findings`
是纯函数；`review_and_revise` 用注入的 `reanswer` / `evidence_fn` 回调，脱离具体答题路径与
SDK 可单测（见 tests/test_review.py）。
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .index import norm_key, parse_feishu_citation
from .model import role_model_settings
from .schema import FenceTolerantOutputSchema
from .tools import _read_doc

logger = logging.getLogger(__name__)

# 复核角色名（model router 的 for_role("reviewer")；OPS_QA_REVIEWER_MODEL 覆盖）。
REVIEWER_ROLE = "reviewer"


# ---------------------------------------------------------------------------
# 配置（缺省关）
# ---------------------------------------------------------------------------


@dataclass
class ReviewConfig:
    """二次复核配置。缺省 enabled=False（不复核，零感知）。"""

    enabled: bool = False
    # 喂给 reviewer 的证据（引用文档内容 + 诊断输出）截断上限，防超长文档撑爆复核上下文。
    max_evidence_chars: int = 8000

    @classmethod
    def from_env(cls) -> ReviewConfig:
        enabled = (os.environ.get("OPS_QA_REVIEW") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        raw = (os.environ.get("OPS_QA_REVIEW_MAX_EVIDENCE") or "").strip()
        try:
            max_chars = int(raw) if raw else 8000
        except ValueError:
            max_chars = 8000
        return cls(enabled=enabled, max_evidence_chars=max_chars)


# ---------------------------------------------------------------------------
# 复核裁决（typed output）
# ---------------------------------------------------------------------------


class ReviewVerdict(BaseModel):
    """复核者对一条答案的裁决。模型必须按此 schema 产出。"""

    verdict: Literal["approve", "revise"] = Field(
        description="approve=答案与证据一致、可交付；revise=发现**具体的证据层面问题**需修正。"
        "只有能指出具体矛盾时才 revise；主观表达/风格/完整性问题一律 approve。"
    )
    findings: list[str] = Field(
        default_factory=list,
        description="verdict=revise 时，逐条列出**具体、锚定证据**的问题（如"
        "'答案说 maxmemory 默认 8gb，但 redis/overview.md 未提及该默认值'）。"
        "approve 时留空。不要写'可以更清楚'这类主观意见。",
    )
    grounded: bool = Field(
        default=True,
        description="答案里的事实性结论是否**全部**被给定证据（引用文档内容 / 诊断输出）支持。",
    )


# ---------------------------------------------------------------------------
# 复核者 agent
# ---------------------------------------------------------------------------

REVIEWER_INSTRUCTIONS = """你是内部运维问答的**二次复核者**。给你一条已生成的答案、以及它所依据的**证据**（引用文档的实际内容 + 实时诊断命令的实际输出）。唯一职责：核对答案里的**事实性结论是否被这些证据支持**，产出结构化裁决。

# 你**只**核对这几类"能对着证据抓实"的问题（发现任一即 verdict=revise）
1. **引用不忠实**：答案说"文档说 X / 根据文档 …"，但证据文档里**根本没有 X**、或说的是**相反**的内容。
2. **诊断-结论矛盾**：结论与诊断输出**自相矛盾**（如结论说"内存正常"，但 `INFO memory` 显示已用 94%）。
3. **编造实时数据**：答案给了具体的"实时/当前"数值，但证据里的诊断输出**并没有**这个数据（凭空捏造）。
4. **无据的事实结论**：某条实质性结论**既无引用文档、也无诊断输出支持**（纯幻觉）。
5. **安全**：答案把**写/变更命令**直接写给用户去跑，而不是走审批（本应提议审批或只给带 ⚠️ 的文字建议）。

# 你**不**做的事（这些一律 verdict=approve，不要 revise）
- 不评价文风、措辞、详略、结构、"能不能更清楚/更完整/再补充"——**主观质量不是你的职责**。
- 不因为"答案没覆盖到某个你觉得相关的点"就 revise（只要它说的**没错**）。
- 不自己另找证据或质疑证据本身；只在**给定证据**范围内核对。
- 反问（要用户补充信息）、友好拒绝、纯问候类答案——不涉及事实结论，approve。

# 判定纪律（重要）
- **宁可漏，不可误伤**：只有能**具体指出**"答案说 A，但证据是 B"时才 revise；拿不准、证据不足以判定时一律 **approve**。
- findings 必须**具体、锚定证据**：写清"答案哪句话"和"证据里实际是什么"，不要写"内存部分不准确"这种空话。
- 你的输出会被用来让答题者**重答一次**，所以 findings 要让它知道**改哪里、依据什么**。

只输出裁决对象（verdict / findings / grounded），不要输出别的。"""


_REVIEW_INPUT_TMPL = """# 用户问题
{question}

# 待核对的答案
{answer}

# 证据（引用文档的实际内容 + 实时诊断输出；只在此范围内核对）
{evidence}

请核对上面的答案是否被证据支持，产出裁决。"""


def build_reviewer_agent(model: Any, *, name: str = "reviewer") -> Agent:
    """构造复核者 agent：无工具（证据内联喂入）、低温、typed `ReviewVerdict` 输出。

    无工具是刻意的——证据由代码 `gather_evidence` 拼好内联给它，让复核**确定、有界**，
    不会自己去读别的文档发散。用 `FenceTolerantOutputSchema` 兼容非 OpenAI provider 的
    不规范 json_schema 输出（与答题契约同源）。
    """
    return Agent(
        name=name,
        instructions=REVIEWER_INSTRUCTIONS,
        tools=[],
        model=model,
        model_settings=role_model_settings(REVIEWER_ROLE),  # 低温：判定要稳、可复现
        output_type=FenceTolerantOutputSchema(ReviewVerdict, strict_json_schema=False),
    )


# ---------------------------------------------------------------------------
# 证据收集 + 纯函数
# ---------------------------------------------------------------------------


# 自由文本答案里的 `（来源：path）` 标注（与 evaluate._CITATION_RE 同款；这里独立一份，
# 避免 bot → evaluate → bot 的循环导入）。全/半角冒号与括号都容忍，逗号分隔多个。
_CITATION_RE = re.compile(r"[（(]\s*来源\s*[:：]\s*([^）)]+)[）)]")


def extract_citations(text: str) -> list[str]:
    """从自由文本答案里抽取 `（来源：path）` 标注的路径（去重保序）。纯函数。"""
    out: list[str] = []
    for m in _CITATION_RE.finditer(text):
        for part in m.group(1).split(","):
            p = part.strip().strip("`").lstrip("/")
            if p and p not in out:
                out.append(p)
    return out


def gather_evidence(
    docs_root: Path,
    citations: list[str],
    diag_outputs: list[str],
    *,
    feishu_answers: dict[str, str] | None = None,
    max_chars: int = 8000,
) -> str:
    """把答案依据的证据拼成给 reviewer 的文本：引用来源的**实际内容** + 诊断**实际输出**。

    这是复核能"对证据核对"而非"凭先验瞎猜"的前提，也是收敛的关键。三类证据：

    - **本地文档**：citations 去重后逐条 `_read_doc` 读原文（不存在的读到 `[未找到]`，正好
      让 reviewer 抓到引用不实）。
    - **飞书文档**（`飞书文档·<组件>` 这类 citation）：本地无文件可读，改用本轮
      `query_feishu_doc` 实际拿回的 markdown（`feishu_answers`，键是组件名）。**不这么做的话
      reviewer 会对每条飞书来源都读到 `[未找到]`、稳定误判"引用不实"并触发无意义的重答。**
      引用了某个组件、本轮却没调过它的工具 → 明确标成"无据可核"，这正是要抓的幻觉。
    - **实时证据**（`diag_outputs`）：本轮 `run_diagnostic` 的输出、`query_database` 的查询
      结果、`query_gateway_trace` 取到的网关链路表。三者都带来源前缀，reviewer 据此核对
      "结论是否被实时数据支持"。

    整体截断到 max_chars 防超长。
    """
    # 键归一化，好让 citation 里的组件名（大小写/目录名）能对上工具调用记录。
    pending = {norm_key(k): (k, v) for k, v in (feishu_answers or {}).items()}
    parts: list[str] = []
    seen: set[str] = set()
    for rel in citations:
        rel = rel.strip()
        if not rel or rel in seen:
            continue
        seen.add(rel)
        component = parse_feishu_citation(rel)
        if component is not None:
            hit = pending.pop(norm_key(component), None)
            body = hit[1] if hit else "（本轮未取得该组件的飞书文档内容——此引用无据可核）"
            parts.append(f"## 引用飞书文档：{rel}\n{body}")
            continue
        parts.append(f"## 引用文档：{rel}\n{_read_doc(docs_root, rel)}")
    # 本轮查到、但答案没显式引用的飞书文档答案：仍是答案的事实依据，一并交给 reviewer。
    for name, answer in pending.values():
        parts.append(f"## 飞书文档答案：{name}（答案未显式引用）\n{answer}")
    for out in diag_outputs:
        parts.append(f"## 实时证据\n{out}")
    if not parts:
        return "（无可核对的证据：答案未引用任何文档、也没有诊断输出。）"
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n…[证据超过 {max_chars} 字符已截断]"
    return text


def is_review_eligible(citations: list[str], risky: bool, clarify: bool) -> bool:
    """是否该复核这条答案。

    - 反问轮（clarify）不复核：它不下事实结论，只是要用户补信息。
    - 有引用 / 跑了诊断 / 提议了写（risky）才复核——问候、纯拒绝这类没有可核对的事实结论。
    """
    if clarify:
        return False
    return bool(citations) or risky


def format_findings(findings: list[str]) -> str:
    """把 findings 列成 markdown 项。空则给一句占位（正常不会走到）。"""
    items = [f.strip() for f in findings if f.strip()]
    return "\n".join(f"- {f}" for f in items) if items else "- （复核未给出具体条目）"


def revise_prompt(note: str) -> str:
    """构造让答题者重答一次的输入（作为对话里的新一轮用户消息）。"""
    return (
        "你上一轮的回答经二次复核，发现以下**证据层面**的问题：\n"
        f"{note}\n\n"
        "请**仅依据文档与实时诊断的证据**修正这些问题后，重新给出完整回答"
        "（沿用原本的格式与规范；不要辩解、不要说明你改了什么，直接给修正后的答案）。"
    )


# ---------------------------------------------------------------------------
# 编排：review-and-revise（revise-once）
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    """一版答案的归一化视图（review 编排只认这几个字段，与具体答题路径解耦）。"""

    text: str  # 用于喂给 reviewer 核对的答案正文
    citations: list[str] = field(default_factory=list)
    risky: bool = False  # 是否涉及实时诊断 / 写操作提议（决定兜底走 A 还是 B）
    clarify: bool = False  # 是否反问轮（反问不复核）
    payload: Any = None  # 承载具体结果对象（AnswerResult / GuardedAnswer / StructuredAnswer）


@dataclass
class ReviewOutcome:
    """复核编排的结果：选定的最终 Draft + 元信息 + 兜底动作。"""

    final: Draft
    reviewed: bool = False  # 是否真跑了复核（不合格/关闭时 False）
    revised: bool = False  # 是否触发了重答
    verdict1: ReviewVerdict | None = None
    verdict2: ReviewVerdict | None = None
    caveat: str | None = None  # A：带注解交付时要附的 findings 文本
    escalate: bool = False  # B：涉及诊断/写且复核仍不过 → 标记需人工复核


async def run_review(
    agent: Agent, question: str, answer_text: str, evidence: str
) -> ReviewVerdict | None:
    """跑一次复核（sessionless，绝不污染答题者的会话）。失败返回 None（上层 fail-open）。"""
    prompt = _REVIEW_INPUT_TMPL.format(question=question, answer=answer_text, evidence=evidence)
    try:
        # 无工具、无会话、无上层 context：复核是独立判定，2 步足够（正常 1 步）。
        result = await Runner.run(agent, input=prompt, max_turns=2)
    except Exception as e:  # noqa: BLE001 —— reviewer 挂了不该掀翻答题，fail-open。
        logger.warning("reviewer 调用失败，fail-open（视作 approve）：%s", str(e)[:200])
        return None
    out = result.final_output
    if isinstance(out, ReviewVerdict):
        return out
    logger.warning("reviewer 输出非 ReviewVerdict（%s），fail-open", type(out).__name__)
    return None


EvidenceFn = Callable[[Draft], str]
ReanswerFn = Callable[[str], Awaitable[Draft]]


async def review_and_revise(
    *,
    agent: Agent,
    question: str,
    draft1: Draft,
    evidence_fn: EvidenceFn,
    reanswer: ReanswerFn,
) -> ReviewOutcome:
    """revise-once 复核编排。见模块 docstring 的控制流。

    `evidence_fn(draft)` 拼该版答案的证据；`reanswer(note)` 让答题者据 findings 重答一次并
    返回新 Draft。二者由调用方（bot）注入，本函数不关心具体答题路径与 SDK。
    """
    verdict1 = await run_review(agent, question, draft1.text, evidence_fn(draft1))
    if verdict1 is None or verdict1.verdict == "approve":
        return ReviewOutcome(final=draft1, reviewed=True, verdict1=verdict1)

    # revise 一次：把 findings 注入让答题者重答。
    note = format_findings(verdict1.findings)
    draft2 = await reanswer(revise_prompt(note))

    verdict2 = await run_review(agent, question, draft2.text, evidence_fn(draft2))
    if verdict2 is None or verdict2.verdict == "approve":
        return ReviewOutcome(
            final=draft2, reviewed=True, revised=True, verdict1=verdict1, verdict2=verdict2
        )

    # 重答后仍不满意——到此为止，不再触发第三轮。按风险决定兜底动作。
    if draft2.risky:
        # B：涉及实时诊断 / 写操作，两个模型都拿不准 → 标记需人工复核。
        return ReviewOutcome(
            final=draft2,
            reviewed=True,
            revised=True,
            verdict1=verdict1,
            verdict2=verdict2,
            escalate=True,
        )
    # A：低风险 → 带注解交付，把决定权交回给人。
    return ReviewOutcome(
        final=draft2,
        reviewed=True,
        revised=True,
        verdict1=verdict1,
        verdict2=verdict2,
        caveat=format_findings(verdict2.findings),
    )
