"""离线评测 harness（差异化 #5）。

ops-qa-bot（Claude SDK 版）一次一个子进程、单模型，做系统性 A/B 评测很别扭。本项目
是进程内库 + provider 可换 + 模式可换（自由文本 / 结构化 / 多 agent），天然适合搭一个
评测台：**同一题集 × 多个配置**跑一遍、打分、出对比报告——把"OpenAI 版到底比 Claude
版好在哪、单 agent vs 多 agent、各 provider"变成可量化的数字。

设计：评分与聚合是**纯函数**（脱离 LLM 可单测），只有"实际跑 bot"和"可选 LLM judge"
需要 API key。核心指标全是确定性的：

- **决策准确率**：decision（answer/clarify/escalate/reject）是否符合预期。
- **转交准确率**：multi/auto 模式下分诊台把问题转交给了正确的处理者（对应组件专家 /
  跨组件协调者 / 分诊自答），据此可量化 auto 自适应路由的准不准。
- **来源命中率**：是否引用了期望组件目录下的文档。
- **来源真实率**：引用的路径是否真实存在（复用 schema.validate_citations）。
- **防幻觉**：该 escalate/reject 的题，是否没有硬编一个带来源的"答案"。
- **成本/时延**：token、轮数、耗时。

结构化模式（#1）下 decision/citations 是类型字段、直接可评；自由文本/多 agent 模式
则从 markers + 正文里抽取（decision 靠启发式）——这本身就是一个发现：结构化输出更可评测。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bot import Markers, OpsQABot
from .model import ModelChoice, resolve_model
from .schema import validate_citations

# 抽取正文里的来源标注：支持全角（来源：x）和半角(来源:x)，逗号分隔多个。
_CITATION_RE = re.compile(r"[（(]\s*来源\s*[:：]\s*([^）)]+)[）)]")
_REJECT_HINTS = ("不在我覆盖", "不在覆盖", "帮不上忙", "范围内", "无法回答", "不在.*范围")


@dataclass
class EvalCase:
    id: str
    question: str
    expected_decision: str | None = None  # answer/clarify/escalate/reject 或 None（不评分）
    expected_component: str | None = None  # 期望被引用的组件目录名，如 "redis"
    # 期望路由（仅 multi/auto 模式评分）：组件目录名（转交给该专家）/ "coordinator"（跨组件）
    # / "self"（分诊台自答，如问候/拒绝）/ None（不评分该项）。
    expected_route: str | None = None
    note: str = ""


@dataclass
class RunOutcome:
    """一道题在某配置下跑出来的归一化结果（与模式无关）。"""

    decision: str
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    usage: dict[str, Any] | None
    num_turns: int | None
    latency_ms: int
    # 归一化后的实际路由（组件名 / "coordinator" / "self"）；None 表示该模式不计路由。
    route: str | None = None


@dataclass
class CaseScore:
    case_id: str
    decision_expected: str | None
    decision_actual: str
    decision_correct: bool | None  # None = 不评分该项
    component_expected: str | None
    component_cited: bool | None  # None = 不评分该项
    route_expected: str | None
    route_actual: str | None
    route_correct: bool | None  # None = 不评分该项（非 multi/auto 模式或题目未标注）
    num_citations: int
    citations_all_valid: bool  # 引用的路径是否全部真实存在（无引用视作 True）
    total_tokens: int
    num_turns: int | None
    latency_ms: int


def load_cases(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=c["id"],
            question=c["question"],
            expected_decision=c.get("expected_decision"),
            expected_component=c.get("expected_component"),
            expected_route=c.get("expected_route"),
            note=c.get("note", ""),
        )
        for c in data["cases"]
    ]


def extract_citations(text: str) -> list[str]:
    """从自由文本答案里抽取 `（来源：path）` 标注的路径（去重保序）。"""
    out: list[str] = []
    for m in _CITATION_RE.finditer(text):
        for part in m.group(1).split(","):
            p = part.strip().strip("`").lstrip("/")
            if p and p not in out:
                out.append(p)
    return out


def infer_decision_freetext(markers: Markers, text: str) -> str:
    """从自由文本的 markers + 正文启发式推断 decision。

    这是自由文本模式的固有脆弱点（结构化模式不需要猜）——评测里如实记录这种推断。
    """
    if markers.clarify:
        return "clarify"
    if markers.escalate is not None:
        return "escalate"
    if any(re.search(h, text) for h in _REJECT_HINTS):
        return "reject"
    return "answer"


def normalize_route(route: str | None) -> str:
    """把 answer() 的 route（最终落点 agent 名）归一成可比对的路由标签。

    `None`（无 handoff，入口 agent 自答）→ "self"；`coordinator` → "coordinator"；
    `<dir>_specialist` → "<dir>"（组件目录名）；其余原样返回。
    """
    if route is None:
        return "self"
    if route == "coordinator":
        return "coordinator"
    if route.endswith("_specialist"):
        return route[: -len("_specialist")]
    return route


def score_case(case: EvalCase, outcome: RunOutcome) -> CaseScore:
    """纯函数评分：不跑模型、不读盘（来源真实性已在 outcome.invalid_citations 里算好）。"""
    decision_correct: bool | None = None
    if case.expected_decision is not None:
        decision_correct = outcome.decision == case.expected_decision

    component_cited: bool | None = None
    if case.expected_component is not None:
        prefix = case.expected_component.rstrip("/") + "/"
        component_cited = any(c.startswith(prefix) for c in outcome.citations)

    # 路由只在能路由的模式下评分（outcome.route 为 None 表示该模式不计路由）。
    route_correct: bool | None = None
    if case.expected_route is not None and outcome.route is not None:
        route_correct = outcome.route == case.expected_route

    usage = outcome.usage or {}
    return CaseScore(
        case_id=case.id,
        decision_expected=case.expected_decision,
        decision_actual=outcome.decision,
        decision_correct=decision_correct,
        component_expected=case.expected_component,
        component_cited=component_cited,
        route_expected=case.expected_route,
        route_actual=outcome.route,
        route_correct=route_correct,
        num_citations=len(outcome.citations),
        citations_all_valid=len(outcome.invalid_citations) == 0,
        total_tokens=int(usage.get("total_tokens", 0) or 0),
        num_turns=outcome.num_turns,
        latency_ms=outcome.latency_ms,
    )


def _rate(numer: int, denom: int) -> float:
    return (numer / denom) if denom else 0.0


def aggregate(scores: list[CaseScore]) -> dict[str, Any]:
    """把一组 CaseScore 聚合成一个配置的汇总指标（纯函数）。"""
    n = len(scores)
    dec_scored = [s for s in scores if s.decision_correct is not None]
    comp_scored = [s for s in scores if s.component_cited is not None]
    route_scored = [s for s in scores if s.route_correct is not None]
    cited = [s for s in scores if s.num_citations > 0]
    return {
        "n": n,
        "decision_accuracy": _rate(
            sum(1 for s in dec_scored if s.decision_correct), len(dec_scored)
        ),
        "decision_scored": len(dec_scored),
        "component_hit_rate": _rate(
            sum(1 for s in comp_scored if s.component_cited), len(comp_scored)
        ),
        "component_scored": len(comp_scored),
        "route_accuracy": _rate(sum(1 for s in route_scored if s.route_correct), len(route_scored)),
        "route_scored": len(route_scored),
        "citation_validity_rate": _rate(sum(1 for s in cited if s.citations_all_valid), len(cited)),
        "cases_with_citations": len(cited),
        "avg_total_tokens": _rate(sum(s.total_tokens for s in scores), n),
        "avg_latency_ms": _rate(sum(s.latency_ms for s in scores), n),
        "avg_turns": _rate(sum(s.num_turns or 0 for s in scores), n),
    }


# ---------------------------------------------------------------------------
# 实际跑 bot（需要 API key）
# ---------------------------------------------------------------------------

# 评测轴 token：路由(single/multi/auto/coordinator) × 输出格式(自由文本/结构化) 两轴正交。
# single 保留友好别名 free（自由文本）/ structured（结构化）；其余路由用 <routing>[+structured]。
MODES = (
    "free",
    "structured",
    "multi",
    "multi+structured",
    "auto",
    "auto+structured",
    "coordinator",
    "coordinator+structured",
)
_ROUTING_MODES = {"multi", "auto"}  # 有分诊路由决策、才计路由准确率


def _parse_eval_mode(mode: str) -> tuple[str, bool]:
    """把评测轴 token 解析成 (路由模式, 是否结构化)。

    `free`/`structured` → single；`<routing>` / `<routing>+structured` → 对应路由 + 格式。
    """
    structured = mode.endswith("structured")
    routing = mode.rsplit("+", 1)[0]
    if routing in ("free", "structured"):
        routing = "single"
    return routing, structured


async def _run_case(
    bot: OpsQABot, routing: str, structured: bool, question: str, docs_root: Path
) -> RunOutcome:
    bot.reset()  # 每题独立会话，避免历史串味
    t0 = time.perf_counter()
    # 仅有分诊路由决策的路由模式记路由；single/coordinator route 置 None → 不计路由准确率。
    scored_route = routing in _ROUTING_MODES
    if structured:
        sa = await bot.answer_structured(question)
        latency = int((time.perf_counter() - t0) * 1000)
        return RunOutcome(
            decision=sa.contract.decision.value,
            answer=sa.contract.answer,
            citations=list(sa.contract.citations),
            invalid_citations=list(sa.invalid_citations),
            usage=sa.usage,
            num_turns=sa.num_turns,
            latency_ms=latency,
            route=normalize_route(sa.route) if scored_route else None,
        )
    r = await bot.answer(question)
    latency = int((time.perf_counter() - t0) * 1000)
    cites = extract_citations(r.text)
    return RunOutcome(
        decision=infer_decision_freetext(r.markers, r.text),
        answer=r.text,
        citations=cites,
        invalid_citations=validate_citations(docs_root, cites),
        usage=r.usage,
        num_turns=r.num_turns,
        latency_ms=latency,
        route=normalize_route(r.route) if scored_route else None,
    )


@dataclass
class ConfigResult:
    label: str
    mode: str
    scores: list[CaseScore] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


async def run_config(
    mode: str,
    cases: list[EvalCase],
    docs_root: Path,
    model_choice: ModelChoice,
    on_case: Any = None,
) -> ConfigResult:
    """在某个模式下跑完整题集并打分。`on_case(case, outcome, score)` 可选回调（进度）。"""
    routing, structured = _parse_eval_mode(mode)
    bot = OpsQABot(docs_root=docs_root, model_choice=model_choice, mode=routing)
    scores: list[CaseScore] = []
    for case in cases:
        outcome = await _run_case(bot, routing, structured, case.question, docs_root)
        score = score_case(case, outcome)
        scores.append(score)
        if on_case is not None:
            on_case(case, outcome, score)
    label = f"{model_choice.description} · {mode}"
    return ConfigResult(label=label, mode=mode, scores=scores, summary=aggregate(scores))


def render_report(results: list[ConfigResult]) -> str:
    """渲染多配置对比报告（纯文本表）。"""
    rows = [
        ("配置", "决策准确", "转交准确", "组件命中", "来源真实", "均tokens", "均轮数", "均耗时ms"),
    ]
    for r in results:
        s = r.summary
        rows.append(
            (
                r.label,
                f"{s['decision_accuracy']:.0%} ({s['decision_scored']})",
                f"{s['route_accuracy']:.0%} ({s['route_scored']})",
                f"{s['component_hit_rate']:.0%} ({s['component_scored']})",
                f"{s['citation_validity_rate']:.0%} ({s['cases_with_citations']})",
                f"{s['avg_total_tokens']:.0f}",
                f"{s['avg_turns']:.1f}",
                f"{s['avg_latency_ms']:.0f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(row))))
    return "\n".join(lines)


def render_case_detail(results: list[ConfigResult]) -> str:
    """逐题逐配置的 decision 对错明细（定位哪些题翻车）。"""
    lines: list[str] = []
    for r in results:
        lines.append(f"\n[{r.label}]")
        for s in r.scores:
            flags = []
            if s.decision_correct is True:
                flags.append("✓决策")
            elif s.decision_correct is False:
                flags.append(f"✗决策(期望{s.decision_expected}/实得{s.decision_actual})")
            if s.route_correct is True:
                flags.append("✓转交")
            elif s.route_correct is False:
                flags.append(f"✗转交(期望{s.route_expected}/实得{s.route_actual})")
            if s.component_cited is True:
                flags.append("✓组件")
            elif s.component_cited is False:
                flags.append(f"✗组件(期望{s.component_expected})")
            if s.num_citations and not s.citations_all_valid:
                flags.append("✗来源不实")
            lines.append(f"  {s.case_id}: {' '.join(flags) or '—'}")
    return "\n".join(lines)


async def run_eval(
    cases_path: Path,
    docs_root: Path,
    modes: list[str],
    model_choice: ModelChoice | None = None,
    verbose: bool = True,
) -> list[ConfigResult]:
    """跑评测：题集 × 给定模式列表，返回各配置结果。"""
    cases = load_cases(cases_path)
    mc = model_choice or resolve_model()
    results: list[ConfigResult] = []
    for mode in modes:
        if verbose:
            print(f"\n=== 跑配置：{mc.description} · {mode}（{len(cases)} 题）===")

        def _progress(case: EvalCase, outcome: RunOutcome, score: CaseScore) -> None:
            if verbose:
                ok = (
                    "·"
                    if score.decision_correct is None
                    else ("✓" if score.decision_correct else "✗")
                )
                print(f"  {ok} {case.id}: {outcome.decision} ({outcome.latency_ms}ms)")

        results.append(await run_config(mode, cases, docs_root, mc, on_case=_progress))
    return results
