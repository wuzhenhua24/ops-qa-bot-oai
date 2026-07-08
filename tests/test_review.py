"""二次复核（review.py）回归测试：证据收集 + 触发门槛 + revise-once 编排的四条终态。

核心用**桩 reviewer / 桩 answerer**（monkeypatch `run_review`）验编排分支，无需真 LLM：
- verdict1=approve → 发原答（不重答）。
- verdict1=revise, verdict2=approve → 重答后通过 → 发重答。
- verdict1=revise, verdict2=revise, risky → B：escalate（需人工复核）。
- verdict1=revise, verdict2=revise, 低风险 → A：带 caveat 交付。
- run_review 返回 None（provider 挂了）→ fail-open 当 approve。

跑法：uv run pytest tests/test_review.py -q
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

import ops_qa_bot_oai.review as R
from ops_qa_bot_oai.review import (
    Draft,
    ReviewConfig,
    ReviewVerdict,
    extract_citations,
    format_findings,
    gather_evidence,
    is_review_eligible,
    review_and_revise,
    revise_prompt,
)

# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_extract_citations():
    text = "结论（来源：redis/x.md）；又（来源：mysql/y.md, redis/x.md）"
    assert extract_citations(text) == ["redis/x.md", "mysql/y.md"]  # 去重保序
    assert extract_citations("没有来源标注的答案") == []


def test_is_review_eligible():
    assert is_review_eligible(["redis/x.md"], False, False) is True  # 有引用
    assert is_review_eligible([], True, False) is True  # 跑了诊断
    assert is_review_eligible([], False, False) is False  # 问候/无据 → 不复核
    assert is_review_eligible(["x"], True, True) is False  # 反问轮 → 不复核


def test_format_findings():
    assert format_findings(["a", " b ", ""]) == "- a\n- b"
    assert "复核未给出" in format_findings([])


def test_revise_prompt_contains_findings():
    p = revise_prompt("- 答案说 X 但证据是 Y")
    assert "答案说 X 但证据是 Y" in p and "仅依据文档与实时诊断" in p


@pytest.fixture()
def docs_root():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "redis").mkdir()
        (root / "redis" / "troubleshooting.md").write_text(
            "# Redis 排查\nmaxmemory 达到上限会触发内存告警。", encoding="utf-8"
        )
        yield root


def test_gather_evidence_reads_docs_and_diag(docs_root: Path):
    ev = gather_evidence(
        docs_root, ["redis/troubleshooting.md"], ["used_memory_human:14.2G"], max_chars=8000
    )
    assert "引用文档：redis/troubleshooting.md" in ev
    assert "内存告警" in ev  # 读到了文档实际内容
    assert "used_memory_human:14.2G" in ev  # 诊断输出


def test_gather_evidence_missing_citation_surfaces(docs_root: Path):
    # 引用了不存在的文档 → 证据里出现 [未找到]，正好让 reviewer 抓到引用不实。
    ev = gather_evidence(docs_root, ["redis/nonexistent.md"], [], max_chars=8000)
    assert "未找到" in ev


def test_gather_evidence_empty():
    ev = gather_evidence(Path("."), [], [], max_chars=8000)
    assert "无可核对的证据" in ev


def test_gather_evidence_truncates(docs_root: Path):
    ev = gather_evidence(docs_root, ["redis/troubleshooting.md"], ["x" * 5000], max_chars=200)
    assert len(ev) <= 260 and "已截断" in ev


# ---------------------------------------------------------------------------
# revise-once 编排（桩 run_review + 桩 reanswer）
# ---------------------------------------------------------------------------


def _stub_reviews(monkeypatch, verdicts: list):
    """把 run_review 打桩成按顺序返回给定裁决（None 表示 fail-open）。记录调用次数。"""
    seq = iter(verdicts)
    calls = {"n": 0}

    async def fake(agent, question, answer_text, evidence):
        calls["n"] += 1
        return next(seq)

    monkeypatch.setattr(R, "run_review", fake)
    return calls


def _make_reanswer(text="ans2", citations=None, risky=False):
    async def reanswer(note: str) -> Draft:
        return Draft(
            text=f"{text}[{note[:4]}]",
            citations=citations if citations is not None else ["redis/troubleshooting.md"],
            risky=risky,
            payload="p2",
        )

    return reanswer


def _run(coro):
    return asyncio.run(coro)


def test_approve_delivers_original(monkeypatch):
    calls = _stub_reviews(monkeypatch, [ReviewVerdict(verdict="approve")])
    d1 = Draft(text="ans1", citations=["redis/troubleshooting.md"], risky=False, payload="p1")
    out = _run(
        review_and_revise(
            agent=None,
            question="q",
            draft1=d1,
            evidence_fn=lambda d: "ev",
            reanswer=_make_reanswer(),
        )
    )
    assert out.reviewed and not out.revised
    assert out.final.payload == "p1" and out.caveat is None and out.escalate is False
    assert calls["n"] == 1  # 只复核一次，没重答


def test_revise_then_approve_delivers_reanswer(monkeypatch):
    _stub_reviews(
        monkeypatch,
        [
            ReviewVerdict(verdict="revise", findings=["X 与证据不符"]),
            ReviewVerdict(verdict="approve"),
        ],
    )
    d1 = Draft(text="ans1", citations=["redis/troubleshooting.md"], risky=False, payload="p1")
    out = _run(
        review_and_revise(
            agent=None,
            question="q",
            draft1=d1,
            evidence_fn=lambda d: "ev",
            reanswer=_make_reanswer(),
        )
    )
    assert out.revised and out.final.payload == "p2"
    assert out.caveat is None and out.escalate is False


def test_revise_still_bad_low_risk_caveats(monkeypatch):
    _stub_reviews(
        monkeypatch,
        [
            ReviewVerdict(verdict="revise", findings=["第一轮问题"]),
            ReviewVerdict(verdict="revise", findings=["仍有问题 Z"]),
        ],
    )
    d1 = Draft(text="ans1", citations=["redis/troubleshooting.md"], risky=False, payload="p1")
    out = _run(
        review_and_revise(
            agent=None,
            question="q",
            draft1=d1,
            evidence_fn=lambda d: "ev",
            reanswer=_make_reanswer(risky=False),
        )
    )
    # A：低风险 → 带 caveat 交付重答，不 escalate
    assert out.revised and out.final.payload == "p2"
    assert out.escalate is False
    assert out.caveat is not None and "仍有问题 Z" in out.caveat


def test_revise_still_bad_risky_escalates(monkeypatch):
    _stub_reviews(
        monkeypatch,
        [
            ReviewVerdict(verdict="revise", findings=["第一轮"]),
            ReviewVerdict(verdict="revise", findings=["诊断结论仍矛盾"]),
        ],
    )
    d1 = Draft(text="ans1", citations=[], risky=True, payload="p1")
    out = _run(
        review_and_revise(
            agent=None,
            question="q",
            draft1=d1,
            evidence_fn=lambda d: "ev",
            reanswer=_make_reanswer(risky=True),  # 重答仍涉及诊断/写
        )
    )
    # B：涉及诊断/写且复核仍不过 → escalate（需人工复核），不带 caveat
    assert out.revised and out.escalate is True and out.caveat is None


def test_reviewer_failure_fails_open(monkeypatch):
    # run_review 返回 None（provider 异常）→ 当 approve，直接发原答，不重答。
    calls = _stub_reviews(monkeypatch, [None])
    d1 = Draft(text="ans1", citations=["redis/troubleshooting.md"], risky=False, payload="p1")
    out = _run(
        review_and_revise(
            agent=None,
            question="q",
            draft1=d1,
            evidence_fn=lambda d: "ev",
            reanswer=_make_reanswer(),
        )
    )
    assert out.reviewed and not out.revised and out.final.payload == "p1"
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_default_off():
    assert ReviewConfig().enabled is False


def test_config_from_env(monkeypatch):
    monkeypatch.delenv("OPS_QA_REVIEW", raising=False)
    assert ReviewConfig.from_env().enabled is False
    monkeypatch.setenv("OPS_QA_REVIEW", "1")
    monkeypatch.setenv("OPS_QA_REVIEW_MAX_EVIDENCE", "1234")
    cfg = ReviewConfig.from_env()
    assert cfg.enabled is True and cfg.max_evidence_chars == 1234


# ---------------------------------------------------------------------------
# bot 层注解落地：_apply_review_outcome 把 caveat（A）/ escalate（B）附到答案 + 置标志
# ---------------------------------------------------------------------------


def test_apply_review_outcome_caveat_and_escalate():
    from ops_qa_bot_oai.bot import AnswerResult, StructuredAnswer
    from ops_qa_bot_oai.review import ReviewOutcome
    from ops_qa_bot_oai.schema import AnswerContract, Decision

    # A：低风险带 caveat → 追加到 text，reviewed/revised=True，needs_human_review=False。
    r = AnswerResult(text="原答案")
    out = ReviewOutcome(
        final=Draft(text="原答案", payload=r),
        reviewed=True,
        revised=True,
        verdict2=ReviewVerdict(verdict="revise", findings=["仍存疑 Q"]),
        caveat=format_findings(["仍存疑 Q"]),
    )
    payload = _apply(out)
    assert payload is r and r.reviewed and r.revised and not r.needs_human_review
    assert "复核提示" in r.text and "仍存疑 Q" in r.text

    # B：涉及诊断/写 escalate → 追加"未通过"横幅 + needs_human_review=True。
    sc = StructuredAnswer(contract=AnswerContract(decision=Decision.answer, answer="结构化答案"))
    out2 = ReviewOutcome(
        final=Draft(text="结构化答案", payload=sc),
        reviewed=True,
        revised=True,
        verdict2=ReviewVerdict(verdict="revise", findings=["诊断矛盾 W"]),
        escalate=True,
    )
    payload2 = _apply(out2)
    assert payload2 is sc and sc.needs_human_review is True
    assert "二次复核未通过" in sc.contract.answer and "诊断矛盾 W" in sc.contract.answer


def _apply(outcome):
    """脱离真 bot 调 _apply_review_outcome：它只用到 self 上的无关字段，用一个假 self。"""
    from ops_qa_bot_oai.bot import OpsQABot

    class _FakeSelf:
        _append_answer_text = staticmethod(OpsQABot._append_answer_text)

    return OpsQABot._apply_review_outcome(_FakeSelf(), outcome)
