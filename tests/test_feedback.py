"""使用者反馈收集回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- log_event：JSON 结构、None 字段剥除；excerpt 折叠截断。
- 卡片纯函数：反馈卡按钮 value（fb 键）、原因表单（白名单选项 + fbr 键）、
  ack 卡、value 解析与其它按钮互不误认。
- handle_feedback_click：asker-only（非提问者点击拒绝且卡片不变）、👍 → ack、
  👎 → 原因表单；事件落日志。
- 原因提交：白名单过滤 + 保序去重、全无效落 invalid、跳过记 skipped。
- runner 全链路：答完发反馈卡（qid 与 qa 事件一致）、反问轮不发卡、
  cardAction 分发（👍 ack 替换 / 👎 表单 / 表单提交 reason 事件）。

跑法：
    uv run pytest tests/test_feedback.py
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

from ops_qa_bot_oai.feishu.feedback import (  # noqa: E402
    excerpt,
    handle_feedback_click,
    handle_feedback_reason_skip,
    handle_feedback_reason_submit,
    log_event,
)
from ops_qa_bot_oai.feishu.render import (  # noqa: E402
    FEEDBACK_REASONS,
    build_feedback_card,
    build_feedback_reason_card,
    parse_feedback_reason_value,
    parse_feedback_value,
)

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")
FB_LOGGER = "ops_qa_bot_oai.feedback"


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _events(caplog) -> list[dict]:
    out = []
    for rec in caplog.records:
        if rec.name == FB_LOGGER:
            out.append(json.loads(rec.message))
    return out


# ---------------------------------------------------------------------------
# log_event / excerpt
# ---------------------------------------------------------------------------


def test_log_event_strips_none_fields(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        log_event("qa", qid="q1", route=None, up=0, escalated_to="ou_x")
    (e,) = _events(caplog)
    assert e == {"event": "qa", "qid": "q1", "up": 0, "escalated_to": "ou_x"}  # None 剥掉、0 保留


def test_excerpt_folds_and_truncates():
    assert excerpt("  a\n\nb\tc  ") == "a b c"
    assert excerpt("x" * 600) == "x" * 500 + "…"
    assert excerpt(None) is None


# ---------------------------------------------------------------------------
# 卡片纯函数 + value 解析
# ---------------------------------------------------------------------------


def test_feedback_card_buttons_and_parse():
    card = build_feedback_card("q1", KEY[1])
    flat = _flat(card)
    assert "👍" in flat and "👎" in flat
    values = [
        b["behaviors"][0]["value"]
        for col in card["body"]["elements"][0]["columns"]
        for b in col["elements"]
    ]
    assert parse_feedback_value(values[0]) == ("q1", "up", KEY[1])
    assert parse_feedback_value(values[1]) == ("q1", "down", KEY[1])
    # 其它家的按钮不误认
    assert parse_feedback_value({"aq": "x"}) is None
    assert parse_feedback_value({"aid": "x", "decision": "approve"}) is None
    assert parse_feedback_value({"fb": "q1", "rating": "sideways"}) is None


def test_reason_card_options_and_parse():
    card = build_feedback_reason_card("q1", KEY[1])
    flat = _flat(card)
    for label in FEEDBACK_REASONS.values():
        assert label in flat
    form = card["body"]["elements"][1]
    submit_value = form["elements"][-1]["behaviors"][0]["value"]
    assert parse_feedback_reason_value(submit_value) == ("q1", "submit", KEY[1])
    skip_value = card["body"]["elements"][-1]["behaviors"][0]["value"]
    assert parse_feedback_reason_value(skip_value) == ("q1", "skip", KEY[1])
    assert parse_feedback_reason_value({"fbr": "q1", "kind": "other"}) is None


# ---------------------------------------------------------------------------
# 点击 / 原因提交处理
# ---------------------------------------------------------------------------


def test_click_up_logs_and_acks(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        card = handle_feedback_click("q1", "up", KEY[1], KEY[1])
    assert "感谢反馈" in _flat(card)
    (e,) = _events(caplog)
    assert e["event"] == "feedback" and e["rating"] == "up" and e["qid"] == "q1"


def test_click_down_returns_reason_form(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        card = handle_feedback_click("q1", "down", KEY[1], KEY[1])
    assert "fb_reason" in _flat(card)  # 原地换成原因表单
    (e,) = _events(caplog)
    assert e["rating"] == "down"


def test_click_by_non_asker_rejected(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        card = handle_feedback_click("q1", "up", "ou_intruder", KEY[1])
    assert "👍" in _flat(card)  # 返回原反馈卡，asker 仍能投票
    (e,) = _events(caplog)
    assert e["event"] == "feedback_rejected"


def test_reason_submit_whitelist_and_dedup(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        card = handle_feedback_reason_submit(
            "q1",
            ["outdated", "evil-injection", "outdated", "incorrect"],
            "  备注  文本 ",
            KEY[1],
            KEY[1],
        )
    assert "已记录" in _flat(card)
    (e,) = _events(caplog)
    assert e["reasons"] == ["outdated", "incorrect"]  # 白名单 + 保序去重
    assert e["reason_labels"] == ["文档过时", "事实错误"]
    assert e["comment"] == "备注 文本"
    assert "invalid" not in e


def test_reason_submit_all_invalid_marks_invalid(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        handle_feedback_reason_submit("q1", ["hack"], None, KEY[1], KEY[1])
    (e,) = _events(caplog)
    assert e["invalid"] is True and "reasons" not in e


def test_reason_skip_logs_skipped(caplog):
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        card = handle_feedback_reason_skip("q1", KEY[1], KEY[1])
    assert "已记录" in _flat(card)
    (e,) = _events(caplog)
    assert e["event"] == "feedback_reason" and e["skipped"] is True


# ---------------------------------------------------------------------------
# runner 全链路
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_cards: list[tuple[str, dict]] = []
        self.updated_cards: list[tuple[str, dict]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return f"ph{len(self.sent_posts)}"

    async def update_post(self, message_id, post):
        self.updated_posts.append((message_id, post))
        return True

    async def send_card(self, chat_id, card, *, parent_id=None):
        self.sent_cards.append((chat_id, card))
        return "c1"

    async def update_card(self, message_id, card):
        self.updated_cards.append((message_id, card))
        return True


class _FakeSession:
    guardrails = False

    def __init__(self, *, clarify=False):
        self._clarify = clarify

    def queued(self, key):
        return False

    async def take_expired_notice(self, key):
        return False

    def register_inflight(self, key, scope):
        return "sid"

    def unregister_inflight(self, key, scope_id):
        pass

    async def answer(self, key, question, approver=None, images=None, on_start=None):
        return SimpleNamespace(
            text="先看 maxmemory。" if not self._clarify else "你的 redis 是 6.x 还是 7.x？",
            markers=SimpleNamespace(escalate=None, clarify=self._clarify, archive_q=None),
            usage={"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 60},
            num_turns=2,
            subtype="success",
            route="redis_specialist",
            agent_usage={"redis_specialist": {"input_tokens": 100, "output_tokens": 20}},
        )


def _runner(session=None):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient()
    r._session = session or _FakeSession()
    r._approvals = SimpleNamespace()
    r._followups = None
    return r


def _text_inbound(text: str):
    from lark_oapi.channel.types import TextContent

    return SimpleNamespace(
        sender=SimpleNamespace(is_bot=False),
        chat_id=KEY[0],
        sender_id=KEY[1],
        message_id="om_msg",
        content=TextContent(raw={"text": text}, text=text),
        mentions=[],
        resources=[],
    )


def _click_event(value: dict, *, clicker=KEY[1], form=None):
    return SimpleNamespace(
        action=SimpleNamespace(value=value),
        operator=SimpleNamespace(open_id=clicker, name="提问者"),
        message_id="om_fb",
        raw={"event": {"action": {"form_value": form or {}}}},
    )


async def test_runner_qa_event_and_feedback_card(caplog):
    r = _runner()
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        await r._handle(_text_inbound("redis 内存告警怎么处理"))

    qa_events = [e for e in _events(caplog) if e["event"] == "qa"]
    assert len(qa_events) == 1
    qa = qa_events[0]
    assert qa["route"] == "redis_specialist"
    assert qa["agent_usage"]["redis_specialist"]["output_tokens"] == 20
    assert qa["usage"]["cached_input_tokens"] == 60
    assert "escalated_to" not in qa and "clarification" not in qa

    # 反馈卡随答案发出，qid 与 qa 事件一致
    assert len(r._client.sent_cards) == 1
    card_flat = _flat(r._client.sent_cards[0][1])
    assert "👍" in card_flat and qa["qid"] in card_flat


async def test_runner_clarify_round_skips_feedback_card(caplog):
    r = _runner(_FakeSession(clarify=True))
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        await r._handle(_text_inbound("备份怎么做"))
    (qa,) = [e for e in _events(caplog) if e["event"] == "qa"]
    assert qa["clarification"] is True
    assert r._client.sent_cards == []  # 反问轮不发反馈卡


async def test_runner_feedback_click_dispatch(caplog):
    r = _runner()
    await r._handle(_text_inbound("q"))
    qid = _flat(r._client.sent_cards[0][1])  # 从卡里抠 qid
    value = r._client.sent_cards[0][1]["body"]["elements"][0]["columns"][0]["elements"][0][
        "behaviors"
    ][0]["value"]
    assert value["rating"] == "up"

    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        await r._on_card_action(_click_event(value))
    assert "感谢反馈" in _flat(r._client.updated_cards[-1][1])
    (e,) = [x for x in _events(caplog) if x["event"] == "feedback"]
    assert e["rating"] == "up" and e["qid"] in qid


async def test_runner_down_then_reason_submit_dispatch(caplog):
    r = _runner()
    await r._handle(_text_inbound("q"))
    down_value = r._client.sent_cards[0][1]["body"]["elements"][0]["columns"][1]["elements"][0][
        "behaviors"
    ][0]["value"]
    qid = down_value["fb"]

    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        await r._on_card_action(_click_event(down_value))  # 👎 → 原因表单
        assert "fb_reason" in _flat(r._client.updated_cards[-1][1])
        await r._on_card_action(  # 提交原因（多选 + 备注）
            _click_event(
                {"fbr": qid, "asker": KEY[1], "kind": "submit"},
                form={"reasons": ["outdated"], "comment": "文档是 5.x 的"},
            )
        )
    assert "已记录" in _flat(r._client.updated_cards[-1][1])
    reason_events = [x for x in _events(caplog) if x["event"] == "feedback_reason"]
    assert reason_events[-1]["reasons"] == ["outdated"]
    assert reason_events[-1]["comment"] == "文档是 5.x 的"


async def test_runner_non_asker_click_keeps_card(caplog):
    r = _runner()
    await r._handle(_text_inbound("q"))
    value = r._client.sent_cards[0][1]["body"]["elements"][0]["columns"][0]["elements"][0][
        "behaviors"
    ][0]["value"]
    with caplog.at_level(logging.INFO, logger=FB_LOGGER):
        await r._on_card_action(_click_event(value, clicker="ou_intruder"))
    assert "👍" in _flat(r._client.updated_cards[-1][1])  # 原卡重建，按钮还在
    assert [x["event"] for x in _events(caplog) if x["event"].startswith("feedback")] == [
        "feedback_rejected"
    ]


async def test_runner_unrecognized_value_falls_through():
    r = _runner()

    calls = []

    async def on_card_action(event):
        calls.append(event)

    r._approvals = SimpleNamespace(on_card_action=on_card_action)
    await r._on_card_action(_click_event({"aid": "x", "decision": "approve"}))
    assert len(calls) == 1  # 不是反馈按钮 → 交给审批中心
