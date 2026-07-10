"""定时跟进链路的回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- FollowupConfig.from_env：开关 / 数值解析 / 非法值回落默认。
- schedule_followup 工具 handler：delay 越界 / task 空 / task 超长 / submitter 异常
  兜底 / 正常登记透传 submitter 返回值。
- FollowupScheduler：登记→到点触发 fire / 取消（cancelled/not_found/not_yours/firing）/
  每人挂起上限 / stop 后拒登记 / fire 抛异常不炸调度器。
- make_followup_submitter：确认语 / 超上限引导语。
- /tasks 卡片渲染 + 取消按钮 value 解析（纯函数）。
- WsRunner：/tasks 指令（未启用/空列表/有列表）、取消按钮回调（asker-only、刷新卡片）、
  到点 _fire_followup 复用答题链路。

定时用真 asyncio.sleep 但 delay 取 0 分钟（sleep(0)），测试瞬时完成。

跑法：
    uv run pytest tests/test_followup.py
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

# SDK tracing 首次使用时构造 httpx client；本地终端若设了 SOCKS 代理（all_proxy）会撞
# socksio ImportError（README「本地开发开了代理时」同一坑）。测试进程内先摘掉代理再
# 关 tracing（直调 function_tool 的执行包装会走 tracing）。
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

from ops_qa_bot_oai.feishu.followup import (  # noqa: E402
    FOLLOWUP_QUESTION_PREFIX,
    FollowupScheduler,
    make_followup_submitter,
)
from ops_qa_bot_oai.feishu.render import (  # noqa: E402
    build_followup_tasks_card,
    parse_card_action_value,
    parse_followup_cancel_value,
)
from ops_qa_bot_oai.followup import (  # noqa: E402
    FollowupConfig,
    FollowupRequest,
    make_schedule_followup_tool,
)

KEY = ("oc_chat", "ou_user")

# 直调 function_tool 的执行包装会走 SDK tracing（内部建 httpx client）；本地终端若设了
# SOCKS 代理会撞 socksio ImportError（README「本地开发开了代理时」同一坑）。测试关掉。
set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# FollowupConfig.from_env
# ---------------------------------------------------------------------------


def test_config_defaults_off(monkeypatch):
    monkeypatch.delenv("OPS_QA_FOLLOWUP", raising=False)
    cfg = FollowupConfig.from_env()
    assert not cfg.enabled
    assert cfg.min_delay_minutes == 1
    assert cfg.max_delay_minutes == 120
    assert cfg.max_pending_per_user == 5


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPS_QA_FOLLOWUP", "1")
    monkeypatch.setenv("OPS_QA_FOLLOWUP_MAX_MINUTES", "60")
    monkeypatch.setenv("OPS_QA_FOLLOWUP_MAX_PENDING", "not-a-number")  # 非法回落默认
    cfg = FollowupConfig.from_env()
    assert cfg.enabled
    assert cfg.max_delay_minutes == 60
    assert cfg.max_pending_per_user == 5


# ---------------------------------------------------------------------------
# schedule_followup 工具（通过 on_invoke_tool 直接调，绕开 LLM）
# ---------------------------------------------------------------------------


def _make_tool(submitter, **cfg):
    config = FollowupConfig(enabled=True, **cfg)
    return make_schedule_followup_tool(config, submitter)


async def _invoke(tool, delay: int, task: str) -> str:
    import json

    from agents.tool_context import ToolContext

    args = json.dumps({"delay_minutes": delay, "task": task})
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="call_1", tool_arguments=args)
    return await tool.on_invoke_tool(ctx, args)


async def test_tool_rejects_out_of_range_delay():
    async def submitter(req):
        raise AssertionError("不该走到登记")

    tool = _make_tool(submitter, min_delay_minutes=1, max_delay_minutes=120)
    out = await _invoke(tool, 0, "查一下")
    assert "1~120" in out
    out = await _invoke(tool, 999, "查一下")
    assert "1~120" in out


async def test_tool_rejects_empty_and_oversize_task():
    async def submitter(req):
        raise AssertionError("不该走到登记")

    tool = _make_tool(submitter)
    assert "缺少 task" in await _invoke(tool, 10, "   ")
    assert "过长" in await _invoke(tool, 10, "x" * 2000)


async def test_tool_submits_and_returns_confirmation():
    seen: list[FollowupRequest] = []

    async def submitter(req):
        seen.append(req)
        return "✅ 已登记"

    tool = _make_tool(submitter)
    out = await _invoke(tool, 20, "  连 10.1.2.3 查 ALTER 进度  ")
    assert out == "✅ 已登记"
    assert seen[0].delay_minutes == 20
    assert seen[0].task == "连 10.1.2.3 查 ALTER 进度"  # 已 strip


async def test_tool_swallows_submitter_error():
    async def submitter(req):
        raise RuntimeError("boom")

    tool = _make_tool(submitter)
    out = await _invoke(tool, 10, "查一下")
    assert "出错" in out  # 返回引导文字而不是抛异常打断 run


# ---------------------------------------------------------------------------
# FollowupScheduler（fire 回调注入，delay=0 瞬时触发）
# ---------------------------------------------------------------------------


async def _drain(scheduler: FollowupScheduler):
    """等所有已登记任务跑完（delay=0 时一个事件循环轮即可）。"""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_scheduler_fires_and_cleans_up():
    fired = []

    async def fire(rec):
        fired.append(rec)

    s = FollowupScheduler(fire)
    assert s.schedule(*KEY, 0, "查 ALTER")
    await _drain(s)
    assert len(fired) == 1
    assert fired[0].task == "查 ALTER"
    assert fired[0].chat_id == KEY[0] and fired[0].asker_id == KEY[1]
    assert s.pending_count(KEY) == 0  # 触发后记录被清


async def test_scheduler_cancel_states():
    async def fire(rec):
        pass

    s = FollowupScheduler(fire)
    assert s.schedule(*KEY, 5, "任务A")
    rec = s.list_pending(KEY)[0]

    assert s.cancel("no-such-id", KEY) == "not_found"
    assert s.cancel(rec.record_id, ("oc_other", "ou_user")) == "not_yours"
    rec.firing = True
    assert s.cancel(rec.record_id, KEY) == "firing"
    rec.firing = False
    assert s.cancel(rec.record_id, KEY) == "cancelled"
    assert s.pending_count(KEY) == 0
    assert s.cancel(rec.record_id, KEY) == "not_found"  # 已取消
    await s.stop()


async def test_scheduler_max_pending_per_user():
    async def fire(rec):
        pass

    s = FollowupScheduler(fire, max_pending_per_user=2)
    assert s.schedule(*KEY, 5, "任务1")
    assert s.schedule(*KEY, 5, "任务2")
    assert not s.schedule(*KEY, 5, "任务3")  # 超上限拒登记
    assert s.schedule("oc_chat", "ou_other", 5, "别人的不受影响")
    await s.stop()


async def test_scheduler_stop_refuses_new_and_cancels_pending():
    async def fire(rec):
        pass

    s = FollowupScheduler(fire)
    assert s.schedule(*KEY, 5, "任务A")
    await s.stop()
    assert s.pending_count(KEY) == 0
    assert not s.schedule(*KEY, 5, "关门后不收")


async def test_scheduler_fire_exception_does_not_leak_record():
    async def fire(rec):
        raise RuntimeError("答题链路炸了")

    s = FollowupScheduler(fire)
    assert s.schedule(*KEY, 0, "任务A")
    await _drain(s)
    assert s.pending_count(KEY) == 0  # finally 里清了记录，不留僵尸


async def test_submitter_confirmation_and_limit():
    async def fire(rec):
        pass

    s = FollowupScheduler(fire, max_pending_per_user=1)
    submit = make_followup_submitter(s, *KEY)
    out = await submit(FollowupRequest(delay_minutes=20, task="查 ALTER"))
    assert "20 分钟" in out and "/tasks" in out
    out2 = await submit(FollowupRequest(delay_minutes=5, task="再来一个"))
    assert "上限" in out2
    await s.stop()


# ---------------------------------------------------------------------------
# /tasks 卡片渲染 + 取消按钮 value 解析（纯函数）
# ---------------------------------------------------------------------------


def _card_texts(card: dict) -> str:
    return "".join(
        el.get("text", {}).get("content", "") for el in card["elements"] if el.get("tag") == "div"
    )


def test_tasks_card_lists_items_with_cancel_buttons():
    items = [
        {"record_id": "r1", "task": "查 ALTER", "remaining_minutes": 18, "firing": False},
        {"record_id": "r2", "task": "查扩容", "remaining_minutes": 0, "firing": True},
    ]
    card = build_followup_tasks_card("ou_user", "oc_chat", items)
    text = _card_texts(card)
    assert "2" in text and "约 18 分钟后执行" in text and "正在执行" in text
    buttons = [el for el in card["elements"] if el.get("tag") == "action"]
    assert len(buttons) == 1  # firing 的那条没有取消按钮
    value = buttons[0]["actions"][0]["value"]
    assert value == {"fua": "r1", "chat": "oc_chat", "asker": "ou_user"}
    # 按钮 value 能被解析回来（与审批按钮解析互不误认）
    assert parse_followup_cancel_value(value) == ("r1", "oc_chat", "ou_user")
    assert parse_card_action_value(value) is None


def test_tasks_card_empty_and_notice():
    card = build_followup_tasks_card("ou_user", "oc_chat", [], notice="✅ 已取消该跟进。")
    text = _card_texts(card)
    assert "已取消" in text and "没有挂起" in text
    assert not [el for el in card["elements"] if el.get("tag") == "action"]


def test_parse_followup_cancel_value_rejects_foreign():
    assert parse_followup_cancel_value({"aid": "x", "decision": "approve"}) is None
    assert parse_followup_cancel_value("not-json{") is None
    assert parse_followup_cancel_value({"fua": "r1"}) is None  # 缺归属字段
    assert parse_followup_cancel_value('{"fua":"r1","chat":"c","asker":"u"}') == ("r1", "c", "u")


# ---------------------------------------------------------------------------
# WsRunner：/tasks 指令、取消回调、到点执行（假 client + 假 session）
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_texts: list[tuple[str, str]] = []
        self.sent_cards: list[tuple[str, dict]] = []
        self.updated_cards: list[tuple[str, dict]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return "ph1"

    async def update_post(self, message_id, post):
        self.updated_posts.append((message_id, post))
        return True

    async def send_text(self, chat_id, text, *, parent_id=None):
        self.sent_texts.append((chat_id, text))
        return "t1"

    async def send_card(self, chat_id, card, *, parent_id=None):
        self.sent_cards.append((chat_id, card))
        return "c1"

    async def update_card(self, message_id, card):
        self.updated_cards.append((message_id, card))
        return True


class _RecordingSession:
    guardrails = False

    def __init__(self):
        self.questions: list[str] = []

    async def answer(self, key, question, approver=None, images=None):
        self.questions.append(question)
        return SimpleNamespace(
            text="跟进结论",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )


def _runner(with_scheduler: bool = True):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient()
    r._session = _RecordingSession()
    r._approvals = SimpleNamespace()
    r._followups = FollowupScheduler(r._fire_followup) if with_scheduler else None
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


async def test_runner_tasks_command_disabled_and_empty():
    r = _runner(with_scheduler=False)
    await r._handle(_text_inbound("/tasks"))
    assert "未启用" in r._client.sent_texts[-1][1]

    r = _runner()
    await r._handle(_text_inbound("跟进任务"))  # 中文触发词
    assert "没有挂起" in r._client.sent_texts[-1][1]
    assert not r._session.questions  # 短路应答，不进答题


async def test_runner_tasks_command_lists_pending():
    r = _runner()
    assert r._followups.schedule(*KEY, 30, "查 ALTER 进度")
    await r._handle(_text_inbound("/tasks"))
    assert r._client.sent_cards, "有挂起任务时应发卡片"
    card = r._client.sent_cards[-1][1]
    assert "查 ALTER 进度" in _card_texts(card)
    await r._followups.stop()


async def test_runner_fire_followup_reuses_answer_flow():
    r = _runner()
    assert r._followups.schedule(*KEY, 0, "连 10.1.2.3 查 ALTER 是否完成")
    await _drain(r._followups)
    # 到点跑了一轮答题：问题带定时跟进前缀 + 原 task
    assert len(r._session.questions) == 1
    q = r._session.questions[0]
    assert q.startswith(FOLLOWUP_QUESTION_PREFIX)
    assert "连 10.1.2.3 查 ALTER 是否完成" in q
    # 结果按正常答案落地（占位被编辑成答案）
    assert r._client.updated_posts


def _cancel_event(record_id: str, *, clicker: str, msg_id: str = "om_card"):
    return SimpleNamespace(
        action=SimpleNamespace(value={"fua": record_id, "chat": KEY[0], "asker": KEY[1]}),
        operator=SimpleNamespace(open_id=clicker, name="值班员"),
        message_id=msg_id,
    )


async def test_runner_cancel_click_asker_only():
    r = _runner()
    assert r._followups.schedule(*KEY, 30, "查 ALTER")
    rec = r._followups.list_pending(KEY)[0]

    # 非登记者点击：忽略（不取消、不刷卡片）
    handled = await r._handle_followup_cancel(_cancel_event(rec.record_id, clicker="ou_other"))
    assert handled and r._followups.pending_count(KEY) == 1
    assert not r._client.updated_cards

    # 登记者点击：取消 + 卡片原地刷新（带结果行、列表已空）
    handled = await r._handle_followup_cancel(_cancel_event(rec.record_id, clicker=KEY[1]))
    assert handled and r._followups.pending_count(KEY) == 0
    msg_id, card = r._client.updated_cards[-1]
    assert msg_id == "om_card"
    assert "已取消" in _card_texts(card) and "没有挂起" in _card_texts(card)
    await r._followups.stop()


# ---------------------------------------------------------------------------
# 装配：工具挂载 + prompt 章节（漏装的后果隐蔽——bot 只会说"我没法定时"）
# ---------------------------------------------------------------------------


def _docs_root(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "INDEX.md").write_text(
        "| 组件 | 目录 | 覆盖内容 | open_id |\n|---|---|---|---|\n"
        "| Redis | `redis/` | 缓存 | ou_aaa |\n",
        encoding="utf-8",
    )
    (tmp_path / "redis").mkdir()
    (tmp_path / "redis" / "overview.md").write_text("# Redis\n", encoding="utf-8")
    return tmp_path


def test_bot_single_mode_mounts_tool_and_prompt(tmp_path):
    from ops_qa_bot_oai.bot import OpsQABot

    async def submitter(req):
        return "ok"

    cfg = FollowupConfig(enabled=True)
    bot = OpsQABot(
        docs_root=_docs_root(tmp_path),
        mode="single",
        followup_config=cfg,
        followup_submitter=submitter,
    )
    names = {getattr(t, "name", "") for t in bot._agent.tools}
    assert "schedule_followup" in names
    assert "定时跟进" in bot._agent.instructions

    # 没有 submitter（CLI 场景）：即使 enabled 也不挂、prompt 不提
    bot2 = OpsQABot(docs_root=_docs_root(tmp_path / "d2"), mode="single", followup_config=cfg)
    names2 = {getattr(t, "name", "") for t in bot2._agent.tools}
    assert "schedule_followup" not in names2
    assert "定时跟进" not in bot2._agent.instructions


def test_specialist_gets_followup_prompt_when_tool_attached():
    from ops_qa_bot_oai.index import Component
    from ops_qa_bot_oai.model import build_model_router
    from ops_qa_bot_oai.orchestration import build_specialist_agent

    async def submitter(req):
        return "ok"

    tool = make_schedule_followup_tool(FollowupConfig(enabled=True), submitter)
    c = Component(name="Redis", dir="redis", source="local", coverage="缓存", open_id="ou_aaa")
    model = build_model_router().for_role("redis")[1]
    agent = build_specialist_agent(c, model, extra_tools=[tool])
    assert "定时跟进" in agent.instructions
    assert "schedule_followup" in {getattr(t, "name", "") for t in agent.tools}


async def test_runner_card_action_dispatch_ignores_foreign_value():
    r = _runner()
    # 审批按钮的 value（aid/decision）不该被跟进取消回调吃掉
    handled = await r._handle_followup_cancel(
        SimpleNamespace(
            action=SimpleNamespace(value={"aid": "a1", "decision": "approve"}),
            operator=SimpleNamespace(open_id="ou_x"),
            message_id="om_card",
        )
    )
    assert not handled
