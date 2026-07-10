"""取消在途提问（/cancel 指令）回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- SessionManager 在途登记表：register / unregister / cancel_inflight（翻标记 +
  cancel 答题 task、计数、清空回收）。
- /cancel 无在途：友好提示"无需取消"，不进答题。
- 取消运行中的答题：答题 task 收到 CancelledError、占位刷成"已取消"、半截结果
  丢弃；取消指令本身得到回执。
- 取消排队中的答题：bot.answer 根本不被调用（在锁上就被打断，零 token）。
- 跨用户隔离：B 发 /cancel 取消不掉 A 的在途答题（按 (chat, user) 隔离）。
- 答题正常结束后 /cancel：在途已注销，回"无需取消"，答案不受影响。
- bot.ask 提前退出（消费方被取消）时调用 SDK RunResultStreaming.cancel()，
  停掉后台 run 不再烧 token。
- 审批挂起中被取消：待批卡片收尾成"作废"，_pending 清空，取消照常传播。

跑法：
    uv run pytest tests/test_cancel_question.py
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

# SDK tracing 首次使用时构造 httpx client；本地终端若设了 SOCKS 代理（all_proxy）会撞
# socksio ImportError（README「本地开发开了代理时」同一坑）。测试进程内先摘掉代理。
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

import ops_qa_bot_oai.feishu.session as session_mod  # noqa: E402
from ops_qa_bot_oai.feishu.approvals import ApprovalCenter  # noqa: E402
from ops_qa_bot_oai.feishu.session import InflightScope, SessionManager  # noqa: E402

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_texts: list[tuple[str, str]] = []
        self.sent_cards: list[tuple[str, dict]] = []
        self.updated_cards: list[tuple[str, dict]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return f"ph{len(self.sent_posts)}"

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


class _HangingBot:
    """模拟跑着的 agent：answer 挂起直到被 cancel。记录是否收到取消。"""

    def __init__(self, **kwargs):
        self.cancelled = False
        self.answer_calls = 0

    async def answer(self, question, images=None):
        self.answer_calls += 1
        try:
            await asyncio.Event().wait()  # 挂到天荒地老，只有 cancel 能结束
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _QuickBot:
    def __init__(self, **kwargs):
        pass

    async def answer(self, question, images=None):
        return SimpleNamespace(
            text="秒答",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )


class _MustNotAnswerBot:
    def __init__(self, **kwargs):
        pass

    async def answer(self, question, images=None):
        raise AssertionError("被取消的排队问题不该开始答题")


def _session_manager(tmp_path, monkeypatch, bot_cls) -> SessionManager:
    """真 SessionManager（真锁 / 真在途登记表），OpsQABot 换成假件。"""
    monkeypatch.setattr(session_mod, "OpsQABot", bot_cls)
    return SessionManager(
        tmp_path,
        model_choice=SimpleNamespace(description="test"),
        mode="single",
        session_db=":memory:",
        guardrails=False,
    )


def _runner(session: SessionManager):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient()
    r._session = session
    r._approvals = SimpleNamespace()
    r._followups = None
    return r


def _text_inbound(text: str, *, sender: str = KEY[1]):
    from lark_oapi.channel.types import TextContent

    return SimpleNamespace(
        sender=SimpleNamespace(is_bot=False),
        chat_id=KEY[0],
        sender_id=sender,
        message_id="om_msg",
        content=TextContent(raw={"text": text}, text=text),
        mentions=[],
        resources=[],
    )


# ---------------------------------------------------------------------------
# 在途登记表（SessionManager 层）
# ---------------------------------------------------------------------------


async def test_inflight_registry_roundtrip(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch, _QuickBot)
    assert sm.cancel_inflight(KEY) == 0

    class _FakeTask:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    s1 = InflightScope()  # 排队中：还没绑 task
    s2 = InflightScope()
    s2.task = _FakeTask()
    id1 = sm.register_inflight(KEY, s1)
    sm.register_inflight(KEY, s2)

    assert sm.cancel_inflight(KEY) == 2
    assert s1.cancelled and s2.cancelled
    assert s2.task.cancel_calls == 1

    sm.unregister_inflight(KEY, id1)
    assert sm.cancel_inflight(KEY) == 1  # 只剩 s2
    sm.unregister_inflight(KEY, "no-such-id")  # 不炸


# ---------------------------------------------------------------------------
# /cancel 指令（runner 层，真 SessionManager + 假 bot）
# ---------------------------------------------------------------------------


async def test_cancel_with_nothing_inflight(tmp_path, monkeypatch):
    r = _runner(_session_manager(tmp_path, monkeypatch, _QuickBot))
    await r._handle(_text_inbound("/cancel"))
    assert "无需取消" in r._client.sent_texts[-1][1]
    assert not r._client.sent_posts  # 短路应答，没发占位、不进答题


async def test_cancel_running_answer(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch, _HangingBot)
    r = _runner(sm)

    task = asyncio.create_task(r._handle(_text_inbound("redis 怎么扩容")))
    await asyncio.sleep(0.05)  # 让答题进入挂起阶段
    await r._handle(_text_inbound("取消"))  # 中文触发词
    await task

    entry = await sm._entry(KEY)
    assert entry.bot.cancelled, "运行中的答题 task 应收到 CancelledError"
    # 取消回执 + 占位收尾成"已取消"（半截结果丢弃、不落错误文案）
    assert "已请求取消" in r._client.sent_texts[-1][1]
    final = _flat(r._client.updated_posts[-1][1])
    assert "已取消" in final
    assert "出错" not in final
    # 在途已注销：再次 /cancel 落空
    await r._handle(_text_inbound("/cancel"))
    assert "无需取消" in r._client.sent_texts[-1][1]


async def test_cancel_queued_answer_never_starts(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch, _MustNotAnswerBot)
    r = _runner(sm)

    entry = await sm._entry(KEY)
    await entry.lock.acquire()  # 模拟前一条问题占着锁
    try:
        task = asyncio.create_task(r._handle(_text_inbound("第二条问题")))
        await asyncio.sleep(0.05)  # 排队等锁中
        await r._handle(_text_inbound("/cancel"))
        await task
    finally:
        entry.lock.release()

    # bot.answer 从未被调用（_MustNotAnswerBot 会 raise），占位收尾成"已取消"
    assert "已请求取消" in r._client.sent_texts[-1][1]
    assert "已取消" in _flat(r._client.updated_posts[-1][1])


async def test_cancel_by_other_user_does_not_touch_askers_question(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch, _HangingBot)
    r = _runner(sm)

    task = asyncio.create_task(r._handle(_text_inbound("redis 怎么扩容")))
    await asyncio.sleep(0.05)
    # B 在同一个群里发 /cancel：查的是 (chat, B) 名下的在途，落空
    await r._handle(_text_inbound("/cancel", sender="ou_intruder"))
    assert "无需取消" in r._client.sent_texts[-1][1]
    entry = await sm._entry(KEY)
    assert not entry.bot.cancelled, "别人取消不掉 A 的在途答题"
    # A 自己取消，正常生效
    await r._handle(_text_inbound("/cancel"))
    await task
    assert entry.bot.cancelled
    assert "已请求取消" in r._client.sent_texts[-1][1]


async def test_cancel_after_done_is_noop(tmp_path, monkeypatch):
    r = _runner(_session_manager(tmp_path, monkeypatch, _QuickBot))
    await r._handle(_text_inbound("快问题"))
    assert "秒答" in _flat(r._client.updated_posts[-1][1])  # 正常答完
    await r._handle(_text_inbound("/cancel"))
    assert "无需取消" in r._client.sent_texts[-1][1]


# ---------------------------------------------------------------------------
# bot.ask：消费方被取消时停掉 SDK 后台 run（真正省 token 的一步）
# ---------------------------------------------------------------------------


async def test_ask_cancels_background_run_on_early_exit(monkeypatch):
    import ops_qa_bot_oai.bot as bot_mod

    class _FakeStreamResult:
        def __init__(self):
            self.is_complete = False
            self.cancel_calls = 0

        def cancel(self, mode="immediate"):
            self.cancel_calls += 1

        async def stream_events(self):
            await asyncio.Event().wait()  # 模拟 run 还在后台跑，事件迟迟不来
            yield  # pragma: no cover

    fake_result = _FakeStreamResult()
    monkeypatch.setattr(
        bot_mod, "Runner", SimpleNamespace(run_streamed=lambda *a, **kw: fake_result)
    )

    bot = bot_mod.OpsQABot.__new__(bot_mod.OpsQABot)
    bot._telemetry = SimpleNamespace(reset_run=lambda: None, route=None, agent_usage=dict)
    bot._agent = None
    bot._context = None
    bot._session = None
    bot._run_config = None
    bot.max_turns = None

    async def consume():
        async for _ in bot.ask("q"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # 进入 stream_events 挂起
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert fake_result.cancel_calls == 1, "提前退出必须 cancel 后台 run，否则白烧 token"


# ---------------------------------------------------------------------------
# 审批挂起中被取消：待批卡片收尾成"作废"
# ---------------------------------------------------------------------------


async def test_approval_request_cancelled_finishes_card():
    client = _FakeClient()
    center = ApprovalCenter(client, approvers=frozenset(), timeout=600.0)

    task = asyncio.create_task(
        center.request("oc_chat", command="rm -f /tmp/x", target="10.0.0.1", reason="清理")
    )
    await asyncio.sleep(0.05)  # 卡片已发出，挂在等拍板
    assert center.pending_count() == 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("取消应照常传播（答题 task 才能整体收尾）")
    assert center.pending_count() == 0
    assert "作废" in _flat(client.updated_cards[-1][1])
