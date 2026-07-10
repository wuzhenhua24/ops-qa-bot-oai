"""同用户连发多条问题的排队占位回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- placeholder_text：queued=True 前缀「🕒 排队中」，默认「🔍 翻文档中」。
- SessionManager.queued：无会话 / 空闲 / 答题中三种状态；纯只读不创建会话。
- runner 全链路：第一条答题中时第二条占位是「排队中」，第一条答完、第二条拿到锁
  开始跑时占位被刷成「翻文档中」，两条答案各自编辑回自己的占位。
- 排队中被 /cancel：占位收尾成「已取消」，不会出现「翻文档中」的误导刷新。

跑法：
    uv run pytest tests/test_queue_placeholder.py
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

import ops_qa_bot_oai.feishu.session as session_mod  # noqa: E402
from ops_qa_bot_oai.feishu.render import placeholder_text  # noqa: E402
from ops_qa_bot_oai.feishu.session import SessionManager  # noqa: E402

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_texts: list[tuple[str, str]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return f"ph{len(self.sent_posts)}"

    async def update_post(self, message_id, post):
        self.updated_posts.append((message_id, post))
        return True

    async def send_text(self, chat_id, text, *, parent_id=None):
        self.sent_texts.append((chat_id, text))
        return "t1"


class _GatedBot:
    """answer 挂在共享 gate 上，测试侧控制第一条何时答完。"""

    gate = asyncio.Event()

    def __init__(self, **kwargs):
        pass

    async def answer(self, question, images=None):
        await _GatedBot.gate.wait()
        return SimpleNamespace(
            text=f"答：{question}",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )


def _session_manager(tmp_path, monkeypatch, bot_cls) -> SessionManager:
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


# ---------------------------------------------------------------------------
# placeholder_text / SessionManager.queued（纯逻辑）
# ---------------------------------------------------------------------------


def test_placeholder_text_queued_prefix():
    assert placeholder_text("Redis 怎么扩容").startswith("🔍 翻文档中")
    assert placeholder_text("Redis 怎么扩容", queued=True).startswith("🕒 排队中")
    assert "Redis 怎么扩容" in placeholder_text("Redis 怎么扩容", queued=True)


async def test_session_queued_states(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch, _GatedBot)
    assert not sm.queued(KEY)  # 无会话：纯只读，不创建
    assert not sm._entries

    entry = await sm._entry(KEY)
    assert not sm.queued(KEY)  # 有会话但空闲
    await entry.lock.acquire()
    assert sm.queued(KEY)  # 锁被占：新问题要排队
    entry.lock.release()
    assert not sm.queued(KEY)


# ---------------------------------------------------------------------------
# runner 全链路：排队占位 + 拿锁后刷新
# ---------------------------------------------------------------------------


async def test_second_question_shows_queued_then_running(tmp_path, monkeypatch):
    _GatedBot.gate = asyncio.Event()
    r = _runner(_session_manager(tmp_path, monkeypatch, _GatedBot))

    t1 = asyncio.create_task(r._handle(_text_inbound("第一条问题")))
    await asyncio.sleep(0.05)  # 第一条占位已发、答题挂在 gate 上
    t2 = asyncio.create_task(r._handle(_text_inbound("第二条问题")))
    await asyncio.sleep(0.05)  # 第二条占位已发、在锁上排队

    ph1 = _flat(r._client.sent_posts[0][1])
    ph2 = _flat(r._client.sent_posts[1][1])
    assert "翻文档中" in ph1 and "第一条问题" in ph1
    assert "排队中" in ph2 and "第二条问题" in ph2

    _GatedBot.gate.set()  # 放行：第一条答完 → 第二条拿到锁开始跑
    await t1
    await t2

    # 第二条拿到锁后占位刷成「翻文档中」（发生在最终答案编辑之前、编辑的是 ph2）
    refresh = [
        (mid, post)
        for mid, post in r._client.updated_posts
        if "翻文档中" in _flat(post) and "第二条问题" in _flat(post)
    ]
    assert refresh and refresh[0][0] == "ph2"
    # 两条答案各自编辑回自己的占位
    finals = {mid: _flat(post) for mid, post in r._client.updated_posts if "答：" in _flat(post)}
    assert "第一条问题" in finals["ph1"]
    assert "第二条问题" in finals["ph2"]


async def test_cancelled_while_queued_never_refreshes_to_running(tmp_path, monkeypatch):
    _GatedBot.gate = asyncio.Event()
    r = _runner(_session_manager(tmp_path, monkeypatch, _GatedBot))

    t1 = asyncio.create_task(r._handle(_text_inbound("第一条问题")))
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(r._handle(_text_inbound("第二条问题")))
    await asyncio.sleep(0.05)

    # 排队期间取消：两条都被停（第一条运行中、第二条排队中）
    await r._handle(_text_inbound("/cancel"))
    await t1
    await t2

    updates = [(mid, _flat(post)) for mid, post in r._client.updated_posts]
    # 两条占位都收尾成「已取消」
    assert any(mid == "ph1" and "已取消" in body for mid, body in updates)
    assert any(mid == "ph2" and "已取消" in body for mid, body in updates)
    # 第二条从未被刷成「翻文档中」（它根本没开始跑）
    assert not any(mid == "ph2" and "翻文档中" in body for mid, body in updates)
