"""会话过期机制（idle_ttl 翻篇 + 过期提示）回归测试（纯逻辑 + 假件）。

覆盖：
- take_expired_notice：无历史 / 未过期 / 过期翻篇（一次性消费，历史被清）。
- /reset 清 _last_seen：主动重置不算过期，下一问不补提示。
- 落盘模式进程重启后的判定回落：内存 _last_seen 空时读 session db 最新消息
  时间戳——陈年历史不被静默恢复；30 分钟内重启则无感接续（落盘的核心价值）。
- _db_last_active：无表 / 无行 / 时间戳解析三种兜底。
- runner 全链路：过期后答案头部挂「已过期」提示，下一问不再挂。
- _evict_idle 顺手剪超保留时长的 _last_seen。

跑法：
    uv run pytest tests/test_session_expiry.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from types import SimpleNamespace

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import SQLiteSession, set_tracing_disabled  # noqa: E402

import ops_qa_bot_oai.feishu.session as session_mod  # noqa: E402
from ops_qa_bot_oai.feishu.session import SessionManager  # noqa: E402

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")
TTL = 1800.0


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


class _FakeBot:
    """假 OpsQABot：答固定文本，reset 清自己挂的 SQLiteSession（对齐真 bot 行为）。"""

    def __init__(self, session=None, **kwargs):
        self._session = session
        self.reset_calls = 0

    async def answer(self, question, images=None):
        return SimpleNamespace(
            text=f"答：{question}",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )

    async def reset(self):
        self.reset_calls += 1
        if self._session is not None:
            await self._session.clear_session()


def _session_manager(tmp_path, monkeypatch, *, session_db=":memory:") -> SessionManager:
    monkeypatch.setattr(session_mod, "OpsQABot", _FakeBot)
    return SessionManager(
        tmp_path,
        idle_ttl=TTL,
        model_choice=SimpleNamespace(description="test"),
        mode="single",
        session_db=session_db,
        guardrails=False,
    )


def _runner(session: SessionManager):
    from ops_qa_bot_oai.feishu.runner import WsRunner

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
# take_expired_notice（内存模式：判定走 _last_seen）
# ---------------------------------------------------------------------------


async def test_no_history_no_notice(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    assert not await sm.take_expired_notice(KEY)  # 从未见过该用户


async def test_recent_activity_no_notice(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    await sm.answer(KEY, "第一问")
    assert not await sm.take_expired_notice(KEY)  # 刚答过


async def test_expired_resets_and_notice_is_one_shot(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    await sm.answer(KEY, "第一问")
    sm._last_seen[KEY] = time.time() - TTL - 1  # 时光倒流：装作 31 分钟没来

    assert await sm.take_expired_notice(KEY)  # 过期：翻篇 + 提示
    entry = await sm._entry(KEY)
    assert entry.bot.reset_calls == 1  # 历史真的被清了
    assert not await sm.take_expired_notice(KEY)  # 一次性：标记已消费


async def test_manual_reset_clears_last_seen(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    await sm.answer(KEY, "第一问")
    sm._last_seen[KEY] = time.time() - TTL - 1
    await sm.reset(KEY)  # 用户主动 /reset
    assert not await sm.take_expired_notice(KEY)  # 不算过期，不补提示


async def test_evict_idle_prunes_last_seen(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    sm._last_seen[KEY] = time.time() - session_mod._LAST_SEEN_RETENTION - TTL - 1
    sm._last_seen[("oc_chat", "ou_fresh")] = time.time()
    await sm._evict_idle()
    assert KEY not in sm._last_seen
    assert ("oc_chat", "ou_fresh") in sm._last_seen


# ---------------------------------------------------------------------------
# 落盘模式：进程重启后的判定回落（_db_last_active）
# ---------------------------------------------------------------------------


async def _seed(db_path: str, *, age_seconds: int) -> None:
    """往 session db 写一条历史，并把时间戳拨到 age_seconds 前（模拟旧会话）。"""
    s = SQLiteSession(session_id=f"{KEY[0]}:{KEY[1]}", db_path=db_path)
    await s.add_items([{"role": "user", "content": "旧问题"}])
    s.close()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE agent_messages SET created_at = datetime('now', ?)",
            (f"-{age_seconds} seconds",),
        )
        conn.commit()
    finally:
        conn.close()


async def test_restart_with_stale_db_history_expires(tmp_path, monkeypatch):
    db = str(tmp_path / "sessions.db")
    await _seed(db, age_seconds=int(TTL) + 600)  # 陈年历史（40 分钟前）
    sm = _session_manager(tmp_path, monkeypatch, session_db=db)  # 模拟重启：_last_seen 空

    assert await sm.take_expired_notice(KEY)  # 从 db 时间戳判出过期，翻篇
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM agent_messages").fetchone()[0]
    conn.close()
    assert n == 0  # db 历史被清，不会再"无感恢复"
    assert not await sm.take_expired_notice(KEY)  # 清完无历史，不再触发


async def test_restart_within_ttl_resumes_silently(tmp_path, monkeypatch):
    db = str(tmp_path / "sessions.db")
    await _seed(db, age_seconds=60)  # 1 分钟前的历史（重启前刚聊过）
    sm = _session_manager(tmp_path, monkeypatch, session_db=db)

    assert not await sm.take_expired_notice(KEY)  # 30 分钟内重启：无感接续
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM agent_messages").fetchone()[0]
    conn.close()
    assert n == 1  # 历史保留


async def test_db_last_active_edge_cases(tmp_path, monkeypatch):
    db = str(tmp_path / "sessions.db")
    sm = _session_manager(tmp_path, monkeypatch, session_db=db)
    assert await sm._db_last_active(KEY) is None  # 表都不存在

    await _seed(db, age_seconds=100)
    assert await sm._db_last_active(KEY) is not None
    assert await sm._db_last_active(("oc_chat", "ou_other")) is None  # 别人的会话无行

    conn = sqlite3.connect(db)
    conn.execute("UPDATE agent_messages SET created_at = 'not-a-timestamp'")
    conn.commit()
    conn.close()
    assert await sm._db_last_active(KEY) is None  # 解析失败按无历史兜底


# ---------------------------------------------------------------------------
# runner 全链路：过期提示挂在答案头部
# ---------------------------------------------------------------------------


async def test_runner_prepends_expired_notice_once(tmp_path, monkeypatch):
    sm = _session_manager(tmp_path, monkeypatch)
    r = _runner(sm)
    await r._handle(_text_inbound("第一问"))
    sm._last_seen[KEY] = time.time() - TTL - 1  # 装作过了 31 分钟

    await r._handle(_text_inbound("接着上面的，然后呢"))
    final = _flat(r._client.updated_posts[-1][1])
    assert "已过期" in final and "30 分钟" in final
    assert "答：接着上面的" in final  # 提示在前、答案照常在后

    await r._handle(_text_inbound("再问一句"))
    assert "已过期" not in _flat(r._client.updated_posts[-1][1])  # 只提示一次
