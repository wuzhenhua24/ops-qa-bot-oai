"""会话管理：按 (chat_id, user_id) 隔离独立的 OpsQABot。

对齐 ops-qa-bot 的会话语义（同群每个用户上下文互不干扰、空闲回收、/reset），但做成
精简版：一个 dict + per-key 锁 + 后台空闲清扫。模型只解析一次（resolve_model），各
会话 bot 复用同一 ModelChoice，避免每会话重建 client。

会话历史走 SDK 的 SQLiteSession（session_id = "chat_id:user_id"）：

- 缺省 `:memory:`（OPS_QA_SESSION_DB 未设）：历史在进程内，行为与旧版一致——
  空闲回收 / 重启即丢。
- 设 `OPS_QA_SESSION_DB=<文件路径>` 后历史落盘：空闲回收只丢 bot 实例（轻），
  同一用户再提问时按 session_id 从 db 恢复上下文接着聊；进程重启同理。

上下文有统一的**过期边界**（take_expired_notice）：距上次答题 ≥ idle_ttl 即翻篇
——清历史开新会话，下一轮答案头部由 runner 挂"已过期"提示。SDK 每轮 run 会把
session 里**全部**历史（含工具调用与返回原文）拼进模型输入，不设过期边界的话落盘
历史会无限累积、token 成本线性涨。因此落盘的价值收敛为：进程在 idle_ttl 内重启时，
活跃会话的上下文不丢。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import SQLiteSession

from ..bot import OpsQABot
from ..model import ModelChoice, env_flag, resolve_mode, resolve_model, resolve_session_db

SessionKey = tuple[str, str]  # (chat_id, user_id)

# _last_seen 内存表的保留时长：超过即随空闲清扫剪掉（防长期运行慢涨）。剪掉后
# 落盘模式仍能从 session db 的消息时间戳判定过期，内存模式则拿不到提示（上下文
# 本来也早丢了）——与 ops-qa-bot 的 24h 保留语义一致。
_LAST_SEEN_RETENTION = 24 * 3600.0


@dataclass
class InflightScope:
    """一条在途提问的取消句柄，挂在 SessionManager 的 inflight 登记表里。

    `task` 是整条 `SessionManager.answer()` 调用的包装 task——排队等锁、流式答题、
    审批挂起都在同一个 task 里，`task.cancel()` 一种机制覆盖全部状态（排队中的
    在锁上被打断，零 token；运行中的由 `bot.ask()` 的 finally 顺带停掉 SDK 后台
    run，见 bot.py）。`cancelled` 供 runner 区分「/cancel 主动取消」和「进程停机
    等外部取消」：前者把占位收尾成"已取消"，后者照常传播。
    """

    cancelled: bool = False
    task: asyncio.Task | None = None


class _Entry:
    __slots__ = ("bot", "lock", "last_used")

    def __init__(self, bot: OpsQABot):
        self.bot = bot
        self.lock = asyncio.Lock()
        self.last_used = time.time()


class SessionManager:
    """按 (chat,user) 维护 OpsQABot；per-key 锁串行同一用户的连发提问。"""

    def __init__(
        self,
        docs_root: Path,
        *,
        idle_ttl: float = 1800.0,
        max_turns: int = 30,
        model_choice: ModelChoice | None = None,
        mode: str | None = None,
        session_db: str | Path | None = None,
        guardrails: bool | None = None,
        followup_submitter_factory: Callable[[SessionKey], Any] | None = None,
    ):
        self.docs_root = docs_root
        self.idle_ttl = idle_ttl
        self.max_turns = max_turns
        self._model_choice = model_choice or resolve_model()
        # 编排模式：飞书无命令行开关，由环境变量 OPS_QA_MODE 控制（与终端 --mode 共用一套
        # .env，缺省 auto）；param 显式传入时优先，便于测试/复用。
        self.mode = resolve_mode() if mode is None else mode
        # 会话历史库：param 显式传入优先，否则读 OPS_QA_SESSION_DB（缺省 :memory:）。
        self.session_db = str(session_db) if session_db is not None else resolve_session_db()
        # 护栏 + 写审批（HITL）：OPS_QA_GUARDRAILS=1 开启（对应终端 --guardrails）。
        # 开启后 answer() 走 answer_guarded，写提议经审批卡片闭环（见 approvals.py）。
        self.guardrails = (
            env_flag("OPS_QA_GUARDRAILS", default=False) if guardrails is None else guardrails
        )
        # 定时跟进 submitter 工厂（runner 在定时器就绪时注入）：按 (chat,user) 造一个
        # 绑定归属的 submitter 传给 bot，schedule_followup 工具据此登记到正确的群与人。
        # None（CLI / 未开启）时 bot 不挂该工具。
        self._followup_submitter_factory = followup_submitter_factory
        self._entries: dict[SessionKey, _Entry] = {}
        self._guard = asyncio.Lock()  # 保护 _entries 结构
        self._sweeper: asyncio.Task | None = None
        # 在途提问登记表：(chat, user) → {scope_id: InflightScope}。/cancel 据此
        # 找到该用户正在处理/排队中的答题 task 并 cancel。纯内存、同 loop 同步
        # 读写（register/cancel 之间无 await），随问题结束即清，无需加锁。
        self._inflight: dict[SessionKey, dict[str, InflightScope]] = {}
        # 上下文过期判定用的活跃时间表：(chat, user) → 上次答题的 unix 时间戳。
        # 独立于 _entries 生命周期（空闲回收不清它），take_expired_notice 据此
        # 判定"过期回来追问"并翻篇。/reset 主动重置会清掉（不算过期）。
        self._last_seen: dict[SessionKey, float] = {}

    @property
    def model_choice(self) -> ModelChoice:
        return self._model_choice

    def _make_session(self, key: SessionKey) -> SQLiteSession:
        return SQLiteSession(session_id=f"{key[0]}:{key[1]}", db_path=self.session_db)

    async def _entry(self, key: SessionKey) -> _Entry:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                factory = self._followup_submitter_factory
                bot = OpsQABot(
                    docs_root=self.docs_root,
                    model_choice=self._model_choice,
                    max_turns=self.max_turns,
                    mode=self.mode,
                    session=self._make_session(key),
                    guardrails=self.guardrails,
                    followup_submitter=factory(key) if factory else None,
                )
                entry = _Entry(bot)
                self._entries[key] = entry
            return entry

    def queued(self, key: SessionKey) -> bool:
        """该 (chat, user) 当前是否有未完成的问题占着 per-key 锁。

        用于占位文本判定：True 表示新进来的问题要排队，前缀用 🕒 排队中；False
        直接 🔍 翻文档中。纯只读检查，不创建会话。
        """
        entry = self._entries.get(key)
        return entry is not None and entry.lock.locked()

    async def answer(
        self, key: SessionKey, question: str, approver=None, images=None, on_start=None
    ):
        """在该会话上答一题（per-key 锁内串行）。

        guardrails 关（默认）→ `bot.answer()` 返回 AnswerResult；
        guardrails 开 → `bot.answer_guarded(approver=...)` 返回 GuardedAnswer（approver
        可为异步，如飞书审批卡片闭环）。两者都有 text/markers/usage/subtype，渲染层通用。

        `images`（list of (media_type, raw_bytes)）透传给 bot，开启视觉路径。
        `on_start`（可选异步回调）在拿到锁、真正开始答题前调用——runner 用它把
        排队中的占位从「🕒 排队中」刷成「🔍 翻文档中」。
        """
        entry = await self._entry(key)
        async with entry.lock:
            entry.last_used = self._last_seen[key] = time.time()
            if on_start is not None:
                await on_start()
            if self.guardrails:
                result = await entry.bot.answer_guarded(question, approver=approver, images=images)
            else:
                result = await entry.bot.answer(question, images=images)
            entry.last_used = self._last_seen[key] = time.time()
            return result

    def register_inflight(self, key: SessionKey, scope: InflightScope) -> str:
        """登记一条在途提问，返回 scope_id（注销时用）。"""
        scope_id = uuid.uuid4().hex[:8]
        self._inflight.setdefault(key, {})[scope_id] = scope
        return scope_id

    def unregister_inflight(self, key: SessionKey, scope_id: str) -> None:
        scopes = self._inflight.get(key)
        if scopes is None:
            return
        scopes.pop(scope_id, None)
        if not scopes:
            self._inflight.pop(key, None)

    def cancel_inflight(self, key: SessionKey) -> int:
        """取消该 (chat, user) 全部在途提问，返回请求取消的条数。

        翻 `cancelled` 标记 + cancel 各自的答题 task。已经答完只是还没注销的
        task，cancel 是 no-op——答案照常送达，不误伤。
        """
        scopes = self._inflight.get(key)
        if not scopes:
            return 0
        for scope in scopes.values():
            scope.cancelled = True
            if scope.task is not None:
                scope.task.cancel()
        return len(scopes)

    async def take_expired_notice(self, key: SessionKey) -> bool:
        """判定"上一轮上下文已过期"；命中则清掉历史（翻篇）并返回 True。

        **必须在 answer() 之前调用**。返回 True 表示本轮按全新会话作答，调用方
        据此在答案头部挂一行"上下文已过期"提示（一次性：翻篇即消费，下一轮
        _last_seen 会被 answer 写回，不再触发）。

        判定：距该用户上次答题 ≥ idle_ttl。上次活跃优先取内存 _last_seen；进程
        重启后内存表是空的，落盘模式下回落到 session db 里最新一条消息的时间戳
        ——否则重启后陈年历史会被静默无感恢复（落盘模式无上限增长的根源）。

        过期处理直接复用 reset：清 db 历史 + 弹掉 _last_seen。即使 entry 还在
        内存里（清扫周期间隙）也照样翻篇，语义统一为"idle_ttl 未活跃即重置"。
        """
        last = self._last_seen.get(key)
        if last is None:
            last = await self._db_last_active(key)
        if last is None or time.time() - last < self.idle_ttl:
            return False
        await self.reset(key)
        return True

    async def _db_last_active(self, key: SessionKey) -> float | None:
        """从 session db 读该会话最新一条消息的时间戳（unix 秒）；无历史返回 None。

        只在落盘模式有意义（内存模式各 SQLiteSession 私有一个 :memory: db，从
        外面连不到）。直查 SDK 的 agent_messages 表（schema 见 SQLiteSession）：
        created_at 是 SQLite CURRENT_TIMESTAMP（UTC）。表不存在（从未答过题）
        或解析失败一律按无历史处理——宁可漏提示，不误清历史。
        """
        if self.session_db == ":memory:":
            return None

        def query() -> str | None:
            conn = sqlite3.connect(self.session_db)
            try:
                row = conn.execute(
                    "SELECT MAX(created_at) FROM agent_messages WHERE session_id = ?",
                    (f"{key[0]}:{key[1]}",),
                ).fetchone()
            finally:
                conn.close()
            return row[0] if row else None

        try:
            raw = await asyncio.to_thread(query)
        except sqlite3.Error:
            return None
        if not raw:
            return None
        try:
            dt = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return dt.timestamp()

    async def reset(self, key: SessionKey) -> bool:
        """清空该会话上下文。

        即使 bot 实例已被空闲回收也要清：落盘模式下历史在 db 里，只看内存 entry 会漏。
        统一走 _entry（不存在则新建）再 reset，语义上"/reset 后一定是新会话"。
        同时清 _last_seen：用户主动 /reset 不算"过期回收"，下一轮提问按全新会话
        处理，不要再追加"上下文已过期"提示徒增噪音。
        """
        entry = await self._entry(key)
        async with entry.lock:
            await entry.bot.reset()
            entry.last_used = time.time()
        self._last_seen.pop(key, None)
        return True

    def active_count(self) -> int:
        return len(self._entries)

    async def start(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            self._sweeper = None

    async def _sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(min(self.idle_ttl, 300.0))
                await self._evict_idle()
        except asyncio.CancelledError:
            pass

    async def _evict_idle(self) -> None:
        now = time.time()
        async with self._guard:
            stale = [k for k, e in self._entries.items() if now - e.last_used > self.idle_ttl]
            for k in stale:
                self._entries.pop(k, None)
        # _last_seen 只为过期提示服务，超过保留时长的顺手剪掉（防长期运行慢涨）。
        cutoff = now - max(self.idle_ttl, _LAST_SEEN_RETENTION)
        for k in [k for k, t in self._last_seen.items() if t < cutoff]:
            self._last_seen.pop(k, None)
