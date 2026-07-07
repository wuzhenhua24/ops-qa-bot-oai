"""会话管理：按 (chat_id, user_id) 隔离独立的 OpsQABot。

对齐 ops-qa-bot 的会话语义（同群每个用户上下文互不干扰、空闲回收、/reset），但做成
精简版：一个 dict + per-key 锁 + 后台空闲清扫。模型只解析一次（resolve_model），各
会话 bot 复用同一 ModelChoice，避免每会话重建 client。

会话历史走 SDK 的 SQLiteSession（session_id = "chat_id:user_id"）：

- 缺省 `:memory:`（OPS_QA_SESSION_DB 未设）：历史在进程内，行为与旧版一致——
  空闲回收 / 重启即丢。
- 设 `OPS_QA_SESSION_DB=<文件路径>` 后历史落盘：空闲回收只丢 bot 实例（轻），
  同一用户再提问时按 session_id 从 db 恢复上下文接着聊；进程重启同理。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from agents import SQLiteSession

from ..bot import OpsQABot
from ..model import ModelChoice, resolve_mode, resolve_model, resolve_session_db

SessionKey = tuple[str, str]  # (chat_id, user_id)


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
        self._entries: dict[SessionKey, _Entry] = {}
        self._guard = asyncio.Lock()  # 保护 _entries 结构
        self._sweeper: asyncio.Task | None = None

    @property
    def model_choice(self) -> ModelChoice:
        return self._model_choice

    def _make_session(self, key: SessionKey) -> SQLiteSession:
        return SQLiteSession(session_id=f"{key[0]}:{key[1]}", db_path=self.session_db)

    async def _entry(self, key: SessionKey) -> _Entry:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                bot = OpsQABot(
                    docs_root=self.docs_root,
                    model_choice=self._model_choice,
                    max_turns=self.max_turns,
                    mode=self.mode,
                    session=self._make_session(key),
                )
                entry = _Entry(bot)
                self._entries[key] = entry
            return entry

    async def answer(self, key: SessionKey, question: str):
        """在该会话上答一题（per-key 锁内串行），返回 AnswerResult。"""
        entry = await self._entry(key)
        async with entry.lock:
            entry.last_used = time.time()
            result = await entry.bot.answer(question)
            entry.last_used = time.time()
            return result

    async def reset(self, key: SessionKey) -> bool:
        """清空该会话上下文。

        即使 bot 实例已被空闲回收也要清：落盘模式下历史在 db 里，只看内存 entry 会漏。
        统一走 _entry（不存在则新建）再 reset，语义上"/reset 后一定是新会话"。
        """
        entry = await self._entry(key)
        async with entry.lock:
            await entry.bot.reset()
            entry.last_used = time.time()
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
