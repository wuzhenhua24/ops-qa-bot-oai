"""会话管理：按 (chat_id, user_id) 隔离独立的 OpsQABot。

对齐 ops-qa-bot 的会话语义（同群每个用户上下文互不干扰、空闲回收、/reset），但做成
精简版：一个 dict + per-key 锁 + 后台空闲清扫。模型只解析一次（resolve_model），各
会话 bot 复用同一 ModelChoice，避免每会话重建 client。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..bot import OpsQABot
from ..model import ModelChoice, resolve_model

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
    ):
        self.docs_root = docs_root
        self.idle_ttl = idle_ttl
        self.max_turns = max_turns
        self._model_choice = model_choice or resolve_model()
        self._entries: dict[SessionKey, _Entry] = {}
        self._guard = asyncio.Lock()  # 保护 _entries 结构
        self._sweeper: asyncio.Task | None = None

    @property
    def model_choice(self) -> ModelChoice:
        return self._model_choice

    async def _entry(self, key: SessionKey) -> _Entry:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                bot = OpsQABot(
                    docs_root=self.docs_root,
                    model_choice=self._model_choice,
                    max_turns=self.max_turns,
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
        """清空该会话上下文。返回是否存在过该会话。"""
        async with self._guard:
            entry = self._entries.get(key)
        if entry is None:
            return False
        async with entry.lock:
            entry.bot.reset()
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
