"""长连接（WebSocket）运行主体：飞书 @机器人 → 答案的核心问答闭环。

用飞书官方推荐的 `lark_oapi.channel.FeishuChannel(transport="ws")` 收事件、发/改消息。
核心闭环：

  群里 @机器人 提问 → 立即发占位消息 → 跑 OpsQABot.answer() → 把占位编辑成最终答案
  （头部 @ 提问者；命中 <<ESCALATE>> 时末尾 @ 负责人）

会话按 (chat,user) 隔离（SessionManager），/reset 开新会话。非文字消息回友好提示。
这是"核心问答闭环"范围——反馈卡 / 追问卡 / 归档暂不做（产品壳层，不影响 SDK 对比）。

飞书开放平台配置（长连接模式）：
- 事件订阅方式选「长连接」（不填 Request URL）
- 订阅事件：`im.message.receive_v1`
- 权限：`im:message`（收发/更新消息）、`im:message.group_at_msg`（群 @ 消息）、
  `im:message:send_as_bot`
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from lark_oapi.channel import FeishuChannel
from lark_oapi.channel.config import (
    ChatQueueConfig,
    PolicyConfig,
    SafetyConfig,
    TextBatchConfig,
)
from lark_oapi.channel.types import InboundMessage, TextContent

from ..model import MODE_LABELS
from .render import (
    RESET_WORDS,
    build_answer_post,
    clean_question,
    escalate_open_id,
    placeholder_text,
)
from .session import SessionManager

logger = logging.getLogger("ops_qa_bot_oai.feishu")

_UNSUPPORTED = "目前只支持文字提问，关键报错请用文字描述。"


class FeishuClient:
    """outbound 轻封装：复用入站的同一个 channel，发/改 post、发文本。"""

    def __init__(self, channel: FeishuChannel):
        self._channel = channel

    @staticmethod
    def _reply_opts(parent_id: str | None) -> dict | None:
        # 引用回复原提问消息（消息头带引用条，不开 thread）。
        return {"reply_to": parent_id, "reply_in_thread": False} if parent_id else None

    async def send_text(
        self, chat_id: str, text: str, *, parent_id: str | None = None
    ) -> str | None:
        try:
            r = await self._channel.send(chat_id, {"text": text}, self._reply_opts(parent_id))
        except Exception:
            logger.exception("send_text failed chat=%s", chat_id)
            return None
        return r.message_id if r.success else None

    async def send_post(
        self, chat_id: str, post: dict, *, parent_id: str | None = None
    ) -> str | None:
        try:
            r = await self._channel.send(chat_id, {"post": post}, self._reply_opts(parent_id))
        except Exception:
            logger.exception("send_post failed chat=%s", chat_id)
            return None
        return r.message_id if r.success else None

    async def update_post(self, message_id: str, post: dict) -> bool:
        try:
            r = await self._channel.edit_message(message_id, {"post": post})
        except Exception:
            logger.exception("update_post failed msg=%s", message_id)
            return False
        return bool(getattr(r, "success", False))


class WsRunner:
    """长连接运行主体：建 channel、注册 message handler、跑核心问答闭环。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        docs_root: Path,
        *,
        idle_ttl: float = 1800.0,
        max_turns: int = 30,
    ):
        docs_root = Path(docs_root).resolve()
        if not (docs_root / "INDEX.md").is_file():
            raise RuntimeError(f"docs_root 缺少 INDEX.md: {docs_root}")
        self._channel = FeishuChannel(
            app_id=app_id,
            app_secret=app_secret,
            transport="ws",
            policy=PolicyConfig(require_mention=False, respond_to_mention_all=False),
            safety=SafetyConfig(
                text_batch=TextBatchConfig(delay_ms=0),
                chat_queue=ChatQueueConfig(enabled=False),
            ),
        )
        self._client = FeishuClient(self._channel)
        self._session = SessionManager(docs_root, idle_ttl=idle_ttl, max_turns=max_turns)
        logger.info(
            "答题模式：%s（模型 %s）",
            MODE_LABELS.get(self._session.mode, self._session.mode),
            self._session.model_choice.description,
        )
        self._channel.on("message", self._on_inbound)
        self._channel.on("reconnecting", lambda: logger.warning("ws reconnecting ..."))
        self._channel.on("reconnected", lambda: logger.info("ws reconnected"))

    async def _on_inbound(self, inbound: InboundMessage) -> None:
        try:
            await self._handle(inbound)
        except Exception:
            logger.exception("inbound handling failed")

    async def _handle(self, inbound: InboundMessage) -> None:
        if getattr(inbound.sender, "is_bot", False):
            return  # 忽略机器人自己/互相 @ 形成的环路
        chat_id = inbound.chat_id
        sender_id = inbound.sender_id
        msg_id = inbound.message_id
        if not chat_id or not sender_id:
            return

        content = inbound.content
        if not isinstance(content, TextContent):
            await self._client.send_text(chat_id, _UNSUPPORTED, parent_id=msg_id)
            return

        raw_text = (content.raw or {}).get("text") or ""
        mention_keys = [m.key for m in (inbound.mentions or []) if getattr(m, "key", None)]
        question = clean_question(raw_text, mention_keys)
        if not question:
            return

        key = (chat_id, sender_id)
        if question in RESET_WORDS:
            await self._session.reset(key)
            await self._client.send_text(chat_id, "（已开启新会话）", parent_id=msg_id)
            return

        logger.info("Q chat=%s user=%s q=%r", chat_id, sender_id, question[:80])

        # 立即占位（post），答完编辑替换。占位以 post 发出，方便后续 edit 成 post。
        ph_post = {
            "zh_cn": {
                "title": "",
                "content": [[{"tag": "text", "text": placeholder_text(question)}]],
            }
        }
        ph_id = await self._client.send_post(chat_id, ph_post, parent_id=msg_id)

        result = await self._session.answer(key, question)
        esc = escalate_open_id(result.markers.escalate)
        final_post = build_answer_post(result.text, asker_id=sender_id, escalate_to=esc)
        if result.subtype == "error_max_turns":
            final_post["zh_cn"]["content"].append(
                [{"tag": "text", "text": "⚠️ 检索步数过多被中断，结论可能不完整。"}]
            )

        edited = await self._client.update_post(ph_id, final_post) if ph_id else False
        if not edited:
            # 占位发送/编辑失败，兜底发新消息。
            await self._client.send_post(chat_id, final_post, parent_id=msg_id)

        u = result.usage or {}
        logger.info(
            "A chat=%s user=%s turns=%s in=%s out=%s",
            chat_id,
            sender_id,
            result.num_turns,
            u.get("input_tokens"),
            u.get("output_tokens"),
        )

    async def _bootstrap(self) -> None:
        await self._session.start()

    async def run(self) -> None:
        await self._channel.connect()
        fut = self._channel.schedule(self._bootstrap())
        await asyncio.wrap_future(fut)
        logger.info("飞书长连接已启动，等待群内 @机器人 提问 …")
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            try:
                fut = self._channel.schedule(self._session.stop())
                await asyncio.wrap_future(fut)
            except Exception:
                logger.exception("teardown failed")
            await self._channel.disconnect()
            logger.info("飞书长连接已停止")
