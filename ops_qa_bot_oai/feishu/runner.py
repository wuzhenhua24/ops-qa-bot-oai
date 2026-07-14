"""长连接（WebSocket）运行主体：飞书 @机器人 → 答案的核心问答闭环。

用飞书官方推荐的 `lark_oapi.channel.FeishuChannel(transport="ws")` 收事件、发/改消息。
核心闭环：

  群里 @机器人 提问 → 立即发占位消息 → 跑 OpsQABot.answer() → 把占位编辑成最终答案
  （头部 @ 提问者；命中 <<ESCALATE>> 时末尾 @ 负责人）

会话按 (chat,user) 隔离（SessionManager），/reset 开新会话；/tasks 列出自己挂起的
定时跟进（带取消按钮，OPS_QA_FOLLOWUP=1 时）；/cancel（或「取消」）停掉自己正在
处理/排队中的提问——发错问题不用干等答完，也不白烧 token；/help（或「帮助」）出
能力清单与指令说明（按实际启用的特性动态拼）。
升级到负责人（<<ESCALATE:ou_xxx:dir>>）时随答案再发一张**问答归档表单卡**：负责人
填答案提交 → 写进 docs/<组件>/qa-archive.md → @ 提问者推送答案（闭环见
feishu/archive.py）。每轮答完随答案发 👍/👎 反馈卡（事件落 feedback.log，报表见
feedback_stats.py）。追问卡暂不做（产品壳层，不影响 SDK 对比）。

支持三种可答题的消息形态（对齐 ops-qa-bot 的视觉路径）：

- text：@bot + 纯文字（原有主线）
- image：单发一张截图（下载 → 视觉答题，要求底层模型支持视觉）
- post：富文本"@bot + 文字 + 截图"（移动端多图常打成这种），文字 + 图一起喂

其余类型（file/sticker/audio…）回友好提示。

飞书开放平台配置（长连接模式）：
- 事件订阅方式选「长连接」（不填 Request URL）
- 订阅事件：`im.message.receive_v1`
- 权限：`im:message`（收发/更新消息）、`im:message.group_at_msg`（群 @ 消息）、
  `im:message:send_as_bot`、`im:resource`（下载消息里的图片）
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from lark_oapi.channel import FeishuChannel
from lark_oapi.channel.config import (
    ChatQueueConfig,
    PolicyConfig,
    SafetyConfig,
    TextBatchConfig,
)
from lark_oapi.channel.types import ImageContent, InboundMessage, PostContent, TextContent

from ..db_query import DB_CHANGE_TOOL_NAME, DbConfig, change_display
from ..diagnostics import DiagConfig
from ..followup import FollowupConfig
from ..gateway_trace import GatewayTraceConfig
from ..index import parse_index_components
from ..model import MODE_LABELS, build_model_router
from ..review import ReviewConfig
from .approvals import ApprovalCenter
from .archive import ArchiveStore, handle_archive_submit, safe_component_dir
from .feedback import (
    excerpt,
    handle_feedback_click,
    handle_feedback_reason_skip,
    handle_feedback_reason_submit,
    log_event,
    setup_feedback_logger,
)
from .followup import (
    FOLLOWUP_QUESTION_PREFIX,
    FollowupRecord,
    FollowupScheduler,
    make_followup_submitter,
)
from .inbound import (
    MAX_IMAGE_BYTES,
    POST_MAX_IMAGES,
    compose_image_question,
    extract_image_caption,
    normalize_image_media_type,
    parse_post_text,
)
from .render import (
    CANCEL_WORDS,
    HELP_WORDS,
    RESET_WORDS,
    TASKS_WORDS,
    build_answer_post,
    build_archive_form_card,
    build_feedback_card,
    build_followup_tasks_card,
    build_help_text,
    clean_question,
    escalate_dir,
    escalate_open_id,
    extract_form_value,
    parse_archive_submit_value,
    parse_feedback_reason_value,
    parse_feedback_value,
    parse_followup_cancel_value,
    placeholder_text,
)
from .session import InflightScope, SessionManager

logger = logging.getLogger("ops_qa_bot_oai.feishu")

_UNSUPPORTED = "目前只支持文字和图片（截图）提问，关键报错请用文字描述或直接贴图。"
# 兜底错误文案：answer() 抛出非预期异常（模型服务异常/超时、provider 5xx、网络抖动）
# 时编辑进占位消息，避免用户对着"🔍 翻文档中"干等。
_ERROR_TEXT = (
    "⚠️ 处理这条问题时出错了（模型服务异常/超时或网络抖动），请稍后重试。"
    "若持续失败，请联系管理员查看服务日志。"
)


def _ph_post(text: str) -> dict:
    """占位/状态提示的 post 负载（单段纯文本）。"""
    return {"zh_cn": {"title": "", "content": [[{"tag": "text", "text": text}]]}}


def _approval_display(req) -> tuple[str, str]:
    """从一条审批请求里取 (命令, 目标) 展示文本。

    `request_write_command` 的参数就是 command/target；`request_db_change` 的参数是
    param/value/host 等结构化字段，用 `change_display` 拼成可读的变更语句 + 目标。
    """
    if getattr(req, "tool_name", "") == DB_CHANGE_TOOL_NAME:
        return change_display(req.arguments)
    return str(req.arguments.get("command", "?")), str(req.arguments.get("target", "?"))


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

    async def send_card(
        self, chat_id: str, card: dict, *, parent_id: str | None = None
    ) -> str | None:
        try:
            r = await self._channel.send(chat_id, {"card": card}, self._reply_opts(parent_id))
        except Exception:
            logger.exception("send_card failed chat=%s", chat_id)
            return None
        return r.message_id if r.success else None

    async def update_card(self, message_id: str, card: dict) -> bool:
        try:
            r = await self._channel.update_card(message_id, card)
        except Exception:
            logger.exception("update_card failed msg=%s", message_id)
            return False
        return bool(getattr(r, "success", False))

    async def download_image(self, message_id: str, file_key: str) -> bytes | None:
        """下载消息里的一张图片，失败返回 None。

        API: GET /open-apis/im/v1/messages/{message_id}/resources/{file_key}?type=image，
        要求应用有 `im:resource` 权限、bot 是消息所在群成员。media_type 由调用方按
        magic bytes 嗅探（API 不暴露 Content-Type）。
        """
        try:
            return await self._channel.download_resource(file_key, "image", message_id=message_id)
        except Exception:
            logger.exception("download_image failed msg=%s key=%s", message_id, file_key)
            return None


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
        self._docs_root = docs_root
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
        # 定时跟进（OPS_QA_FOLLOWUP=1）：内存定时器 + 按 (chat,user) 绑定的 submitter 工厂。
        # 到点回调 _fire_followup 复用 _answer_flow 整条答题落地链路。未开启时 factory=None，
        # SessionManager 建 bot 不注入 submitter → schedule_followup 工具不挂载，零感知。
        self._followup_config = FollowupConfig.from_env()
        self._followups: FollowupScheduler | None = None
        followup_factory = None
        if self._followup_config.enabled:
            self._followups = FollowupScheduler(
                self._fire_followup,
                max_pending_per_user=self._followup_config.max_pending_per_user,
            )

            def followup_factory(key):
                return make_followup_submitter(self._followups, key[0], key[1])

        self._session = SessionManager(
            docs_root,
            idle_ttl=idle_ttl,
            max_turns=max_turns,
            followup_submitter_factory=followup_factory,
        )
        # 写操作审批闭环（HITL）：guardrails 开启时生效。cardAction 回调常驻注册（无
        # 在途审批时回调是 no-op），审批人白名单 / 超时由 ApprovalCenter 读环境变量。
        self._approvals = ApprovalCenter(self._client)
        # 问答归档：升级到负责人的问题挂一张表单卡，负责人答完写进
        # docs/<组件>/qa-archive.md 并推给提问者（闭环见 feishu/archive.py）。
        self._archives = ArchiveStore()
        # 反馈事件日志（qa / feedback / archive / cancelled…，JSON lines）：
        # 反馈卡和 feedback_stats 报表都建在它上面。路径可用 OPS_QA_FEEDBACK_LOG 覆盖。
        fb_path = setup_feedback_logger()
        logger.info(
            "反馈事件日志：%s（报表：uv run python -m ops_qa_bot_oai.feedback_stats）", fb_path
        )
        logger.info(
            "答题模式：%s（模型 %s）",
            MODE_LABELS.get(self._session.mode, self._session.mode),
            self._session.model_choice.description,
        )
        if self._session.mode != "single":
            # 多模型路由回显（CLI 有横幅、飞书此前没有）：OPS_QA_MODEL_<组件> 这类覆盖
            # 键没对上目录名时是**静默闲置**的，不打这行就无从确认配置是否生效。
            # roles 按当前模式实际会建的角色拼；reviewer 在开复核时一并列出。
            try:
                roles = ["coordinator"] if self._session.mode == "coordinator" else ["triage"]
                if self._session.mode == "auto":
                    roles.append("coordinator")
                roles += [c.dir for c in parse_index_components(docs_root)]
                if ReviewConfig.from_env().enabled:
                    roles.append("reviewer")
                logger.info("模型路由：%s", build_model_router().describe(roles))
            except Exception as e:  # 回显失败不拦启动；reviewer 独立端点配错会在这里显形
                logger.warning("模型路由解析失败（答题时会再次报错）：%s", e)
        logger.info(
            "会话历史：%s",
            "内存（重启即丢；设 OPS_QA_SESSION_DB 可落盘）"
            if self._session.session_db == ":memory:"
            else f"落盘 {self._session.session_db}（重启/回收后可恢复上下文）",
        )
        if self._session.guardrails:
            who = (
                "、".join(sorted(self._approvals.approvers))
                or "群内任何人（未设 OPS_QA_APPROVERS）"
            )
            logger.info(
                "护栏 + 写审批：开（审批人：%s；超时 %.0fs 自动驳回）",
                who,
                self._approvals.timeout,
            )
        # 实时诊断（OPS_QA_DIAG=1）：各会话 bot 自己从环境读同一份配置（session 不传 diag_config，
        # 走 DiagConfig.from_env），这里只做启动日志回显。写命令在诊断里识别后引导走写审批。
        diag = DiagConfig.from_env()
        if diag.enabled:
            how = (
                "模拟执行（未配 jumphost）"
                if diag.use_mock
                else f"真实 ssh 经跳板机 {diag.jumphost}"
            )
            hosts = "、".join(diag.allowed_hosts) if diag.allowed_hosts else "不限（仍拒生产）"
            logger.info("实时诊断：开（测试环境只读；%s；目标白名单：%s）", how, hosts)
        # 网关链路排查（OPS_QA_GW_TRACE=1）：同上，各会话 bot 自己从环境读配置。这里回显的
        # 「挂在组件 X 上」尤其值得看一眼——组件目录名配错时工具会静默挂空（OpsQABot 构造时
        # 另有 WARNING，但那要等第一个问题进来才打）。
        gw = GatewayTraceConfig.from_env()
        if gw.enabled:
            how = "模拟链路数据（未配 base_url）" if gw.use_mock else f"真实 cat 平台 {gw.base_url}"
            logger.info("网关链路排查：开（%s；挂在组件 `%s` 的专家上）", how, gw.component)
        # 二次复核（OPS_QA_REVIEW=1）：各会话 bot 自己从环境读配置，这里只做启动日志回显。
        if ReviewConfig.from_env().enabled:
            logger.info("二次复核：开（另一模型证据核对，revise-once 后交付；诊断/写不过转人工）")
        if self._followups is not None:
            logger.info(
                "定时跟进：开（%d~%d 分钟，每人挂起上限 %d；内存定时器，重启丢未触发任务）",
                self._followup_config.min_delay_minutes,
                self._followup_config.max_delay_minutes,
                self._followup_config.max_pending_per_user,
            )
        # /help 用的特性开关快照（env 是启动期静态的，存一份免得每次重读）。
        # 参数变更申请要 db + guardrails 同时开（request_db_change 只在护栏模式挂载）。
        db_cfg = DbConfig.from_env()
        self._help_features = {
            "diag": diag.enabled,
            "gw_trace": gw.enabled,
            "db": db_cfg.enabled,
            "db_change": db_cfg.enabled and self._session.guardrails,
            "followup": self._followups is not None,
        }
        self._channel.on("message", self._on_inbound)
        self._channel.on("cardAction", self._on_card_action)
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

        extracted = await self._extract_question(inbound, chat_id, msg_id)
        if extracted is None:
            return
        question, images = extracted

        key = (chat_id, sender_id)
        # 帮助指令：能力清单 + 指令说明。短路应答、零 LLM 成本；放最前——一句
        # /help 不该动会话状态（不建 session、不消费过期提示）。
        if not images and question.lower() in HELP_WORDS:
            await self._client.send_post(
                chat_id,
                build_answer_post(self._help_text(), asker_id=sender_id),
                parent_id=msg_id,
            )
            return

        if not images and question in RESET_WORDS:
            await self._session.reset(key)
            await self._client.send_text(chat_id, "（已开启新会话）", parent_id=msg_id)
            return

        # 跟进任务管理指令：列出自己挂起的定时跟进（带取消按钮）。短路应答、零 LLM 成本。
        if not images and question.lower() in TASKS_WORDS:
            await self._send_tasks_list(chat_id, sender_id, msg_id)
            return

        # 取消在途提问：停掉自己全部处理中/排队中的答题（发错问题不用干等、不白烧
        # token）。各条问题的占位由它们自己的 _answer_flow 收尾成"已取消"，这里只回
        # 执行结果。按 (chat, user) 隔离，取消不掉别人的提问。
        if not images and question.lower() in CANCEL_WORDS:
            n = self._session.cancel_inflight(key)
            if n == 0:
                reply = "当前没有正在处理中的问题，无需取消。"
            else:
                reply = f"🛑 已请求取消你 {n} 条处理中/排队中的问题，对应消息会标记为已取消。"
            logger.info("cancel inflight: chat=%s user=%s n=%d", chat_id, sender_id, n)
            await self._client.send_text(chat_id, reply, parent_id=msg_id)
            return

        await self._answer_flow(
            chat_id, sender_id, question, images=images or None, parent_id=msg_id
        )

    async def _answer_flow(
        self,
        chat_id: str,
        sender_id: str,
        question: str,
        *,
        images: list[tuple[str, bytes]] | None = None,
        parent_id: str | None = None,
    ) -> None:
        """占位 → 答题 → 答案编辑回占位的完整落地流程。

        入站消息（_handle）与定时跟进到点（_fire_followup）共用这一条：到点等于
        "以用户名义发了一条新问题"，占位/审批/@ 提问者/审批轨迹全套行为一致。
        """
        key = (chat_id, sender_id)
        msg_id = parent_id
        # qid 贯穿一轮问答：qa 事件、反馈卡、被踩原因都用它关联（报表回填原题）。
        qid = uuid.uuid4().hex[:12]
        logger.info(
            "Q chat=%s user=%s images=%d q=%r",
            chat_id,
            sender_id,
            len(images or []),
            question[:80],
        )

        # 上下文过期判定（一次性消费，须在 answer 之前）：距上次答题 ≥ idle_ttl
        # 即翻篇——清历史开新会话，本轮答案头部挂"已过期"提示，让用户立刻知道
        # 那句「接着上面的」bot 没拿到上下文、是按全新问题答的。
        session_expired = await self._session.take_expired_notice(key)
        if session_expired:
            logger.info("session expired, starting fresh: chat=%s user=%s", chat_id, sender_id)

        # 立即占位（post），答完编辑替换。占位以 post 发出，方便后续 edit 成 post。
        # 同用户前一条还没答完时本条要排队等锁：占位前缀用「🕒 排队中」，拿到锁
        # 开始跑时（on_start 回调）再刷成「🔍 翻文档中」，让用户分辨哪条真的在跑。
        queued = self._session.queued(key)
        ph_id = await self._client.send_post(
            chat_id, _ph_post(placeholder_text(question, queued=queued)), parent_id=msg_id
        )

        on_start = None
        if queued and ph_id:

            async def on_start():
                await self._client.update_post(ph_id, _ph_post(placeholder_text(question)))

        # 写审批 approver（guardrails 开启时用）：占位改成等待提示 → 发审批卡片 →
        # 等值班人点按钮（超时驳回）。run 在 answer_guarded 的中断循环里挂起等它返回。
        approver = None
        if self._session.guardrails:

            async def approver(req):
                args = req.arguments
                if ph_id:
                    await self._client.update_post(
                        ph_id, _ph_post("⏳ agent 提议了写操作，等待审批…")
                    )
                command, target = _approval_display(req)
                return await self._approvals.request(
                    chat_id,
                    command=command,
                    target=target,
                    reason=str(args.get("reason", "")),
                    asker_id=sender_id,
                    parent_id=msg_id,
                )

        # 在途登记：/cancel 据此找到这条提问并 cancel 掉答题 task。整条 answer()
        # （排队等锁 / 流式答题 / 审批挂起）都在这个内层 task 里，一种机制覆盖全部
        # 状态；register 与 scope.task 赋值之间无 await，不会被 /cancel 插进来。
        scope = InflightScope()
        scope_id = self._session.register_inflight(key, scope)
        try:
            task = asyncio.ensure_future(
                self._session.answer(
                    key, question, approver=approver, images=images, on_start=on_start
                )
            )
            scope.task = task
            result = await task
        except asyncio.CancelledError:
            # 只吞 /cancel 主动取消（scope.cancelled 已翻）；停机等外部取消照常传播。
            # 注意被取消的是内层 task，当前协程自身并未被 cancel，吞掉是安全的。
            if not scope.cancelled:
                task.cancel()  # 被取消的是外层（如停机）：别把答题 task 留成孤儿
                raise
            cancel_post = build_answer_post(
                "❌ 已取消这条问题。要重新提问直接发新消息即可。", asker_id=sender_id
            )
            await self._deliver(chat_id, ph_id, cancel_post, parent_id=msg_id)
            log_event(
                "cancelled",
                qid=qid,
                chat_id=chat_id,
                user_id=sender_id,
                question=excerpt(question, 200),
            )
            logger.info(
                "question cancelled by user: chat=%s user=%s q=%r",
                chat_id,
                sender_id,
                question[:80],
            )
            return
        except Exception:
            # answer() 内部只兜了 max_turns / 护栏，其余异常（provider 5xx、鉴权失败、
            # 超时、网络抖动、ModelBehaviorError…）会抛到这里。不接住的话占位消息会永远
            # 停在"🔍 翻文档中"，用户干等还不知道出错——把占位编辑成错误提示兜底。
            # 用 Exception（非 BaseException）：不吞 asyncio.CancelledError，优雅停机不受影响。
            logger.exception(
                "answer failed chat=%s user=%s q=%r", chat_id, sender_id, question[:80]
            )
            log_event(
                "qa_error",
                qid=qid,
                chat_id=chat_id,
                user_id=sender_id,
                question=excerpt(question, 200),
            )
            err_post = build_answer_post(_ERROR_TEXT, asker_id=sender_id)
            await self._deliver(chat_id, ph_id, err_post, parent_id=msg_id)
            return
        finally:
            self._session.unregister_inflight(key, scope_id)
        esc = escalate_open_id(result.markers.escalate)
        answer_text = result.text
        # 过期提示挂在答案最前面（post 第一段正文，最显眼）。只在成功路径挂：
        # 答题失败时标记已消费也不补挂——历史已翻篇，错误文案里再解释反而添乱。
        if session_expired:
            idle_minutes = max(1, int(self._session.idle_ttl // 60))
            answer_text = (
                f"⏱️ 上一轮上下文已过期（超过 {idle_minutes} 分钟未活跃，已自动开新会话），"
                f"本次按新问题处理。\n\n{answer_text}"
            )
        final_post = build_answer_post(answer_text, asker_id=sender_id, escalate_to=esc)
        # 审批轨迹（仅 GuardedAnswer 有这些字段）：黑名单自动驳回 / 人工拍板结果。
        for req, reason in getattr(result, "blacklist_rejections", None) or []:
            cmd = _approval_display(req)[0]
            final_post["zh_cn"]["content"].append(
                [{"tag": "text", "text": f"⛔ 提议 `{cmd}` 被自动驳回（{reason}）。"}]
            )
        for req, ok in getattr(result, "approvals", None) or []:
            cmd = _approval_display(req)[0]
            # db 参数变更批准后可能已真执行（配了 admin 账号时），执行结果在答案正文里；
            # 这里只记审批结论，不断言"待人工执行"。
            mark = "✅ 已批准" if ok else "🚫 已驳回"
            final_post["zh_cn"]["content"].append(
                [{"tag": "text", "text": f"审批：`{cmd}` → {mark}"}]
            )
        if result.subtype == "error_max_turns":
            final_post["zh_cn"]["content"].append(
                [{"tag": "text", "text": "⚠️ 检索步数过多被中断，结论可能不完整。"}]
            )

        await self._deliver(chat_id, ph_id, final_post, parent_id=msg_id)

        clarify = bool(getattr(result.markers, "clarify", False))
        # qa 事件：一轮问答的完整快照。比参考项目多记 route / agent_usage / 复核
        # 元信息 / 缓存 token——报表据此能看路由分布、按 agent 成本拆分、复核触发率。
        log_event(
            "qa",
            qid=qid,
            chat_id=chat_id,
            user_id=sender_id,
            question=excerpt(question),
            answer_excerpt=excerpt(result.text),
            subtype=result.subtype,
            escalated_to=esc,
            clarification=True if clarify else None,
            images_attached=len(images) if images else None,
            session_expired=True if session_expired else None,
            route=getattr(result, "route", None),
            num_turns=result.num_turns,
            usage=result.usage,
            agent_usage=getattr(result, "agent_usage", None),
            reviewed=True if getattr(result, "reviewed", False) else None,
            revised=True if getattr(result, "revised", False) else None,
            needs_human_review=True if getattr(result, "needs_human_review", False) else None,
            approvals=len(getattr(result, "approvals", None) or []) or None,
            blacklist_rejections=len(getattr(result, "blacklist_rejections", None) or []) or None,
        )

        # 反馈卡：每轮答完随答案发 👍/👎（asker-only 校验在回调侧）。反问轮跳过——
        # 让用户专注答反问，别对半截流程打分。发卡失败不影响答案交付。
        if not clarify:
            try:
                await self._client.send_card(
                    chat_id, build_feedback_card(qid, sender_id), parent_id=msg_id
                )
            except Exception:
                logger.exception("send feedback card failed: chat=%s", chat_id)

        # 问答归档：实际 @ 了负责人时（<<ESCALATE:ou_xxx:dir>>，非 none）再发一张
        # 归档表单卡——负责人答完提交，答案写进 docs/<组件>/qa-archive.md 并推给
        # 提问者；归档进文档后 bot 的检索能命中，下次同样的问题不再升级。
        if esc:
            try:
                await self._send_archive_form(chat_id, sender_id, question, result, esc, msg_id)
            except Exception:
                # 表单卡发不出不影响主答案交付（升级 @ 已经送达），只丢这次沉淀机会。
                logger.exception("send archive form failed: chat=%s", chat_id)

        u = result.usage or {}
        logger.info(
            "A chat=%s user=%s turns=%s in=%s out=%s",
            chat_id,
            sender_id,
            result.num_turns,
            u.get("input_tokens"),
            u.get("output_tokens"),
        )

    async def _extract_question(
        self, inbound: InboundMessage, chat_id: str, msg_id: str | None
    ) -> tuple[str, list[tuple[str, bytes]]] | None:
        """按消息形态抽出 (问题文本, 图片列表)；不可答题时回友好提示并返回 None。

        - text  → 剥 @bot 占位后的纯文字，无图。
        - image → 下载单图走视觉路径；下载失败/超大/为空回提示不进答题。引导问题用
          caption（转发/第三方客户端偶有）或 DEFAULT_IMAGE_PROMPT。
        - post  → AST 抽文字 + 从 inbound.resources 下载图（最多 POST_MAX_IMAGES 张，
          单张失败只丢那一张不阻塞）；文字和图都为空才回 unsupported。
        - 其它  → unsupported 提示。
        """
        content = inbound.content

        if isinstance(content, TextContent):
            raw_text = (content.raw or {}).get("text") or ""
            mention_keys = [m.key for m in (inbound.mentions or []) if getattr(m, "key", None)]
            question = clean_question(raw_text, mention_keys)
            return (question, []) if question else None

        if isinstance(content, ImageContent):
            if not content.image_key or not msg_id:
                return None
            img, err = await self._fetch_image(msg_id, content.image_key)
            if img is None:
                await self._client.send_text(chat_id, err or _UNSUPPORTED, parent_id=msg_id)
                return None
            caption = extract_image_caption(content.raw or {})
            return compose_image_question(caption, 1), [img]

        if isinstance(content, PostContent):
            text = parse_post_text(content.post or {})
            keys = [
                r.file_key for r in (inbound.resources or []) if r.type == "image" and r.file_key
            ]
            if len(keys) > POST_MAX_IMAGES:
                logger.warning(
                    "post 图片超上限，截断 %d → %d：chat=%s", len(keys), POST_MAX_IMAGES, chat_id
                )
                keys = keys[:POST_MAX_IMAGES]
            images: list[tuple[str, bytes]] = []
            for k in keys if msg_id else []:
                img, err = await self._fetch_image(msg_id, k)
                if img is None:
                    logger.warning("post 图片跳过（%s）：chat=%s key=%s", err, chat_id, k)
                    continue
                images.append(img)
            if not text and not images:
                # 纯 sticker/表情/仅链接等：明确回不支持，避免 bot 静默无反应。
                await self._client.send_text(chat_id, _UNSUPPORTED, parent_id=msg_id)
                return None
            if not images:
                return text, []
            return compose_image_question(text, len(images)), images

        await self._client.send_text(chat_id, _UNSUPPORTED, parent_id=msg_id)
        return None

    async def _fetch_image(
        self, msg_id: str, file_key: str
    ) -> tuple[tuple[str, bytes] | None, str | None]:
        """下载并校验一张图。成功返回 ((media_type, bytes), None)；失败返回 (None, 提示文案)。"""
        data = await self._client.download_image(msg_id, file_key)
        if data is None:
            return None, "📎 图片读取失败，请把截图里的关键报错或现象用文字描述后再发。"
        if not data:
            return None, "图片内容为空，请重新发送 🙏"
        if len(data) > MAX_IMAGE_BYTES:
            size_mb = len(data) / (1024 * 1024)
            return None, (
                f"图片太大（{size_mb:.1f}MB，上限 {MAX_IMAGE_BYTES // (1024 * 1024)}MB）🙏 "
                "请压缩后重发，或把关键内容用文字描述出来。"
            )
        return (normalize_image_media_type("", data), data), None

    # ------------------------------------------------------------------
    # 定时跟进：到点执行 / /tasks 列表 / 取消按钮回调
    # ------------------------------------------------------------------

    async def _fire_followup(self, rec: FollowupRecord) -> None:
        """定时跟进到点：用存好的 task 跑一轮全新答题，结果 @ 发起人推回原群。

        复用 `_answer_flow` 整条落地链路（占位/审批/@ 提问者），到点等于"以用户名义
        发了一条新问题"。无 parent（原消息可能已隔很久，按 top-level 发更自然）。
        `_answer_flow` 内部已把答题异常兜成错误 post；走到这里的 except 是占位都没发
        出去之类的通道故障——补一条失败提示，静默失败等于放用户鸽子。
        """
        question = FOLLOWUP_QUESTION_PREFIX + rec.task
        try:
            await self._answer_flow(rec.chat_id, rec.asker_id, question)
        except Exception:
            logger.exception(
                "scheduled followup failed: id=%s chat=%s user=%s",
                rec.record_id,
                rec.chat_id,
                rec.asker_id,
            )
            task_short = rec.task if len(rec.task) <= 100 else rec.task[:100] + "…"
            notice = build_answer_post(
                f"⚠️ 你之前登记的定时跟进刚才执行失败了（任务：{task_short}）。"
                "请直接把要查的事再发我一遍，我立刻查。",
                asker_id=rec.asker_id,
            )
            await self._client.send_post(rec.chat_id, notice)

    def _pending_items(self, key: tuple[str, str]) -> list[dict]:
        """该 (chat, user) 挂起的跟进，整理成 build_followup_tasks_card 要的字典列表。"""
        if self._followups is None:
            return []
        return [
            {
                "record_id": r.record_id,
                "task": r.task,
                "remaining_minutes": self._followups.remaining_minutes(r),
                "firing": r.firing,
            }
            for r in self._followups.list_pending(key)
        ]

    async def _send_tasks_list(self, chat_id: str, sender_id: str, msg_id: str | None) -> None:
        """/tasks 指令：列出自己挂起的定时跟进（每条带取消按钮）。"""
        if self._followups is None:
            await self._client.send_text(
                chat_id, "定时跟进功能未启用，没有可管理的跟进任务。", parent_id=msg_id
            )
            return
        items = self._pending_items((chat_id, sender_id))
        if not items:
            await self._client.send_text(
                chat_id,
                "你当前没有挂起的定时跟进。需要时直接说「X 分钟后帮我再看看 …」即可登记。",
                parent_id=msg_id,
            )
            return
        card = build_followup_tasks_card(sender_id, chat_id, items)
        await self._client.send_card(chat_id, card, parent_id=msg_id)

    def _help_text(self) -> str:
        """/help 文案：组件清单每次现读 INDEX.md（mtime 缓存，加组件即时生效），
        特性开关用启动期快照。"""
        try:
            components = [c.name for c in parse_index_components(self._docs_root)]
        except Exception:
            components = []  # 解析失败降级成不列组件，帮助本身照常给
        idle_minutes = max(1, int(self._session.idle_ttl // 60))
        return build_help_text(components, idle_minutes=idle_minutes, **self._help_features)

    async def _on_card_action(self, event) -> None:
        """cardAction 统一入口：跟进取消 → 归档提交 → 反馈 → 写审批中心，依次尝试。

        各家按自己的按钮 value 结构识别（fua vs aq vs fb/fbr vs aid/decision），
        认不出即交给下一家，互不干扰。
        """
        if await self._handle_followup_cancel(event):
            return
        if await self._handle_archive_submit(event):
            return
        if await self._handle_feedback(event):
            return
        await self._approvals.on_card_action(event)

    async def _handle_feedback(self, event) -> bool:
        """反馈卡回调：👍/👎 点击（fb）与原因表单提交/跳过（fbr）；认不出返回 False。

        asker-only 校验、白名单过滤、事件落日志都在 feedback 模块；这里只做飞书
        I/O——用返回的卡片原地替换（👎 换原因表单、提交/跳过换 ack、非提问者点击
        换回原卡保持可用）。
        """
        value = getattr(getattr(event, "action", None), "value", None)
        clicker = getattr(getattr(event, "operator", None), "open_id", "") or ""
        msg_id = getattr(event, "message_id", None)

        parsed = parse_feedback_value(value)
        if parsed is not None:
            qid, rating, asker = parsed
            card = handle_feedback_click(qid, rating, clicker, asker)
        else:
            parsed_r = parse_feedback_reason_value(value)
            if parsed_r is None:
                return False
            qid, kind, asker = parsed_r
            if kind == "skip":
                card = handle_feedback_reason_skip(qid, clicker, asker)
            else:
                form = extract_form_value(getattr(event, "raw", None))
                raw_reasons = form.get("reasons")
                # multi_select 正常回传 list[str]；防御性兼容单字符串（SDK 抖动/回放）。
                if isinstance(raw_reasons, str):
                    raw_reasons = [raw_reasons]
                reasons = [r for r in raw_reasons or [] if isinstance(r, str)]
                comment = str(form.get("comment") or "") or None
                card = handle_feedback_reason_submit(qid, reasons, comment, clicker, asker)
        if msg_id:
            await self._client.update_card(msg_id, card)
        return True

    async def _send_archive_form(
        self, chat_id: str, asker_id: str, question: str, result, owner_id: str, parent_id
    ) -> None:
        """升级交付后：登记 pending + 给负责人发归档表单卡。"""
        component_dir = safe_component_dir(self._docs_root, escalate_dir(result.markers.escalate))
        # 表单预填标题：优先 LLM 给的归一化标题（ARCHIVE_Q），没给退回用户原话。
        question_default = getattr(result.markers, "archive_q", None) or question
        rec = self._archives.register(
            chat_id=chat_id,
            asker_id=asker_id,
            question=question,
            question_default=question_default,
            owner_id=owner_id,
            component_dir=component_dir,
            parent_msg_id=parent_id,
        )
        card = build_archive_form_card(
            rec.qid, rec.question_default, owner_id, rec.archive_path_repr
        )
        await self._client.send_card(chat_id, card, parent_id=parent_id)
        logger.info(
            "archive form sent: qid=%s owner=%s target=%s", rec.qid, owner_id, rec.archive_path_repr
        )

    async def _handle_archive_submit(self, event) -> bool:
        """归档表单「提交并归档」回调；不是归档按钮返回 False 交回上游。

        owner-only 校验、写盘、幂等都在 archive.handle_archive_submit 里；这里只做
        飞书 I/O：用返回的卡片原地替换表单（ack 或非负责人误点时重建的表单），有
        通知任务就把答案 @ 提问者推回原群。
        """
        qid = parse_archive_submit_value(getattr(getattr(event, "action", None), "value", None))
        if qid is None:
            return False
        form = extract_form_value(getattr(event, "raw", None))
        clicker = getattr(getattr(event, "operator", None), "open_id", "") or ""
        card, notify = await handle_archive_submit(
            self._archives,
            self._docs_root,
            qid=qid,
            question=str(form.get("question") or ""),
            answer=str(form.get("answer") or ""),
            clicker_id=clicker,
        )
        msg_id = getattr(event, "message_id", None)
        if msg_id:
            await self._client.update_card(msg_id, card)
        if notify is not None:
            n_chat, n_post, n_parent = notify
            await self._client.send_post(n_chat, n_post, parent_id=n_parent)
        return True

    async def _handle_followup_cancel(self, event) -> bool:
        """/tasks 卡片「取消」按钮回调；不是取消按钮返回 False 交回上游。

        asker-only：非登记者点击直接忽略（卡片保持原样，不给反馈）。处理后用**刷新
        过的列表卡**原地替换——列表是会过期的快照（其他任务可能已触发），每次点击
        顺带刷新比留着陈旧列表强。
        """
        parsed = parse_followup_cancel_value(getattr(getattr(event, "action", None), "value", None))
        if parsed is None:
            return False
        record_id, chat_id, asker_id = parsed
        clicker = getattr(getattr(event, "operator", None), "open_id", "") or ""
        if not clicker or clicker != asker_id:
            logger.info(
                "followup cancel rejected (not asker): id=%s clicker=%s", record_id, clicker
            )
            return True
        if self._followups is None:
            return True  # 功能已关（重启换配置），卡片是旧的——忽略
        key = (chat_id, asker_id)
        status = self._followups.cancel(record_id, key)
        notice = {
            "cancelled": "✅ 已取消该跟进。",
            "not_found": "⏰ 该跟进已执行完成或已被取消。",
            "not_yours": "⚠️ 这条跟进不是在本群登记的，无法在这里取消。",
            "firing": "⏳ 这条跟进已经在执行了，结果马上会发出来，取消不了。",
        }[status]
        logger.info("followup cancel: id=%s chat=%s user=%s → %s", record_id, *key, status)
        msg_id = getattr(event, "message_id", None)
        if msg_id:
            card = build_followup_tasks_card(
                asker_id, chat_id, self._pending_items(key), notice=notice
            )
            await self._client.update_card(msg_id, card)
        return True

    async def _deliver(
        self, chat_id: str, ph_id: str | None, post: dict, *, parent_id: str | None = None
    ) -> None:
        """最终 post 落地：优先编辑占位消息；没有占位或编辑失败则发新消息兜底。
        成功答案与错误提示走同一条落地路径。"""
        edited = await self._client.update_post(ph_id, post) if ph_id else False
        if not edited:
            await self._client.send_post(chat_id, post, parent_id=parent_id)

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
                # 先停"还会往群里发消息"的定时跟进，再关会话清扫。
                if self._followups is not None:
                    fut = self._channel.schedule(self._followups.stop())
                    await asyncio.wrap_future(fut)
                fut = self._channel.schedule(self._session.stop())
                await asyncio.wrap_future(fut)
            except Exception:
                logger.exception("teardown failed")
            await self._channel.disconnect()
            logger.info("飞书长连接已停止")
