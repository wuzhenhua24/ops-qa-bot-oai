"""定时跟进的内存定时器 + 按 (chat, asker) 绑定的 submitter 工厂。

`FollowupScheduler` 只管**计时与记录**：登记（schedule）、列表（list_pending）、
取消（cancel）、收尾（stop）。到点执行什么由构造时注入的 `fire(record)` 回调决定
——飞书接入层（runner）把"跑一轮答题并推回群"的协程给进来，本模块不 import
答题链路，可用假 fire 单测（对比参考项目：FollowupScheduler 直接调
handle_question、住在 feishu_core 里，测它就得把整条答题链搭起来）。

MVP 是纯内存态：进程重启丢未触发任务（与参考项目一致；20 分钟级场景够用，
真有痛点再落 SQLite 重载）。任务跑在创建它的 asyncio loop 上（channel 后台
loop——submitter 在 agent 工具调用里被 await，那整条链跑在后台 loop），与所有
业务同 loop，锁语义一致。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..followup import FollowupRequest

logger = logging.getLogger("ops_qa_bot_oai.feishu.followup")

SessionKey = tuple[str, str]  # (chat_id, asker_id)

# 到点时喂回答题流程的问题前缀：把"这是定时跟进、现在到点了、请实际去查一次"的
# 语境讲清楚，引导 agent 真去调工具（而不是只复述怎么查）；紧跟 agent 当初写的
# 自包含 task。前缀放最前，占位摘要里也会露出"【定时跟进】"让用户认得出。
FOLLOWUP_QUESTION_PREFIX = (
    "【定时跟进】用户之前要求过一会儿自动跟进下面这件事，现在时间到了，"
    "请**立即实际执行一次检查**（按需调用你拥有的检索/诊断工具真的去查，"
    "不要只复述怎么查），把当前结果直接告诉他；如果一次查询还看不出最终结论，"
    "如实说明当前状态即可。\n"
    "注意：系统已自动 @ 发起人、并会标注这是定时跟进的结果，所以你**只输出检查"
    "结论本身**，不要在开头写「@用户」这类字面（写了也不会变成真的 @），"
    "也不要自己加「定时跟进结果」之类的标题。\n\n任务："
)


@dataclass
class FollowupRecord:
    """一笔挂起的定时跟进的元数据，给 /tasks 列表与取消用。

    `firing` 在 sleep 结束、进入实际执行阶段时置 True——执行中的跟进不可取消
    （取消会把跑到一半的答题掐死，且结果马上就出了）。
    """

    record_id: str
    chat_id: str
    asker_id: str
    task: str
    fire_at: float  # 预计触发时刻（unix ts），列表里算"还剩几分钟"
    handle: asyncio.Task | None = None
    firing: bool = False


class FollowupScheduler:
    """内存定时器：schedule() 登记"N 分钟后触发 fire(record)"的后台任务。

    `fire` 由接入层注入（如 runner 的"用存好的 task 跑一轮答题并 @ asker 推回群"）；
    fire 抛异常时调度器只记日志——用户可见的失败提示由 fire 自己兜（接入层才知道
    怎么给用户发消息）。挂起计数从 records 现算，登记/触发/取消天然一致。
    """

    def __init__(
        self,
        fire: Callable[[FollowupRecord], Awaitable[None]],
        *,
        max_pending_per_user: int = 5,
    ):
        self._fire = fire
        self.max_pending_per_user = max_pending_per_user
        # record_id → 挂起（含正在执行）的跟进；触发完成/取消时移除
        self._records: dict[str, FollowupRecord] = {}
        self._closing = False

    def pending_count(self, key: SessionKey) -> int:
        return sum(1 for r in self._records.values() if (r.chat_id, r.asker_id) == key)

    def list_pending(self, key: SessionKey) -> list[FollowupRecord]:
        """该 (chat, asker) 挂起的跟进（按触发时间升序），给 /tasks 渲染。"""
        items = [r for r in self._records.values() if (r.chat_id, r.asker_id) == key]
        items.sort(key=lambda r: r.fire_at)
        return items

    @staticmethod
    def remaining_minutes(rec: FollowupRecord) -> int:
        """还剩几分钟触发（向上取整；不足 1 分钟按 0）。"""
        return max(0, int((rec.fire_at - time.time() + 59) // 60))

    def schedule(self, chat_id: str, asker_id: str, delay_minutes: int, task: str) -> bool:
        """登记一笔定时跟进。超上限 / 正在关闭返回 False（不登记）。"""
        if self._closing:
            return False
        if self.pending_count((chat_id, asker_id)) >= self.max_pending_per_user:
            return False
        rec = FollowupRecord(
            record_id=uuid.uuid4().hex[:8],
            chat_id=chat_id,
            asker_id=asker_id,
            task=task,
            fire_at=time.time() + delay_minutes * 60,
        )
        self._records[rec.record_id] = rec
        rec.handle = asyncio.create_task(self._run(rec, delay_minutes))
        return True

    def cancel(self, record_id: str, key: SessionKey) -> str:
        """取消一笔挂起的跟进。返回状态字符串：

        - "cancelled"：取消成功
        - "not_found"：不存在（已触发完成 / 已被取消 / id 错）
        - "not_yours"：record 不属于该 (chat, asker)——卡片转发到别处被点等
        - "firing"：已进入执行阶段，不可取消（结果马上会发出来）
        """
        rec = self._records.get(record_id)
        if rec is None:
            return "not_found"
        if (rec.chat_id, rec.asker_id) != key:
            return "not_yours"
        if rec.firing:
            return "firing"
        self._records.pop(record_id, None)
        if rec.handle is not None:
            rec.handle.cancel()
        logger.info("scheduled followup cancelled: id=%s chat=%s user=%s", record_id, *key)
        return "cancelled"

    async def _run(self, rec: FollowupRecord, delay_minutes: int) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)
            rec.firing = True  # 进入执行阶段，cancel 从此拒绝
            logger.info(
                "scheduled followup firing: id=%s chat=%s user=%s delay=%dmin",
                rec.record_id,
                rec.chat_id,
                rec.asker_id,
                delay_minutes,
            )
            await self._fire(rec)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 用户可见的失败提示由 fire 自己兜；走到这说明 fire 连兜底都没接住。
            logger.exception(
                "scheduled followup failed: id=%s chat=%s user=%s",
                rec.record_id,
                rec.chat_id,
                rec.asker_id,
            )
        finally:
            self._records.pop(rec.record_id, None)

    async def stop(self) -> None:
        """取消所有未触发/在跑的跟进任务（进程收尾用）。"""
        self._closing = True
        recs = list(self._records.values())
        for r in recs:
            if r.handle is not None:
                r.handle.cancel()
        for r in recs:
            if r.handle is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await r.handle
        self._records.clear()


def make_followup_submitter(scheduler: FollowupScheduler, chat_id: str, asker_id: str):
    """造一个按 (chat, asker) 绑定的 FollowupSubmitter，注入给 `schedule_followup` 工具。

    收到工具校验好的 FollowupRequest → 登记到 scheduler，返回给 agent 的确认/拒绝
    文字。超每人上限时返回引导文字（让 agent 如实告诉用户精简或稍后手动问）。
    """

    async def submit(req: FollowupRequest) -> str:
        pending = scheduler.pending_count((chat_id, asker_id))
        if pending >= scheduler.max_pending_per_user:
            return (
                f"⚠️ 你当前已有 {pending} 个定时跟进在排队（上限 "
                f"{scheduler.max_pending_per_user}），暂时不能再加。"
                "请等其中一个到点完成、或用 /tasks 取消一个，"
                "或这次先不定时、过会儿手动再来问我。"
            )
        if not scheduler.schedule(chat_id, asker_id, req.delay_minutes, req.task):
            return (
                "⚠️ 暂时无法登记定时跟进（服务可能正在重启）。"
                "请如实告诉用户这次没能定时，让他过会儿手动再来问我一次。"
            )
        logger.info(
            "scheduled followup registered: chat=%s user=%s delay=%dmin",
            chat_id,
            asker_id,
            req.delay_minutes,
        )
        return (
            f"✅ 已登记定时跟进：{req.delay_minutes} 分钟后我会自动执行一次检查，"
            "并把结果发到本群 @ 你。（在那之前你可以随时手动来问；"
            "发 /tasks 可以查看或取消挂起的跟进。）"
        )

    return submit
