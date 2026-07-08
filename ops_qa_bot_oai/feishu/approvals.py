"""飞书写操作审批闭环（HITL）：发卡片 → 等值班人点按钮 → 恢复挂起的 run。

OpenAI Agents SDK 的 `needs_approval` 让 run 在写提议处挂起（interruption）；
`OpsQABot.answer_guarded` 接受**异步 approver**——本模块提供那个 approver：

  agent 提议写命令 → run 挂起 → 群里发审批卡片（命令/目标/理由 + 批准/驳回按钮）
  → 值班人点按钮（cardAction 回调）→ resolve Future → run 续跑 → 答案编辑回占位消息

要点：
- **超时默认驳回**（OPS_QA_APPROVAL_TIMEOUT 秒，缺省 600）：无人拍板不放行，
  卡片改成"超时驳回"结果卡。
- **审批人白名单**（OPS_QA_APPROVERS，逗号分隔 open_id）：设了则只有名单内的人
  点按钮有效，其他人点了在卡片上提示无权限；不设则群里任何人可拍板（演示模式）。
- 拍板后卡片**原地替换**成结果卡（按钮移除，防重复点击 / 双人竞态：第一个有效
  点击生效，后到的发现 Future 已 resolve 即忽略）。

黑名单命中的毁灭性命令在 `answer_guarded` 的审批前短路就驳回了，根本不会走到
这里发卡片——审批卡片只为"有风险但可能合理"的命令出现（三层分级见 guardrails.py）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from .render import build_approval_card, build_approval_result_card, parse_card_action_value

logger = logging.getLogger("ops_qa_bot_oai.feishu.approvals")

DEFAULT_APPROVAL_TIMEOUT = 600.0


def resolve_approvers() -> frozenset[str]:
    """从 OPS_QA_APPROVERS 解析审批人 open_id 白名单；空集 = 不限（任何人可拍板）。"""
    raw = (os.environ.get("OPS_QA_APPROVERS") or "").strip()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def resolve_approval_timeout() -> float:
    raw = (os.environ.get("OPS_QA_APPROVAL_TIMEOUT") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_APPROVAL_TIMEOUT
    except ValueError:
        return DEFAULT_APPROVAL_TIMEOUT


@dataclass
class _Pending:
    """一张在途审批卡片。"""

    command: str
    target: str
    reason: str
    card_msg_id: str | None
    future: asyncio.Future[bool] = field(default_factory=asyncio.Future)
    operator_name: str = ""  # 拍板人（结果卡展示用）


class ApprovalCenter:
    """管理在途审批：request() 发卡等结果；on_card_action() 是 cardAction 回调。

    `client` 需要 send_card(chat_id, card, parent_id=None) -> msg_id | None 与
    update_card(message_id, card) -> bool 两个方法（FeishuClient 提供；测试可用假件）。
    """

    def __init__(
        self,
        client: Any,
        *,
        approvers: frozenset[str] | None = None,
        timeout: float | None = None,
    ):
        self._client = client
        self.approvers = resolve_approvers() if approvers is None else approvers
        self.timeout = resolve_approval_timeout() if timeout is None else timeout
        self._pending: dict[str, _Pending] = {}

    def pending_count(self) -> int:
        return len(self._pending)

    async def request(
        self,
        chat_id: str,
        *,
        command: str,
        target: str,
        reason: str,
        asker_id: str | None = None,
        parent_id: str | None = None,
    ) -> bool:
        """发审批卡片并等待拍板；超时/发卡失败一律驳回（安全默认）。"""
        aid = uuid.uuid4().hex[:12]
        card = build_approval_card(
            approval_id=aid, command=command, target=target, reason=reason, asker_id=asker_id
        )
        msg_id = await self._client.send_card(chat_id, card, parent_id=parent_id)
        if msg_id is None:
            logger.error("审批卡片发送失败，按驳回处理：%s", command)
            return False
        pend = _Pending(command=command, target=target, reason=reason, card_msg_id=msg_id)
        self._pending[aid] = pend
        try:
            approved = await asyncio.wait_for(pend.future, timeout=self.timeout)
        except asyncio.TimeoutError:
            approved = False
            await self._finish_card(
                pend, approved=False, note=f"⏰ {self.timeout:.0f}s 内无人拍板，超时自动驳回。"
            )
            logger.warning("审批超时自动驳回：%s", command)
            return False
        finally:
            self._pending.pop(aid, None)
        await self._finish_card(pend, approved=approved)
        return approved

    async def on_card_action(self, event: Any) -> None:
        """cardAction 回调：解析按钮 value → 校验白名单 → resolve 对应 Future。"""
        parsed = parse_card_action_value(getattr(getattr(event, "action", None), "value", None))
        if parsed is None:
            return  # 不是审批按钮（其他卡片交互），忽略
        aid, approved = parsed
        pend = self._pending.get(aid)
        if pend is None or pend.future.done():
            return  # 已拍板/超时/重复点击，忽略
        operator = getattr(event, "operator", None)
        open_id = getattr(operator, "open_id", "") or ""
        if self.approvers and open_id not in self.approvers:
            logger.info("非审批人点击（%s），忽略", open_id)
            return
        pend.operator_name = getattr(operator, "name", None) or open_id
        pend.future.set_result(approved)
        logger.info(
            "审批拍板：%s → %s（by %s）", pend.command, "批准" if approved else "驳回", open_id
        )

    async def _finish_card(self, pend: _Pending, *, approved: bool, note: str = "") -> None:
        """把待批卡片原地替换成结果卡（移除按钮）。失败仅记日志，不影响主流程。"""
        if not pend.card_msg_id:
            return
        card = build_approval_result_card(
            command=pend.command,
            target=pend.target,
            reason=pend.reason,
            approved=approved,
            operator_name=pend.operator_name,
            note=note,
        )
        try:
            await self._client.update_card(pend.card_msg_id, card)
        except Exception:
            logger.exception("审批结果卡更新失败 msg=%s", pend.card_msg_id)
