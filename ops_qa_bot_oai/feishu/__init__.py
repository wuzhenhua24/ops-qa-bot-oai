"""飞书长连接接入（核心问答闭环 + 写操作审批 HITL）。

需要 `lark-oapi`：`uv sync --extra feishu`。入口见项目根 `run_ws.py`。
"""

from .approvals import ApprovalCenter
from .render import (
    RESET_WORDS,
    build_answer_post,
    build_approval_card,
    build_approval_result_card,
    clean_question,
    escalate_open_id,
    parse_card_action_value,
    placeholder_text,
)
from .runner import FeishuClient, WsRunner
from .session import SessionManager

__all__ = [
    "WsRunner",
    "FeishuClient",
    "SessionManager",
    "ApprovalCenter",
    "clean_question",
    "escalate_open_id",
    "build_answer_post",
    "build_approval_card",
    "build_approval_result_card",
    "parse_card_action_value",
    "placeholder_text",
    "RESET_WORDS",
]
