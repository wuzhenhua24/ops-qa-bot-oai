"""飞书长连接接入（核心问答闭环）。

需要 `lark-oapi`：`uv sync --extra feishu`。入口见项目根 `run_ws.py`。
"""

from .render import (
    RESET_WORDS,
    build_answer_post,
    clean_question,
    escalate_open_id,
    placeholder_text,
)
from .runner import FeishuClient, WsRunner
from .session import SessionManager

__all__ = [
    "WsRunner",
    "FeishuClient",
    "SessionManager",
    "clean_question",
    "escalate_open_id",
    "build_answer_post",
    "placeholder_text",
    "RESET_WORDS",
]
