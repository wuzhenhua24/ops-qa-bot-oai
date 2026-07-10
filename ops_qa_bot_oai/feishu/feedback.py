"""使用者反馈收集：事件日志（feedback.log）+ 👍/👎 卡片点击处理。

三层结构（对齐 ops-qa-bot，事件字段按本项目更丰富的遥测扩展）：

1. **事件日志**：`log_event(name, **fields)` 往专用 logger 写一行
   `时间戳 + JSON`。事件类型：qa（每轮答题，含路由/按 agent 用量/复核元信息——
   这些是参考项目没有的维度）、qa_error、cancelled、feedback（up/down）、
   feedback_reason（原因多选+备注）、feedback_rejected、archive。
2. **反馈卡**：答完随答案发「👍 有帮助 / 👎 待改进」（asker-only）；👎 原地换成
   原因表单（白名单多选 + 可选备注 + 提交/跳过）。卡片渲染在 render.py。
3. **离线统计**：feedback_stats.py 解析日志出报表（满意率/被踩清单/路由分布/
   按 agent 成本拆分等）。

日志文件由 `setup_feedback_logger` 配置（WsRunner 启动时调用，路径读
OPS_QA_FEEDBACK_LOG，缺省 logs/feedback.log）；未配置时事件走普通 logging
传播（测试里 caplog 可捕获），不丢也不崩。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .render import (
    FEEDBACK_REASONS,
    build_feedback_ack_card,
    build_feedback_card,
    build_feedback_reason_card,
)

logger = logging.getLogger("ops_qa_bot_oai.feishu.feedback")

# 专用事件 logger：只发 JSON 行，handler 由 setup_feedback_logger 挂。
feedback_logger = logging.getLogger("ops_qa_bot_oai.feedback")


def setup_feedback_logger(log_path: str | Path | None = None) -> Path:
    """给事件 logger 挂文件 handler（幂等），返回实际日志路径。

    路径优先级：参数 > OPS_QA_FEEDBACK_LOG > logs/feedback.log。事件行自带
    时间戳前缀（feedback_stats 按前 10 字符取日期），不随根 logger 格式走；
    propagate 关掉，事件不刷进运行日志。
    """
    path = Path(log_path or os.environ.get("OPS_QA_FEEDBACK_LOG") or "logs/feedback.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    for h in feedback_logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == resolved:
            return path  # 已挂过同一文件（重复调用 / 多 runner），幂等跳过
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    feedback_logger.addHandler(handler)
    feedback_logger.setLevel(logging.INFO)
    feedback_logger.propagate = False
    return path


def log_event(event: str, **fields) -> None:
    """写一行结构化事件（JSON）。值为 None 的字段剥掉，行保持紧凑可 jq。"""
    payload = {"event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    feedback_logger.info(json.dumps(payload, ensure_ascii=False))


def excerpt(text: str | None, limit: int = 500) -> str | None:
    """日志摘要：折叠空白 + 截断。None 进 None 出（配合 log_event 剥 None）。"""
    if text is None:
        return None
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "…"


def handle_feedback_click(
    qid: str, rating: str, clicker_id: str | None, asker_id: str | None
) -> dict:
    """处理 👍/👎 点击，返回应替换原卡片的卡片。

    - 非提问者点击：拒绝（记 feedback_rejected），返回原反馈卡保持按钮可用——
      群里的卡谁都能点，不拦会污染 rating。
    - 👍：记 feedback(up)，返回 ack 卡（流程结束）。
    - 👎：记 feedback(down)，返回原因收集表单（第二跳回调记 reason）。
    """
    if clicker_id and asker_id and clicker_id != asker_id:
        log_event(
            "feedback_rejected", qid=qid, rating=rating, clicker_id=clicker_id, asker_id=asker_id
        )
        logger.info("feedback rejected (not asker): qid=%s by=%s", qid, clicker_id)
        return build_feedback_card(qid, asker_id)
    log_event("feedback", qid=qid, rating=rating, clicker_id=clicker_id, asker_id=asker_id)
    logger.info("feedback qid=%s rating=%s by=%s", qid, rating, clicker_id)
    if rating == "down":
        return build_feedback_reason_card(qid, asker_id)
    return build_feedback_ack_card("up")


def handle_feedback_reason_submit(
    qid: str | None,
    reasons: list[str] | None,
    comment: str | None,
    clicker_id: str | None,
    asker_id: str | None,
) -> dict:
    """处理 👎 原因表单提交，返回最终 ack 卡。

    reasons 按白名单过滤 + 保序去重；全部无效（None / 注入 / SDK 字段名变了）
    写一行 invalid 标记日志但仍返回 ack，别让 UI 卡死。非提问者提交返回原表单
    （不顶掉 asker 的表单、不污染数据）。
    """
    if clicker_id and asker_id and clicker_id != asker_id:
        log_event("feedback_rejected", qid=qid, clicker_id=clicker_id, asker_id=asker_id)
        return build_feedback_reason_card(qid or "", asker_id)
    cleaned: list[str] = []
    for r in reasons or []:
        if r in FEEDBACK_REASONS and r not in cleaned:
            cleaned.append(r)
    valid = bool(cleaned)
    log_event(
        "feedback_reason",
        qid=qid,
        reasons=cleaned if valid else None,
        reason_labels=[FEEDBACK_REASONS[r] for r in cleaned] if valid else None,
        comment=excerpt(comment) if comment else None,
        clicker_id=clicker_id,
        invalid=True if not valid else None,
    )
    logger.info("feedback reasons qid=%s reasons=%s", qid, cleaned)
    return build_feedback_ack_card("down")


def handle_feedback_reason_skip(
    qid: str | None, clicker_id: str | None, asker_id: str | None
) -> dict:
    """原因表单点「跳过」：记 skipped 事件返回 ack。

    skipped 比例是"这个二次表单值不值得留"的判据——绝大多数都跳过说明时机
    或选项不对。非提问者跳过返回原表单。
    """
    if clicker_id and asker_id and clicker_id != asker_id:
        log_event("feedback_rejected", qid=qid, clicker_id=clicker_id, asker_id=asker_id)
        return build_feedback_reason_card(qid or "", asker_id)
    log_event("feedback_reason", qid=qid, skipped=True, clicker_id=clicker_id)
    return build_feedback_ack_card("down")
