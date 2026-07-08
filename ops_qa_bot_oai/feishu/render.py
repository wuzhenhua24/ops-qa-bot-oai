"""飞书消息渲染：问题清洗 + markdown→post + @ 段拼装。

纯逻辑（除 markdown→post 调 lark-oapi 外都无外部依赖），便于单测。markdown 解析
委托给 lark-oapi 的 `markdown_to_post_ast`（覆盖全、跟着 SDK 演进），本模块只负责
把答案正文包成飞书 post，并按需在头部 @ 提问者、在升级时 @ 负责人。
"""

from __future__ import annotations

from typing import Any

# 会话重置关键词（与 CLI 对齐）。
RESET_WORDS = {"/reset", "/new", "新对话", "重置"}


def clean_question(raw_text: str, mention_keys: list[str]) -> str:
    """从入站文本里剥掉 @bot 占位符（mentions[].key），返回纯问题文本。

    飞书把 @机器人 在原始文本里表示成 `@_user_1` 之类的占位 key；逐个 replace 掉。
    """
    q = raw_text or ""
    for key in mention_keys:
        if key:
            q = q.replace(key, "")
    return q.strip()


def escalate_open_id(payload: str | None) -> str | None:
    """从 ESCALATE marker 负载里抠出负责人 open_id。

    负载形如 `ou_xxx:redis`（owner:目录）或 `none`/空。返回 `ou_xxx` 或 None。
    """
    if not payload:
        return None
    head = payload.split(":", 1)[0].strip()
    return head if head.startswith("ou_") else None


def _render_body(markdown: str) -> list[list[dict[str, Any]]]:
    """markdown → 飞书 post 段落列表（委托 lark-oapi）。"""
    from lark_oapi.channel.outbound.markdown import markdown_to_post_ast

    ast = markdown_to_post_ast(markdown or "")
    content = ast.get("zh_cn", {}).get("content") or []
    return content if content else [[{"tag": "text", "text": markdown or ""}]]


def build_answer_post(
    markdown: str,
    *,
    asker_id: str | None = None,
    escalate_to: str | None = None,
    title: str = "",
) -> dict[str, Any]:
    """把答案 markdown 包成飞书 post：头部 @提问者，升级时末尾 @负责人。

    返回结构可直接作为 `channel.send/edit_message` 的 `{"post": ...}` 负载。
    """
    paragraphs: list[list[dict[str, Any]]] = []
    if asker_id:
        # 头部 @ 提问者 + 一个空格，群里多人并行提问时一眼看出归属。
        paragraphs.append([{"tag": "at", "user_id": asker_id}, {"tag": "text", "text": " "}])
    paragraphs.extend(_render_body(markdown))
    if escalate_to:
        paragraphs.append(
            [
                {"tag": "text", "text": "📌 文档未覆盖，已请 "},
                {"tag": "at", "user_id": escalate_to},
                {"tag": "text", "text": " 协助处理。"},
            ]
        )
    return {"zh_cn": {"title": title, "content": paragraphs}}


def placeholder_text(question: str, *, clarifying: bool = False) -> str:
    """占位消息文案：收到即发，答完编辑替换。"""
    snippet = question.strip().replace("\n", " ")
    if len(snippet) > 24:
        snippet = snippet[:24] + "…"
    return f"🔍 翻文档中：'{snippet}'"


# ---------------------------------------------------------------------------
# 写操作审批卡片（HITL）：提议 → 待批卡片（带按钮）→ 拍板后改成结果卡片
# ---------------------------------------------------------------------------


def _approval_fields(command: str, target: str, reason: str) -> dict[str, Any]:
    md = f"**目标**：{target}\n**命令**：`{command}`\n**理由**：{reason}"
    return {"tag": "div", "text": {"tag": "lark_md", "content": md}}


def build_approval_card(
    *, approval_id: str, command: str, target: str, reason: str, asker_id: str | None = None
) -> dict[str, Any]:
    """待批卡片：命令详情 + 批准/驳回按钮。按钮 value 带 approval_id + decision，
    cardAction 回调据此定位待批项。纯函数，可单测。"""
    elements: list[dict[str, Any]] = []
    if asker_id:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"提问者：<at id={asker_id}></at>"},
            }
        )
    elements.append(_approval_fields(command, target, reason))
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 批准"},
                    "type": "primary",
                    "value": {"aid": approval_id, "decision": "approve"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🚫 驳回"},
                    "type": "danger",
                    "value": {"aid": approval_id, "decision": "reject"},
                },
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {"tag": "plain_text", "content": "⚠️ 写操作审批：agent 提议执行变更"},
        },
        "elements": elements,
    }


def build_approval_result_card(
    *,
    command: str,
    target: str,
    reason: str,
    approved: bool,
    operator_name: str = "",
    note: str = "",
) -> dict[str, Any]:
    """结果卡片：拍板/超时后替换待批卡片（按钮移除，防重复点击）。纯函数，可单测。"""
    verdict = "✅ 已批准（登记为待人工执行）" if approved else "🚫 已驳回"
    who = f"，由 {operator_name} 拍板" if operator_name else ""
    lines = f"{verdict}{who}。" + (f"\n{note}" if note else "")
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green" if approved else "red",
            "title": {"tag": "plain_text", "content": "写操作审批结果"},
        },
        "elements": [
            _approval_fields(command, target, reason),
            {"tag": "div", "text": {"tag": "lark_md", "content": lines}},
        ],
    }


def parse_card_action_value(value: Any) -> tuple[str, bool] | None:
    """从 cardAction 的按钮 value 解析 (approval_id, 是否批准)；不是审批按钮返回 None。

    飞书可能把 value 原样回传 dict，也可能是 JSON 字符串——两种都容。纯函数，可单测。
    """
    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, dict):
        return None
    aid = value.get("aid")
    decision = value.get("decision")
    if not aid or decision not in ("approve", "reject"):
        return None
    return str(aid), decision == "approve"
