"""飞书消息渲染：问题清洗 + markdown→post + @ 段拼装。

纯逻辑（除 markdown→post 调 lark-oapi 外都无外部依赖），便于单测。markdown 解析
委托给 lark-oapi 的 `markdown_to_post_ast`（覆盖全、跟着 SDK 演进），本模块只负责
把答案正文包成飞书 post，并按需在头部 @ 提问者、在升级时 @ 负责人。
"""

from __future__ import annotations

from typing import Any

# 会话重置关键词（与 CLI 对齐）。
RESET_WORDS = {"/reset", "/new", "新对话", "重置"}
# 跟进任务管理指令：列出自己挂起的定时跟进（带取消按钮）。匹配前先 lower。
TASKS_WORDS = {"/tasks", "跟进任务"}
# 取消在途提问：停掉自己正在处理/排队中的提问（发错了不用干等）。匹配前先 lower。
CANCEL_WORDS = {"/cancel", "取消"}


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


def placeholder_text(question: str, *, queued: bool = False) -> str:
    """占位消息文案：收到即发，答完编辑替换。

    queued=True 表示同用户前一条问题还没答完、本条在排队等锁，前缀用「🕒 排队中」
    让用户分辨哪条占位真的在跑；拿到锁开始答题时 runner 会再刷一次成「🔍 翻文档中」
    （见 SessionManager.answer 的 on_start 回调）。
    """
    snippet = question.strip().replace("\n", " ")
    if len(snippet) > 24:
        snippet = snippet[:24] + "…"
    icon = "🕒 排队中" if queued else "🔍 翻文档中"
    return f"{icon}：'{snippet}'"


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


def _card_value_dict(value: Any) -> dict | None:
    """cardAction 按钮 value 归一成 dict：飞书可能原样回传 dict，也可能是 JSON 字符串。"""
    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def parse_card_action_value(value: Any) -> tuple[str, bool] | None:
    """从 cardAction 的按钮 value 解析 (approval_id, 是否批准)；不是审批按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    aid = value.get("aid")
    decision = value.get("decision")
    if not aid or decision not in ("approve", "reject"):
        return None
    return str(aid), decision == "approve"


# ---------------------------------------------------------------------------
# 定时跟进：/tasks 任务列表卡（每条带取消按钮）
# ---------------------------------------------------------------------------


def _task_excerpt(task: str, limit: int = 80) -> str:
    task = " ".join(task.split())
    return task if len(task) <= limit else task[:limit] + "…"


def build_followup_tasks_card(
    asker_id: str,
    chat_id: str,
    items: list[dict[str, Any]],
    *,
    notice: str | None = None,
) -> dict[str, Any]:
    """/tasks 的跟进任务列表卡：每条挂起跟进一行状态 + 一颗取消按钮。

    `items` 每项：record_id / task / remaining_minutes / firing。执行中（firing）
    的条目只展示状态、不给取消按钮——那一轮已经在跑了。`notice` 是取消回调刷新
    卡片时挂在顶部的结果行（"✅ 已取消…"）；列表为空时整卡收尾成纯文本。
    asker-only 校验在回调侧做，按钮 value 带回 record_id 与归属。纯函数，可单测。
    """
    elements: list[dict[str, Any]] = []
    if notice:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": notice}})
    if not items:
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": "当前没有挂起的定时跟进了。"}}
        )
    else:
        head = f"⏰ 你有 **{len(items)}** 个挂起的定时跟进："
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": head}})
        for i, item in enumerate(items, 1):
            if item.get("firing"):
                status = "⏳ 正在执行，结果马上发出"
            else:
                m = int(item.get("remaining_minutes", 0))
                status = f"约 {m} 分钟后执行" if m > 0 else "即将执行（不足 1 分钟）"
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{i}.** {status}\n{_task_excerpt(str(item.get('task', '')))}",
                    },
                }
            )
            if not item.get("firing"):
                elements.append(
                    {
                        "tag": "action",
                        "actions": [
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "🗑 取消这条跟进"},
                                "type": "default",
                                "value": {
                                    "fua": str(item.get("record_id", "")),
                                    "chat": chat_id,
                                    "asker": asker_id,
                                },
                            }
                        ],
                    }
                )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "⏰ 定时跟进任务"},
        },
        "elements": elements,
    }


def parse_followup_cancel_value(value: Any) -> tuple[str, str, str] | None:
    """从 cardAction 按钮 value 解析 (record_id, chat_id, asker_id)；不是取消按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    rid, chat, asker = value.get("fua"), value.get("chat"), value.get("asker")
    if not rid or not chat or not asker:
        return None
    return str(rid), str(chat), str(asker)
