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
