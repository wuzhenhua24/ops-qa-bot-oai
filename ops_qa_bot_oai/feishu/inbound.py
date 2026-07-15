"""入站消息解析：图片 / 富文本（post）→ 可答题的 (问题文本, 图片) 形态。

纯逻辑（无 IO、无 SDK 依赖），便于单测。移植自 `ops-qa-bot`（Claude 版）
feishu_core.py 的同名函数，行为对齐：

- media type 按 magic bytes 嗅探（下载 API 不暴露 Content-Type）；
- post AST 抽纯文本时跳过 @（路由标记不是问题主语）和 img（图走 resources）；
- 只发图无文字时用 DEFAULT_IMAGE_PROMPT 推动 agent"先识别再查文档"。
"""

from __future__ import annotations

# 视觉输入支持的图片类型：OpenAI vision 支持 png/jpeg/gif/webp（与 Anthropic 同集合）。
# 超过 5MB 大概率被 API 拒，让 bot 友好提示用户压缩，而不是甩 LLM 报错原文。
SUPPORTED_IMAGE_TYPES: set[str] = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# 用户只发图无文字时的默认引导问题："识别 → 找文档 → 答"。
DEFAULT_IMAGE_PROMPT = (
    "用户发了一张运维相关的截图。请先识别图中的关键信息"
    "（报错文本、命令、指标值、配置项、UI 状态等），"
    "再按这些线索去文档里查解决办法。"
)

# post 消息一次最多读取的图片张数。移动端"@bot + 多张截图"会打成 post，单条问题
# 塞多图既贵也违反视觉上下文人体工学；超出截断走日志告警，不报给用户。
POST_MAX_IMAGES = 5


def normalize_image_media_type(content_type: str, data: bytes) -> str:
    """归一化 image media_type 到 vision API 接受的集合内。

    优先看给定的 Content-Type；不在白名单时按 magic bytes 嗅探（PNG/JPEG/GIF/WebP
    都有唯一头）；都识别不出来 fallback 到 image/jpeg（飞书截图绝大多数是 jpeg）。
    """
    if content_type in SUPPORTED_IMAGE_TYPES:
        return content_type
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def extract_image_caption(content_dict: dict) -> str | None:
    """从 image 消息 content 里抽 caption-like 字段。

    飞书标准 image 消息 schema 只有 `{"image_key": "..."}`，没有 caption 字段；
    但转发 / 富文本中转 / 部分第三方客户端会把说明文字塞到 caption / text /
    description 之一。全部尝试，第一个非空字符串就用，都没有返回 None。
    """
    for k in ("caption", "text", "description"):
        v = content_dict.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _iter_post_documents(post: dict) -> list[dict]:
    """post AST 两种形态归一：返回 locale doc 列表。

    可能是单 locale（``{"title":..., "content":[[...]]}``）或带 locale 包裹
    （``{"zh_cn": {...}}``），与 SDK ``normalize/converters/post.py`` 行为一致。
    """
    if not isinstance(post, dict) or not post:
        return []
    if "content" in post:
        return [post]
    return [doc for doc in post.values() if isinstance(doc, dict)]


# @所有人 在 post AST 的 ``at`` 元素里 ``user_id`` 实测取值。线上抓包（2026-06-24，
# ops-qa-bot）是 ``@_all``；SDK 文档的 ``is_mention_all`` 又按 ``all`` 判——两种都收，
# 避免再被 wire 取值变体绕过。普通 @某人 是 open_id（``ou_xxx``），不会撞这两个值。
_POST_AT_ALL_IDS = frozenset({"all", "@_all"})


def post_mention_all_ids(post: dict) -> tuple[bool, set[str]]:
    """扫 post AST 的 ``at`` 元素，返回 ``(是否含@所有人, 其余被@到的 id 集合)``。

    富文本里 @所有人 是 ``{"tag":"at","user_id":"@_all"}``、@某人是 ``user_id=ou_xxx``。
    SDK 的 ``mentioned_all`` 只认纯文本里的 ``@_all`` 占位符——post 的 ``at`` 元素被
    ``converters/post.py`` 直接渲染成字面 ``@所有人``，all 信号在转换时就丢了，于是
    post 形态的 @所有人（典型场景：@所有人 + 截图发通知）绕过 channel PolicyGate
    （``policy_mention_all_blocked`` 不触发）。这里直接走 raw AST 兜底识别，不依赖
    SDK 的 mention 抽取。

    ``user_id`` 字段在飞书 post wire 里装的就是 open_id（``ou_xxx``）；@所有人 是
    ``_POST_AT_ALL_IDS`` 里的特例。``open_id`` 作历史形态兜底。
    """
    has_all = False
    at_ids: set[str] = set()
    for doc in _iter_post_documents(post):
        for para in doc.get("content") or []:
            if not isinstance(para, list):
                continue
            for el in para:
                if not isinstance(el, dict) or el.get("tag") != "at":
                    continue
                uid = el.get("user_id") or el.get("open_id") or ""
                if uid in _POST_AT_ALL_IDS:
                    has_all = True
                elif uid:
                    at_ids.add(uid)
    return has_all, at_ids


def parse_post_text(post_ast: dict) -> str:
    """从 post AST 抽出供 LLM 阅读的纯文本（image_key 由 ``inbound.resources``
    给出，不在这里重复抽）。

    走 AST 而不复用 SDK 的 ``PostContent.text``：SDK 的 flatten 把 ``tag:at``
    渲染成 ``@user_name`` 占位、把 ``tag:img`` 渲染成 ``[image]`` 文本——bot
    希望 @ 整段跳过（多数场景就是 @bot 自己）、image 单独走 resources。

    - tag:text / tag:a → 取显示文本（href 不暴露给 LLM，简化注入面）
    - tag:at / tag:img → 跳过
    """
    text_lines: list[str] = []
    for doc in _iter_post_documents(post_ast):
        for para in doc.get("content") or []:
            if not isinstance(para, list):
                continue
            line_parts: list[str] = []
            for el in para:
                if not isinstance(el, dict):
                    continue
                if el.get("tag") in ("text", "a"):
                    t = el.get("text")
                    if isinstance(t, str):
                        line_parts.append(t)
                # at / img / emotion / hr 等暂不抽取
            line = "".join(line_parts).strip()
            if line:
                text_lines.append(line)
    return "\n".join(text_lines).strip()


def compose_image_question(text: str | None, n_images: int) -> str:
    """把「用户文字 + 附图」拼成引导问题。

    有文字时把文字当真正的问题、附一段"先识图再查文档"的格式说明；只有图时
    退化成 DEFAULT_IMAGE_PROMPT。
    """
    text = (text or "").strip()
    if not text:
        return DEFAULT_IMAGE_PROMPT
    unit = "一张截图" if n_images == 1 else f"{n_images} 张截图"
    return (
        f"{text}\n\n"
        f"（同时附了{unit}。先从图里识别关键信息（报错 / 命令 / 指标 / 配置等），"
        "再结合上面的问题去文档里查解决办法。）"
    )
