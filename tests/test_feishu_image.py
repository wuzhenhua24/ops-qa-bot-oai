"""图片提问链路的回归测试（纯逻辑 + 假 channel 的 runner 分发，无需 LLM / 网络 / 飞书）。

覆盖：
- normalize_image_media_type：白名单直通 / magic bytes 嗅探 / 兜底 jpeg。
- extract_image_caption：caption/text/description 依次尝试 / 全空返回 None。
- parse_post_text：text+a 拼接 / at 与 img 跳过 / 两种 AST 形态（带 locale 包裹与否）。
- compose_image_question：有文字 / 无文字（DEFAULT_IMAGE_PROMPT）/ 多图张数。
- build_user_input：无图返回原字符串；有图返回单条 user 消息，图块在前、文本在后，
  data URI 前缀正确。
- WsRunner._handle 分发：image 消息走视觉答题 / 下载失败与超大图回友好提示不进答题 /
  post 文字+图组合 / post 单图失败只丢那一张 / 纯 sticker post 回 unsupported。

跑法：
    uv run pytest tests/test_feishu_image.py
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

from lark_oapi.channel.types import ImageContent, PostContent, ResourceDescriptor

from ops_qa_bot_oai.bot import build_user_input
from ops_qa_bot_oai.feishu.inbound import (
    DEFAULT_IMAGE_PROMPT,
    compose_image_question,
    extract_image_caption,
    normalize_image_media_type,
    parse_post_text,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
_GIF = b"GIF89a" + b"\x00" * 8
_WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8


# ---------------------------------------------------------------------------
# normalize_image_media_type
# ---------------------------------------------------------------------------


def test_media_type_whitelist_passthrough():
    assert normalize_image_media_type("image/webp", _PNG) == "image/webp"


def test_media_type_magic_sniffing():
    assert normalize_image_media_type("", _PNG) == "image/png"
    assert normalize_image_media_type("", _JPEG) == "image/jpeg"
    assert normalize_image_media_type("", _GIF) == "image/gif"
    assert normalize_image_media_type("", _WEBP) == "image/webp"


def test_media_type_fallback_jpeg():
    # Content-Type 不在白名单 + 嗅探不出 → 兜底 jpeg（飞书截图绝大多数是 jpeg）
    assert normalize_image_media_type("application/octet-stream", b"garbage") == "image/jpeg"


# ---------------------------------------------------------------------------
# extract_image_caption
# ---------------------------------------------------------------------------


def test_caption_priority_and_strip():
    assert extract_image_caption({"caption": " 报错截图 ", "text": "x"}) == "报错截图"
    assert extract_image_caption({"text": "描述"}) == "描述"
    assert extract_image_caption({"description": "d"}) == "d"


def test_caption_absent_or_blank():
    assert extract_image_caption({"image_key": "k"}) is None
    assert extract_image_caption({"caption": "   ", "text": ""}) is None
    assert extract_image_caption({"caption": 123}) is None


# ---------------------------------------------------------------------------
# parse_post_text
# ---------------------------------------------------------------------------

_POST_DOC = {
    "title": "t",
    "content": [
        [
            {"tag": "at", "user_id": "ou_bot"},
            {"tag": "text", "text": "Redis 报这个错 "},
            {"tag": "a", "text": "看链接", "href": "https://example.com"},
        ],
        [{"tag": "img", "image_key": "img_k1"}],
        [{"tag": "text", "text": "  怎么处理？"}],
    ],
}


def test_parse_post_text_skips_at_and_img():
    text = parse_post_text(_POST_DOC)
    assert text == "Redis 报这个错 看链接\n怎么处理？"
    assert "ou_bot" not in text and "img_k1" not in text
    assert "https://example.com" not in text  # href 不暴露给 LLM


def test_parse_post_text_locale_wrapped():
    assert parse_post_text({"zh_cn": _POST_DOC}) == parse_post_text(_POST_DOC)


def test_parse_post_text_empty_and_malformed():
    assert parse_post_text({}) == ""
    assert parse_post_text({"content": ["not-a-list", [{"tag": "img"}]]}) == ""


# ---------------------------------------------------------------------------
# compose_image_question
# ---------------------------------------------------------------------------


def test_compose_image_question_no_text_uses_default():
    assert compose_image_question(None, 1) == DEFAULT_IMAGE_PROMPT
    assert compose_image_question("  ", 3) == DEFAULT_IMAGE_PROMPT


def test_compose_image_question_with_text():
    q = compose_image_question("这是什么报错？", 1)
    assert q.startswith("这是什么报错？")
    assert "一张截图" in q
    q3 = compose_image_question("看看", 3)
    assert "3 张截图" in q3


# ---------------------------------------------------------------------------
# build_user_input
# ---------------------------------------------------------------------------


def test_build_user_input_text_only_is_passthrough():
    assert build_user_input("问题") == "问题"
    assert build_user_input("问题", None) == "问题"
    assert build_user_input("问题", []) == "问题"


def test_build_user_input_with_images():
    items = build_user_input("问题", [("image/png", _PNG), ("image/jpeg", _JPEG)])
    assert isinstance(items, list) and len(items) == 1
    msg = items[0]
    assert msg["role"] == "user"
    blocks = msg["content"]
    # 图块在前、文本在后（先看图再读引导问题）
    assert [b["type"] for b in blocks] == ["input_image", "input_image", "input_text"]
    assert blocks[-1]["text"] == "问题"
    b64 = base64.b64encode(_PNG).decode("ascii")
    assert blocks[0]["image_url"] == f"data:image/png;base64,{b64}"
    assert blocks[0]["detail"] == "auto"


# ---------------------------------------------------------------------------
# WsRunner._handle 分发（假 client + 假 session，不建真 channel）
# ---------------------------------------------------------------------------


class _FakeClient:
    """假 FeishuClient：记录发出/更新的消息，图片下载查 `images` 字典（None=下载失败）。"""

    def __init__(self, images: dict[str, bytes | None] | None = None):
        self.images = images or {}
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_texts: list[tuple[str, str]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return "ph1"

    async def update_post(self, message_id, post):
        self.updated_posts.append((message_id, post))
        return True

    async def send_text(self, chat_id, text, *, parent_id=None):
        self.sent_texts.append((chat_id, text))
        return "t1"

    async def download_image(self, message_id, file_key):
        return self.images.get(file_key)


class _RecordingSession:
    """假 SessionManager：记录 answer() 收到的 (question, images)。"""

    guardrails = False

    def __init__(self):
        self.calls: list[tuple[str, list | None]] = []

    async def answer(self, key, question, approver=None, images=None):
        self.calls.append((question, images))
        return SimpleNamespace(
            text="答案",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )


def _runner(images: dict[str, bytes | None] | None = None):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient(images)
    r._session = _RecordingSession()
    r._approvals = SimpleNamespace()
    return r


def _inbound(content, resources=()):
    return SimpleNamespace(
        sender=SimpleNamespace(is_bot=False),
        chat_id="oc_chat",
        sender_id="ou_user",
        message_id="om_msg",
        content=content,
        mentions=[],
        resources=list(resources),
    )


async def test_runner_image_message_answers_with_vision():
    r = _runner({"k1": _PNG})
    await r._handle(_inbound(ImageContent(raw={"image_key": "k1"}, image_key="k1")))
    assert len(r._session.calls) == 1
    question, images = r._session.calls[0]
    assert DEFAULT_IMAGE_PROMPT == question  # 无 caption → 默认引导问题
    assert images == [("image/png", _PNG)]
    assert r._client.updated_posts, "答案未编辑回占位"


async def test_runner_image_caption_becomes_question():
    r = _runner({"k1": _JPEG})
    content = ImageContent(raw={"image_key": "k1", "caption": "这是什么报错"}, image_key="k1")
    await r._handle(_inbound(content))
    question, images = r._session.calls[0]
    assert question.startswith("这是什么报错")
    assert images[0][0] == "image/jpeg"


async def test_runner_image_download_failure_replies_hint():
    r = _runner({})  # k1 缺失 → download_image 返回 None
    await r._handle(_inbound(ImageContent(raw={"image_key": "k1"}, image_key="k1")))
    assert not r._session.calls, "下载失败不应进答题"
    assert any("图片读取失败" in t for _, t in r._client.sent_texts)


async def test_runner_image_too_large_replies_hint():
    from ops_qa_bot_oai.feishu.inbound import MAX_IMAGE_BYTES

    r = _runner({"k1": b"\xff\xd8\xff" + b"\x00" * MAX_IMAGE_BYTES})
    await r._handle(_inbound(ImageContent(raw={"image_key": "k1"}, image_key="k1")))
    assert not r._session.calls
    assert any("图片太大" in t for _, t in r._client.sent_texts)


async def test_runner_post_text_plus_images():
    post = {
        "zh_cn": {
            "title": "",
            "content": [
                [
                    {"tag": "at", "user_id": "ou_bot"},
                    {"tag": "text", "text": "redis 报这个错怎么办"},
                ],
                [{"tag": "img", "image_key": "k1"}],
            ],
        }
    }
    r = _runner({"k1": _PNG, "k2": None})  # k2 下载失败 → 只丢那一张
    content = PostContent(raw={}, post=post)
    resources = [
        ResourceDescriptor(type="image", file_key="k1"),
        ResourceDescriptor(type="image", file_key="k2"),
    ]
    await r._handle(_inbound(content, resources))
    question, images = r._session.calls[0]
    assert question.startswith("redis 报这个错怎么办")
    assert "ou_bot" not in question
    assert images == [("image/png", _PNG)]


async def test_runner_post_without_usable_content_replies_unsupported():
    from ops_qa_bot_oai.feishu.runner import _UNSUPPORTED

    r = _runner({})
    await r._handle(_inbound(PostContent(raw={}, post={"zh_cn": {"title": "", "content": []}})))
    assert not r._session.calls
    assert (r._client.sent_texts or [("", "")])[-1][1] == _UNSUPPORTED
