"""post（富文本）形态 @所有人 过滤的回归测试（移植自 ops-qa-bot 同名用例）。

主因：@所有人 过滤交给 channel PolicyGate（``respond_to_mention_all=False``），但
PolicyGate 的 ``mentioned_all`` 只认纯文本里的 ``@_all`` 占位符——post 里 @所有人
是 ``{"tag":"at","user_id":"@_all"}`` 元素，被 SDK 的 post 转换器渲染成字面
``@所有人``，all 信号丢失，于是 post 形态（典型场景：@所有人 + 截图发通知）漏过
PolicyGate，bot 把全员广播当问题答了。

修复：`_extract_question` 的 PostContent 分支用 ``post_mention_all_ids`` 走 raw AST
兜底——纯 @所有人 广播静默不答；"@所有人 + 同时单独 @bot" 仍放行（对齐 text 路径
语义）；bot identity 未解析时 fail-safe 按广播丢。

跑法：
    uv run pytest tests/test_post_mention_all.py
"""

from __future__ import annotations

from types import SimpleNamespace

from lark_oapi.channel.types import PostContent, ResourceDescriptor

from ops_qa_bot_oai.feishu.inbound import post_mention_all_ids

BOT = "ou_bot_self"

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


# ---------------------------------------------------------------------------
# 纯函数：post_mention_all_ids 直接吃 AST
# ---------------------------------------------------------------------------


def test_at_all_only():
    post = {"title": "", "content": [[
        {"tag": "at", "user_id": "all", "user_name": "所有人"},
        {"tag": "text", "text": " 各同事注意"},
    ]]}
    has_all, ids = post_mention_all_ids(post)
    assert has_all is True
    assert ids == set()


def test_at_all_plus_bot():
    post = {"title": "", "content": [[
        {"tag": "at", "user_id": "all", "user_name": "所有人"},
        {"tag": "at", "user_id": BOT, "user_name": "ops-qa-bot"},
        {"tag": "text", "text": " 帮我看下"},
    ]]}
    has_all, ids = post_mention_all_ids(post)
    assert has_all is True
    assert BOT in ids


def test_at_someone_not_all():
    post = {"title": "", "content": [[
        {"tag": "at", "user_id": "ou_zhang", "user_name": "张三"},
        {"tag": "text", "text": " 你看下"},
    ]]}
    has_all, ids = post_mention_all_ids(post)
    assert has_all is False
    assert ids == {"ou_zhang"}


def test_no_at():
    post = {"title": "", "content": [[{"tag": "text", "text": "纯文字"}]]}
    has_all, ids = post_mention_all_ids(post)
    assert has_all is False
    assert ids == set()


def test_locale_wrapped_ast():
    # 飞书最新 wire / SDK fixture 形态：locale key 包一层
    post = {"zh_cn": {"title": "", "content": [[
        {"tag": "at", "user_id": "all"},
        {"tag": "text", "text": " hi"},
    ]]}}
    has_all, _ = post_mention_all_ids(post)
    assert has_all is True


def test_open_id_fallback_field():
    # 历史形态：at 元素用 open_id 字段而非 user_id
    post = {"content": [[{"tag": "at", "open_id": "all"}]]}
    has_all, _ = post_mention_all_ids(post)
    assert has_all is True


def test_at_all_real_wire_underscore():
    # 线上真实 wire（2026-06-24 ops-qa-bot 抓包）：@所有人 的 user_id 是 "@_all"
    # 而非文档约定的 "all"——关键回归，最初漏拦就是因为只认了 "all"
    post = {"content": [[
        {"tag": "at", "user_id": "@_all", "user_name": "所有人", "style": []},
        {"tag": "text", "text": " ", "style": []},
    ]]}
    has_all, ids = post_mention_all_ids(post)
    assert has_all is True
    assert ids == set()


# ---------------------------------------------------------------------------
# 端到端：WsRunner._handle 的 PostContent 分支 drop / 放行
# （假 client + 假 session + 假 channel，复用 test_feishu_image 的桩风格）
# ---------------------------------------------------------------------------


class _FakeClient:
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
    guardrails = False

    def __init__(self):
        self.calls: list[tuple[str, list | None]] = []

    def queued(self, key):
        return False

    async def take_expired_notice(self, key):
        return False

    def register_inflight(self, key, scope):
        return "sid"

    def unregister_inflight(self, key, scope_id):
        pass

    async def answer(self, key, question, approver=None, images=None, on_start=None):
        self.calls.append((question, images))
        return SimpleNamespace(
            text="答案",
            markers=SimpleNamespace(escalate=None),
            usage=None,
            num_turns=1,
            subtype="success",
        )


def _runner(images: dict[str, bytes | None] | None = None, *, bot_open_id: str | None = BOT):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient(images)
    r._session = _RecordingSession()
    r._approvals = SimpleNamespace()
    identity = SimpleNamespace(open_id=bot_open_id, user_id=None) if bot_open_id else None
    r._channel = SimpleNamespace(bot_identity=identity)
    return r


def _inbound(content, resources=(), mentions=()):
    return SimpleNamespace(
        sender=SimpleNamespace(is_bot=False),
        chat_id="oc_chat",
        sender_id="ou_asker",
        message_id="om_msg",
        content=content,
        mentions=list(mentions),
        resources=list(resources),
    )


async def test_runner_at_all_plus_image_dropped():
    # 线上 case：@所有人(user_id="@_all") + 图 + 文字，未单独 @bot → 静默不答题，
    # 也不下载图、不回任何提示。
    post = {"title": "", "content": [
        [{"tag": "at", "user_id": "@_all", "user_name": "所有人", "style": []},
         {"tag": "text", "text": " ", "style": []}],
        [{"tag": "img", "image_key": "k1", "width": 1263, "height": 347}],
        [{"tag": "text", "text": " 各同事，禁止压测。", "style": []}],
    ]}
    r = _runner({"k1": _PNG})
    await r._handle(_inbound(
        PostContent(raw={}, post=post),
        resources=[ResourceDescriptor(type="image", file_key="k1")],
    ))
    assert not r._session.calls, "post @所有人 广播不应进答题"
    assert not r._client.sent_texts and not r._client.sent_posts, "广播应静默丢弃"


async def test_runner_pure_at_all_text_dropped():
    post = {"content": [[
        {"tag": "at", "user_id": "all", "user_name": "所有人"},
        {"tag": "text", "text": " 分支环境请关闭回收。"},
    ]]}
    r = _runner()
    await r._handle(_inbound(PostContent(raw={}, post=post)))
    assert not r._session.calls


async def test_runner_at_all_plus_bot_answered():
    # @所有人 + 同时单独 @bot → 当定向求助，仍答题（对齐 text 路径）
    post = {"content": [[
        {"tag": "at", "user_id": "all", "user_name": "所有人"},
        {"tag": "at", "user_id": BOT, "user_name": "ops-qa-bot"},
        {"tag": "text", "text": " 帮我看下这个报错"},
    ]]}
    r = _runner()
    await r._handle(_inbound(
        PostContent(raw={}, post=post),
        mentions=[SimpleNamespace(key="@_user_1", open_id=BOT, name="bot")],
    ))
    assert len(r._session.calls) == 1
    assert r._session.calls[0][0].startswith("帮我看下这个报错")


async def test_runner_at_all_plus_bot_answered_even_if_mentions_empty():
    # 兜底：飞书未填 mentions 数组时，仍靠 AST 里的 bot at 元素识别 @bot
    post = {"content": [[
        {"tag": "at", "user_id": "all"},
        {"tag": "at", "user_id": BOT},
        {"tag": "text", "text": " 看下"},
    ]]}
    r = _runner()
    await r._handle(_inbound(PostContent(raw={}, post=post)))
    assert len(r._session.calls) == 1


async def test_runner_no_at_all_answered():
    # 普通 post（无 @所有人）：正常答题，不受影响
    post = {"content": [[{"tag": "text", "text": "这个接口怎么调用"}]]}
    r = _runner()
    await r._handle(_inbound(PostContent(raw={}, post=post)))
    assert len(r._session.calls) == 1


async def test_runner_at_all_unresolved_bot_identity_dropped():
    # bot identity 未解析时 fail-safe：@所有人 一律 drop，宁可漏答不误答广播
    post = {"content": [[
        {"tag": "at", "user_id": "all"},
        {"tag": "at", "user_id": BOT},
        {"tag": "text", "text": " 看下"},
    ]]}
    r = _runner(bot_open_id=None)
    await r._handle(_inbound(PostContent(raw={}, post=post)))
    assert not r._session.calls
