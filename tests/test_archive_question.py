"""问答归档闭环回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- parse_markers 的 ARCHIVE_Q 解析：命中/未命中/空白折叠/空负载/超长截断/不吃尖括号，
  以及与 ESCALATE 共存互不干扰。
- safe_component_dir：合法目录 / 不存在 / 路径穿越与非法字符一律拒。
- ArchiveStore：登记/取/弹、TTL 过期、容量上限逐出最老。
- write_qa_archive：块结构、qid 幂等、连续追加。
- 渲染纯函数：表单卡（预填/按钮 value）、提交 value 解析、通知 post、form_value 抽取。
- handle_archive_submit：参数缺失/过期/非负责人（重建表单不顶掉）/空答案/超长/
  成功写盘+通知+清 pending/重复提交/标题回退。
- runner 全链路：升级答案 → 表单卡发出 → 负责人提交 → qa-archive.md 落盘 +
  ack 替换表单 + @ 提问者推送答案。

跑法：
    uv run pytest tests/test_archive_question.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

from ops_qa_bot_oai.bot import parse_markers  # noqa: E402
from ops_qa_bot_oai.feishu.archive import (  # noqa: E402
    ArchiveStore,
    handle_archive_submit,
    safe_component_dir,
    write_qa_archive,
)
from ops_qa_bot_oai.feishu.render import (  # noqa: E402
    build_archive_form_card,
    build_archive_notify_post,
    escalate_dir,
    extract_form_value,
    parse_archive_submit_value,
)

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")
OWNER = "ou_owner_redis_01"


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _docs_root(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "redis").mkdir(parents=True)
    (root / "INDEX.md").write_text("# 索引\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# parse_markers：ARCHIVE_Q
# ---------------------------------------------------------------------------


def test_archive_q_hit_and_escalate_untouched():
    text = "找不到相关内容。\n\n<<ESCALATE:ou_x:redis>>\n<<ARCHIVE_Q:Redis 集群跨机房迁移步骤>>"
    cleaned, markers = parse_markers(text)
    assert markers.archive_q == "Redis 集群跨机房迁移步骤"
    assert markers.escalate == "ou_x:redis"
    assert "<<ARCHIVE_Q" not in cleaned and "<<ESCALATE" not in cleaned


def test_archive_q_miss():
    cleaned, markers = parse_markers("普通答案，没有标记。")
    assert markers.archive_q is None
    assert cleaned == "普通答案，没有标记。"


def test_archive_q_collapses_whitespace():
    _, markers = parse_markers("x <<ARCHIVE_Q:  Redis   内存\t告警  >> y")
    assert markers.archive_q == "Redis 内存 告警"


def test_archive_q_empty_after_strip():
    cleaned, markers = parse_markers("x <<ARCHIVE_Q:   >> y")
    assert markers.archive_q is None
    assert "<<ARCHIVE_Q" not in cleaned  # 空负载照样剥掉


def test_archive_q_overlong_truncated():
    _, markers = parse_markers(f"<<ARCHIVE_Q:{'标' * 200}>>")
    assert markers.archive_q is not None
    assert len(markers.archive_q) == 101  # 100 + "…"
    assert markers.archive_q.endswith("…")


def test_archive_q_does_not_eat_angle_brackets():
    _, markers = parse_markers("<<ARCHIVE_Q:bad<value>>")
    assert markers.archive_q is None  # 负载不吃 < >，不会误匹配


# ---------------------------------------------------------------------------
# escalate_dir / safe_component_dir
# ---------------------------------------------------------------------------


def test_escalate_dir_split():
    assert escalate_dir("ou_x:redis") == "redis"
    assert escalate_dir("ou_x") is None
    assert escalate_dir("none") is None
    assert escalate_dir(None) is None


def test_safe_component_dir(tmp_path):
    root = _docs_root(tmp_path)
    assert safe_component_dir(root, "redis") == "redis"
    assert safe_component_dir(root, " redis/ ") == "redis"  # 容忍空白和尾斜杠
    assert safe_component_dir(root, "mysql") is None  # 目录不存在
    assert safe_component_dir(root, "../etc") is None  # 穿越
    assert safe_component_dir(root, "a/b") is None  # 多级
    assert safe_component_dir(root, ".hidden") is None  # 点开头
    assert safe_component_dir(root, "") is None
    assert safe_component_dir(root, None) is None


# ---------------------------------------------------------------------------
# ArchiveStore
# ---------------------------------------------------------------------------


def _register(store: ArchiveStore, **overrides):
    kw = dict(
        chat_id=KEY[0],
        asker_id=KEY[1],
        question="redis 跨机房咋迁",
        question_default="Redis 集群跨机房迁移步骤",
        owner_id=OWNER,
        component_dir="redis",
        parent_msg_id="om_q",
    )
    kw.update(overrides)
    return store.register(**kw)


def test_store_roundtrip_and_ttl():
    store = ArchiveStore(ttl=1000.0)
    rec = _register(store)
    assert store.get(rec.qid) is rec
    assert store.get("nope") is None
    assert store.get(None) is None
    rec.created_at = time.time() - 2000  # 时光倒流：过期
    assert store.get(rec.qid) is None
    assert store.pending_count() == 0  # 过期即清


def test_store_capacity_evicts_oldest():
    store = ArchiveStore(max_pending=2)
    r1, r2 = _register(store), _register(store)
    r3 = _register(store)
    assert store.get(r1.qid) is None  # 最老的被逐出
    assert store.get(r2.qid) and store.get(r3.qid)


def test_archive_path_repr():
    store = ArchiveStore()
    assert _register(store).archive_path_repr == "redis/qa-archive.md"
    assert _register(store, component_dir=None).archive_path_repr == "qa-archive.md"


# ---------------------------------------------------------------------------
# write_qa_archive
# ---------------------------------------------------------------------------


async def test_write_block_and_idempotent(tmp_path):
    path = tmp_path / "redis" / "qa-archive.md"
    wrote = await write_qa_archive(
        path,
        qid="q1",
        question="Redis 迁移步骤",
        answer="先建从库再切流。",
        owner_id=OWNER,
        asker_id=KEY[1],
    )
    assert wrote
    content = path.read_text(encoding="utf-8")
    assert "## Q: Redis 迁移步骤" in content
    assert f"回答者：<@{OWNER}>" in content and f"提问者：<@{KEY[1]}>" in content
    assert "qid: q1" in content and "先建从库再切流。" in content

    assert not await write_qa_archive(  # 同 qid 幂等跳过
        path, qid="q1", question="改标题也不重写", answer="x", owner_id=OWNER, asker_id=None
    )
    assert await write_qa_archive(  # 新 qid 追加
        path, qid="q2", question="第二问", answer="第二答", owner_id=OWNER, asker_id=None
    )
    content = path.read_text(encoding="utf-8")
    assert content.count("## Q:") == 2
    assert "改标题也不重写" not in content


# ---------------------------------------------------------------------------
# 渲染纯函数
# ---------------------------------------------------------------------------


def test_form_card_prefill_and_button_value():
    card = build_archive_form_card("qid123", "标" * 150, OWNER, "redis/qa-archive.md")
    flat = _flat(card)
    assert "redis/qa-archive.md" in flat and OWNER in flat
    form = card["body"]["elements"][1]
    q_input, a_input, btn = form["elements"]
    assert q_input["default_value"].endswith("…") and len(q_input["default_value"]) == 101
    assert a_input["input_type"] == "multiline_text"
    assert btn["behaviors"][0]["value"] == {"aq": "qid123"}
    assert parse_archive_submit_value(btn["behaviors"][0]["value"]) == "qid123"


def test_parse_submit_value_rejects_others():
    assert parse_archive_submit_value({"aid": "x", "decision": "approve"}) is None  # 审批按钮
    assert parse_archive_submit_value({"fua": "x"}) is None  # 跟进取消按钮
    assert parse_archive_submit_value("not-json") is None
    assert parse_archive_submit_value('{"aq": "q9"}') == "q9"  # JSON 字符串形态


def test_notify_post_structure():
    post = build_archive_notify_post(
        asker_id=KEY[1],
        owner_id=OWNER,
        question="Redis 迁移步骤",
        answer_markdown="1. 建从库\n2. 切流",
        archive_rel="redis/qa-archive.md",
    )
    paras = post["zh_cn"]["content"]
    assert paras[0][0] == {"tag": "at", "user_id": KEY[1]}  # 第一段 @ asker 才有提醒
    flat = _flat(post)
    assert "Q: Redis 迁移步骤" in flat and "redis/qa-archive.md" in flat and OWNER in flat


def test_extract_form_value_shapes():
    raw = {"event": {"action": {"form_value": {"question": "q", "answer": "a"}}}}
    assert extract_form_value(raw) == {"question": "q", "answer": "a"}
    assert extract_form_value({"event": {"action": {}}}) == {}
    assert extract_form_value(None) == {}
    assert extract_form_value("garbage") == {}


# ---------------------------------------------------------------------------
# handle_archive_submit
# ---------------------------------------------------------------------------


async def _submit(store, root, qid, *, question="", answer="整理后的答案", clicker=OWNER):
    return await handle_archive_submit(
        store, root, qid=qid, question=question, answer=answer, clicker_id=clicker
    )


async def test_submit_missing_and_expired_qid(tmp_path):
    root = _docs_root(tmp_path)
    store = ArchiveStore()
    card, notify = await _submit(store, root, None)
    assert "参数缺失" in _flat(card) and notify is None
    card, notify = await _submit(store, root, "no-such")
    assert "已过期" in _flat(card) and notify is None


async def test_submit_by_non_owner_keeps_form(tmp_path):
    root = _docs_root(tmp_path)
    store = ArchiveStore()
    rec = _register(store)
    card, notify = await _submit(store, root, rec.qid, clicker="ou_intruder")
    assert notify is None
    assert "archive_form" in _flat(card)  # 重建表单卡，负责人还能填
    assert store.get(rec.qid) is not None  # pending 没被顶掉
    assert not (root / "redis" / "qa-archive.md").exists()


async def test_submit_validation(tmp_path):
    root = _docs_root(tmp_path)
    store = ArchiveStore()
    rec = _register(store)
    card, _ = await _submit(store, root, rec.qid, answer="   ")
    assert "不能为空" in _flat(card)
    card, _ = await _submit(store, root, rec.qid, answer="x" * 10_001)
    assert "过长" in _flat(card)
    assert store.get(rec.qid) is not None  # 校验失败不消费 pending


async def test_submit_success_writes_notifies_and_pops(tmp_path):
    root = _docs_root(tmp_path)
    store = ArchiveStore()
    rec = _register(store)
    card, notify = await _submit(store, root, rec.qid, question="Redis 跨机房迁移（改后标题）")
    assert "✅" in _flat(card) and "redis/qa-archive.md" in _flat(card)
    content = (root / "redis" / "qa-archive.md").read_text(encoding="utf-8")
    assert "## Q: Redis 跨机房迁移（改后标题）" in content  # 用负责人改后的标题
    assert notify is not None
    n_chat, n_post, n_parent = notify
    assert n_chat == KEY[0] and n_parent == "om_q"
    assert "Redis 跨机房迁移（改后标题）" in _flat(n_post)
    assert store.get(rec.qid) is None  # 已消费
    # 再提交：pending 已清 → 过期 ack，不重复写、不重复通知
    card, notify = await _submit(store, root, rec.qid)
    assert "已过期" in _flat(card) and notify is None


async def test_submit_empty_question_falls_back_to_default(tmp_path):
    root = _docs_root(tmp_path)
    store = ArchiveStore()
    rec = _register(store, component_dir=None)
    card, notify = await _submit(store, root, rec.qid, question="   ")
    assert "✅" in _flat(card)
    content = (root / "qa-archive.md").read_text(encoding="utf-8")  # 无目录落根目录
    assert "## Q: Redis 集群跨机房迁移步骤" in content  # 回退到预填标题


# ---------------------------------------------------------------------------
# runner 全链路
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.updated_posts: list[tuple[str, dict]] = []
        self.sent_cards: list[tuple[str, dict]] = []
        self.updated_cards: list[tuple[str, dict]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return f"ph{len(self.sent_posts)}"

    async def update_post(self, message_id, post):
        self.updated_posts.append((message_id, post))
        return True

    async def send_card(self, chat_id, card, *, parent_id=None):
        self.sent_cards.append((chat_id, card))
        return "c1"

    async def update_card(self, message_id, card):
        self.updated_cards.append((message_id, card))
        return True


class _EscalatingSession:
    """假 SessionManager：答案带升级标记 + 归档标题。"""

    guardrails = False

    def queued(self, key):
        return False

    async def take_expired_notice(self, key):
        return False

    def register_inflight(self, key, scope):
        return "sid"

    def unregister_inflight(self, key, scope_id):
        pass

    async def answer(self, key, question, approver=None, images=None, on_start=None):
        return SimpleNamespace(
            text="文档中未找到相关内容，建议联系负责人。",
            markers=SimpleNamespace(
                escalate=f"{OWNER}:redis",
                clarify=False,
                archive_q="Redis 集群跨机房迁移步骤",
            ),
            usage=None,
            num_turns=1,
            subtype="success",
        )


def _runner(tmp_path):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient()
    r._session = _EscalatingSession()
    r._approvals = SimpleNamespace()
    r._followups = None
    r._docs_root = _docs_root(tmp_path)
    r._archives = ArchiveStore()
    return r


def _text_inbound(text: str):
    from lark_oapi.channel.types import TextContent

    return SimpleNamespace(
        sender=SimpleNamespace(is_bot=False),
        chat_id=KEY[0],
        sender_id=KEY[1],
        message_id="om_msg",
        content=TextContent(raw={"text": text}, text=text),
        mentions=[],
        resources=[],
    )


def _submit_event(qid: str, *, clicker: str = OWNER, question="", answer="整理后的答案"):
    return SimpleNamespace(
        action=SimpleNamespace(value={"aq": qid}),
        operator=SimpleNamespace(open_id=clicker, name="负责人"),
        message_id="om_form",
        raw={"event": {"action": {"form_value": {"question": question, "answer": answer}}}},
    )


async def test_runner_end_to_end_archive_loop(tmp_path):
    r = _runner(tmp_path)
    await r._handle(_text_inbound("我们生产那套 redis7 想迁到新机房 咋整啊"))

    # 升级答案已交付（@ 负责人），随后发了归档表单卡
    assert "已请" in _flat(r._client.updated_posts[-1][1])
    assert len(r._client.sent_cards) == 1
    form_flat = _flat(r._client.sent_cards[0][1])
    assert "问答归档" in form_flat and "Redis 集群跨机房迁移步骤" in form_flat
    assert r._archives.pending_count() == 1
    qid = next(iter(r._archives._pending))

    # 负责人填写提交（走 cardAction 统一入口）
    await r._on_card_action(_submit_event(qid, answer="1. 新机房建从库\n2. 校验后切流"))

    content = (r._docs_root / "redis" / "qa-archive.md").read_text(encoding="utf-8")
    assert "新机房建从库" in content
    assert "✅" in _flat(r._client.updated_cards[-1][1])  # 表单被替换成 ack
    # @ 提问者的推送
    notify_flat = _flat(r._client.sent_posts[-1][1])
    assert KEY[1] in notify_flat and "已答复" in notify_flat
    assert r._archives.pending_count() == 0


async def test_runner_submit_by_non_owner_via_dispatch(tmp_path):
    r = _runner(tmp_path)
    await r._handle(_text_inbound("redis 跨机房迁移"))
    qid = next(iter(r._archives._pending))

    await r._on_card_action(_submit_event(qid, clicker="ou_intruder"))
    assert "archive_form" in _flat(r._client.updated_cards[-1][1])  # 表单重建，没顶掉
    assert r._archives.pending_count() == 1
    assert not (r._docs_root / "redis" / "qa-archive.md").exists()
