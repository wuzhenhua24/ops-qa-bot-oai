"""/help 指令回归测试（纯逻辑 + 假件，无需 LLM / 网络 / 飞书）。

覆盖：
- build_help_text：常驻行；可选特性行按开关出现/消失；组件清单后缀；
  /tasks 指令行只在定时跟进开启时列；idle 分钟数渲染。
- runner：/help（大小写、中文触发词）短路应答——不进答题、不发占位、不动会话
  状态；帮助内容按 INDEX.md 组件与特性快照拼出；INDEX 解析失败降级不列组件。

跑法：
    uv run pytest tests/test_help_command.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from agents import set_tracing_disabled  # noqa: E402

from ops_qa_bot_oai.feishu.render import build_help_text  # noqa: E402

set_tracing_disabled(True)

KEY = ("oc_chat", "ou_asker")

_INDEX_MD = """# 索引

| 组件 | 来源 | 目录 | 飞书文档 | 覆盖内容 | 负责人 | open_id |
|------|------|------|----------|----------|--------|---------|
| Redis | local | `redis/` | - | 缓存 | 张三 | ou_owner_1 |
| MySQL | local | `mysql/` | - | 数据库 | 李四 | ou_owner_2 |
"""


def _flat(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# build_help_text（纯函数）
# ---------------------------------------------------------------------------


def test_help_text_base_lines_always_present():
    out = build_help_text([])
    assert "@ 我 + 问题" in out
    assert "文档问答" in out and "当前覆盖" not in out  # 空组件列表不出后缀
    assert "看图提问" in out
    assert "/reset" in out and "/cancel" in out and "/help" in out
    assert "👍/👎" in out  # 反馈入口说明
    assert "沉淀回文档库" in out  # 归档闭环说明
    assert "30 分钟没动静" in out  # 默认 idle
    # 可选特性默认全关，不出现
    for absent in ("实时诊断", "网关链路", "数据库实时分析", "参数变更", "定时跟进", "/tasks"):
        assert absent not in out


def test_help_text_optional_features_and_components():
    out = build_help_text(
        ["Redis", "MySQL"],
        diag=True,
        gw_trace=True,
        db=True,
        db_change=True,
        followup=True,
        idle_minutes=10,
    )
    assert "当前覆盖：Redis、MySQL" in out
    assert "实时诊断" in out and "网关链路排查" in out
    assert "数据库实时分析" in out and "参数变更申请" in out
    assert "定时跟进" in out and "`/tasks`" in out
    assert "10 分钟没动静" in out


def test_help_text_tasks_line_gated_by_followup():
    assert "/tasks" not in build_help_text([], followup=False)
    assert build_help_text([], followup=True).count("/tasks") == 2  # 能力行 + 指令行


# ---------------------------------------------------------------------------
# runner：/help 短路应答
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self):
        self.sent_posts: list[tuple[str, dict]] = []
        self.sent_texts: list[tuple[str, str]] = []

    async def send_post(self, chat_id, post, *, parent_id=None):
        self.sent_posts.append((chat_id, post))
        return "ph1"

    async def send_text(self, chat_id, text, *, parent_id=None):
        self.sent_texts.append((chat_id, text))
        return "t1"


class _MustNotAnswerSession:
    guardrails = False
    idle_ttl = 1800.0

    def queued(self, key):
        raise AssertionError("/help 不该走到答题链路")

    async def take_expired_notice(self, key):
        raise AssertionError("/help 不该消费过期提示")

    async def answer(self, key, question, approver=None, images=None, on_start=None):
        raise AssertionError("/help 不该进答题")


def _runner(tmp_path):
    from ops_qa_bot_oai.feishu.runner import WsRunner

    (tmp_path / "INDEX.md").write_text(_INDEX_MD, encoding="utf-8")
    r = WsRunner.__new__(WsRunner)  # 跳过 __init__，不建真 channel
    r._client = _FakeClient()
    r._session = _MustNotAnswerSession()
    r._approvals = SimpleNamespace()
    r._followups = None
    r._docs_root = tmp_path
    r._help_features = {
        "diag": True,
        "gw_trace": False,
        "db": False,
        "db_change": False,
        "followup": False,
    }
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


async def test_runner_help_short_circuits(tmp_path):
    r = _runner(tmp_path)
    await r._handle(_text_inbound("/help"))
    assert len(r._client.sent_posts) == 1  # 只有帮助 post，没有占位
    flat = _flat(r._client.sent_posts[0][1])
    assert "运维问答机器人" in flat
    assert "Redis、MySQL" in flat  # 组件清单来自 INDEX.md
    assert "实时诊断" in flat  # 开了的特性出现
    assert "网关链路" not in flat  # 关着的不出现
    assert KEY[1] in flat  # @ 提问者


async def test_runner_help_trigger_words(tmp_path):
    r = _runner(tmp_path)
    for word in ("HELP", "帮助", "Help"):
        await r._handle(_text_inbound(word))
    assert len(r._client.sent_posts) == 3


async def test_runner_help_survives_bad_index(tmp_path):
    r = _runner(tmp_path)
    (tmp_path / "INDEX.md").write_text("没有组件表", encoding="utf-8")
    await r._handle(_text_inbound("/help"))
    flat = _flat(r._client.sent_posts[-1][1])
    assert "文档问答" in flat and "当前覆盖" not in flat  # 降级不列组件，帮助照常
