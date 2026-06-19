"""文档检索工具 + 标记解析的回归测试（纯逻辑，无需 LLM / 网络）。

覆盖：
- _read_doc：正常读取 / 不存在 / 路径越界。
- _glob_docs：命中 / 无匹配 / `..` 逃逸被挡。
- _grep_docs：命中行格式 / 限定子目录 / 无命中 / 越界 / 非法正则。
- parse_markers：ESCALATE / CLARIFY / FOLLOWUPS 解析 + 白名单过滤 + 文本剥离。

跑法：
    uv run pytest
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ops_qa_bot_oai.bot import parse_markers
from ops_qa_bot_oai.model import normalize_openai_base_url
from ops_qa_bot_oai.orchestration import parse_index_components
from ops_qa_bot_oai.schema import AnswerContract, Decision, Followup, validate_citations
from ops_qa_bot_oai.tools import _glob_docs, _grep_docs, _read_doc

_INDEX = """# 索引
| 组件 | 目录 | open_id |
|------|------|---------|
| Redis | `redis/` | ou_aaa |
"""

_REDIS_OVERVIEW = "# Redis 概览\n\nRedis 是内存缓存。\nmaxmemory 默认无上限。\n"
_REDIS_TROUBLE = "# 故障排查\n\n## 内存告警\n监控报 memory > 85%。\n慢查询用 SLOWLOG GET。\n"
_MYSQL_OVERVIEW = "# MySQL 概览\n\n主从复制基础。\n慢查询日志 slow_query_log。\n"


@pytest.fixture()
def docs_root() -> Path:
    d = Path(tempfile.mkdtemp(prefix="opsqa_oai_test_"))
    (d / "INDEX.md").write_text(_INDEX, encoding="utf-8")
    (d / "redis").mkdir()
    (d / "redis" / "overview.md").write_text(_REDIS_OVERVIEW, encoding="utf-8")
    (d / "redis" / "troubleshooting.md").write_text(_REDIS_TROUBLE, encoding="utf-8")
    (d / "mysql").mkdir()
    (d / "mysql" / "overview.md").write_text(_MYSQL_OVERVIEW, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# read_doc
# ---------------------------------------------------------------------------


def test_read_doc_ok(docs_root: Path):
    out = _read_doc(docs_root, "redis/overview.md")
    assert "Redis 是内存缓存" in out


def test_read_doc_index_at_root(docs_root: Path):
    assert "索引" in _read_doc(docs_root, "INDEX.md")


def test_read_doc_missing(docs_root: Path):
    out = _read_doc(docs_root, "redis/nope.md")
    assert out.startswith("[未找到]")


def test_read_doc_path_traversal_blocked(docs_root: Path):
    # 试图读 docs_root 之外的文件
    out = _read_doc(docs_root, "../../../etc/passwd")
    assert out.startswith("[错误]")
    assert "越界" in out


# ---------------------------------------------------------------------------
# glob_docs
# ---------------------------------------------------------------------------


def test_glob_docs_dir(docs_root: Path):
    out = _glob_docs(docs_root, "redis/*.md")
    lines = set(out.splitlines())
    assert "redis/overview.md" in lines
    assert "redis/troubleshooting.md" in lines
    assert "mysql/overview.md" not in lines


def test_glob_docs_recursive(docs_root: Path):
    out = _glob_docs(docs_root, "**/*.md")
    assert "redis/overview.md" in out
    assert "mysql/overview.md" in out


def test_glob_docs_no_match(docs_root: Path):
    assert _glob_docs(docs_root, "kafka/*.md").startswith("[无匹配]")


def test_glob_docs_escape_blocked(docs_root: Path):
    # ../ 逃逸不应列出 docs_root 之外的文件
    out = _glob_docs(docs_root, "../*")
    assert out.startswith("[无匹配]")


# ---------------------------------------------------------------------------
# grep_docs
# ---------------------------------------------------------------------------


def test_grep_docs_hit_format(docs_root: Path):
    out = _grep_docs(docs_root, "慢查询")
    # 命中两个文件，格式为 路径:行号: 内容
    assert "redis/troubleshooting.md:" in out
    assert "mysql/overview.md:" in out
    assert ": " in out


def test_grep_docs_scoped_to_subdir(docs_root: Path):
    out = _grep_docs(docs_root, "慢查询", path="redis")
    assert "redis/troubleshooting.md:" in out
    assert "mysql/" not in out


def test_grep_docs_no_hit(docs_root: Path):
    assert _grep_docs(docs_root, "kubernetes").startswith("[无命中]")


def test_grep_docs_regex_alternation(docs_root: Path):
    out = _grep_docs(docs_root, "maxmemory|memory")
    assert "redis/" in out


def test_grep_docs_bad_regex(docs_root: Path):
    assert _grep_docs(docs_root, "[unclosed").startswith("[错误]")


def test_grep_docs_path_traversal_blocked(docs_root: Path):
    out = _grep_docs(docs_root, "x", path="../..")
    assert out.startswith("[错误]")


# ---------------------------------------------------------------------------
# parse_markers
# ---------------------------------------------------------------------------


def test_parse_escalate():
    text = "文档中未找到相关内容。\n\n<<ESCALATE:ou_abc:redis>>"
    cleaned, m = parse_markers(text)
    assert m.escalate == "ou_abc:redis"
    assert "<<ESCALATE" not in cleaned
    assert "文档中未找到相关内容" in cleaned


def test_parse_clarify():
    cleaned, m = parse_markers("你用 6.x 还是 7.x？\n\n<<CLARIFY>>")
    assert m.clarify is True
    assert "<<CLARIFY>>" not in cleaned


def test_parse_followups_whitelist_and_dedup():
    text = "扩容步骤...\n\n<<FOLLOWUPS:rollback|risks|rollback|bogus|commands|related>>"
    cleaned, m = parse_markers(text)
    # 去重 + 白名单过滤 + 最多 3 个
    assert m.followups == ["rollback", "risks", "commands"]
    assert "bogus" not in m.followups
    assert "<<FOLLOWUPS" not in cleaned


def test_parse_no_markers():
    cleaned, m = parse_markers("普通答案，无标记。")
    assert m.escalate is None
    assert m.clarify is False
    assert m.followups == []
    assert cleaned == "普通答案，无标记。"


# ---------------------------------------------------------------------------
# normalize_openai_base_url（OpenAI 协议 base_url 容错，以智谱 URL 为例）
# ---------------------------------------------------------------------------


def test_base_url_strips_chat_completions_suffix():
    # 用户直接粘贴完整 URL（智谱 OpenAI 格式）
    assert (
        normalize_openai_base_url("https://open.bigmodel.cn/api/paas/v4/chat/completions")
        == "https://open.bigmodel.cn/api/paas/v4"
    )


def test_base_url_already_prefix_unchanged():
    assert (
        normalize_openai_base_url("https://open.bigmodel.cn/api/paas/v4")
        == "https://open.bigmodel.cn/api/paas/v4"
    )


def test_base_url_trailing_slash_trimmed():
    assert (
        normalize_openai_base_url("https://open.bigmodel.cn/api/paas/v4/chat/completions/")
        == "https://open.bigmodel.cn/api/paas/v4"
    )


def test_base_url_strips_responses_suffix():
    # Responses 协议端点：client 会自己追加 /responses，base_url 里多余的要剥掉
    assert (
        normalize_openai_base_url("https://api.example.com/v1/responses")
        == "https://api.example.com/v1"
    )


# ---------------------------------------------------------------------------
# anthropic provider 的 Bearer 鉴权开关（以火山引擎为例）
# ---------------------------------------------------------------------------


def test_anthropic_bearer_promotes_key_to_auth_token(monkeypatch):
    # OPS_QA_ANTHROPIC_AUTH=bearer 时，应把 OPS_QA_API_KEY 提升成 ANTHROPIC_AUTH_TOKEN
    # 并不再以 api_key 传入（这样 litellm 才发 Authorization: Bearer 而非 x-api-key）。
    from ops_qa_bot_oai.model import resolve_model

    for k in (
        "OPS_QA_PROVIDER",
        "OPS_QA_MODEL",
        "OPS_QA_BASE_URL",
        "OPS_QA_API_KEY",
        "OPS_QA_ANTHROPIC_AUTH",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPS_QA_PROVIDER", "anthropic")
    monkeypatch.setenv("OPS_QA_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding")
    monkeypatch.setenv("OPS_QA_API_KEY", "volc-key")
    monkeypatch.setenv("OPS_QA_ANTHROPIC_AUTH", "bearer")
    monkeypatch.setenv("OPS_QA_MODEL", "claude-opus-4-8")

    mc = resolve_model()
    # key 被提升到 ANTHROPIC_AUTH_TOKEN
    import os

    assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == "volc-key"
    # LitellmModel 不再持有 api_key（改由 auth_token 走 Bearer）
    assert mc.model.api_key is None
    assert mc.model.model == "anthropic/claude-opus-4-8"
    assert mc.model.base_url == "https://ark.cn-beijing.volces.com/api/coding"


def test_anthropic_default_uses_x_api_key(monkeypatch):
    # 默认（不设 OPS_QA_ANTHROPIC_AUTH）保持 x-api-key：api_key 照常传入。
    from ops_qa_bot_oai.model import resolve_model

    for k in (
        "OPS_QA_PROVIDER",
        "OPS_QA_MODEL",
        "OPS_QA_BASE_URL",
        "OPS_QA_API_KEY",
        "OPS_QA_ANTHROPIC_AUTH",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPS_QA_PROVIDER", "anthropic")
    monkeypatch.setenv("OPS_QA_API_KEY", "k")
    monkeypatch.setenv("OPS_QA_MODEL", "claude-opus-4-8")

    mc = resolve_model()
    assert mc.model.api_key == "k"


# ---------------------------------------------------------------------------
# 结构化输出契约（差异化 #1）：AnswerContract 解析 + 来源真实性校验
# ---------------------------------------------------------------------------


def test_contract_parses_from_json():
    # 模型产出的 JSON 应能按 schema 解析，enum 字段被正确收敛
    c = AnswerContract.model_validate(
        {
            "decision": "answer",
            "answer": "先看 maxmemory。",
            "citations": ["redis/troubleshooting.md"],
            "followups": ["troubleshoot", "commands"],
            "confidence": 0.8,
        }
    )
    assert c.decision is Decision.answer
    assert c.followups == [Followup.troubleshoot, Followup.commands]
    assert c.escalate_to == ""  # 默认值


def test_contract_defaults_minimal():
    # 只给必填项，其余走默认（reject/clarify 常见）
    c = AnswerContract.model_validate({"decision": "reject", "answer": "不在范围内。"})
    assert c.citations == []
    assert c.confidence == 0.0


def test_validate_citations_all_real(docs_root):
    invalid = validate_citations(docs_root, ["redis/overview.md", "mysql/overview.md"])
    assert invalid == []


def test_validate_citations_flags_missing_and_traversal(docs_root):
    invalid = validate_citations(
        docs_root,
        ["redis/overview.md", "redis/ghost.md", "../../../etc/passwd"],
    )
    # 真实的不报，编造的 + 越界的都报
    assert "redis/overview.md" not in invalid
    assert "redis/ghost.md" in invalid
    assert "../../../etc/passwd" in invalid


# ---------------------------------------------------------------------------
# 多 agent 编排（差异化 #3）：INDEX.md 组件解析
# ---------------------------------------------------------------------------

_INDEX_FULL = """# 索引
| 组件 | 来源 | 目录 | 飞书文档 | 覆盖内容 | 负责人 | open_id |
|------|------|------|----------|----------|--------|---------|
| Redis | local | `redis/` | - | Redis 集群运维、故障排查 | 张三 | ou_aaa |
| MySQL | local | `mysql/` | - | 主从、备份、慢查询 | 李四 | ou_bbb |
| Nginx | feishu | `nginx/` | docx_X | 网关配置（飞书） | 赵六 | ou_ccc |
"""


def test_parse_components_fields_and_dir_cleaned(tmp_path):
    (tmp_path / "INDEX.md").write_text(_INDEX_FULL, encoding="utf-8")
    comps = parse_index_components(tmp_path)
    by_name = {c.name: c for c in comps}
    assert set(by_name) == {"Redis", "MySQL", "Nginx"}
    assert by_name["Redis"].dir == "redis"  # 去掉了反引号和斜杠
    assert by_name["Redis"].source == "local"
    assert by_name["Redis"].open_id == "ou_aaa"
    assert by_name["Nginx"].source == "feishu"


def test_parse_components_local_only_index(docs_root):
    # fixture 的 INDEX 没有「来源」列 → 一律按 local
    comps = parse_index_components(docs_root)
    assert all(c.source == "local" for c in comps)
    assert "redis" in {c.dir for c in comps}


def test_parse_components_missing_index_empty(tmp_path):
    assert parse_index_components(tmp_path) == []
