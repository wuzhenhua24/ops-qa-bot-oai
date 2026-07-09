"""飞书文档问答接入的回归测试。

覆盖四层：

- `index`：INDEX.md 的组件表解析（飞书文档列、双别名、mtime 缓存）与来源标识往返。
- `DocQAClient.ask`：200 成功 / 200 空答案 / 401 / 422 / 500 / 504 / 超时 / 连接错误的映射。
- `query_feishu_doc_core`：成功透传；未登记组件 / 缺 question / 上游失败都**返回文字**而不抛。
- **集成点**（这几条是本项目独有、参考项目没有的硬校验，最容易在移植时踩坑）：
  `schema.validate_citations` 认飞书来源、`review.gather_evidence` 不把飞书引用读成 `[未找到]`、
  `orchestration` 按来源分配工具集。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest

from ops_qa_bot_oai.actions import WriteCommandLog, make_write_command_tool
from ops_qa_bot_oai.doc_qa import (
    DocQAClient,
    DocQAConfig,
    DocQAError,
    DocQALog,
    make_feishu_doc_tool,
    query_feishu_doc_core,
)
from ops_qa_bot_oai.index import (
    Component,
    feishu_citation,
    feishu_registry,
    parse_feishu_citation,
    parse_index_components,
)
from ops_qa_bot_oai.orchestration import build_specialist_agent, routable_components
from ops_qa_bot_oai.review import gather_evidence
from ops_qa_bot_oai.schema import validate_citations

# ---------------------------------------------------------------------------
# 夹具：临时 docs_root + INDEX.md
# ---------------------------------------------------------------------------

_INDEX_WITH_FEISHU = """# 索引

| 组件 | 来源 | 目录 | 飞书文档 | 覆盖内容 | 负责人 | open_id |
|------|------|------|----------|----------|--------|---------|
| Redis | local | `redis/` | - | 缓存 | 张三 | ou_aaa |
| Nginx | feishu | `nginx/` | docx_ABC, docx_DEF | 网关 | 赵六 | ou_bbb |
| Gateway | feishu | `gw/` | `docx_XYZ` | 网关2 | 钱七 | ou_ccc |
"""

_INDEX_LOCAL_ONLY = """# 索引

| 组件 | 目录 | 覆盖内容 | 负责人 | open_id |
|------|------|----------|--------|---------|
| Redis | `redis/` | 缓存 | 张三 | ou_aaa |
"""

# 登记为 feishu 但「飞书文档」列空着 —— 配置错，不该进注册表。
_INDEX_FEISHU_NO_TOKEN = """# 索引

| 组件 | 来源 | 目录 | 飞书文档 | 覆盖内容 | 负责人 | open_id |
|------|------|------|----------|----------|--------|---------|
| Nginx | feishu | `nginx/` | - | 网关 | 赵六 | ou_bbb |
"""


def _docs_root(tmp_path: Path, index_body: str) -> Path:
    (tmp_path / "INDEX.md").write_text(index_body, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# INDEX.md 解析
# ---------------------------------------------------------------------------


def test_parses_feishu_docs_column(tmp_path: Path):
    comps = {c.name: c for c in parse_index_components(_docs_root(tmp_path, _INDEX_WITH_FEISHU))}
    assert comps["Nginx"].docs == ("docx_ABC", "docx_DEF")
    assert comps["Gateway"].docs == ("docx_XYZ",)  # 去掉 backtick
    assert comps["Nginx"].is_feishu and not comps["Redis"].is_feishu


def test_local_row_dash_cell_is_not_a_token(tmp_path: Path):
    """`-` 是"没有"的占位写法，不该被切成一个名为 "-" 的 doc token。"""
    comps = {c.name: c for c in parse_index_components(_docs_root(tmp_path, _INDEX_WITH_FEISHU))}
    assert comps["Redis"].docs == ()


def test_registry_only_feishu_rows(tmp_path: Path):
    reg = feishu_registry(_docs_root(tmp_path, _INDEX_WITH_FEISHU))
    assert {c.name for c in reg.values()} == {"Nginx", "Gateway"}
    assert "redis" not in reg


def test_registry_alias_by_name_and_dir(tmp_path: Path):
    reg = feishu_registry(_docs_root(tmp_path, _INDEX_WITH_FEISHU))
    assert reg["nginx"].name == "Nginx"  # 组件名（小写）
    assert reg["gw"].name == "Gateway"  # 目录名
    assert reg["gateway"].name == "Gateway"  # 组件名


def test_registry_skips_feishu_row_without_token(tmp_path: Path):
    assert feishu_registry(_docs_root(tmp_path, _INDEX_FEISHU_NO_TOKEN)) == {}


def test_registry_local_only_index_is_empty(tmp_path: Path):
    assert feishu_registry(_docs_root(tmp_path, _INDEX_LOCAL_ONLY)) == {}


def test_missing_index_returns_empty(tmp_path: Path):
    assert parse_index_components(tmp_path) == []
    assert feishu_registry(tmp_path) == {}


def test_cache_invalidates_on_mtime_change(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_LOCAL_ONLY)
    assert feishu_registry(root) == {}
    (root / "INDEX.md").write_text(_INDEX_WITH_FEISHU, encoding="utf-8")
    # 某些文件系统 mtime 粒度较粗，强制推进避免同秒未刷新
    future = time.time() + 2
    os.utime(root / "INDEX.md", (future, future))
    assert {c.name for c in feishu_registry(root).values()} == {"Nginx", "Gateway"}


# ---------------------------------------------------------------------------
# 来源标识
# ---------------------------------------------------------------------------


def test_citation_roundtrip():
    assert parse_feishu_citation(feishu_citation("Nginx")) == "Nginx"


@pytest.mark.parametrize(
    "raw", ["飞书文档·Nginx", "飞书文档:Nginx", "飞书文档：Nginx", "飞书文档 Nginx"]
)
def test_citation_tolerates_separators(raw: str):
    """模型未必打得出 `·`，几种常见写法都得认，否则来源护栏会把好答案拦掉。"""
    assert parse_feishu_citation(raw) == "Nginx"


@pytest.mark.parametrize("raw", ["redis/overview.md", "INDEX.md", "飞书文档", ""])
def test_citation_rejects_non_feishu(raw: str):
    assert parse_feishu_citation(raw) is None


# ---------------------------------------------------------------------------
# DocQAClient：HTTP 层的错误映射
# ---------------------------------------------------------------------------


def _client(handler, token: str | None = "t0ken") -> DocQAClient:
    cfg = DocQAConfig(base_url="http://doc-qa.test", token=token, timeout=5)
    return DocQAClient(cfg, transport=httpx.MockTransport(handler))


def test_client_success_sends_docs_and_auth():
    def h(req: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(req.content)
        assert body["q"] == "怎么配限流"
        assert body["docs"] == ["docx_ABC"]
        assert body["req_id"] == "r-1"
        assert req.headers["Authorization"] == "Bearer t0ken"
        assert str(req.url) == "http://doc-qa.test/doc_qa"
        return httpx.Response(200, json={"ok": True, "req_id": "r-1", "answer": "用 limit_req"})

    out = asyncio.run(_client(h).ask(["docx_ABC"], "怎么配限流", req_id="r-1"))
    assert out == "用 limit_req"


def test_client_no_token_omits_auth_header():
    def h(req: httpx.Request) -> httpx.Response:
        assert "Authorization" not in req.headers
        return httpx.Response(200, json={"ok": True, "answer": "a"})

    assert asyncio.run(_client(h, token=None).ask(["d"], "q")) == "a"


def test_client_200_empty_answer_raises():
    def h(req):
        return httpx.Response(200, json={"ok": True, "answer": "   "})

    with pytest.raises(DocQAError) as e:
        asyncio.run(_client(h).ask(["d"], "q"))
    assert "升级规则" in e.value.agent_hint


@pytest.mark.parametrize("status", [401, 422, 500, 504])
def test_client_error_statuses_raise_with_hint(status: int):
    def h(req):
        return httpx.Response(status, json={"ok": False, "error": "x", "req_id": "r"})

    with pytest.raises(DocQAError) as e:
        asyncio.run(_client(h).ask(["d"], "q"))
    assert "升级规则" in e.value.agent_hint


def test_client_ok_false_with_200_raises():
    """状态码 200 但 ok=false：不能当成功透传。"""

    def h(req):
        return httpx.Response(200, json={"ok": False, "error": "no permission"})

    with pytest.raises(DocQAError):
        asyncio.run(_client(h).ask(["d"], "q"))


def test_client_non_json_body_raises():
    def h(req):
        return httpx.Response(200, text="<html>502 bad gateway</html>")

    with pytest.raises(DocQAError):
        asyncio.run(_client(h).ask(["d"], "q"))


def test_client_timeout_and_connect_error_raise():
    def timeout(req):
        raise httpx.TimeoutException("timed out", request=req)

    def refused(req):
        raise httpx.ConnectError("refused", request=req)

    for h in (timeout, refused):
        with pytest.raises(DocQAError):
            asyncio.run(_client(h).ask(["d"], "q"))


def test_client_requires_base_url():
    with pytest.raises(ValueError):
        DocQAClient(DocQAConfig(base_url=None))


# ---------------------------------------------------------------------------
# 真实 HTTP 往返：起一个本地 /doc_qa stub，走真的 socket + 真的 httpx 传输层。
# MockTransport 短路了连接与序列化，这里补上那一段（也是"没有 mock 客户端"后，
# 想在本地端到端验证这条链路的正确方式）。
# ---------------------------------------------------------------------------


class _StubDocQAServer(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802  (BaseHTTPRequestHandler 的既定命名)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _StubDocQAServer.received = {
            "path": self.path,
            "body": json.loads(body),
            "auth": self.headers.get("Authorization"),
        }
        payload = json.dumps({"ok": True, "req_id": "r-9", "answer": "限流用 limit_req"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *args):  # 静音 stderr 访问日志
        pass


@contextmanager
def _stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _StubDocQAServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_end_to_end_against_a_real_local_doc_qa_service(tmp_path: Path):
    """整条链路走真 HTTP：注册表解析 → POST /doc_qa → 渲染来源标识 → 记进证据日志。"""
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    with _stub_server() as base_url:
        cfg = DocQAConfig(base_url=base_url, token="t0ken", timeout=10)
        log = DocQALog()
        # 注入一个**真的** AsyncHTTPTransport：httpx 只在 `transport is None` 时才挂载
        # 环境/系统代理（macOS 的 trust_env 会读系统代理设置，不止读环境变量），显式给
        # transport 就直连 127.0.0.1，测试不受开发机代理配置影响。上面那些 MockTransport
        # 用例能过，也是同一个原因。
        client = DocQAClient(cfg, transport=httpx.AsyncHTTPTransport())
        out = asyncio.run(query_feishu_doc_core(client, log, root, "Nginx", "怎么配限流"))

    sent = _StubDocQAServer.received
    assert sent["path"] == "/doc_qa"
    assert sent["body"]["docs"] == ["docx_ABC", "docx_DEF"]  # token 由代码解析，不是 agent 传的
    assert sent["body"]["q"] == "怎么配限流"
    assert sent["auth"] == "Bearer t0ken"

    assert out.startswith("（来源：飞书文档·Nginx）")
    assert "限流用 limit_req" in out
    # 复核证据链路：答案进了 log，且引用能被校验为真
    assert log.answers()["Nginx"].endswith("限流用 limit_req")
    assert validate_citations(root, ["飞书文档·Nginx"]) == []


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_disabled_by_default(monkeypatch):
    for k in ("OPS_QA_DOC_QA_BASE_URL", "OPS_QA_DOC_QA_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert not DocQAConfig.from_env().enabled


def test_config_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("OPS_QA_DOC_QA_BASE_URL", "http://x.test/")
    cfg = DocQAConfig.from_env()
    assert cfg.base_url == "http://x.test"  # 否则拼出 //doc_qa
    assert cfg.enabled


def test_config_bad_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("OPS_QA_DOC_QA_TIMEOUT", "not-a-number")
    assert DocQAConfig.from_env().timeout == 60.0


# ---------------------------------------------------------------------------
# 工具核心：失败一律返回文字，不抛（抛会打断 agent 这一轮）
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, answer: str | None = None, exc: Exception | None = None):
        self._answer = answer
        self._exc = exc
        self.calls: list[tuple] = []

    async def ask(self, docs, question, req_id=None):
        self.calls.append((tuple(docs), question, req_id))
        if self._exc is not None:
            raise self._exc
        return self._answer


def _core(root: Path, client, component: str, question: str, log: DocQALog | None = None) -> str:
    return asyncio.run(query_feishu_doc_core(client, log or DocQALog(), root, component, question))


def test_core_success_resolves_tokens_and_renders_citation(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    fake = _FakeClient(answer="用 limit_req 模块")
    log = DocQALog()
    out = _core(root, fake, "Nginx", "怎么配限流", log)
    # agent 只传组件名，token 由代码查注册表解析
    assert fake.calls[0][0] == ("docx_ABC", "docx_DEF")
    # 抬头就是模型该照抄的来源标识
    assert out.startswith("（来源：飞书文档·Nginx）")
    assert "用 limit_req 模块" in out
    assert log.calls[0].component == "Nginx"


def test_core_lookup_by_dir_name(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    fake = _FakeClient(answer="ok")
    assert "ok" in _core(root, fake, "gw", "q")
    assert fake.calls[0][0] == ("docx_XYZ",)


def test_core_unknown_component_does_not_call_upstream(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    fake = _FakeClient(answer="x")
    out = _core(root, fake, "Redis", "q")  # Redis 是 local 来源
    assert "未登记" in out
    assert "Nginx" in out and "Gateway" in out  # 告诉 agent 有哪些飞书组件
    assert not fake.calls


def test_core_missing_question_does_not_call_upstream(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    fake = _FakeClient(answer="x")
    assert "缺少 question" in _core(root, fake, "Nginx", "   ")
    assert not fake.calls


def test_core_upstream_failure_returns_hint_not_raises(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    fake = _FakeClient(exc=DocQAError("boom"))
    log = DocQALog()
    out = _core(root, fake, "Nginx", "怎么配", log)
    assert "升级规则" in out
    assert not log.calls  # 失败不该进证据


def test_core_picks_up_index_edit_without_restart(tmp_path: Path):
    """INDEX.md 改了登记不必重启进程——注册表每次调用都重查（mtime 缓存）。"""
    root = _docs_root(tmp_path, _INDEX_LOCAL_ONLY)
    fake = _FakeClient(answer="ok")
    assert "未登记" in _core(root, fake, "Nginx", "q")
    (root / "INDEX.md").write_text(_INDEX_WITH_FEISHU, encoding="utf-8")
    future = time.time() + 2
    os.utime(root / "INDEX.md", (future, future))
    assert "ok" in _core(root, fake, "Nginx", "q")


def test_log_answers_merges_multiple_rounds(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    log = DocQALog()
    _core(root, _FakeClient(answer="第一轮"), "Nginx", "q1", log)
    _core(root, _FakeClient(answer="第二轮"), "Nginx", "q2", log)
    merged = log.answers()["Nginx"]
    assert "第一轮" in merged and "第二轮" in merged  # 两轮都是答案的依据


def test_log_answers_since_slices_current_round(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    log = DocQALog()
    _core(root, _FakeClient(answer="上一轮"), "Nginx", "q1", log)
    mark = len(log.calls)
    _core(root, _FakeClient(answer="本轮"), "Nginx", "q2", log)
    assert "上一轮" not in log.answers(mark)["Nginx"]


# ---------------------------------------------------------------------------
# 集成点 1：来源校验认飞书来源，但不给它开后门
# ---------------------------------------------------------------------------


def test_validate_citations_accepts_registered_feishu_component(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    assert validate_citations(root, ["飞书文档·Nginx"]) == []


def test_validate_citations_rejects_unregistered_feishu_component(tmp_path: Path):
    """编一个没登记的组件，和编一个不存在的文件路径一样要被抓出来。"""
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    assert validate_citations(root, ["飞书文档·Postgres"]) == ["飞书文档·Postgres"]


def test_validate_citations_rejects_local_component_cited_as_feishu(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    assert validate_citations(root, ["飞书文档·Redis"]) == ["飞书文档·Redis"]


def test_validate_citations_still_checks_local_paths(tmp_path: Path):
    root = _docs_root(tmp_path, _INDEX_WITH_FEISHU)
    (root / "redis").mkdir()
    (root / "redis" / "overview.md").write_text("x", encoding="utf-8")
    assert validate_citations(root, ["redis/overview.md"]) == []
    assert validate_citations(root, ["redis/nope.md", "../etc/passwd"]) == [
        "redis/nope.md",
        "../etc/passwd",
    ]


# ---------------------------------------------------------------------------
# 集成点 2：复核证据——飞书引用不能被读成 [未找到]
# ---------------------------------------------------------------------------


def test_gather_evidence_uses_feishu_answer_not_filesystem(tmp_path: Path):
    ev = gather_evidence(
        tmp_path,
        ["飞书文档·Nginx"],
        [],
        feishu_answers={"Nginx": "limit_req 配在 http 段"},
    )
    assert "limit_req 配在 http 段" in ev
    assert "[未找到]" not in ev  # 否则 reviewer 会稳定误判"引用不实"


def test_gather_evidence_flags_feishu_citation_without_a_call(tmp_path: Path):
    """引用了某组件、本轮却没调过它的工具 —— 正是要抓的幻觉。"""
    ev = gather_evidence(tmp_path, ["飞书文档·Nginx"], [], feishu_answers={})
    assert "无据可核" in ev


def test_gather_evidence_includes_uncited_feishu_answer(tmp_path: Path):
    ev = gather_evidence(tmp_path, [], [], feishu_answers={"Nginx": "限流配置"})
    assert "限流配置" in ev and "未显式引用" in ev


def test_gather_evidence_matches_component_case_insensitively(tmp_path: Path):
    ev = gather_evidence(tmp_path, ["飞书文档·nginx"], [], feishu_answers={"Nginx": "答案正文"})
    assert "答案正文" in ev and "无据可核" not in ev


def test_gather_evidence_mixes_local_and_feishu(tmp_path: Path):
    (tmp_path / "redis").mkdir()
    (tmp_path / "redis" / "overview.md").write_text("maxmemory 说明", encoding="utf-8")
    ev = gather_evidence(
        tmp_path,
        ["redis/overview.md", "飞书文档·Nginx"],
        ["free -h 输出"],
        feishu_answers={"Nginx": "limit_req"},
    )
    assert "maxmemory 说明" in ev and "limit_req" in ev and "free -h 输出" in ev


# ---------------------------------------------------------------------------
# 集成点 3：编排按来源分配工具集
# ---------------------------------------------------------------------------


def _comp(name: str, source: str) -> Component:
    return Component(
        name=name,
        dir=name.lower(),
        source=source,
        coverage="",
        open_id="ou_x",
        docs=("docx_A",) if source == "feishu" else (),
    )


def test_routable_skips_feishu_when_tool_absent():
    comps = [_comp("Redis", "local"), _comp("Nginx", "feishu")]
    assert [c.name for c in routable_components(comps, None)] == ["Redis"]


def test_routable_includes_feishu_when_tool_present():
    comps = [_comp("Redis", "local"), _comp("Nginx", "feishu")]
    assert [c.name for c in routable_components(comps, object())] == ["Redis", "Nginx"]


def test_routable_treats_unknown_source_as_local():
    """「来源」列写错的行按 local 处理，而不是从名单里静悄悄消失。"""
    comps = [_comp("Redis", "lcoal")]  # typo
    assert [c.name for c in routable_components(comps, None)] == ["Redis"]


def _tool_names(agent) -> list[str]:
    return [t.name for t in agent.tools]


def test_feishu_specialist_has_only_the_doc_qa_tool():
    """机制保证而非 prompt 自律：feishu 专家物理上没有文档检索工具，想 read_doc 也调不到。"""
    tool = make_feishu_doc_tool(DocQAConfig(base_url="http://doc-qa.test"), DocQALog())
    agent = build_specialist_agent(_comp("Nginx", "feishu"), "gpt-5", feishu_tool=tool)
    assert _tool_names(agent) == ["query_feishu_doc"]


def test_local_specialist_never_gets_the_doc_qa_tool():
    tool = make_feishu_doc_tool(DocQAConfig(base_url="http://doc-qa.test"), DocQALog())
    agent = build_specialist_agent(_comp("Redis", "local"), "gpt-5", feishu_tool=tool)
    assert _tool_names(agent) == ["read_doc", "glob_docs", "grep_docs"]


def test_feishu_specialist_without_tool_is_a_build_error():
    """漏传工具是编排 bug，不该悄悄建出一个查不到东西的专家。"""
    with pytest.raises(ValueError, match="feishu"):
        build_specialist_agent(_comp("Nginx", "feishu"), "gpt-5", feishu_tool=None)


def test_cross_cutting_tools_compose_onto_feishu_specialist():
    """写审批 / 实时诊断与「知识来源」正交：飞书来源的组件照样能诊断、能提议变更。"""
    tool = make_feishu_doc_tool(DocQAConfig(base_url="http://doc-qa.test"), DocQALog())
    extra = [make_write_command_tool(WriteCommandLog())]
    agent = build_specialist_agent(
        _comp("Nginx", "feishu"), "gpt-5", extra_tools=extra, feishu_tool=tool
    )
    assert _tool_names(agent) == ["query_feishu_doc", "request_write_command"]


def test_feishu_specialist_instructions_teach_the_citation_format():
    tool = make_feishu_doc_tool(DocQAConfig(base_url="http://doc-qa.test"), DocQALog())
    agent = build_specialist_agent(_comp("Nginx", "feishu"), "gpt-5", feishu_tool=tool)
    assert "飞书文档·Nginx" in agent.instructions
    assert "没有对话记忆" in agent.instructions  # question 必须自包含
