"""数据库诊断（db_query.py）回归测试：纯函数校验 + 客户端流程 + 审批前校验。

安全语义两个方向都锁（对齐参考项目 ops-qa-bot 的 test_db_query.py 用例结构）：

- **必须拦**：白名单外 host、多语句拼接、非法标识符/端口、带引号的参数值、
  密码出现在 argv、mock 模式下的写 SQL。
- **不许误杀**：CIDR 内 IP、末尾分号、整数/容量/枚举参数值、正常只读查询。

失败路径断言"返回引导文字而不抛"（query_database_core 的契约）。
纯函数 + 注入 executor，无需 LLM / 真数据库。跑法：uv run pytest tests/test_db_query.py -q
"""

from __future__ import annotations

import asyncio
import os

import pytest

from ops_qa_bot_oai.actions import WriteCommandLog
from ops_qa_bot_oai.db_query import (
    DatabaseClient,
    DatabaseQueryError,
    DbConfig,
    DbCreds,
    DbQueryLog,
    build_argv,
    build_change_sql,
    change_display,
    host_allowed,
    mock_query_output,
    query_database_core,
    resolve_kind,
    resolve_port,
    sanitize_sql,
    validate_change_args,
    validate_param_value,
)

# ---------------------------------------------------------------------------
# resolve_kind / resolve_port
# ---------------------------------------------------------------------------


def test_resolve_kind():
    assert resolve_kind("mysql", None) == "mysql"
    assert resolve_kind("MySQL", "") == "mysql"
    assert resolve_kind("oceanbase", "mysql") == "ob_mysql"
    assert resolve_kind("ob", "oracle") == "ob_oracle"
    # oceanbase 未给 mode 时缺省按 mysql 模式
    assert resolve_kind("oceanbase", None) == "ob_mysql"


@pytest.mark.parametrize(
    "db_type,mode",
    [("postgres", None), ("", None), ("oceanbase", "pg")],
)
def test_resolve_kind_rejects(db_type, mode):
    with pytest.raises(DatabaseQueryError):
        resolve_kind(db_type, mode)


def test_resolve_port_defaults_and_bounds():
    assert resolve_port("mysql", None) == 3306
    assert resolve_port("ob_mysql", None) == 2883
    assert resolve_port("ob_oracle", "") == 2883
    assert resolve_port("mysql", 3307) == 3307
    assert resolve_port("mysql", "2883") == 2883
    for bad in (0.5, "abc", -1, 65536):
        if bad in (None, "", 0):
            continue
        with pytest.raises(DatabaseQueryError):
            resolve_port("mysql", bad)


# ---------------------------------------------------------------------------
# host 白名单
# ---------------------------------------------------------------------------


def test_host_allowed():
    allowed = ("10.1.0.0/16", "192.168.3.7", "db-test-01")
    assert host_allowed("10.1.2.3", allowed)  # CIDR 命中
    assert host_allowed("192.168.3.7", allowed)  # 精确 IP
    assert host_allowed("db-test-01", allowed)  # 精确主机名
    assert not host_allowed("10.2.0.1", allowed)  # 网段外
    assert not host_allowed("db-prod-01", allowed)  # 名单外主机名
    assert not host_allowed("10.1.2.3", ())  # 空白名单 = 全拒（fail-closed）
    assert not host_allowed("", allowed)


# ---------------------------------------------------------------------------
# sanitize_sql：一次一条语句
# ---------------------------------------------------------------------------


def test_sanitize_sql_accepts_trailing_semicolon():
    assert sanitize_sql("SHOW PROCESSLIST;") == "SHOW PROCESSLIST"
    assert sanitize_sql("  SELECT 1  ") == "SELECT 1"


def test_sanitize_sql_rejects_multi_statement():
    with pytest.raises(DatabaseQueryError):
        sanitize_sql("SELECT 1; DROP TABLE t")
    with pytest.raises(DatabaseQueryError):
        sanitize_sql("")
    with pytest.raises(DatabaseQueryError):
        sanitize_sql("SELECT '" + "x" * 9000 + "'")


# ---------------------------------------------------------------------------
# build_argv：密码不进 argv
# ---------------------------------------------------------------------------


def test_build_argv_no_password_and_client_choice():
    argv = build_argv("mysql", "10.1.2.3", 3306, "ro_user", "SELECT 1", 30)
    assert argv[0] == "mysql"
    assert "SELECT 1" in argv
    assert not any("password" in a.lower() or "秘密" in a for a in argv)
    argv_ob = build_argv("ob_oracle", "10.1.2.3", 2883, "ro@t1#c1", "SELECT 1 FROM dual", 30)
    assert argv_ob[0] == "obclient"
    assert "ro@t1#c1" in argv_ob
    # connect-timeout 被夹在 [2, 10]
    assert any(a == "--connect-timeout=10" for a in argv_ob)


# ---------------------------------------------------------------------------
# 参数变更：值校验 + 拼语句
# ---------------------------------------------------------------------------


def test_validate_param_value_accepts_common_values():
    for v in ("1000", "256M", "ON", "READ-COMMITTED", "10s", "0.75"):
        validate_param_value("max_connections", v)


def test_validate_param_value_rejects_injection():
    for v in ("1'; DROP TABLE t; --", "x` y", "a;b", "'quoted'"):
        with pytest.raises(DatabaseQueryError):
            validate_param_value("max_connections", v)
    with pytest.raises(DatabaseQueryError):
        validate_param_value("bad name!", "1")


def test_build_change_sql():
    assert build_change_sql("mysql", "max_connections", "1000") == (
        "SET GLOBAL max_connections = 1000"
    )
    assert build_change_sql("mysql", "innodb_buffer_pool_size", "256M") == (
        "SET GLOBAL innodb_buffer_pool_size = '256M'"
    )
    assert build_change_sql("ob_mysql", "memory_limit", "8G") == (
        "ALTER SYSTEM SET memory_limit = '8G'"
    )


def test_change_display_never_raises():
    cmd, target = change_display(
        {"db_type": "mysql", "host": "10.1.2.3", "port": 3306,
         "param": "max_connections", "value": "1000"}
    )
    assert cmd == "SET GLOBAL max_connections = 1000"
    assert "10.1.2.3:3306" in target
    # 非法值不抛，退化成字面展示
    cmd2, _ = change_display({"db_type": "mysql", "host": "x", "param": "p", "value": "1'; --"})
    assert "未通过校验" in cmd2


# ---------------------------------------------------------------------------
# validate_change_args（审批前短路用）
# ---------------------------------------------------------------------------


def _cfg(**kw) -> DbConfig:
    base = dict(
        enabled=True,
        allowed_hosts=("10.1.0.0/16",),
        mysql_ro=DbCreds(user="ro", password="pw"),
    )
    base.update(kw)
    return DbConfig(**base)


def test_validate_change_args_ok():
    args = {"db_type": "mysql", "host": "10.1.2.3", "param": "max_connections", "value": "500"}
    assert validate_change_args(args) is None
    assert validate_change_args(args, _cfg()) is None


def test_validate_change_args_rejects():
    assert validate_change_args({}) is not None  # 缺必填
    assert (
        validate_change_args(
            {"db_type": "mysql", "host": "h", "param": "p", "value": "bad;value"}
        )
        is not None
    )  # 值非法
    assert (
        validate_change_args(
            {"db_type": "oceanbase", "mode": "mysql", "host": "h", "param": "p", "value": "1"}
        )
        is not None
    )  # OB 缺 tenant/cluster
    # host 白名单（带 config 时检查）
    args = {"db_type": "mysql", "host": "10.9.9.9", "param": "p", "value": "1"}
    assert validate_change_args(args, _cfg()) is not None
    # mock 模式（无只读账号）跳过白名单
    assert validate_change_args(args, DbConfig(enabled=True)) is None


# ---------------------------------------------------------------------------
# DatabaseClient：注入 executor 的真实路径
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def make_fake_executor(rc=0, out="Id\tUser\n1\tapp", err="", capture=None):
    async def fake(argv, password, timeout):
        if capture is not None:
            capture.append((argv, password, timeout))
        return rc, out, err

    return fake


def test_client_run_success_and_creds_isolation():
    capture: list = []
    cfg = _cfg(executor=make_fake_executor(capture=capture))
    client = DatabaseClient(cfg)
    text = _run(
        client.run(
            db_type="mysql", mode=None, host="10.1.2.3", port=None, tenant="", cluster="",
            sql="SHOW PROCESSLIST;",
        )
    )
    assert "Id" in text
    argv, password, _ = capture[0]
    assert password == "pw"  # 密码走独立通道
    assert "pw" not in argv  # 绝不进 argv
    assert "-P" in argv and "3306" in argv  # 端口缺省


def test_client_run_rejects_host_outside_allowlist():
    client = DatabaseClient(_cfg(executor=make_fake_executor()))
    with pytest.raises(DatabaseQueryError) as ei:
        _run(
            client.run(
                db_type="mysql", mode=None, host="10.9.9.9", port=None, tenant="", cluster="",
                sql="SELECT 1",
            )
        )
    assert "允许范围" in ei.value.agent_hint


def test_client_run_engine_error_becomes_hint():
    cfg = _cfg(
        executor=make_fake_executor(rc=1, out="", err="ERROR 1146 (42S02): Table 'x' doesn't exist")
    )
    client = DatabaseClient(cfg)
    with pytest.raises(DatabaseQueryError) as ei:
        _run(
            client.run(
                db_type="mysql", mode=None, host="10.1.2.3", port=None, tenant="", cluster="",
                sql="SELECT * FROM x",
            )
        )
    # 引导文字带报错原文 + "别急着下无权限结论"的经验
    assert "1146" in ei.value.agent_hint
    assert "无权限" in ei.value.agent_hint


def test_client_run_ob_requires_tenant_cluster():
    client = DatabaseClient(
        _cfg(ob_mysql_ro=DbCreds(user="ro", password="pw"), executor=make_fake_executor())
    )
    with pytest.raises(DatabaseQueryError):
        _run(
            client.run(
                db_type="oceanbase", mode="mysql", host="10.1.2.3", port=2883,
                tenant="", cluster="", sql="SELECT 1",
            )
        )
    capture: list = []
    client2 = DatabaseClient(
        _cfg(
            ob_mysql_ro=DbCreds(user="ro", password="pw"),
            executor=make_fake_executor(capture=capture),
        )
    )
    _run(
        client2.run(
            db_type="oceanbase", mode="mysql", host="10.1.2.3", port=2883,
            tenant="t1", cluster="c1", sql="SELECT 1",
        )
    )
    argv, _, _ = capture[0]
    assert "ro@t1#c1" in argv  # 连接串定向到租户#集群
    assert argv[0] == "obclient"


def test_client_run_truncates_long_output():
    cfg = _cfg(max_result_chars=100, executor=make_fake_executor(out="x" * 500))
    client = DatabaseClient(cfg)
    text = _run(
        client.run(
            db_type="mysql", mode=None, host="10.1.2.3", port=None, tenant="", cluster="",
            sql="SELECT 1",
        )
    )
    assert "已截断" in text and len(text) < 300


# ---------------------------------------------------------------------------
# mock 模式（未配只读账号时的降级）
# ---------------------------------------------------------------------------


def test_mock_mode_labels_fake_data_and_rejects_writes():
    client = DatabaseClient(DbConfig(enabled=True))  # 无账号 → use_mock
    text = _run(
        client.run(
            db_type="mysql", mode=None, host="whatever", port=None, tenant="", cluster="",
            sql="SHOW PROCESSLIST",
        )
    )
    assert "模拟数据" in text
    with pytest.raises(DatabaseQueryError):  # 写 SQL 与真实引擎行为一致：拒
        _run(
            client.run(
                db_type="mysql", mode=None, host="whatever", port=None, tenant="", cluster="",
                sql="SET GLOBAL max_connections = 1",
            )
        )


def test_mock_query_output_samples():
    assert "Id" in mock_query_output("SHOW PROCESSLIST")
    assert "max_connections" in mock_query_output("SHOW GLOBAL VARIABLES LIKE 'max%'")
    assert "模拟数据" in mock_query_output("SELECT 1")


# ---------------------------------------------------------------------------
# query_database_core：失败返回引导文字而不抛 + 日志记录
# ---------------------------------------------------------------------------


def test_core_returns_hint_instead_of_raising():
    log = DbQueryLog()
    client = DatabaseClient(_cfg(executor=make_fake_executor()))
    # host 越界 → 返回引导文字
    out = _run(
        query_database_core(client, log, db_type="mysql", sql="SELECT 1", host="10.9.9.9")
    )
    assert "允许范围" in out
    assert log.results == []  # 失败不记日志
    # 缺必填
    out2 = _run(query_database_core(client, log, db_type="", sql="SELECT 1", host="h"))
    assert "必填" in out2


def test_core_success_logs_evidence():
    log = DbQueryLog()
    client = DatabaseClient(
        _cfg(executor=make_fake_executor(out="Variable\tValue\nmax_connections\t512"))
    )
    out = _run(
        query_database_core(
            client, log, db_type="mysql", sql="SHOW GLOBAL VARIABLES LIKE 'max_connections';",
            host="10.1.2.3",
        )
    )
    assert "（数据库实时数据：10.1.2.3）" in out  # 来源抬头（模型照抄进答案）
    assert len(log.results) == 1
    rec = log.results[0]
    assert rec.host == "10.1.2.3" and "max_connections" in rec.output
    # 复核证据渲染带 host/SQL 上下文
    ev = log.outputs()
    assert len(ev) == 1 and "10.1.2.3" in ev[0] and "512" in ev[0]
    assert log.outputs(since=1) == []  # _EvidenceMark 切片语义


# ---------------------------------------------------------------------------
# 参数变更：prepare/execute 路径
# ---------------------------------------------------------------------------


def test_prepare_change_validates_and_builds_sql():
    client = DatabaseClient(_cfg())
    kind, port, sql = client.prepare_change(
        db_type="mysql", mode=None, host="10.1.2.3", port=None, tenant="", cluster="",
        param="max_connections", value="1000",
    )
    assert (kind, port, sql) == ("mysql", 3306, "SET GLOBAL max_connections = 1000")
    with pytest.raises(DatabaseQueryError):  # host 白名单外
        client.prepare_change(
            db_type="mysql", mode=None, host="10.9.9.9", port=None, tenant="", cluster="",
            param="p", value="1",
        )


def test_execute_change_uses_admin_creds():
    capture: list = []
    cfg = _cfg(
        mysql_admin=DbCreds(user="admin", password="adminpw"),
        executor=make_fake_executor(out="", capture=capture),
    )
    client = DatabaseClient(cfg)
    assert client.can_execute_change("mysql")
    assert not client.can_execute_change("ob_mysql")  # 该类型没配 admin
    out = _run(
        client.execute_change(
            "mysql", "10.1.2.3", 3306, "", "", "SET GLOBAL max_connections = 1000"
        )
    )
    assert "执行成功" in out
    argv, password, _ = capture[0]
    assert password == "adminpw" and "admin" in argv and "adminpw" not in argv


def test_write_log_shared_type():
    # request_db_change 复用 WriteCommandLog：登记结构与 request_write_command 一致。
    log = WriteCommandLog()
    assert log.requests == []


# ---------------------------------------------------------------------------
# DbConfig.from_env
# ---------------------------------------------------------------------------


def _clear_db_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("OPS_QA_DB"):
            monkeypatch.delenv(k, raising=False)


def test_db_config_from_env(monkeypatch):
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("OPS_QA_DB", "1")
    monkeypatch.setenv("OPS_QA_DB_ALLOWED_HOSTS", "10.1.0.0/16, db-test-01")
    monkeypatch.setenv("OPS_QA_DB_MYSQL_RO_USER", "ro")
    monkeypatch.setenv("OPS_QA_DB_MYSQL_RO_PASSWORD", "pw")
    monkeypatch.setenv("OPS_QA_DB_TIMEOUT", "12")
    cfg = DbConfig.from_env()
    assert cfg.enabled
    assert cfg.allowed_hosts == ("10.1.0.0/16", "db-test-01")
    assert cfg.mysql_ro.configured and not cfg.ob_mysql_ro.configured
    assert cfg.query_timeout == 12.0
    assert not cfg.use_mock


def test_db_config_default_off(monkeypatch):
    _clear_db_env(monkeypatch)
    cfg = DbConfig.from_env()
    assert not cfg.enabled
    assert cfg.use_mock  # 没配账号 → 即便手动 enabled 也走模拟
