"""数据库诊断（测试环境，只读）+ 参数变更审批：`query_database` / `request_db_change` 工具。

对标 `ops-qa-bot`（Claude Agent SDK 版）的 db_query.py——asker 在问题里给连接信息
（IP、端口、租户、集群），bot 用部署机本地的 `mysql` / `obclient` 客户端、以 DBA 预建的
**只读账号**连上目标库跑诊断 SQL，让 agent 在排查"CPU 高 / 连接数高 / 慢查询"这类问题时
自由迭代地查（`SHOW PROCESSLIST` → 看可疑 query → 翻 `gv$ob_sql_audit` / `sys` /
`performance_schema`…），而不必预先写死 SQL。

参考项目已验证的安全设计**原样保留**（这些是纯 stdlib 函数，与 SDK 无关，直接移植）：

- **只读由数据库引擎强制，不靠解析 SQL**：连接用的是每个实例上预建的只读账号（只有
  SELECT/SHOW/PROCESS 权限），写操作被引擎直接拒。所以不做 SQL 白/黑名单——既不误杀
  诊断语句，也没有黑名单 fail-open 的风险。唯一拦的是多语句拼接（`;`）。
- **凭据工具内注入，LLM 全程看不到**：账号密码来自配置，按 (引擎, 模式) 选对应账号；
  密码经 `MYSQL_PWD` 环境变量传给 client，不进命令行 argv（防 `ps` 泄露）、不进上下文。
- **目标受 IP 白名单约束**（`allowed_hosts`，IP / CIDR / 精确主机名）。
- **create_subprocess_exec 而非 shell**：argv 列表传参，SQL 作为单个 `-e` 实参；
  租户/集群/账号名做字符集校验，防被拼进 `user@tenant#cluster`。
- **失败返回引导文字而不抛异常**：让 agent 据数据库报错自己改 SQL 重试或走升级规则。

在 OpenAI Agents SDK 上做的三点改进：

1. **参数变更审批复用 `needs_approval`（HITL 一等机制）**：参考项目为 `request_db_change`
   手工搭了 DbChangeSubmitter Protocol + 确认卡 + pending 登记 + 飞书回调执行一整条链路
   （因为 Claude SDK 的 hook 无法挂起 run）。这里工具标 `needs_approval=True`，run 在提议处
   挂起、走 `answer_guarded` 已有的审批闭环（飞书发卡 → 值班人点按钮 → resume），审批决定
   与审计天然落在 RunState 上——一个工具函数替代整条手工胶水链。
2. **查询结果进二次复核证据链**：`DbQueryLog` 记录本轮查询输出，`review.gather_evidence`
   把它与诊断输出并列喂给 reviewer——"结论与 processlist 证据矛盾"这类幻觉会被抓住。
   参考项目的复核不覆盖 DB 证据。
3. **executor 可注入 + mock 降级**：未配只读账号（或 `OPS_QA_DB_MOCK=1`）时返回**标注
   模拟数据**的假结果，链路无真实数据库也能端到端跑通（对齐 `DiagConfig` 的姿态——DB
   实时状态与假的 `free -h` 输出同属无害模拟，不是 doc_qa 那种"假知识库"）；单测注入
   假 executor，不需要真数据库。

姿态与项目其它可选能力一致：**`OPS_QA_DB` 缺省关**，关闭时工具不挂载、prompt 不加章节，
零感知。实现拆成"纯函数核心 + 薄 function_tool 包装"两层（对齐 diagnostics.py / doc_qa.py）。
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agents import RunContextWrapper, function_tool

from .actions import WriteCommandLog, WriteRequest
from .model import env_flag
from .tools import DocsContext

logger = logging.getLogger(__name__)

# 工具名常量：orchestration / bot 按名字判断专家挂了哪些横切工具（与 DIAG_TOOL_NAME 同理）。
DB_TOOL_NAME = "query_database"
DB_CHANGE_TOOL_NAME = "request_db_change"

# 连接类型（引擎 + OceanBase 模式）。决定用哪个 client、哪套账号、哪种方言。
_KIND_MYSQL = "mysql"  # 原生 MySQL，用 mysql client
_KIND_OB_MYSQL = "ob_mysql"  # OceanBase mysql 模式，用 obclient
_KIND_OB_ORACLE = "ob_oracle"  # OceanBase oracle 模式，用 obclient

# 账号 / 租户 / 集群名允许的字符：拼进 `user@tenant#cluster` 前做校验，防注入。
_IDENT_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_MAX_IDENT_LEN = 128

# SQL 长度上限：诊断语句远不到这个量级，超长当异常拒。
_MAX_SQL_LEN = 8000

# connect 阶段超时上限：不让它吃满整条 query_timeout（query 本身可能慢）。
_MAX_CONNECT_TIMEOUT = 10

# 参数值允许的字符集刻意收紧：覆盖常见取值（数字、枚举、容量 "256M"、时长 "10s"、
# 隔离级别 "READ-COMMITTED"、路径/比例等），但**禁掉引号/分号/反引号/括号**——值要被
# 拼进 bot 自己构造的 `SET GLOBAL`/`ALTER SYSTEM SET` 语句，禁掉这些就堵死了注入面。
_VALUE_RE = re.compile(r"^[A-Za-z0-9_.\-+:/ ]+$")
_MAX_VALUE_LEN = 256
# 纯整数值不加引号（如 max_connections=1000）；其余一律单引号包裹（值已过 _VALUE_RE，
# 不含单引号，包裹安全）。
_INT_VALUE_RE = re.compile(r"^-?\d+$")

# 只读 SQL 起始动词：仅 **mock 模式**用它模拟"只读账号会被引擎拒写"的行为；
# 真实路径的只读由数据库账号权限强制，不解析 SQL。
_READ_ONLY_SQL_RE = re.compile(
    r"^\s*\(?\s*(select|show|desc|describe|explain|analyze\s+table)\b", re.I
)


class DatabaseQueryError(Exception):
    """数据库查询/校验失败。`agent_hint` 是给 agent 看的引导文字。"""

    def __init__(self, log_detail: str, agent_hint: str):
        super().__init__(log_detail)
        self.agent_hint = agent_hint


_HINT_HOST = (
    "目标地址不在允许范围内（本工具只允许连测试环境白名单内的数据库实例）。"
    "请确认用户给的 IP 是测试环境实例；若确属测试环境但被拒，是部署侧 "
    "OPS_QA_DB_ALLOWED_HOSTS 没覆盖该网段，按升级规则通知运维补白名单，不要换地址重试。"
)
_HINT_TIMEOUT = (
    "连接或查询超时。可能是实例不可达、负载过高，或这条 SQL 太重。"
    "可以让用户确认连接信息（IP/端口/租户/集群）是否正确，或把查询收窄"
    "（加 WHERE/LIMIT）后重试；多次超时按升级规则通知 DBA。"
)
_HINT_CLIENT_MISSING = (
    "数据库客户端（mysql/obclient）未安装在部署机上。按升级规则通知运维，不要重试。"
)


def _hint_no_creds(kind: str) -> str:
    return (
        f"部署侧没有为该数据库类型（{kind}）配置只读账号，无法查询。"
        "请按升级规则通知运维补配 OPS_QA_DB_*_RO_USER / OPS_QA_DB_*_RO_PASSWORD，"
        "不要换种方式重试。"
    )


def _hint_query_failed(detail: str) -> str:
    short = detail.strip().replace("\n", " ")
    if len(short) > 500:
        short = short[:500] + "…"
    return (
        f"数据库返回错误：{short or '（无错误详情）'}。"
        "这只是**这一条 SQL** 的问题，不代表连接或整体权限有问题——请换种写法**继续**"
        "排查，不要因为一两次报错就放弃或直接下「无权限」结论：\n"
        "- 表/视图不存在（如 ORA-00942、MySQL 1146）：多半是**对象名不对**。不同库/模式"
        "视图名不一样，OceanBase 的动态视图常带 `OB` 前缀（如 GV$OB_PROCESSLIST、"
        "GV$OB_LOCKS），别照搬标准 Oracle 的 V$SESSION。**先从数据字典查实际存在的对象名"
        "再查**：oracle 模式 `SELECT view_name FROM dba_views WHERE view_name LIKE "
        "'%SESSION%'`（或 %LOCK%/%PROCESS%），mysql 模式用 `SHOW TABLES` / 查 "
        "information_schema；或换 OB 专用视图名重试，**多试几种再下结论**。"
        "（注意：oracle 模式 ORA-00942 既可能是名字错、也可能是无权访问，光看报错分不清，"
        "所以不能一见 ORA-00942 就当没权限。）\n"
        "- 语法/列名错（如 ORA-00904、MySQL 1064/1054）：按报错修正后重试。\n"
        "- 只有在「确属连接失败」或「在确认存在的对象上、换多种写法仍明确被拒访问」时，"
        "才按升级规则通知 DBA。"
    )


# ---------------------------------------------------------------------------
# 纯函数核心（移植自参考项目，脱离 SDK 可直接单测）
# ---------------------------------------------------------------------------


def resolve_kind(db_type: str, mode: str | None) -> str:
    """(引擎, 模式) → 连接类型。非法组合抛 DatabaseQueryError。"""
    db_type = (db_type or "").strip().lower()
    mode = (mode or "").strip().lower() or "mysql"
    if db_type == "mysql":
        return _KIND_MYSQL
    if db_type in ("oceanbase", "ob"):
        if mode == "mysql":
            return _KIND_OB_MYSQL
        if mode == "oracle":
            return _KIND_OB_ORACLE
        raise DatabaseQueryError(
            f"bad oceanbase mode: {mode!r}",
            "OceanBase 必须指定 mode 为 mysql 或 oracle。请确认目标租户是哪种模式。",
        )
    raise DatabaseQueryError(
        f"bad db_type: {db_type!r}",
        "db_type 只支持 mysql 或 oceanbase。请确认目标数据库类型。",
    )


def host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """host 是否落在白名单内：先精确字符串匹配（覆盖主机名），再按 IP/CIDR 匹配。"""
    host = (host or "").strip()
    if not host or not allowed:
        return False
    if host in allowed:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # 非 IP 且不在精确名单里 → 拒
    for entry in allowed:
        try:
            net = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if ip in net:
            return True
    return False


def _validate_identifier(value: str, kind_label: str) -> None:
    if not value or not _IDENT_RE.match(value) or len(value) > _MAX_IDENT_LEN:
        raise DatabaseQueryError(
            f"bad {kind_label}: {value!r}",
            f"{kind_label} 含非法字符或为空（只允许字母/数字/._-）。请核对连接信息。",
        )


def resolve_port(kind: str, port: object) -> int:
    """校验端口；缺省时按连接类型取默认（MySQL 3306 / OceanBase 2883）。"""
    if port in (None, "", 0):
        return 3306 if kind == _KIND_MYSQL else 2883
    try:
        p = int(port)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        p = -1
    if not (1 <= p <= 65535):
        raise DatabaseQueryError(
            f"bad port: {port!r}",
            "端口非法（应为 1-65535 的整数）。请确认用户给的端口，OceanBase 通常是 2883。",
        )
    return p


def sanitize_sql(sql: str) -> str:
    """去首尾空白 + 去末尾分号；含内嵌分号（多语句）则拒。

    只读由引擎保证，这里只守"一次一条语句"的契约——多语句拼接既破坏结果可读性，
    也是常见的注入/堆叠手法，直接拒掉最干净。
    """
    s = (sql or "").strip()
    if not s:
        raise DatabaseQueryError("empty sql", "调用缺少 SQL 语句。")
    if len(s) > _MAX_SQL_LEN:
        raise DatabaseQueryError("sql too long", "SQL 过长，疑似异常。请精简查询后重试。")
    stripped = s.rstrip("; \t\r\n")
    if ";" in stripped:
        raise DatabaseQueryError(
            "multi-statement sql rejected",
            "一次只能查一条语句（检测到多条用 ; 拼接）。请拆成多次调用，每次一条。",
        )
    return stripped


def build_argv(
    kind: str, host: str, port: int, conn_user: str, sql: str, query_timeout: float
) -> list[str]:
    """拼 client 命令 argv（密码不在其中，走 MYSQL_PWD 环境变量）。"""
    connect_timeout = max(2, min(_MAX_CONNECT_TIMEOUT, int(query_timeout)))
    client = "mysql" if kind == _KIND_MYSQL else "obclient"
    return [
        client,
        "-h",
        host,
        "-P",
        str(port),
        "-u",
        conn_user,
        f"--connect-timeout={connect_timeout}",
        "-e",
        sql,
    ]


def validate_param_value(param: str, value: str) -> None:
    """校验参数名 + 目标值，非法抛 DatabaseQueryError（带 agent_hint）。"""
    _validate_identifier(param, "param")
    v = (value or "").strip()
    if not v:
        raise DatabaseQueryError("empty value", "缺少目标值（value）。请确认要把参数改成多少。")
    if len(v) > _MAX_VALUE_LEN:
        raise DatabaseQueryError("value too long", "目标值过长，疑似异常。请核对后重试。")
    if not _VALUE_RE.match(v):
        raise DatabaseQueryError(
            f"bad value: {value!r}",
            "目标值含不被允许的字符（只允许字母/数字/._-+:/ 和空格，不能有引号/分号等）。"
            "请确认值是否正确；带特殊字符的复杂值暂不支持自动改，需 DBA 人工执行。",
        )


def _value_literal(value: str) -> str:
    """把已校验的值渲染成 SQL 字面量：纯整数不加引号，其余单引号包裹。"""
    v = value.strip()
    return v if _INT_VALUE_RE.match(v) else f"'{v}'"


def build_change_sql(kind: str, param: str, value: str) -> str:
    """按连接类型拼参数变更语句（param/value 须已过校验）。

    - MySQL：`SET GLOBAL <param> = <literal>`
    - OceanBase（mysql/oracle 模式）：`ALTER SYSTEM SET <param> = <literal>`
      连接串已定向到 `user@tenant#cluster`，以该租户身份执行、作用于本租户。
    """
    literal = _value_literal(value)
    if kind == _KIND_MYSQL:
        return f"SET GLOBAL {param} = {literal}"
    return f"ALTER SYSTEM SET {param} = {literal}"


# ---------------------------------------------------------------------------
# 配置（缺省关，从环境变量解析）
# ---------------------------------------------------------------------------


@dataclass
class DbCreds:
    """一种连接类型（引擎 + OceanBase 模式）的一套账号。"""

    user: str | None = None
    password: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.user and self.password)


# 执行器签名：(argv, password, timeout) → (returncode, stdout, stderr)。
# 密码单独传（由执行器经 MYSQL_PWD 注入），保证它永远不进 argv。可注入以便单测。
DbExecutor = Callable[[list[str], str, float], Awaitable[tuple[int | None, str, str]]]


@dataclass
class DbConfig:
    """数据库诊断配置。缺省 enabled=False（工具不挂载、prompt 不加章节，零感知）。"""

    enabled: bool = False
    allowed_hosts: tuple[str, ...] = ()  # IP / CIDR / 精确主机名；真实模式下空 = 全拒
    query_timeout: float = 30.0
    max_result_chars: int = 20000
    mock: bool = False  # 显式强制模拟执行（OPS_QA_DB_MOCK=1）
    mysql_ro: DbCreds = field(default_factory=DbCreds)
    ob_mysql_ro: DbCreds = field(default_factory=DbCreds)
    ob_oracle_ro: DbCreds = field(default_factory=DbCreds)
    # 参数变更审批的 admin（写）账号：未配时变更在批准后登记为"待 DBA 人工执行"。
    mysql_admin: DbCreds = field(default_factory=DbCreds)
    ob_mysql_admin: DbCreds = field(default_factory=DbCreds)
    ob_oracle_admin: DbCreds = field(default_factory=DbCreds)
    executor: DbExecutor | None = None  # 注入自定义执行器（单测用）

    @property
    def use_mock(self) -> bool:
        """真连库还是模拟：显式 mock、或一套只读账号都没配时走模拟（标注假数据）。"""
        return self.mock or not any(
            c.configured for c in (self.mysql_ro, self.ob_mysql_ro, self.ob_oracle_ro)
        )

    @classmethod
    def from_env(cls) -> DbConfig:
        """从 OPS_QA_DB* 环境变量解析。缺省全关。"""

        def _list(name: str) -> tuple[str, ...]:
            raw = (os.environ.get(name) or "").strip()
            return tuple(x.strip() for x in raw.split(",") if x.strip()) if raw else ()

        def _num(name: str, default: float) -> float:
            raw = (os.environ.get(name) or "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        def _creds(prefix: str) -> DbCreds:
            return DbCreds(
                user=(os.environ.get(f"{prefix}_USER") or "").strip() or None,
                password=(os.environ.get(f"{prefix}_PASSWORD") or "").strip() or None,
            )

        return cls(
            enabled=env_flag("OPS_QA_DB", default=False),
            allowed_hosts=_list("OPS_QA_DB_ALLOWED_HOSTS"),
            query_timeout=_num("OPS_QA_DB_TIMEOUT", 30.0),
            max_result_chars=int(_num("OPS_QA_DB_MAX_CHARS", 20000)),
            mock=env_flag("OPS_QA_DB_MOCK", default=False),
            mysql_ro=_creds("OPS_QA_DB_MYSQL_RO"),
            ob_mysql_ro=_creds("OPS_QA_DB_OB_MYSQL_RO"),
            ob_oracle_ro=_creds("OPS_QA_DB_OB_ORACLE_RO"),
            mysql_admin=_creds("OPS_QA_DB_MYSQL_ADMIN"),
            ob_mysql_admin=_creds("OPS_QA_DB_OB_MYSQL_ADMIN"),
            ob_oracle_admin=_creds("OPS_QA_DB_OB_ORACLE_ADMIN"),
        )


# ---------------------------------------------------------------------------
# 执行器：真子进程 / 模拟
# ---------------------------------------------------------------------------


async def subprocess_executor(
    argv: list[str], password: str, timeout: float
) -> tuple[int | None, str, str]:
    """真执行器：起本机 mysql/obclient 子进程，密码经 MYSQL_PWD 注入（不进 argv）。"""
    env = dict(os.environ)
    env["MYSQL_PWD"] = password or ""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as e:
        raise DatabaseQueryError(f"client binary missing: {e!r}", _HINT_CLIENT_MISSING) from e
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError) as e:
        proc.kill()
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        raise DatabaseQueryError(f"db query timeout: {e!r}", _HINT_TIMEOUT) from e
    return (
        proc.returncode,
        (out or b"").decode("utf-8", errors="replace"),
        (err or b"").decode("utf-8", errors="replace"),
    )


_MOCK_NOTE = "\n（模拟数据：未配置只读账号；配好 OPS_QA_DB_*_RO_USER/PASSWORD 后走真实查询）"


def mock_query_output(sql: str) -> str:
    """按 SQL 关键词产出带标注的假结果，让链路无真实数据库也能端到端演示/单测。"""
    up = sql.upper()
    if "PROCESSLIST" in up:
        body = (
            "Id\tUser\tHost\tdb\tCommand\tTime\tState\tInfo\n"
            "12\tapp\t10.0.0.8:51322\torders\tQuery\t35\tSending data\t"
            "SELECT * FROM orders WHERE status = 'PAID'\n"
            "15\tapp\t10.0.0.9:50144\torders\tSleep\t120\t\tNULL"
        )
    elif "VARIABLES" in up or "PARAMETERS" in up:
        body = "Variable_name\tValue\nmax_connections\t512"
    elif up.lstrip("( \t").startswith("EXPLAIN"):
        body = (
            "id\tselect_type\ttable\ttype\trows\tExtra\n"
            "1\tSIMPLE\torders\tALL\t1204520\tUsing where"
        )
    else:
        body = "col1\tcol2\nv1\tv2\n（模拟结果集：2 rows）"
    return body + _MOCK_NOTE


# ---------------------------------------------------------------------------
# 客户端：账号选择 → 校验 → 执行
# ---------------------------------------------------------------------------


class DatabaseClient:
    """本机 mysql / obclient 的薄封装：选账号 → 校验 → 跑执行器 → 返回文本。"""

    def __init__(self, config: DbConfig):
        self._config = config
        self._allowed = config.allowed_hosts
        self._timeout = config.query_timeout
        self._max_chars = config.max_result_chars
        self._creds = {
            _KIND_MYSQL: config.mysql_ro,
            _KIND_OB_MYSQL: config.ob_mysql_ro,
            _KIND_OB_ORACLE: config.ob_oracle_ro,
        }
        self._admin_creds = {
            _KIND_MYSQL: config.mysql_admin,
            _KIND_OB_MYSQL: config.ob_mysql_admin,
            _KIND_OB_ORACLE: config.ob_oracle_admin,
        }

    def _resolve_conn(
        self, kind: str, creds: DbCreds, host: str, port: object, tenant: str, cluster: str
    ) -> tuple[int, str]:
        """校验 host 白名单/端口/标识符，拼出 (port_int, conn_user)。只读/admin 共用。"""
        if not host_allowed(host, self._allowed):
            raise DatabaseQueryError(f"host not allowed: {host}", _HINT_HOST)
        p = resolve_port(kind, port)
        _validate_identifier(creds.user or "", "user")
        if kind in (_KIND_OB_MYSQL, _KIND_OB_ORACLE):
            _validate_identifier(tenant, "tenant")
            _validate_identifier(cluster, "cluster")
            conn_user = f"{creds.user}@{tenant}#{cluster}"
        else:
            conn_user = creds.user or ""
        return p, conn_user

    async def _exec_sql(
        self, kind: str, host: str, port: int, conn_user: str, password: str, sql: str
    ) -> str:
        """跑一条语句，返回截断后的 stdout（可能为空串）。失败抛 DatabaseQueryError。"""
        argv = build_argv(kind, host.strip(), port, conn_user, sql, self._timeout)
        executor = self._config.executor or subprocess_executor
        rc, out, err = await executor(argv, password, self._timeout)
        if rc != 0:
            detail = err.strip()
            logger.warning("db query rc=%s: %s", rc, detail)
            raise DatabaseQueryError(
                f"db query failed rc={rc}: {detail}", _hint_query_failed(detail)
            )
        text = out.strip()
        if len(text) > self._max_chars:
            text = (
                text[: self._max_chars]
                + "\n…（结果过长，已截断；请用更具体的条件或 LIMIT 收窄查询）"
            )
        return text

    async def run(
        self,
        *,
        db_type: str,
        mode: str | None,
        host: str,
        port: object,
        tenant: str,
        cluster: str,
        sql: str,
    ) -> str:
        """连库跑一条只读 SQL，返回结果文本。失败抛 DatabaseQueryError（带 agent_hint）。"""
        kind = resolve_kind(db_type, mode)
        clean_sql = sanitize_sql(sql)
        if self._config.use_mock:
            # 模拟"只读账号会被引擎拒写"：mock 下没有引擎，行为得和真实路径一致，
            # 否则模型会以为写 SQL 执行成功了。
            if not _READ_ONLY_SQL_RE.match(clean_sql):
                raise DatabaseQueryError(
                    "mock: non-readonly sql rejected",
                    "该 SQL 不是只读查询，只读账号会被数据库引擎直接拒绝。"
                    "改参数请用 request_db_change 工具（若可用）走审批；其它写/变更"
                    "按「写操作建议输出格式」给文字建议由 DBA 人工执行。",
                )
            return mock_query_output(clean_sql)
        creds = self._creds[kind]
        if not creds.configured:
            raise DatabaseQueryError(f"no ro creds for {kind}", _hint_no_creds(kind))
        p, conn_user = self._resolve_conn(kind, creds, host, port, tenant, cluster)
        logger.info(
            "db query: kind=%s host=%s port=%d user=%s sql_len=%d",
            kind,
            host,
            p,
            conn_user,
            len(clean_sql),
        )
        text = await self._exec_sql(kind, host, p, conn_user, creds.password or "", clean_sql)
        return text or "（查询执行成功，但没有返回任何行。）"

    # -- 参数变更（needs_approval 工具的执行体在批准后调这两个） ------------------

    def prepare_change(
        self,
        *,
        db_type: str,
        mode: str | None,
        host: str,
        port: object,
        tenant: str,
        cluster: str,
        param: str,
        value: str,
    ) -> tuple[str, int, str]:
        """确定性校验 + 拼变更语句（无 I/O，不写库）。返回 (kind, port, sql)。

        与参考项目的 prepare_change 的差别：不再预读参数现值——那需要在发审批卡**之前**
        连一次库，而 needs_approval 挂起点在工具执行之前，卡片由接入层直接从工具参数渲染。
        """
        kind = resolve_kind(db_type, mode)
        p = resolve_port(kind, port)
        if kind in (_KIND_OB_MYSQL, _KIND_OB_ORACLE):
            _validate_identifier(tenant, "tenant")
            _validate_identifier(cluster, "cluster")
        validate_param_value(param, value)
        if not self._config.use_mock and not host_allowed(host, self._allowed):
            raise DatabaseQueryError(f"host not allowed: {host}", _HINT_HOST)
        return kind, p, build_change_sql(kind, param.strip(), value.strip())

    def can_execute_change(self, kind: str) -> bool:
        """是否具备真正执行变更的条件（配了该类型的 admin 账号且非模拟模式）。"""
        return not self._config.use_mock and self._admin_creds[kind].configured

    async def execute_change(
        self, kind: str, host: str, port: object, tenant: str, cluster: str, sql: str
    ) -> str:
        """用 admin 账号执行一条已审批的变更语句。失败抛 DatabaseQueryError。"""
        admin = self._admin_creds[kind]
        if not admin.configured:
            raise DatabaseQueryError(
                f"no admin creds for {kind}",
                f"未配置 {kind} 的 admin 账号，无法执行变更。",
            )
        p, conn_user = self._resolve_conn(kind, admin, host, port, tenant, cluster)
        logger.info(
            "db admin change: kind=%s host=%s port=%d user=%s sql=%r", kind, host, p, conn_user, sql
        )
        text = await self._exec_sql(kind, host, p, conn_user, admin.password or "", sql)
        # SET GLOBAL / ALTER SYSTEM SET 正常无返回行；有输出就一并带回展示
        return text or "（执行成功，无返回行。）"


# ---------------------------------------------------------------------------
# 查询日志（供二次复核取证据 / 观测）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DbQueryRecord:
    """一次成功的数据库查询。"""

    kind: str
    host: str
    sql: str
    output: str


@dataclass
class DbQueryLog:
    """收集本会话跑过的数据库查询。

    与 DocQALog 同理，存在的理由不只是观测：`review.gather_evidence` 复核答案时要核对
    "结论是否被实时证据支持"，数据库查询输出与诊断输出同属实时证据，记在这里喂给复核者。
    """

    results: list[DbQueryRecord] = field(default_factory=list)

    def outputs(self, since: int = 0) -> list[str]:
        """`since` 之后的查询输出（带 host/SQL 上下文），喂给 `review.gather_evidence`。"""
        return [
            f"（数据库 {r.host}）`{_sql_preview(r.sql)}`：\n{r.output}"
            for r in self.results[since:]
            if r.output
        ]


def _sql_preview(sql: str) -> str:
    return sql if len(sql) <= 120 else sql[:120] + "…"


def render_db_result(host: str, sql: str, text: str) -> str:
    """把查询结果渲染成给模型的文本，抬头即来源标识（对齐 diagnostics.render_result）。"""
    return f"（数据库实时数据：{host}）`{_sql_preview(sql)}`\n{text or '（无返回行）'}"


# ---------------------------------------------------------------------------
# 审批前校验 + 展示（供 bot / feishu 接入层用）
# ---------------------------------------------------------------------------


def validate_change_args(args: dict, config: DbConfig | None = None) -> str | None:
    """确定性校验一笔参数变更提议的参数。非法返回可读原因，合法返回 None。

    给 `answer_guarded` 的审批循环在**发卡前**短路用（与禁止清单短路同一姿态）：
    校验不过的提议直接驳回、不打扰审批人。传 `config` 时额外做 host 白名单检查。
    纯函数（不连库），工具执行体内还会再校验一遍（纵深防御）。
    """
    db_type = str(args.get("db_type") or "").strip()
    host = str(args.get("host") or "").strip()
    param = str(args.get("param") or "").strip()
    value = str(args.get("value") or "").strip()
    if not db_type or not host or not param or not value:
        return "缺少必填参数（db_type / host / param / value）"
    try:
        kind = resolve_kind(db_type, str(args.get("mode") or ""))
        resolve_port(kind, args.get("port"))
        if kind in (_KIND_OB_MYSQL, _KIND_OB_ORACLE):
            _validate_identifier(str(args.get("tenant") or ""), "tenant")
            _validate_identifier(str(args.get("cluster") or ""), "cluster")
        validate_param_value(param, value)
    except DatabaseQueryError as e:
        return e.agent_hint
    if config is not None and not config.use_mock and not host_allowed(host, config.allowed_hosts):
        return _HINT_HOST
    return None


def change_display(args: dict) -> tuple[str, str]:
    """把一笔变更提议的参数渲染成 (变更语句, 目标) 两段展示文本（审批卡/审计行用）。

    确定性、不抛：校验不过时退化成 `param = value（未通过校验）` 的字面展示。
    """
    db_type = str(args.get("db_type") or "?").strip()
    mode = str(args.get("mode") or "").strip()
    host = str(args.get("host") or "?").strip()
    port = args.get("port")
    tenant = str(args.get("tenant") or "").strip()
    target = db_type + (f"/{mode}" if mode else "") + f" @ {host}"
    if port:
        target += f":{port}"
    if tenant:
        target += f" tenant={tenant}"
    param = str(args.get("param") or "?").strip()
    value = str(args.get("value") or "?").strip()
    try:
        kind = resolve_kind(db_type, mode)
        validate_param_value(param, value)
        cmd = build_change_sql(kind, param, value)
    except DatabaseQueryError:
        cmd = f"{param} = {value}（未通过校验）"
    return cmd, target


# ---------------------------------------------------------------------------
# 纯核心 + function_tool 包装
# ---------------------------------------------------------------------------


async def query_database_core(
    client: DatabaseClient,
    log: DbQueryLog,
    *,
    db_type: str,
    sql: str,
    host: str,
    port: object = None,
    mode: str | None = None,
    tenant: str | None = None,
    cluster: str | None = None,
) -> str:
    """查询工具的纯核心：校验 + 执行 + 记日志，返回给模型的文本。

    失败一律返回引导文字而不抛——抛会打断 agent 这一轮，返回提示则让它自己按引导
    处理（改 SQL 重试 / 走升级规则）。脱离 SDK 可直接 await 单测。
    """
    db_type = (db_type or "").strip()
    host = (host or "").strip()
    if not db_type or not host or not (sql or "").strip():
        return "调用缺少必填参数：db_type / host / sql。请补齐后重试。"
    try:
        text = await client.run(
            db_type=db_type,
            mode=mode,
            host=host,
            port=port,
            tenant=(tenant or "").strip(),
            cluster=(cluster or "").strip(),
            sql=sql,
        )
    except DatabaseQueryError as e:
        logger.warning("db query failed: %s", e)
        return e.agent_hint
    kind = resolve_kind(db_type, mode)  # run 已校验过，这里不会抛
    log.results.append(
        DbQueryRecord(kind=kind, host=host, sql=sanitize_sql(sql), output=text)
    )
    return render_db_result(host, sanitize_sql(sql), text)


def make_query_database_tool(client: DatabaseClient, log: DbQueryLog):
    """构造 `query_database` 工具（只读分析）。"""

    @function_tool
    async def query_database(
        ctx: RunContextWrapper[DocsContext],
        db_type: str,
        sql: str,
        host: str,
        port: int | None = None,
        mode: str | None = None,
        tenant: str | None = None,
        cluster: str | None = None,
    ) -> str:
        """连接**测试环境**的数据库，执行一条**只读** SQL 做分析/排查，返回结果文本。

        适用：用户报告某个数据库实例的实时问题（CPU 高、连接数高、慢查询、锁等待、空间
        增长等）并给出了连接信息。这类排查通常要多次调用、迭代收敛——先看面上（如
        SHOW PROCESSLIST / gv$ob_processlist），再挑可疑点深入（EXPLAIN、慢查询视图、
        gv$ob_sql_audit、sys/performance_schema 等），每次调用跑**一条**语句。

        只读由数据库账号权限强制：写/变更语句（INSERT/UPDATE/DELETE/DDL、SET GLOBAL 等）
        会被引擎直接拒。用户要求改参数时改用 request_db_change 工具（若可用）；杀 session、
        kill query、加索引等其它变更按「写操作建议输出格式」给文字建议由 DBA 人工执行。

        方言注意：OceanBase oracle 模式没有 SHOW，用数据字典/动态性能视图（gv$ 视图、
        dba_* 等），且查 dual 而非空 FROM；OB 动态视图常带 OB 前缀（GV$OB_PROCESSLIST）。
        出错时本工具返回引导提示（含数据库报错原文），据此修正 SQL 或按升级规则处理。

        Args:
            db_type: 数据库类型，mysql 或 oceanbase。
            sql: 要执行的**那一条**只读 SQL（不要用 ; 拼多条）。
            host: 目标实例 IP（以用户给的为准，不要自己猜）。
            port: 端口（用户提供；缺省 MySQL 3306 / OceanBase 2883）。
            mode: OceanBase 租户模式，mysql 或 oracle；db_type=oceanbase 时必给。
            tenant: OceanBase 租户名；db_type=oceanbase 时必给。
            cluster: OceanBase 集群名；db_type=oceanbase 时必给。
        """
        return await query_database_core(
            client,
            log,
            db_type=db_type,
            sql=sql,
            host=host,
            port=port,
            mode=mode,
            tenant=tenant,
            cluster=cluster,
        )

    return query_database


def make_db_change_tool(client: DatabaseClient, write_log: WriteCommandLog):
    """构造 `request_db_change` 工具：needs_approval=True，批准后才走到执行体。

    执行体内：确定性校验 + 拼 SQL → 配了 admin 账号则真执行，否则登记为"已批准、
    待 DBA 人工执行"（与 request_write_command 的登记姿态一致）。两种结局都记入
    `write_log`，GuardedAnswer.approved_writes 统一呈现。

    校验不过的提议通常在 `answer_guarded` 的审批循环里就被 `validate_change_args`
    短路驳回了（不打扰审批人）；这里再校验一遍是纵深防御。
    """

    @function_tool(needs_approval=True)
    async def request_db_change(
        ctx: RunContextWrapper[DocsContext],
        db_type: str,
        host: str,
        param: str,
        value: str,
        reason: str,
        port: int | None = None,
        mode: str | None = None,
        tenant: str | None = None,
        cluster: str | None = None,
    ) -> str:
        """申请修改测试环境某个数据库实例的**参数**，走人工审批后执行。

        调用本工具会挂起等值班人审批——你（agent）只是提议，绝不自己执行；批准前什么
        都不会改。只在用户明确要求**改某个库参数**（如调 max_connections、改 OB 的某个
        system parameter）并给了实例连接信息时用。杀 session、kill query、加索引、
        DML/DDL 等其它变更**不要**用本工具，改用 request_write_command（若可用）或按
        「写操作建议输出格式」给文字建议。纯知识/排查问题更不要调。

        工具返回执行/登记结果，把它如实转达给用户；被驳回时不要重复提交，改给文字建议。

        Args:
            db_type: 数据库类型，mysql 或 oceanbase。
            host: 目标实例 IP（用户提供）。
            param: 要修改的参数名（如 max_connections）。
            value: 目标值（如 1000、256M、ON；不能含引号/分号）。
            reason: 为什么需要这次变更（结合文档/查询证据给出依据，审批人会看到）。
            port: 端口（用户提供；缺省 MySQL 3306 / OceanBase 2883）。
            mode: OceanBase 租户模式，mysql 或 oracle；db_type=oceanbase 时必给。
            tenant: OceanBase 租户名；db_type=oceanbase 时必给。
            cluster: OceanBase 集群名；db_type=oceanbase 时必给。
        """
        # 走到这里说明已通过人工审批（needs_approval 在执行前挂起）。
        try:
            kind, p, sql = client.prepare_change(
                db_type=db_type,
                mode=mode,
                host=host,
                port=port,
                tenant=(tenant or "").strip(),
                cluster=(cluster or "").strip(),
                param=param,
                value=value,
            )
        except DatabaseQueryError as e:
            logger.warning("db change validation failed post-approval: %s", e)
            return f"[未执行] {e.agent_hint}"

        target = f"{db_type}@{host}:{p}"
        if not client.can_execute_change(kind):
            write_log.requests.append(
                WriteRequest(command=sql, target=target, reason=reason, approved=True)
            )
            return (
                f"已批准并登记参数变更（部署侧未配置 {kind} 的 admin 账号，需 DBA 人工执行）："
                f"在 {target} 执行 `{sql}`。请如实告知用户：已获批准、待 DBA 执行，不要谎称已改好。"
            )
        try:
            out = await client.execute_change(
                kind, host, p, (tenant or "").strip(), (cluster or "").strip(), sql
            )
        except DatabaseQueryError as e:
            logger.warning("db change execution failed: %s", e)
            return (
                f"[已批准但执行失败] `{sql}`：{e.agent_hint}\n"
                "请如实告知用户执行失败及原因，按升级规则通知 DBA 人工处理。"
            )
        write_log.requests.append(
            WriteRequest(command=sql, target=target, reason=reason, approved=True)
        )
        return f"已批准并执行参数变更：在 {target} 执行 `{sql}`。执行结果：{out}"

    return request_db_change
