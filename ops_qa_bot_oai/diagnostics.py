"""实时诊断（测试环境，只读）：`run_diagnostic(host, command)` 工具。

对标 `ops-qa-bot`（Claude Agent SDK 版）的「实时诊断」——让 agent ssh 到测试环境机器
跑只读命令、把实时状态叠加到基于文档的回答上。参考项目的做法是：暴露 Claude 内置的
`Bash` 工具让模型自己写 `ssh jumphost "ssh <target> '<cmd>'"`，再用一个只看命令字符串的
PreToolUse hook（`_block_write_bash_hook`）**黑名单**兜底拦写命令；只读命令直接跑。

本项目在 OpenAI Agents SDK 上把同一能力做得更稳、更安全，利用了这套 SDK 给的原语：

1. **结构化工具参数取代自由 Bash**：工具签名是 `run_diagnostic(host, command)`，模型只描述
   "在哪台机、跑哪条只读命令"，**跳板机 / 嵌套 ssh 语法由代码在底层拼**（见
   `ssh_executor`）——模型再也不会写错两跳 ssh、也无法改用 `ssh -J` 绕过认证拓扑。参考
   项目那句"prompt 里硬性要求 agent 必须这么写"在这里变成了机制保证。

2. **白名单优先（默认拒绝）取代黑名单**：`classify_diagnostic_command` 只放行**已知只读**
   命令（系统 free/df/ss…、日志 tail/grep、redis-cli 只读、mysql 的 SELECT/SHOW/DESC/
   EXPLAIN），其余一律拒。黑名单（参考项目 `_WRITE_PATTERNS`）天然有泄漏面——任何没被
   列进去的写命令都会漏过；白名单是正向安全模型，未知即拒。禁止清单
   （`detect_forbidden_command`，毁灭性命令）作为**第二层兜底**保留。

3. **命令三分层，写命令复用现有审批（HITL）**：只读→执行；**识别到的写命令**（重启 /
   CONFIG SET / 改库参数…）→ 提示模型改用 `request_write_command` 走 needs_approval
   审批（不在诊断工具里跑）；毁灭性命令→直接拒。参考项目对写命令只有"deny → 退化文字
   建议"这一条死路，这里能把它接进你已经建好的审批闭环。

4. **quote-aware 分词取代裸正则**：用 `shlex`（punctuation_chars）尊重引号来切命令+运算符，
   `awk '$3 > 100'` 里的 `>` 不会被当重定向、`grep set` 不会被当 redis 写、`free; rm` 里的 `;`
   会被切成独立 token 拦下——参考项目在正则黑名单上踩过的这些误杀/漏拦坑，这里从解析层就没有了。

5. **目标机校验落到代码**：生产机（名字带 prod/production/正式）由工具直接拒、可选的
   `allowed_hosts` 白名单限定可诊断目标——不再只靠 prompt 自律。

姿态与项目其它可选能力一致：**`OPS_QA_DIAG` 缺省关**，关闭时工具不挂载、prompt 不加章节，
零感知。未配 jumphost 时自动降级为**模拟执行**（标「模拟数据」），方便本地/演示先跑起来。

实现拆成"纯函数核心 + 薄 function_tool 包装"两层（对齐 tools.py / actions.py）：
`classify_diagnostic_command` / `validate_host` / `run_diagnostic_core` 都可脱离 SDK 单测。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch

from agents import RunContextWrapper, function_tool

from .guardrails import detect_forbidden_command, forbidden_write_command_guardrail
from .tools import DocsContext

logger = logging.getLogger(__name__)

# 工具名常量：orchestration 按名字判断专家挂了哪些横切工具（决定 prompt 加哪段）。
DIAG_TOOL_NAME = "run_diagnostic"

# 诊断输出上限：远端命令 stdout 可能很长（top、CLIENT LIST），截断防撑爆上下文。
_DEFAULT_OUTPUT_LIMIT = 4000
# 单条诊断命令的执行超时（秒）：ssh 卡住 / 远端 hang 时兜底。
_DEFAULT_TIMEOUT = 15.0
# 生产环境判定关键词：命中即拒（诊断只允许测试环境）。
_DEFAULT_PROD_PATTERNS = ("prod", "production", "正式")


# ---------------------------------------------------------------------------
# 只读命令白名单（默认拒绝的正向安全模型）
# ---------------------------------------------------------------------------
#
# 每个管道段的**基础命令**必须落在这里（或是下面的 DB / Redis 客户端并通过子检查），
# 否则整条命令按"未知命令，不在只读白名单内"拒绝。这与参考项目的黑名单相反：参考项目
# 枚举"坏命令"、漏一个就放行；这里枚举"好命令"、没列进来就默认拒。

# 系统只读诊断命令 + 只读文本过滤器（后者也可作为管道下游段，如 `top | head`）。
_READ_ONLY_BASE: frozenset[str] = frozenset(
    {
        # 资源 / 进程 / 网络
        "free",
        "top",
        "htop",
        "uptime",
        "vmstat",
        "iostat",
        "mpstat",
        "pidstat",
        "sar",
        "df",
        "du",
        "ps",
        "pstree",
        "netstat",
        "ss",
        "lsof",
        "nproc",
        "lscpu",
        "lsblk",
        "lsof",
        "dmesg",
        "journalctl",
        "uname",
        "hostname",
        "date",
        "w",
        "who",
        "whoami",
        "id",
        "env",
        "printenv",
        # 文件 / 日志读取 + 只读文本过滤
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "grep",
        "egrep",
        "fgrep",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "column",
        "awk",
        "nl",
        "stat",
        "echo",
        "true",
        # 只读的 sed 用法（-i 原地改写在下面 detect_write_command 里被判为写）
        "sed",
        # 大数据 / 中间件只读脚本（--describe / --list 等，写子命令另拦）
        "kafka-topics.sh",
        "kafka-consumer-groups.sh",
    }
)

# 数据库客户端：允许调用，但必须能校验出**只读 SQL**（SELECT/SHOW/DESC/EXPLAIN），
# 否则拒——SQL 的写面是开放式的，不能靠枚举。
_DB_CLIENTS: frozenset[str] = frozenset({"mysql", "obclient", "mariadb"})

# Redis 客户端：读子命令数量庞大（几百个），写子命令集合明确且已被 detect_write_command
# / detect_forbidden_command 覆盖——所以这里对 redis-cli 用"允许，写/毁灭由检测器拦"的
# 姿态（forbidden/write 在 classify 里先于白名单判定，read 面则整体放行）。
_REDIS_CLIENTS: frozenset[str] = frozenset({"redis-cli", "valkey-cli", "keydb-cli"})

# 只读 SQL 起始动词（去掉前导空白/括号后匹配）。
_READ_ONLY_SQL_RE = re.compile(
    r"^\s*\(?\s*(select|show|desc|describe|explain|analyze\s+table)\b", re.I
)

# 重定向 token（`>&` 是 shlex 合并出的 fd-dup 运算符，如 `2>&1` → ['2','>&','1']）。
_REDIRECT_OPS: frozenset[str] = frozenset({">", ">>", ">&"})
# 命令分隔 / 子 shell / 输入重定向：一律拒（不是可审批的写，是注入面）。
_REJECT_OPS: frozenset[str] = frozenset({";", "&", "&&", "||", "(", ")", "<", "<<", "<<<"})


# ---------------------------------------------------------------------------
# 写命令检测（识别到的写 → 路由到审批）。移植自参考项目 ops-qa-bot 的 _WRITE_PATTERNS，
# 语义从"命中即 deny"改成"命中即判为 write，交由上层路由到 request_write_command 审批"。
# 毁灭性子集（rm 根目录 / FLUSH / DROP DATABASE / mkfs / dd 直写块设备）由 guardrails 的
# detect_forbidden_command 单独识别、且在 classify 里**先于**本检测判定，所以那些会落到
# forbidden 而非 write。这里保留它们在正则里无妨（先判 forbidden，走不到这一层）。
# ---------------------------------------------------------------------------
_WRITE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 文件系统破坏性写
    (re.compile(r"\brm\b", re.I), "rm"),
    (re.compile(r"\bmv\b", re.I), "mv"),
    (re.compile(r"\bcp\b", re.I), "cp"),
    (re.compile(r"\bdd\b", re.I), "dd"),
    (re.compile(r"\bmkfs\b", re.I), "mkfs"),
    (re.compile(r"\bchmod\b", re.I), "chmod"),
    (re.compile(r"\bchown\b", re.I), "chown"),
    (re.compile(r"\btruncate\b", re.I), "truncate"),
    (re.compile(r"\bshred\b", re.I), "shred"),
    (re.compile(r"\bunlink\b", re.I), "unlink"),
    (re.compile(r"\btee\b", re.I), "tee"),
    # sed 原地改文件（-i / -ri / -i.bak / --in-place）；只读过滤的 sed 不算写
    (re.compile(r"\bsed\s+(-[a-zA-Z]*i[a-zA-Z]*\b|--in-place)", re.I), "sed -i"),
    # dmesg 清缓冲区（-c / -C / --clear / --read-clear）——读带清也算写
    (re.compile(r"\bdmesg\s+(-[a-zA-Z]*[cC]\b|--clear|--read-clear)", re.I), "dmesg 清缓冲"),
    # 文件外传 / 覆盖
    (re.compile(r"\bscp\b", re.I), "scp"),
    (re.compile(r"\brsync\b", re.I), "rsync"),
    # crontab 除 -l（只读列出）外都是装载/删除
    (re.compile(r"\bcrontab\b(?!\s+-l\b)", re.I), "crontab 写操作"),
    # 进程 / 服务管理
    (re.compile(r"\bkill(all)?\b", re.I), "kill/killall"),
    (re.compile(r"\bpkill\b", re.I), "pkill"),
    (re.compile(r"\breboot\b", re.I), "reboot"),
    (re.compile(r"\bshutdown\b", re.I), "shutdown"),
    (re.compile(r"\bhalt\b", re.I), "halt"),
    (
        re.compile(r"\bsystemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask)\b", re.I),
        "systemctl 写操作",
    ),
    (re.compile(r"\bservice\s+\S+\s+(start|stop|restart|reload)\b", re.I), "service 写操作"),
    # Redis 管理写命令
    (re.compile(r"\bCONFIG\s+SET\b", re.I), "CONFIG SET"),
    (re.compile(r"\bCONFIG\s+REWRITE\b", re.I), "CONFIG REWRITE"),
    (
        re.compile(r"\bCLUSTER\s+(FORGET|RESET|FAILOVER|ADDSLOTS|DELSLOTS|SETSLOT)\b", re.I),
        "CLUSTER 写",
    ),
    (
        re.compile(r"\bDEBUG\s+(SLEEP|RELOAD|LOADAOF|FLUSHALL|SET-ACTIVE-EXPIRE)\b", re.I),
        "DEBUG 写",
    ),
    # Redis 数据写（SET/DEL 一族）：锚定在 redis-cli 调用段内，[^|;&]* 不跨管道/分号，
    # 避免 `redis-cli INFO | grep set`、名字带 set 的 key 被误杀。超通用词还要求后跟参数。
    (
        re.compile(
            r"redis-cli\b[^|;&]*"
            r"\b(SET|DEL|INCR|DECR|APPEND|MOVE|COPY|RENAME|EXPIRE|PERSIST)"
            r"\s+\S+",
            re.I,
        ),
        "Redis 写命令",
    ),
    (
        re.compile(
            r"redis-cli\b[^|;&]*\b("
            r"SETEX|PSETEX|SETNX|MSET|MSETNX|GETSET|GETDEL|SETRANGE|SETBIT"
            r"|UNLINK|PEXPIRE|EXPIREAT|PEXPIREAT|RENAMENX"
            r"|MIGRATE|RESTORE|SWAPDB"
            r"|INCRBY|DECRBY|INCRBYFLOAT"
            r"|LPUSH|RPUSH|LPOP|RPOP|LSET|LREM|LTRIM|LINSERT|LMOVE|RPOPLPUSH"
            r"|SADD|SREM|SPOP|SMOVE|SINTERSTORE|SUNIONSTORE|SDIFFSTORE"
            r"|HSET|HMSET|HSETNX|HDEL|HINCRBY|HINCRBYFLOAT"
            r"|ZADD|ZREM|ZINCRBY|ZPOPMIN|ZPOPMAX|ZREMRANGEBY\w+|ZRANGESTORE"
            r"|XADD|XDEL|XTRIM|XSETID|XGROUP"
            r"|PFADD|PFMERGE|GEOADD"
            r"|BGSAVE|BGREWRITEAOF"
            r"|SLAVEOF|REPLICAOF"
            r"|EVAL|EVALSHA|FCALL"
            r"|SCRIPT\s+(LOAD|FLUSH|KILL)"
            r"|FUNCTION\s+(LOAD|DELETE|FLUSH|RESTORE)"
            r"|ACL\s+(SETUSER|DELUSER|LOAD|SAVE)"
            r")\b",
            re.I,
        ),
        "Redis 写命令",
    ),
    # SQL 写（带后续修饰避免 SHOW CREATE TABLE 误中）。SET GLOBAL/PERSIST、ALTER SYSTEM
    # 是参数变更——正是 request_write_command 审批链路要接的东西。
    (
        re.compile(
            r"\b(INSERT\s+INTO"
            r"|REPLACE\s+INTO"
            r"|UPDATE\s+\S+\s+SET"
            r"|DELETE\s+FROM"
            r"|LOAD\s+DATA"
            r"|RENAME\s+TABLE"
            r"|DROP\s+(TABLE|DATABASE|INDEX|VIEW|PROCEDURE|USER)"
            r"|ALTER\s+(TABLE|DATABASE|USER|SYSTEM|TENANT)"
            r"|TRUNCATE\s+(TABLE\s+)?\S+"
            r"|SET\s+(GLOBAL|PERSIST|SESSION)"
            r"|GRANT\s+.+\s+ON"
            r"|REVOKE\s+.+\s+ON)\b",
            re.I,
        ),
        "SQL 写操作",
    ),
    (
        re.compile(r"(?<!\bSHOW\s)\bCREATE\s+(TABLE|DATABASE|INDEX|VIEW|PROCEDURE|USER)\b", re.I),
        "SQL 写操作",
    ),
    # sudo：即便部署账号能 sudo 也走审批，不在只读诊断里直接跑
    (re.compile(r"\bsudo\b"), "sudo"),
]


def detect_write_command(command: str) -> str | None:
    """检测命令是否命中"识别到的写"清单。命中返回原因标签，否则 None。纯函数，可单测。"""
    for pattern, label in _WRITE_PATTERNS:
        if pattern.search(command):
            return label
    return None


# ---------------------------------------------------------------------------
# 命令分类（纯函数）：只读白名单 + 写路由 + 禁止兜底
# ---------------------------------------------------------------------------

# 分类结果四态：
#   allow     —— 只读、在白名单内 → 执行
#   write     —— 识别到的写命令 → 路由到 request_write_command 审批（不在诊断里跑）
#   forbidden —— 毁灭性命令 → 直接拒（不可执行、不可提议）
#   reject    —— 未知/不在白名单/含危险 shell 结构 → 拒（引导模型换只读命令或走文字建议）
_KIND_ALLOW = "allow"
_KIND_WRITE = "write"
_KIND_FORBIDDEN = "forbidden"
_KIND_REJECT = "reject"


@dataclass(frozen=True)
class DiagVerdict:
    """`classify_diagnostic_command` 的判定：类别 + 可读原因标签。"""

    kind: str  # allow / write / forbidden / reject
    label: str | None = None


def _basename(token: str) -> str:
    """取命令基名：去路径（/usr/bin/free → free），保留脚本名的 .sh 后缀。"""
    return token.rsplit("/", 1)[-1]


def _tokenize(command: str) -> list[str]:
    """shell 式分词（quote-aware + 运算符独立成 token）。

    用 `shlex` 的 `punctuation_chars=True`：`();<>|&` 被当运算符切出来（`&&`/`>>`/`>&`
    等连写会合并成单 token），而引号内的 `>` `;` 仍留在 token 里——`awk '$3 > 100'` 的 `>`
    不会被误判成重定向。这是相比参考项目"裸正则扫命令串"最实的收益：注入面从解析层就切干净。
    引号不配对时抛 ValueError，由 classify 捕获转成 reject。
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _find_forbidden_operator(tokens: list[str]) -> str | None:
    """扫描分词后的 token，找危险 shell 结构。命中返回可读原因，否则 None。

    放行：单个 `|`（管道，另行按段校验）、只读重定向（`> /dev/null`、`2>&1` 这类 fd-dup）。
    拦截：命令分隔 `;`/`&&`/`||`/`&`、子 shell `(...)`、命令替换 `$(...)`/反引号、
    输入重定向 `<`、写文件重定向（`>`/`>>` 到 /dev/null 以外）。
    """
    for i, t in enumerate(tokens):
        if t == "|":
            continue  # 管道由 _split_pipeline 按段校验
        if t in _REJECT_OPS:
            return t
        if "`" in t:
            return "反引号命令替换"
        if t in _REDIRECT_OPS:
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            if t == ">&":
                # fd 复制（2>&1 / 1>&2 / >&-）：目标必须是 fd 数字或 `-`。
                if target not in ("0", "1", "2", "-"):
                    return f"重定向 {t}{target}"
            elif target != "/dev/null":
                # 写文件重定向：只放行 /dev/null（丢弃输出），其余（写真实路径）拒。
                return f"重定向 {t} {target}"
    return None


def _split_pipeline(tokens: list[str]) -> list[list[str]]:
    """按 `|` token 把命令切成若干管道段。"""
    segments: list[list[str]] = []
    cur: list[str] = []
    for t in tokens:
        if t == "|":
            segments.append(cur)
            cur = []
        else:
            cur.append(t)
    segments.append(cur)
    return segments


def _extract_sql(seg: list[str]) -> str | None:
    """从 mysql/obclient 段里抠出 -e / --execute 传入的 SQL；取不到返回 None。"""
    for i, tok in enumerate(seg):
        if tok in ("-e", "--execute") and i + 1 < len(seg):
            return seg[i + 1]
        if tok.startswith("-e") and len(tok) > 2:
            return tok[2:]
        if tok.startswith("--execute="):
            return tok[len("--execute=") :]
    return None


def _is_read_only_sql(sql: str) -> bool:
    return bool(_READ_ONLY_SQL_RE.match(sql))


def classify_diagnostic_command(command: str) -> DiagVerdict:
    """把一条诊断命令分类为 allow / write / forbidden / reject。纯函数，可单测。

    判定优先级（顺序即安全语义）：
    1. **禁止清单**（毁灭性）→ forbidden。最先判，即使命令别的地方能解析成只读也拒。
    2. **无法安全解析**（引号不配对）/ **危险 shell 结构**（`;` `$()` 反引号 写重定向…）→ reject。
       shell 注入不是"可审批的写"，直接拒。
    3. **识别到的写**（重启 / CONFIG SET / SQL 写 / sudo…）→ write（路由到审批）。
    4. 每个管道段的基础命令都在只读白名单内（DB 客户端另需只读 SQL）→ allow。
    5. 其余（未知命令 / 不在白名单）→ reject。
    """
    command = command.strip()
    if not command:
        return DiagVerdict(_KIND_REJECT, "空命令")

    # 1) 禁止清单兜底（毁灭性）：用原始串正则，即便分词异常也先拦下。
    forbidden = detect_forbidden_command(command)
    if forbidden:
        return DiagVerdict(_KIND_FORBIDDEN, forbidden)

    # 2) quote-aware 分词。运算符独立成 token，引号内的 > ; 仍留在 token 里（见 _tokenize）。
    try:
        tokens = _tokenize(command)
    except ValueError:
        return DiagVerdict(_KIND_REJECT, "命令无法安全解析（引号可能不配对）")
    if not tokens:
        return DiagVerdict(_KIND_REJECT, "空命令")
    if op := _find_forbidden_operator(tokens):
        return DiagVerdict(_KIND_REJECT, f"命令含禁止的 shell 结构：{op}")

    # 3) 识别到的写 → 审批（先于白名单，否则 mysql SET GLOBAL 会掉进"只读 SQL 校验失败"拒绝）。
    if wr := detect_write_command(command):
        return DiagVerdict(_KIND_WRITE, wr)

    # 4) 逐段白名单校验（默认拒绝）。
    for seg in _split_pipeline(tokens):
        if not seg:
            return DiagVerdict(_KIND_REJECT, "空的管道段")
        base = _basename(seg[0])
        if base in _REDIS_CLIENTS:
            continue  # 读面庞大；写/毁灭已在上面被 forbidden/write 拦掉
        if base in _DB_CLIENTS:
            sql = _extract_sql(seg)
            if sql is None:
                return DiagVerdict(_KIND_REJECT, f"{base} 需用 -e 传入可校验的只读 SQL（未见 -e）")
            if not _is_read_only_sql(sql):
                return DiagVerdict(
                    _KIND_REJECT,
                    f"{base} 的 SQL 不是只读查询（只允许 SELECT/SHOW/DESC/EXPLAIN）",
                )
            continue
        if base not in _READ_ONLY_BASE:
            return DiagVerdict(_KIND_REJECT, f"`{base}` 不在只读诊断白名单内")

    return DiagVerdict(_KIND_ALLOW)


# ---------------------------------------------------------------------------
# 目标机校验（生产拒答 + 可选白名单）
# ---------------------------------------------------------------------------

# 目标机名**正向白名单**：必须以字母/数字开头（挡掉前导 `-` 的 ssh 选项注入，如
# `-oProxyCommand=…`——shlex.quote 只防 shell 拆词、防不住 ssh 自己把 `-x` 当选项解析），
# 其后只允许字母/数字/`.`/`_`/`@`/`-`。这比"列非法字符"的黑名单更稳：任何 shell 元字符、
# 空格、前导连字符都因不在字符集里而被拒。（IPv6 的 `:` 暂不支持，需要时再放开。）
_HOST_VALID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


def validate_host(host: str, config: DiagConfig) -> str | None:
    """校验目标机：命中返回拒绝原因（可读），合法返回 None。纯函数，可单测。

    - 生产机（名字带 prod/production/正式）直接拒——诊断只允许测试环境。
    - 配了 allowed_hosts 时，host 必须匹配其中一个 glob（如 `10.1.*`、`*-test-*`）。
    - host 走正向字符白名单（`_HOST_VALID_RE`）：必须字母/数字开头（挡前导 `-` 的 ssh 选项
      注入），只含字母/数字/`.`/`_`/`@`/`-`；shell 元字符/空格/前导连字符一律拒。
    """
    host = host.strip()
    if not host:
        return "未指定目标机器"
    low = host.lower()
    for pat in config.prod_patterns:
        if pat and pat.lower() in low:
            return f"`{host}` 疑似生产环境（命中 `{pat}`）；诊断只允许测试环境，请联系运维"
    if not _HOST_VALID_RE.match(host):
        return f"目标机名不合法（须字母/数字开头，只含字母数字 . _ @ -）：{host}"
    if config.allowed_hosts:
        if not any(fnmatch(host, p) or fnmatch(low, p.lower()) for p in config.allowed_hosts):
            return f"`{host}` 不在允许诊断的目标名单内（OPS_QA_DIAG_ALLOWED_HOSTS）"
    return None


# ---------------------------------------------------------------------------
# 执行结果 + 执行器（可注入）
# ---------------------------------------------------------------------------


@dataclass
class DiagResult:
    """一次只读诊断的执行结果。"""

    host: str
    command: str
    ok: bool
    exit_code: int | None
    output: str  # 远端 stdout+stderr（已截断）
    note: str = ""  # 附加说明（超时 / 模拟数据 / 未配置等）


# 执行器签名：给定 host + 只读 command + 配置，返回 DiagResult。可注入以便单测/演示。
DiagExecutor = Callable[[str, str, "DiagConfig"], Awaitable[DiagResult]]


def _truncate(text: str, limit: int) -> str:
    text = text.rstrip()
    if len(text) > limit:
        return text[:limit] + f"\n…[输出超过 {limit} 字符已截断]"
    return text


def build_nested_ssh_argv(host: str, command: str, jumphost: str) -> list[str]:
    """把 host + 只读命令拼成"经跳板机的嵌套 ssh" argv（不经部署机本地 shell）。

    等价于参考项目要求模型手写的 `ssh jumphost "ssh <host> '<command>'"`，但由代码用
    `shlex.quote` 安全拼接：外层 argv = ["ssh", "--", jumphost, inner]；inner 在 jumphost 的
    登录 shell 里被解析成 `ssh -- <host> <quoted-command>`，内层 ssh 再把整条命令带到 target。
    模型永远不碰 ssh / 跳板机语法，也就无从写错两跳或改用 ssh -J 绕过认证拓扑。

    两处 `--` 是**纵深防御**：ssh 的 `--` 终止选项解析，即便某个以 `-` 开头的 host 侥幸绕过
    `validate_host`，也只会被当目标机名、不会被 ssh 当选项（挡 `-oProxyCommand=…` 这类注入）。
    validate_host 是第一道（正向白名单拒前导 `-`），这里是第二道。
    """
    inner = f"ssh -- {shlex.quote(host)} {shlex.quote(command)}"
    return ["ssh", "--", jumphost, inner]


async def ssh_executor(host: str, command: str, config: DiagConfig) -> DiagResult:
    """真执行器：经跳板机嵌套 ssh 跑只读命令，带超时与输出截断。"""
    argv = build_nested_ssh_argv(host, command, config.jumphost)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:
        return DiagResult(host, command, False, None, "", note=f"无法启动 ssh：{e}")
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=config.timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return DiagResult(
            host, command, False, None, "", note=f"执行超时（>{config.timeout:.0f}s）"
        )
    text = _truncate(out.decode("utf-8", errors="replace"), config.output_limit)
    return DiagResult(host, command, proc.returncode == 0, proc.returncode, text)


async def mock_executor(host: str, command: str, config: DiagConfig) -> DiagResult:
    """模拟执行器：未配 jumphost（或显式 OPS_QA_DIAG_MOCK=1）时用，产出**带标注的假数据**，
    让"实时诊断"链路在没有真实运维基建时也能端到端跑起来（本地开发 / 演示 / 单测）。"""
    base = _basename(command.split()[0]) if command.strip() else ""
    samples = {
        "free": "              total    used    free\nMem:           15Gi    14Gi   498Mi",
        "redis-cli": "used_memory_human:14.20G\nmaxmemory_human:15.00G\nmaxmemory_policy:noevict",
        "mysql": "Id  User  Host  db  Command  Time  State\n12  app  x  db  Query  35  Sending",
        "df": "Filesystem      Size  Used Avail Use%\n/dev/vda1        50G   47G  1.2G  98%",
        "top": "load average: 7.82, 6.10, 4.55\n%Cpu(s): 88.1 us,  9.2 sy",
    }
    body = samples.get(base, f"$ {command}\n(模拟输出：本机未配置 jumphost，返回示例数据)")
    return DiagResult(
        host,
        command,
        ok=True,
        exit_code=0,
        output=_truncate(body, config.output_limit),
        note="模拟数据（未配置 jumphost；设 OPS_QA_DIAG_JUMPHOST 后走真实 ssh）",
    )


# ---------------------------------------------------------------------------
# 配置（缺省关，从环境变量解析）
# ---------------------------------------------------------------------------


@dataclass
class DiagConfig:
    """实时诊断配置。缺省 enabled=False（工具不挂载、prompt 不加章节，零感知）。"""

    enabled: bool = False
    jumphost: str = ""  # 跳板机 ssh 别名（~/.ssh/config 里配好），空则降级为模拟执行
    allowed_hosts: tuple[str, ...] = ()  # 允许诊断的目标 glob；空 = 不限（仍拒生产）
    prod_patterns: tuple[str, ...] = _DEFAULT_PROD_PATTERNS
    timeout: float = _DEFAULT_TIMEOUT
    output_limit: int = _DEFAULT_OUTPUT_LIMIT
    mock: bool = False  # 显式强制模拟执行（OPS_QA_DIAG_MOCK=1）
    executor: DiagExecutor | None = None  # 注入自定义执行器（单测/演示用）

    @property
    def use_mock(self) -> bool:
        """真跑 ssh 还是模拟：显式 mock、或没配 jumphost 时走模拟。"""
        return self.mock or not self.jumphost

    async def run(self, host: str, command: str) -> DiagResult:
        """按配置选执行器跑一条已判定为只读的命令。"""
        ex = self.executor or (mock_executor if self.use_mock else ssh_executor)
        return await ex(host, command, self)

    @classmethod
    def from_env(cls) -> DiagConfig:
        """从 OPS_QA_DIAG* 环境变量解析。缺省全关。"""

        def _flag(name: str) -> bool:
            return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")

        def _list(name: str) -> tuple[str, ...]:
            raw = (os.environ.get(name) or "").strip()
            return tuple(x.strip() for x in raw.split(",") if x.strip()) if raw else ()

        def _float(name: str, default: float) -> float:
            raw = (os.environ.get(name) or "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        prod = _list("OPS_QA_DIAG_PROD_PATTERNS") or _DEFAULT_PROD_PATTERNS
        return cls(
            enabled=_flag("OPS_QA_DIAG"),
            jumphost=(os.environ.get("OPS_QA_DIAG_JUMPHOST") or "").strip(),
            allowed_hosts=_list("OPS_QA_DIAG_ALLOWED_HOSTS"),
            prod_patterns=prod,
            timeout=_float("OPS_QA_DIAG_TIMEOUT", _DEFAULT_TIMEOUT),
            mock=_flag("OPS_QA_DIAG_MOCK"),
        )


@dataclass
class DiagnosticLog:
    """收集本会话跑过的只读诊断（含被拒的判定），便于观测/测试。"""

    results: list[DiagResult] = field(default_factory=list)
    rejections: list[tuple[str, str, DiagVerdict]] = field(
        default_factory=list
    )  # (host, cmd, verdict)


# ---------------------------------------------------------------------------
# 纯核心 + function_tool 包装
# ---------------------------------------------------------------------------


def render_result(result: DiagResult) -> str:
    """把执行结果渲染成给模型的文本（模型据此把实时数据整合进答案）。"""
    header = f"（实时数据：{result.host}）命令 `{result.command}`"
    if result.note:
        header += f"（{result.note}）"
    status = "" if result.ok else f"\n[命令未成功，退出码 {result.exit_code}]"
    body = result.output or "(无输出)"
    return f"{header}：\n{body}{status}"


async def run_diagnostic_core(
    config: DiagConfig, log: DiagnosticLog, host: str, command: str
) -> str:
    """诊断工具的纯核心：校验 host + 分类命令 + （只读则）执行，返回给模型的文本。

    脱离 SDK 可直接 await 单测（工具层只是把 ctx 取出后转调本函数）。
    """
    reason = validate_host(host, config)
    if reason:
        log.rejections.append((host, command, DiagVerdict(_KIND_REJECT, reason)))
        return f"[拒绝] {reason}。"

    verdict = classify_diagnostic_command(command)
    if verdict.kind == _KIND_FORBIDDEN:
        log.rejections.append((host, command, verdict))
        return (
            f"[拒绝] 该命令属毁灭性操作（{verdict.label}），不可执行、也不可提议执行。"
            "请改为文字说明风险与更安全的替代方案，并标 ⚠️。"
        )
    if verdict.kind == _KIND_WRITE:
        log.rejections.append((host, command, verdict))
        return (
            f"[需审批·未执行] 这是**写/变更**命令（{verdict.label}），只读诊断工具不会执行它。"
            "若确需变更，请改用 `request_write_command` 工具**提议**（会挂起等人工审批）；"
            "若无该工具，则以文字建议形式给出（标 ⚠️ 风险、由人工执行），不要在诊断里跑。"
        )
    if verdict.kind == _KIND_REJECT:
        log.rejections.append((host, command, verdict))
        return (
            f"[拒绝] {verdict.label}。实时诊断只允许**白名单内的只读命令**"
            "（系统 free/df/ss、日志 tail/grep、redis-cli 只读子命令、mysql 的 SELECT/SHOW 等）。"
            "请换一条只读命令，或直接基于文档作答。"
        )

    # allow：执行。
    result = await config.run(host, command)
    log.results.append(result)
    logger.info("diag host=%s cmd=%r ok=%s note=%s", host, command, result.ok, result.note)
    return render_result(result)


def make_diagnostic_tool(config: DiagConfig, log: DiagnosticLog):
    """构造 `run_diagnostic(host, command)` 工具（只读执行 / 写路由审批 / 毁灭性拒）。

    - 参数是结构化的 host + command，跳板机嵌套 ssh 由底层拼（模型不碰 ssh 语法）。
    - 叠挂 `forbidden_write_command_guardrail`（SDK tool-level guardrail）作毁灭性命令的
      **第二层兜底**：与写审批工具同源，随工具走——即便 core 的判定被绕过也执行不到。
    """

    @function_tool(tool_input_guardrails=[forbidden_write_command_guardrail])
    async def run_diagnostic(ctx: RunContextWrapper[DocsContext], host: str, command: str) -> str:
        """到**测试环境**机器上跑**一条只读**诊断命令，返回实时状态（叠加到基于文档的回答上）。

        只在用户明确问"当前/此刻/正在"的状态、且给了具体机器（IP/主机名）时用；纯知识问题
        不要调。只允许只读命令（系统 free/top/df/ss、日志 tail/grep、redis-cli 只读子命令、
        mysql 的 SELECT/SHOW/DESC/EXPLAIN 等）；写/变更命令请改用 request_write_command。

        Args:
            host: 目标机器（用户给的 IP 或主机名，以用户为准，不要自己猜；生产机会被拒）。
            command: 要在该机上跑的**那一条只读命令**（如 `free -h`、`redis-cli INFO`）。
                     不要写 ssh / 跳板机语法——底层会自动经跳板机执行。
        """
        return await run_diagnostic_core(config, log, host, command)

    return run_diagnostic
