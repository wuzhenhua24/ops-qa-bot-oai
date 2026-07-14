"""`INDEX.md` 路由表解析：组件注册表 + 飞书文档来源 + 飞书来源的引用标识。

原先 `parse_index_components` 住在 `orchestration.py` 里，只在建 agent 图时用一次。
接入飞书文档问答（`doc_qa.py`）之后，同一张表要被三处读到：

- `orchestration`：为每个组件建专家（local 组件挂文档检索工具，feishu 组件挂 `query_feishu_doc`）。
- `doc_qa`：`query_feishu_doc` 每次调用时按组件名解析出 doc token。
- `schema.validate_citations`：核对 `飞书文档·<组件>` 这类来源是否**真的登记过**。

所以把解析下沉成一个**不依赖 agents SDK 的叶子模块**，三边共用同一份结果——参考项目
`ops-qa-bot` 里 `parse_feishu_registry` 与 `_index_owner_to_dirs` 各解析一遍同一张表
（列定位逻辑重复、行为可能漂移），这里合成一处。

带 mtime 缓存：`query_feishu_doc` 是每次工具调用都要查注册表的（这样改了 INDEX.md 的
登记不必重启进程），文件没变时直接返回上次的解析结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# 「来源」列的取值。缺省 local（列不存在时也按 local 处理，兼容没有该列的旧 INDEX.md）。
LOCAL_SOURCE = "local"
FEISHU_SOURCE = "feishu"

# 飞书来源的答案没有本地文件路径可引用，用「飞书文档·<组件名>」当来源标识。它和本地
# 相对路径一样会被 `schema.validate_citations` 核对——不是绕过校验，而是把"来源必须真实
# 存在"从"文件存在"扩展到"组件已在 INDEX.md 登记"。
CITATION_PREFIX = "飞书文档·"

# 解析模型写出来的来源标识。模型未必打得出 `·`（U+00B7），容忍常见的几个分隔符和无分隔符。
_CITATION_RE = re.compile(r"^飞书文档\s*[·・:：\-]?\s*(\S.*)$")

# 「飞书文档」单元格里多个 doc token / url 的分隔符：中英文逗号、顿号、分号、空白。
_DOC_SPLIT_RE = re.compile(r"[,，、;；\s]+")

# 表格里表示"没有"的单元格写法，归一成空。
_EMPTY_CELLS = frozenset({"", "-", "—", "–", "无", "n/a"})


@dataclass
class Component:
    """`INDEX.md` 里登记的一个组件。"""

    name: str  # 组件名，如 "Redis"
    dir: str  # 目录名（不带斜杠），如 "redis"
    source: str  # "local" / "feishu"
    coverage: str  # 覆盖内容描述
    open_id: str  # 负责人 open_id（升级用）
    docs: tuple[str, ...] = ()  # 飞书 doc token / url（仅 source=feishu 有）

    @property
    def is_feishu(self) -> bool:
        return self.source == FEISHU_SOURCE


def norm_key(s: str) -> str:
    """组件名 / 目录名归一化成查找 key：去 backtick、去首尾斜杠和空白、转小写。"""
    return s.strip().strip("`").strip("/").strip().lower()


# 函数调用的工具名只允许 [A-Za-z0-9_]。
_IDENT_RE = re.compile(r"[^0-9A-Za-z_]")


def safe_ident(s: str) -> str:
    """把组件目录名转成函数调用安全的标识符（非字母/数字/下划线的字符 → `_`）。

    专家 agent 名（SDK 会自动生成 handoff 工具 `transfer_to_<agent名>`）和协调者的
    `ask_<dir>` 工具名都受工具命名约束。目录名带连字符等字符时（如 `anti-asset`），
    SDK 会自己转换并打 WARNING，且分诊 prompt 里的转交目标名会与实际工具名对不上——
    在源头转掉。评测比对 route 时用同一函数清洗 expected_route，保证能对上。
    """
    return _IDENT_RE.sub("_", s)


def feishu_citation(component_name: str) -> str:
    """构造飞书来源的引用标识，如 `飞书文档·Nginx`。"""
    return f"{CITATION_PREFIX}{component_name}"


def parse_feishu_citation(citation: str) -> str | None:
    """从引用标识里抠出组件名；不是飞书来源标识则返回 None（按本地路径处理）。"""
    m = _CITATION_RE.match(citation.strip().strip("`"))
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# INDEX.md 组件表解析（mtime 缓存）
# ---------------------------------------------------------------------------

_cache: dict[Path, tuple[float, list[Component]]] = {}


def _split_row(line: str) -> list[str]:
    """拆一行 markdown 表格为去空白的单元格列表。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _clean_dir(raw: str) -> str:
    """把 "`redis/`" 这种目录单元格归一成 "redis"。"""
    return raw.strip().strip("`").strip().rstrip("/").lstrip("/")


def _cell(raw: str) -> str:
    """去 backtick + 把 `-` / `无` 这类占位写法归一成空串。"""
    v = raw.strip().strip("`").strip()
    return "" if v.lower() in _EMPTY_CELLS else v


def _parse_rows(lines: list[str]) -> list[Component]:
    """从 INDEX.md 的行里解析组件表。

    按表头名定位列（容忍列顺序/有无变化）：组件 / 来源 / 目录 / 飞书文档 / 覆盖内容 / open_id。
    缺「来源」列时一律按 local 处理；缺 open_id / 飞书文档列时留空。只有能解析出目录名的
    行才算数（目录名是升级标记 `<<ESCALATE:ou_xxx:目录>>` 和专家 agent 命名的依据）。
    """
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []

    header = _split_row(rows[0])

    def col(*keywords: str) -> int:
        for i, h in enumerate(header):
            hl = h.lower()
            if any(k in hl for k in keywords):
                return i
        return -1

    i_name = col("组件")
    i_src = col("来源")
    i_dir = col("目录")
    i_docs = col("飞书文档", "飞书")
    i_cov = col("覆盖")
    i_oid = col("open_id", "openid", "open id")

    components: list[Component] = []
    for ln in rows[1:]:
        # 跳过分隔行（形如 |---|---|）。
        if set(ln.strip()) <= set("|-: "):
            continue
        cells = _split_row(ln)

        def get(idx: int, _cells: list[str] = cells) -> str:
            return _cells[idx] if 0 <= idx < len(_cells) else ""

        name = get(i_name)
        dir_ = _clean_dir(get(i_dir))
        if not name or not dir_:
            continue
        source = (get(i_src) or LOCAL_SOURCE).strip("`").strip().lower()
        docs = tuple(t for t in _DOC_SPLIT_RE.split(_cell(get(i_docs))) if t)
        components.append(
            Component(
                name=name,
                dir=dir_,
                source=source,
                coverage=get(i_cov),
                open_id=get(i_oid),
                docs=docs,
            )
        )
    return components


def parse_index_components(docs_root: Path) -> list[Component]:
    """解析 `docs_root/INDEX.md` 的组件表，返回组件列表（INDEX.md 缺失时返回空）。

    mtime 缓存：文件没改直接返回上次结果的副本。读盘失败回退到上次缓存（或空表）。
    """
    index_path = docs_root / "INDEX.md"
    try:
        mtime = index_path.stat().st_mtime
    except OSError:
        return []
    cached = _cache.get(index_path)
    if cached is not None and cached[0] == mtime:
        return list(cached[1])
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return list(_cache.get(index_path, (0.0, []))[1])
    components = _parse_rows(lines)
    _cache[index_path] = (mtime, components)
    return list(components)


def feishu_registry(docs_root: Path) -> dict[str, Component]:
    """飞书来源组件的查找表：`{归一化 key: Component}`。

    只收「来源」= feishu 且「飞书文档」列非空的行——登记为 feishu 却没填 token 的行进不来，
    `query_feishu_doc` 会以"未登记"提示引导 agent 走升级规则，而不是拿着空 token 去打上游。

    组件名和目录名都建别名（`nginx` 和 `Nginx` 都能命中），agent 传哪个都认。
    """
    registry: dict[str, Component] = {}
    for c in parse_index_components(docs_root):
        if not c.is_feishu or not c.docs:
            continue
        registry[norm_key(c.name)] = c
        if dir_key := norm_key(c.dir):
            registry.setdefault(dir_key, c)
    return registry
