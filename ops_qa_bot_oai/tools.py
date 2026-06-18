"""文档检索工具：read_doc / glob_docs / grep_docs。

这是本项目与 `ops-qa-bot`（Claude Agent SDK）最关键的差异点。Claude Agent SDK
自带 `Read`/`Glob`/`Grep` 内置工具，agent 开箱即用；OpenAI Agents SDK **不提供**
文件系统工具，必须用 `@function_tool` 自己实现，并自己负责沙箱与防越权。

设计目标是让这三个工具的语义尽量贴近 Claude SDK 的内置版，从而把对比变量收敛到
"两个 SDK 的 agent loop / 工具调度 / prompt 适配"，而不是"工具能力本身不同"：

- `read_doc`：读单个 md 文件（对标 Read）。
- `glob_docs`：按 glob pattern 列文件（对标 Glob）。
- `grep_docs`：跨文档正则搜关键词（对标 Grep）。

实现拆成"纯函数核心 + 薄 function_tool 包装"两层：核心 `_read_doc/_glob_docs/
_grep_docs` 接收 `docs_root: Path`，便于脱离 SDK 直接单测沙箱逻辑；工具层只负责从
`RunContextWrapper[DocsContext]` 取出 docs_root 后转交。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agents import RunContextWrapper, function_tool

# 单文件读取上限：运维 md 通常很短，给个宽松上限兜底，防超大文件把上下文撑爆。
_MAX_READ_CHARS = 60_000
# grep 返回的最多命中行数，防关键词过宽时刷屏。
_MAX_GREP_HITS = 80
# 一次 grep 扫描的最大文件数兜底。
_MAX_GREP_FILES = 500


@dataclass
class DocsContext:
    """run context：携带文档根目录，供工具解析相对路径。

    `Runner.run(..., context=DocsContext(docs_root=...))` 注入；工具签名首参
    `ctx: RunContextWrapper[DocsContext]` 即可拿到（SDK 自动注入、不进 JSON schema）。
    """

    docs_root: Path


def _resolve_within(docs_root: Path, rel: str) -> Path:
    """把相对 docs_root 的路径解析成绝对路径，并强制约束在 docs_root 子树内。

    解析真实路径（resolve）后做前缀校验，挡 `..` 越权和符号链接逃逸。越界抛
    ValueError，由各核心函数捕获转成给 agent 的错误文本（而不是 500）。
    """
    root = docs_root.resolve()
    # 允许 agent 传 "redis/overview.md" 或 "./redis/overview.md"；去掉开头斜杠避免被当绝对路径。
    candidate = (root / rel.lstrip("/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"路径越界（必须在文档根目录内）：{rel}")
    return candidate


def _iter_md_files(base: Path) -> list[Path]:
    """列出 base 下所有 .md 文件（含子目录），排序稳定。"""
    return sorted(p for p in base.rglob("*.md") if p.is_file())


# ---------------------------------------------------------------------------
# 纯函数核心（脱离 SDK 可直接单测）
# ---------------------------------------------------------------------------


def _read_doc(docs_root: Path, path: str) -> str:
    try:
        target = _resolve_within(docs_root, path)
    except ValueError as e:
        return f"[错误] {e}"
    if not target.is_file():
        return f"[未找到] 文件不存在：{path}"
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[错误] 文件不是 UTF-8 文本（可能是二进制 / 图片），无法按文档读取：{path}"
    if len(text) > _MAX_READ_CHARS:
        text = text[:_MAX_READ_CHARS] + f"\n\n[已截断：文件超过 {_MAX_READ_CHARS} 字符]"
    return text


def _glob_docs(docs_root: Path, pattern: str) -> str:
    root = docs_root.resolve()
    # glob 模式里可能带 ../ 试图逃逸，统一去掉开头斜杠后用 Path.glob，再对结果做子树校验。
    matches: list[str] = []
    for p in sorted(root.glob(pattern.lstrip("/"))):
        try:
            rp = p.resolve()
        except OSError:
            continue
        if (rp == root or root in rp.parents) and p.is_file():
            matches.append(str(p.relative_to(root)))
    if not matches:
        return f"[无匹配] 没有文件匹配 pattern：{pattern}"
    return "\n".join(matches)


def _grep_docs(docs_root: Path, pattern: str, path: str | None = None) -> str:
    root = docs_root.resolve()
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"[错误] 正则无效：{e}"

    if path:
        try:
            base = _resolve_within(root, path)
        except ValueError as e:
            return f"[错误] {e}"
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = _iter_md_files(base)
        else:
            return f"[未找到] 路径不存在：{path}"
    else:
        files = _iter_md_files(root)

    files = files[:_MAX_GREP_FILES]
    hits: list[str] = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = f.relative_to(root)
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
                if len(hits) >= _MAX_GREP_HITS:
                    hits.append(f"[已截断：命中超过 {_MAX_GREP_HITS} 行]")
                    return "\n".join(hits)
    if not hits:
        scope = f"（范围：{path}）" if path else ""
        return f"[无命中] 没有内容匹配：{pattern}{scope}"
    return "\n".join(hits)


# ---------------------------------------------------------------------------
# function_tool 包装层（agent 实际挂载的）
# ---------------------------------------------------------------------------


@function_tool
def read_doc(ctx: RunContextWrapper[DocsContext], path: str) -> str:
    """读取单个文档文件的全文内容。

    Args:
        path: 相对文档根目录的路径，例如 "redis/troubleshooting.md" 或 "INDEX.md"。
    """
    return _read_doc(ctx.context.docs_root, path)


@function_tool
def glob_docs(ctx: RunContextWrapper[DocsContext], pattern: str) -> str:
    """按 glob pattern 列出文档根目录下匹配的文件路径（相对路径，每行一个）。

    Args:
        pattern: glob 模式，相对文档根目录。例如 "redis/*.md"、"**/*.md"、"*/overview.md"。
    """
    return _glob_docs(ctx.context.docs_root, pattern)


@function_tool
def grep_docs(ctx: RunContextWrapper[DocsContext], pattern: str, path: str | None = None) -> str:
    """在文档里跨文件正则搜索关键词，返回命中的 文件:行号: 内容。

    Args:
        pattern: 正则表达式（大小写不敏感）。例如 "慢查询"、"maxmemory|内存"。
        path: 可选，限定搜索范围的子目录或文件（相对文档根目录）。省略则搜全部 .md。
    """
    return _grep_docs(ctx.context.docs_root, pattern, path)


# agent 默认挂载的文档检索工具集。对标 Claude SDK 的 Read/Glob/Grep。
DOC_TOOLS = [read_doc, glob_docs, grep_docs]
