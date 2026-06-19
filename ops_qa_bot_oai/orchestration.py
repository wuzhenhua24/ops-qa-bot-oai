"""多 agent 编排（差异化 #3）：分诊台 Triage → 组件专家（handoffs）。

ops-qa-bot（Claude SDK 版）用一个巨型 system prompt 同时承担「路由 + 各组件答题」，
组件一多 prompt 越堆越长、上下文也越吃越多。OpenAI Agents SDK 的 handoffs 让我们把
它拆开：

- 一个轻量 **Triage agent** 只做路由——按问题关键词 handoff 给对应组件专家；问候/
  能力介绍/范围外的它自己短答。
- 每个 **组件专家 agent** 只挂自己组件的文档作用域 + 更窄更准的 prompt，互不干扰。

组件**从 `INDEX.md` 动态解析**（保持"加组件=改 INDEX、不改代码"的理念），每个
`local` 来源的组件生成一个专家。专家 agent 是可组合的 Python 对象，instructions 可
按组件定制——这种掌控度是 Claude SDK 的 CLI 托管 subagent 给不了的。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import Agent, Model
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from .tools import DOC_TOOLS, DocsContext


@dataclass
class Component:
    """INDEX.md 里登记的一个组件。"""

    name: str  # 组件名，如 "Redis"
    dir: str  # 目录名（不带斜杠），如 "redis"
    source: str  # "local" / "feishu"
    coverage: str  # 覆盖内容描述
    open_id: str  # 负责人 open_id（升级用）


def _split_row(line: str) -> list[str]:
    """拆一行 markdown 表格为去空白的单元格列表。"""
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _clean_dir(raw: str) -> str:
    """把 "`redis/`" 这种目录单元格归一成 "redis"。"""
    return raw.strip().strip("`").strip().rstrip("/").lstrip("/")


def parse_index_components(docs_root: Path) -> list[Component]:
    """解析 `INDEX.md` 的组件表，返回组件列表。

    按表头名定位列（容忍列顺序/有无变化）：组件 / 来源 / 目录 / 覆盖内容 / open_id。
    缺「来源」列时一律按 local 处理；缺 open_id 列时留空。只有能解析出目录名的行才算数。
    """
    index_path = docs_root / "INDEX.md"
    if not index_path.is_file():
        return []
    lines = index_path.read_text(encoding="utf-8").splitlines()
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
    i_cov = col("覆盖")
    i_oid = col("open_id", "openid", "open id")

    components: list[Component] = []
    # 跳过表头行 + 分隔行（形如 |---|---|）。
    for ln in rows[1:]:
        if set(ln.strip()) <= set("|-: "):
            continue
        cells = _split_row(ln)

        def get(idx: int) -> str:
            return cells[idx] if 0 <= idx < len(cells) else ""

        name = get(i_name)
        dir_ = _clean_dir(get(i_dir))
        if not name or not dir_:
            continue
        source = (get(i_src) or "local").lower()
        components.append(
            Component(
                name=name,
                dir=dir_,
                source=source,
                coverage=get(i_cov),
                open_id=get(i_oid),
            )
        )
    return components


def _specialist_instructions(c: Component) -> str:
    owner = c.open_id or "（INDEX.md 未登记 open_id）"
    return f"""你是 **{c.name}** 运维问答专家，只负责 {c.name} 这一个组件。它的文档全部在 `{c.dir}/` 目录下。

# 工作流程
1. 用 `glob_docs("{c.dir}/*.md")` 列出本组件文档，`read_doc` 读相关篇目；拿不准用 `grep_docs(pattern, path="{c.dir}")` 搜。
2. **只看 `{c.dir}/` 下的文档**，不要去读别的组件目录。
3. 基于读到的内容回答。

# 回答规范
- **引用来源**：每个事实结论后附 `（来源：{c.dir}/<文件>.md）`。
- **找不到就说找不到**：文档没有就明说"文档中未找到相关内容"，**不要编**，并建议联系负责人（open_id: {owner}）。
- **危险操作**（删除/重启/flush/改主库等）显式标 ⚠️ 风险，并引用文档里的对应警告。
- 中文、简洁、分点。
- 信息不足以准确回答（缺版本/环境/报错码且会让答案分叉）时，先反问 1-2 个关键点，不要硬答。
"""


def build_specialist_agent(c: Component, model: str | Model) -> Agent[DocsContext]:
    """为一个组件构造专家 agent：作用域限定在它自己的目录。"""
    return Agent[DocsContext](
        name=f"{c.dir}_specialist",
        handoff_description=f"{c.name} 运维问题（{c.coverage}）",
        instructions=_specialist_instructions(c),
        tools=list(DOC_TOOLS),
        model=model,
    )


def _triage_instructions(components: list[Component]) -> str:
    lines = [f"- **{c.name}**（转交目标：{c.dir}_specialist）：{c.coverage}" for c in components]
    roster = "\n".join(lines) if lines else "（INDEX.md 未解析到组件）"
    body = f"""你是内部运维问答的**分诊台**。你自己不查组件文档、不回答组件细节问题——你的职责是把问题**转交（handoff）给正确的组件专家**。

# 可转交的组件专家
{roster}

# 路由规则
- **运维组件问题** → 立即 handoff 给最匹配的那个专家（按问题里的组件名 / 关键词判断），**不要自己用工具查文档作答**。
- **问候 / 致谢 / 闲聊性收尾**（"你好"、"在吗"、"谢谢"）→ 自己简短回应，1-2 句，引导用户提具体运维问题。
- **询问能力 / "你能做什么"** → 自己回答：列出上面覆盖的组件 + 给 1-2 个示例问题（可 `read_doc("INDEX.md")` 核对）。
- **与运维无关的请求**（写代码 / 翻译 / 通用知识）→ 自己友好拒绝，说明只覆盖运维组件问答。
- **跨多个组件 / 归属判断不明** → 先用一句话反问澄清主问哪个组件；用户讲清后再 handoff。宁可漏问别滥问：能判断出主组件就直接转。

转交后由专家基于该组件文档作答，你不需要再介入。"""
    return prompt_with_handoff_instructions(body)


def build_triage_agent(
    docs_root: Path, model: str | Model
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造分诊 agent（持有各 local 组件专家的 handoff）+ 返回解析到的组件。

    只为 `local` 来源的组件建专家（feishu 来源无本地文档，本核心版不支持）。
    """
    components = parse_index_components(docs_root)
    local = [c for c in components if c.source == "local"]
    specialists = [build_specialist_agent(c, model) for c in local]
    triage = Agent[DocsContext](
        name="triage",
        instructions=_triage_instructions(local),
        tools=list(DOC_TOOLS),  # 仅用于"能力介绍"时读 INDEX；组件问题一律转交
        handoffs=list(specialists),
        model=model,
    )
    return triage, local
