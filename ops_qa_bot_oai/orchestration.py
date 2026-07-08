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

from .diagnostics import DIAG_TOOL_NAME
from .model import ModelRouter, role_model_settings
from .prompt import STRUCTURED_CONTRACT_SUFFIX, diagnostics_prompt_section
from .tools import DOC_TOOLS, DocsContext

# 写审批工具名（actions.make_write_command_tool 产出）。orchestration 按名字判断专家挂了
# 哪些横切工具，从而决定 prompt 加哪段——避免把布尔标志一路穿过 4 个 build 函数。
_WRITE_TOOL_NAME = "request_write_command"


def _tool_names(tools: list | None) -> set[str]:
    return {getattr(t, "name", "") for t in (tools or [])}


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


def _specialist_instructions(
    c: Component,
    *,
    has_write_tool: bool = False,
    has_diag_tool: bool = False,
    structured: bool = False,
) -> str:
    owner = c.open_id or "（INDEX.md 未登记 open_id）"
    # 带写审批工具时，明确引导专家把"写/变更"走 request_write_command（挂起等人批），而不是
    # 把命令直接写进答案——否则"危险操作标 ⚠️ 风险"那句会把它带向纯文字，工具形同虚设。
    write_block = (
        "\n- **需要执行写/变更命令时**（重启 / 改配置 / flush / 删数据 / 改库参数等），"
        "不要把命令直接写进答案让用户去跑，而是调用 `request_write_command` 工具**提议**——"
        "它会挂起等人工审批，批准后才登记执行；只读诊断 / 纯知识问答不要用它。"
        if has_write_tool
        else ""
    )
    # 带实时诊断工具时，追加「实时诊断」章节（与 single 模式同款，has_write_tool 决定写命令去处）。
    diag_section = (
        diagnostics_prompt_section(has_write_tool=has_write_tool) if has_diag_tool else ""
    )
    # 结构化时来源走契约的 citations 字段，正文别写行内来源——否则跟契约要求打架、把模型带偏。
    cite_line = (
        "- **来源**：填到契约的 `citations` 字段（`{dir}/<文件>.md` 相对路径），正文里"
        "**不要**写 `（来源：...）`。".format(dir=c.dir)
        if structured
        else f"- **引用来源**：每个事实结论后附 `（来源：{c.dir}/<文件>.md）`。"
    )
    return f"""你是 **{c.name}** 运维问答专家，只负责 {c.name} 这一个组件。它的文档全部在 `{c.dir}/` 目录下。

# 工作流程
1. 用 `glob_docs("{c.dir}/*.md")` 列出本组件文档，`read_doc` 读相关篇目；拿不准用 `grep_docs(pattern, path="{c.dir}")` 搜。
2. **只看 `{c.dir}/` 下的文档**，不要去读别的组件目录。
3. 基于读到的内容回答。

# 回答规范
{cite_line}
- **找不到就说找不到**：文档没有就明说"文档中未找到相关内容"，**不要编**，并建议联系负责人（open_id: {owner}）。
- **危险操作**（删除/重启/flush/改主库等）显式标 ⚠️ 风险，并引用文档里的对应警告。{write_block}
- 中文、简洁、分点。
- 信息不足以准确回答（缺版本/环境/报错码且会让答案分叉）时，先反问 1-2 个关键点，不要硬答。
{diag_section}{STRUCTURED_CONTRACT_SUFFIX if structured else ""}"""


def build_specialist_agent(
    c: Component,
    model: str | Model,
    *,
    extra_tools: list | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
) -> Agent[DocsContext]:
    """为一个组件构造专家 agent：作用域限定在它自己的目录。

    `extra_tools` 用于把横切工具（如写操作审批 `request_write_command`）挂到专家上——
    多/自适应/协调者模式下真正答题、可能提议写操作的是专家，护栏得挂在这一层。

    `output_type` 非空时专家作为**终端 agent** 产出结构化契约（multi/auto 下 handoff 后
    由专家收尾）——此时叠加契约字段引导、并挂 `output_guardrails`。coordinator 模式下
    专家是被 as_tool 调用、返回文字喂协调者，不传 output_type。
    """
    structured = output_type is not None
    names = _tool_names(extra_tools)
    return Agent[DocsContext](
        name=f"{c.dir}_specialist",
        handoff_description=f"{c.name} 运维问题（{c.coverage}）",
        instructions=_specialist_instructions(
            c,
            has_write_tool=_WRITE_TOOL_NAME in names,
            has_diag_tool=DIAG_TOOL_NAME in names,
            structured=structured,
        ),
        tools=list(DOC_TOOLS) + list(extra_tools or []),
        model=model,
        model_settings=role_model_settings(c.dir),
        output_type=output_type,
        output_guardrails=list(output_guardrails or []),
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
    docs_root: Path,
    router: ModelRouter,
    *,
    input_guardrails: list | None = None,
    specialist_extra_tools: list | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造分诊 agent（持有各 local 组件专家的 handoff）+ 返回解析到的组件。

    多模型路由（#2）：分诊用 `router.for_role("triage")`，每个专家用
    `router.for_role(component.dir)`——无覆盖时都回退到默认模型（等价单模型）。
    只为 `local` 来源的组件建专家（feishu 来源无本地文档，本核心版不支持）。

    护栏（横切，与编排模式正交）：`input_guardrails` 挂在入口分诊 agent 上（输入护栏
    只在入口 agent 对用户输入跑一次）；`specialist_extra_tools`（如写审批工具）挂到各专家。

    结构化输出（与路由正交）：`output_type` 非空时，各专家（handoff 后终端）与分诊自身
    （自答场景）都产出该契约类型，`output_guardrails`（来源护栏）挂到这些终端 agent。
    """
    structured = output_type is not None
    components = parse_index_components(docs_root)
    local = [c for c in components if c.source == "local"]
    specialists = [
        build_specialist_agent(
            c,
            router.for_role(c.dir)[1],
            extra_tools=specialist_extra_tools,
            output_type=output_type,
            output_guardrails=output_guardrails,
        )
        for c in local
    ]
    instructions = _triage_instructions(local)
    if structured:
        instructions += STRUCTURED_CONTRACT_SUFFIX
    triage = Agent[DocsContext](
        name="triage",
        instructions=instructions,
        tools=list(DOC_TOOLS),  # 仅用于"能力介绍"时读 INDEX；组件问题一律转交
        handoffs=list(specialists),
        model=router.for_role("triage")[1],
        model_settings=role_model_settings("triage"),  # 低温：路由决策要稳定
        input_guardrails=list(input_guardrails or []),
        output_type=output_type,
        output_guardrails=list(output_guardrails or []),
    )
    return triage, local


# ---------------------------------------------------------------------------
# 跨组件协作（coordinator + agents-as-tools）
# ---------------------------------------------------------------------------
#
# 与 triage 的根本区别：triage 用 handoff（转交后控制权不回来），适合"路由到唯一
# 专家"；跨组件诊断需要"问多个专家再综合"，所以用 agents-as-tools——协调者把每个
# 专家当**工具**调用，自己保留控制权，收齐各组件证据后综合出根因链。这是单一巨型
# prompt（原项目那种）很难做干净的场景：每个专家独立作用域、独立上下文，协调者只
# 负责拆解与综合。


def _coordinator_instructions(components: list[Component]) -> str:
    lines = [f"- **{c.name}**（工具 `ask_{c.dir}`）：{c.coverage}" for c in components]
    roster = "\n".join(lines) if lines else "（INDEX.md 未解析到组件）"
    return f"""你是内部运维问答的**跨组件协调者**，专门处理"一个现象可能牵涉多个组件"的复杂排查。你不直接查文档，而是把问题拆给对应的**组件专家工具**，收齐证据后综合出根因。

# 可调用的组件专家工具
{roster}

# 工作流程
1. **拆解现象**：判断这个现象**可能涉及哪些组件**。运维问题常是跨层的——例如"接口偶发失败"可能同时牵涉网关（看到上游实例偶发不健康）和容器平台（实例 OOM 重启）。
2. **并行求证**：对每个相关组件，调用其 `ask_<组件>` 工具，传一个**自包含的子问题**（写清现象 + 要它从本组件角度查什么）。可以调用多个工具。每个工具会返回该组件文档依据下的发现。
3. **综合根因**：把各组件的发现**串成因果链**，指出最可能的根因和证据链路。例如「容器层 A 实例周期性 OOM 重启 → 重启期间网关健康检查判其 unhealthy 并摘流 → 命中该实例的请求偶发 5xx」。
4. **给处置建议**：基于综合结论给排查/处置建议；涉及变更（改配置/扩资源/重启）只给文字建议，标 ⚠️ 风险，不代为执行。

# 回答规范
- **标清每条证据来自哪个组件**：如「（网关）上游实例偶发 unhealthy」「（容器）该实例 OOMKilled」。专家返回里带的 `（来源：xxx.md）` 一并保留。
- **证据不足时**：某组件查不到相关内容就如实说明，不要替它编；只综合**真实拿到的**证据。若只够支撑单一组件，就直说"目前证据只指向 X 组件"。
- **不要无差别调用所有专家**：只调可能相关的。纯单组件问题（"redis 内存告警怎么处理"）也可以只调一个专家，但这种场景通常更适合直接问对应组件。
- 中文、结构清晰：先给**结论/根因链**，再列**各组件证据**，最后给**处置建议**。
- 与运维无关的请求友好拒绝，不调用任何专家工具。"""


def build_coordinator_agent(
    docs_root: Path,
    router: ModelRouter,
    *,
    specialist_max_turns: int = 12,
    handoff_description: str | None = None,
    input_guardrails: list | None = None,
    specialist_extra_tools: list | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
    agent_tool_hooks: object | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造跨组件协调者：各 local 组件专家以 `as_tool` 暴露给协调者调用。

    每个专家工具名 `ask_<dir>`；专家用 `router.for_role(dir)` 的模型，协调者用
    `router.for_role("coordinator")`（无覆盖回退默认）。`as_tool` 会把父级的
    DocsContext 透传给专家，专家的 read_doc/grep_docs 据此拿到 docs_root。

    `handoff_description` 仅在 auto 模式下把协调者当作分诊台的 handoff 目标时用到——
    分诊台据此判断"何时转交给它"。作为独立入口（coordinator 模式）时留空即可。

    护栏：`input_guardrails` 挂在入口协调者上（coordinator 模式下它是入口）；
    `specialist_extra_tools`（写审批工具）挂到 as_tool 的专家——SDK 会把嵌套子 run 的
    needs_approval 中断冒到顶层，由 answer_guarded 统一 approve/reject。

    结构化输出：`output_type` 非空时**只有协调者**（终端、负责综合）产出契约并挂
    `output_guardrails`；as_tool 专家仍返回文字喂给协调者，不套 output_type。

    遥测：`agent_tool_hooks`（RunHooks 实例）注入 as_tool 的嵌套子 run——SDK 的
    `as_tool(hooks=...)` 是构建期参数，run 级 hooks 不会自动透传到子 run；构图时
    注入同一实例，专家子 run 的 LLM 调用才能按 agent 归账（见 hooks.py）。
    """
    structured = output_type is not None
    components = parse_index_components(docs_root)
    local = [c for c in components if c.source == "local"]
    specialist_tools = [
        build_specialist_agent(
            c, router.for_role(c.dir)[1], extra_tools=specialist_extra_tools
        ).as_tool(
            tool_name=f"ask_{c.dir}",
            tool_description=(
                f"就某现象咨询 {c.name} 组件专家（覆盖：{c.coverage}）。"
                f"传一个自包含的子问题；返回该组件文档依据下的发现。"
            ),
            max_turns=specialist_max_turns,
            hooks=agent_tool_hooks,  # type: ignore[arg-type]
        )
        for c in local
    ]
    coord_instructions = _coordinator_instructions(local)
    if structured:
        coord_instructions += STRUCTURED_CONTRACT_SUFFIX
    coordinator = Agent[DocsContext](
        name="coordinator",
        handoff_description=handoff_description,
        instructions=coord_instructions,
        tools=specialist_tools,
        model=router.for_role("coordinator")[1],
        # parallel_tool_calls=True：prompt 里的"并行求证"落到调用参数上——一轮里同时
        # 发起多个 ask_<组件>，跨组件排查不用串行等每个专家。
        model_settings=role_model_settings("coordinator"),
        input_guardrails=list(input_guardrails or []),
        output_type=output_type,
        output_guardrails=list(output_guardrails or []),
    )
    return coordinator, local


# ---------------------------------------------------------------------------
# 自适应分诊（auto）：分诊台 = 单专家 handoff + 跨组件协调者 handoff
# ---------------------------------------------------------------------------
#
# 面向真实使用（非评测）：让分诊台按问题自适应选架构，而不是让用户去配"用 multi 还是
# coordinator"。大多数问题落在单个组件 → handoff 给该专家（便宜、聚焦）；少数横跨多组件
# 的现象 → handoff 给 coordinator 综合根因。auto 相当于 multi 多挂一个"跨组件逃生口"。

_AUTO_COORDINATOR_HANDOFF_DESC = (
    "跨组件综合排查：当一个现象可能牵涉**多个**组件、根因跨越组件边界、"
    "或单个专家不足以定位时，转交给它——它会并行咨询多个专家并综合根因链。"
)


def _auto_triage_instructions(components: list[Component]) -> str:
    lines = [f"- **{c.name}**（转交目标：{c.dir}_specialist）：{c.coverage}" for c in components]
    roster = "\n".join(lines) if lines else "（INDEX.md 未解析到组件）"
    body = f"""你是内部运维问答的**分诊台**（自适应路由）。你自己不查组件文档、不回答组件细节——职责是把问题**转交（handoff）给正确的处理者**。

# 可转交对象
## 单组件专家（问题明确落在某一个组件时转给它）
{roster}
## 跨组件协调者（转交目标：coordinator）
当现象可能牵涉**多个**组件、根因跨越组件边界、或单个专家不足以定位时，转交给它——它会并行咨询多个专家、综合根因链。

# 路由规则
- 问题明确落在**某一个**组件 → handoff 给该组件专家（**大多数问题属此类，优先走这条**）。
- 现象**横跨多个**组件 / 根因归属不清 / 需综合多个组件证据 → handoff 给**跨组件协调者**。
- **问候 / 致谢 / 闲聊**（"你好"、"谢谢"）→ 自己简短回应，1-2 句，引导用户提具体运维问题。
- **询问能力 / "你能做什么"** → 自己回答：列出上面覆盖的组件 + 1-2 个示例问题（可 `read_doc("INDEX.md")` 核对）。
- **与运维无关的请求**（写代码 / 翻译 / 通用知识）→ 自己友好拒绝，说明只覆盖运维组件问答。
- 拿不准是单组件还是跨组件时，**倾向先转给单组件专家**（更聚焦）；确有跨组件迹象再转协调者。

转交后由处理者作答，你不需要再介入。"""
    return prompt_with_handoff_instructions(body)


def build_auto_agent(
    docs_root: Path,
    router: ModelRouter,
    *,
    specialist_max_turns: int = 12,
    input_guardrails: list | None = None,
    specialist_extra_tools: list | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
    agent_tool_hooks: object | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造自适应分诊 agent：handoffs = 各组件专家 + 跨组件协调者。

    分诊按问题决定转交给单个专家（常见）还是 coordinator（跨组件）。复用
    `build_specialist_agent` / `build_coordinator_agent`，不重写编排。

    护栏：`input_guardrails` 只挂在入口分诊台（输入护栏在入口跑一次即可）；写审批工具
    （`specialist_extra_tools`）挂到两条路径的专家——直接 handoff 的专家、以及 coordinator
    逃生口内 as_tool 的专家。

    结构化输出：`output_type` 非空时，三类终端都产出契约——直接 handoff 的专家、coordinator
    逃生口（其内部 as_tool 专家仍回文字）、以及分诊自答。
    """
    structured = output_type is not None
    components = parse_index_components(docs_root)
    local = [c for c in components if c.source == "local"]
    specialists = [
        build_specialist_agent(
            c,
            router.for_role(c.dir)[1],
            extra_tools=specialist_extra_tools,
            output_type=output_type,
            output_guardrails=output_guardrails,
        )
        for c in local
    ]
    coordinator, _ = build_coordinator_agent(
        docs_root,
        router,
        specialist_max_turns=specialist_max_turns,
        handoff_description=_AUTO_COORDINATOR_HANDOFF_DESC,
        specialist_extra_tools=specialist_extra_tools,
        output_type=output_type,
        output_guardrails=output_guardrails,
        agent_tool_hooks=agent_tool_hooks,
    )
    instructions = _auto_triage_instructions(local)
    if structured:
        instructions += STRUCTURED_CONTRACT_SUFFIX
    triage = Agent[DocsContext](
        name="triage",
        instructions=instructions,
        tools=list(DOC_TOOLS),  # 仅用于"能力介绍"时读 INDEX；组件问题一律转交
        handoffs=[*specialists, coordinator],
        model=router.for_role("triage")[1],
        model_settings=role_model_settings("triage"),  # 低温：路由决策要稳定
        input_guardrails=list(input_guardrails or []),
        output_type=output_type,
        output_guardrails=list(output_guardrails or []),
    )
    return triage, local
