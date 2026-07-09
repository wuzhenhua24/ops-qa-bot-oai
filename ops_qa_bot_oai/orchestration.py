"""多 agent 编排（差异化 #3）：分诊台 Triage → 组件专家（handoffs）。

ops-qa-bot（Claude SDK 版）用一个巨型 system prompt 同时承担「路由 + 各组件答题」，
组件一多 prompt 越堆越长、上下文也越吃越多。OpenAI Agents SDK 的 handoffs 让我们把
它拆开：

- 一个轻量 **Triage agent** 只做路由——按问题关键词 handoff 给对应组件专家；问候/
  能力介绍/范围外的它自己短答。
- 每个 **组件专家 agent** 只挂自己组件的文档作用域 + 更窄更准的 prompt，互不干扰。

组件**从 `INDEX.md` 动态解析**（保持"加组件=改 INDEX、不改代码"的理念），每个组件生成
一个专家。专家 agent 是可组合的 Python 对象，instructions 可按组件定制——这种掌控度是
Claude SDK 的 CLI 托管 subagent 给不了的。

**来源异构**（差异化的一个副产品）：`local` 组件的专家挂文档检索工具（read_doc/glob_docs/
grep_docs），`feishu` 组件的专家**只挂 `query_feishu_doc`**。参考项目 ops-qa-bot 是单一巨型
prompt，只能靠自律条款"来源=feishu 时不要用 Glob/Read/Grep"来区分；这里 feishu 专家物理上
就没有文档检索工具，规则从 prompt 自律变成了机制保证（与 diagnostics 用白名单取代黑名单
是同一个姿态）。跨来源问题（Redis 本地 + Nginx 飞书）由 coordinator 分别咨询再合并。

**输出契约要各发一份**：拆成独立 agent 的代价是每个 agent 只看得见自己的 instructions。
single 模式那份 `SYSTEM_PROMPT_TEMPLATE` 里的 `<<CLARIFY>>` / `<<ESCALATE>>` / `<<FOLLOWUPS>>`
契约，专家和协调者是读不到的——所以这里必须把契约再发给它们（`_tail` / `free_text_markers_section`）。
漏掉这一步的后果很隐蔽：答案看着正常，只是标记从不出现，飞书的 @负责人 永远不触发。
"""

from __future__ import annotations

from pathlib import Path

from agents import Agent, Model
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from .diagnostics import DIAG_TOOL_NAME
from .index import Component, feishu_citation, parse_index_components
from .model import ModelRouter, role_model_settings
from .prompt import (
    STRUCTURED_CONTRACT_SUFFIX,
    TRIAGE_CLARIFY_NOTE,
    diagnostics_prompt_section,
    escalate_marker,
    free_text_markers_section,
)
from .tools import DOC_TOOLS, DocsContext

__all__ = [
    "Component",
    "parse_index_components",
    "build_specialist_agent",
    "build_triage_agent",
    "build_coordinator_agent",
    "build_auto_agent",
]

# 写审批工具名（actions.make_write_command_tool 产出）。orchestration 按名字判断专家挂了
# 哪些横切工具，从而决定 prompt 加哪段——避免把布尔标志一路穿过 4 个 build 函数。
_WRITE_TOOL_NAME = "request_write_command"


def _tool_names(tools: list | None) -> set[str]:
    return {getattr(t, "name", "") for t in (tools or [])}


def routable_components(components: list[Component], feishu_tool: object | None) -> list[Component]:
    """筛出能建专家的组件：非 feishu 一律能建；feishu 只有在挂了 `query_feishu_doc` 时才能建。

    没配 doc_qa 服务（`feishu_tool is None`）时 feishu 组件被跳过——分诊台名单里不会出现它，
    问到它会走"不在覆盖范围"的友好拒绝。这比建一个查不到东西的专家诚实。

    「来源」列写错（既不是 local 也不是 feishu）的行按 local 处理，与 index.py 里"缺列即
    local"的容错一致：至少能建出专家去 glob 它的目录，而不是从名单里静悄悄消失。
    """
    return [c for c in components if not c.is_feishu or feishu_tool is not None]


def _write_block(has_write_tool: bool) -> str:
    """带写审批工具时，明确引导专家把"写/变更"走 request_write_command（挂起等人批），而不是
    把命令直接写进答案——否则"危险操作标 ⚠️ 风险"那句会把它带向纯文字，工具形同虚设。"""
    if not has_write_tool:
        return ""
    return (
        "\n- **需要执行写/变更命令时**（重启 / 改配置 / flush / 删数据 / 改库参数等），"
        "不要把命令直接写进答案让用户去跑，而是调用 `request_write_command` 工具**提议**——"
        "它会挂起等人工审批，批准后才登记执行；只读诊断 / 纯知识问答不要用它。"
    )


def _tail(
    *, has_write_tool: bool, has_diag_tool: bool, structured: bool, escalate_rule: str
) -> str:
    """专家 instructions 的公共结尾：实时诊断章节（若挂了工具）+ 输出契约。

    输出契约二选一，互斥：结构化模式用 `AnswerContract` 的字段（decision/escalate_to/
    followups）；自由文本模式用 `<<CLARIFY>>` / `<<ESCALATE>>` / `<<FOLLOWUPS>>` 标记。
    专家是独立 agent，拿不到 single 模式那份 SYSTEM_PROMPT_TEMPLATE，标记契约必须在这里给
    ——否则它们永远不发标记，飞书的 @负责人 就永远不触发。
    """
    diag = diagnostics_prompt_section(has_write_tool=has_write_tool) if has_diag_tool else ""
    contract = (
        STRUCTURED_CONTRACT_SUFFIX
        if structured
        else free_text_markers_section(escalate_rule=escalate_rule)
    )
    return f"{diag}{contract}"


def _specialist_escalate_rule(c: Component, *, source_phrase: str) -> str:
    """某组件专家的升级规则：标记字面量在构建期算好，模型只需照抄（见 prompt.escalate_marker）。"""
    marker = escalate_marker(c.open_id, c.dir)
    unnamed = "" if c.open_id.startswith("ou_") else "（该组件在 INDEX.md 未登记 open_id，故不 @ 人）"
    return f"""- {source_phrase}确实没有相关内容时：先回复「文档中未找到相关内容」+ 一两句说明你查了什么，然后在末尾独立一行原样输出 `{marker}`{unnamed}。
- 这条标记**本组件固定就是它**，照抄即可：不要改动、不要换成别的 open_id 或目录、不要自己去 INDEX.md 里另找。"""


def _not_found_line(c: Component, *, structured: bool) -> str:
    """「找不到怎么办」那条规范。两种模式下**升级的出口不同**，措辞必须跟着变。

    自由文本模式绝不能让它在正文里写 `（open_id: ou_x）`——那与「答案塑形标记」里"不要在正文
    里直接写 @ou_xxx"直接打架，而且系统只认 `<<ESCALATE:...>>` 标记来 @ 人，正文里写了也 @
    不到。结构化模式则把 open_id / 目录直接填进契约字段，同样在构建期算好、不让模型去查表。
    """
    head = '- **找不到就说找不到**：确实没有就明说"文档中未找到相关内容"，**不要编**'
    if not structured:
        return f"{head}，并按下面「答案塑形标记」里的升级规则通知负责人（正文里不要写 @ 或 open_id）。"
    to = f'`escalate_to="{c.open_id}"`' if c.open_id.startswith("ou_") else "`escalate_to=\"\"`"
    return f'{head}，改用 `decision=escalate`，并填 {to} 与 `escalate_dir="{c.dir}"`（这两个值固定，照抄）。'


def _specialist_instructions(
    c: Component,
    *,
    has_write_tool: bool = False,
    has_diag_tool: bool = False,
    structured: bool = False,
) -> str:
    """本地 markdown 来源的组件专家：作用域限定在自己的目录，挂文档检索工具。"""
    write_block = _write_block(has_write_tool)
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
{_not_found_line(c, structured=structured)}
- **危险操作**（删除/重启/flush/改主库等）显式标 ⚠️ 风险，并引用文档里的对应警告。{write_block}
- 中文、简洁、分点。
- 信息不足以准确回答（缺版本/环境/报错码且会让答案分叉）时，先反问 1-2 个关键点，不要硬答。
{_tail(
        has_write_tool=has_write_tool,
        has_diag_tool=has_diag_tool,
        structured=structured,
        escalate_rule=_specialist_escalate_rule(c, source_phrase=f"`{c.dir}/` 下的文档里"),
    )}"""


def _feishu_specialist_instructions(
    c: Component,
    *,
    has_write_tool: bool = False,
    has_diag_tool: bool = False,
    structured: bool = False,
) -> str:
    """飞书文档来源的组件专家：唯一知识入口是 `query_feishu_doc`，本地没有它的 md 文件。

    与本地专家的三处实质差异，都源于「上游是个无状态问答服务」而非「一堆可反复检索的文件」：
    - **question 必须自包含**：上游没有对话记忆，追问轮要把上下文折进这一条 question。
    - **来源标识不是路径**：写 `飞书文档·<组件>`（见 index.CITATION_PREFIX），会被 citations
      校验按 INDEX.md 的登记核对。
    - **一次问不到就升级**：不像本地文档可以换关键词再 grep 一轮，这里换着问法反复调上游
      既慢又贵，且上游已经跑过一轮 agent 了。
    """
    citation = feishu_citation(c.name)
    write_block = _write_block(has_write_tool)
    cite_line = (
        f"- **来源**：填到契约的 `citations` 字段，固定写 `{citation}`（**不要**写本地路径——"
        "你没有本地文档）；正文里**不要**写 `（来源：...）`。"
        if structured
        else f"- **引用来源**：每个事实结论后附 `（来源：{citation}）`；不要编造飞书文档的链接或路径。"
    )
    return f"""你是 **{c.name}** 运维问答专家，只负责 {c.name} 这一个组件。它的运维知识维护在**飞书文档**里，本地**没有**它的 md 文件。

# 工作流程
1. 调 `query_feishu_doc(component="{c.name}", question=...)` 取答案。
2. `question` 必须是**自包含的完整问题**：这个服务**没有对话记忆**。用户追问（"那它怎么回滚"）或补齐反问信息后，要把前几轮的关键上下文（版本、报错、已确认的环境）**折进这一条 question**，不能只发依赖上文的半句。
3. 你**没有** `read_doc` / `glob_docs` / `grep_docs`——不要试图去读本地文件，`{c.dir}/` 目录不存在。
4. 基于工具返回的 markdown 组织回答，规范与本地文档一致。

# 回答规范
{cite_line}
{_not_found_line(c, structured=structured)} 工具返回"未能取得…"/"未登记…"、或返回内容明显答非所问，都算取不到。**不要**换个问法反复重试——上游每次调用内部都会跑一轮 agent，换问法重试既慢又贵。
- **危险操作**（删除/重启/reload/改配置/限流调整等）显式标 ⚠️ 风险，并引用文档里的对应警告。{write_block}
- 中文、简洁、分点。
- 信息不足以准确回答（缺版本/环境/报错码且会让答案分叉）时，先反问 1-2 个关键点，不要硬答——**反问不要调工具**，先把问题问清楚再去查。
{_tail(
        has_write_tool=has_write_tool,
        has_diag_tool=has_diag_tool,
        structured=structured,
        escalate_rule=_specialist_escalate_rule(c, source_phrase="该组件的飞书文档里"),
    )}"""


def build_specialist_agent(
    c: Component,
    model: str | Model,
    *,
    extra_tools: list | None = None,
    feishu_tool: object | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
) -> Agent[DocsContext]:
    """为一个组件构造专家 agent：作用域限定在它自己的知识来源。

    **来源决定工具集**：`local` 组件挂 `DOC_TOOLS`（read/glob/grep，作用域是它自己的目录）；
    `feishu` 组件只挂 `feishu_tool`（`query_feishu_doc`）——它本地没有 md 文件，给它文档检索
    工具只会白费 round。feishu 组件必须传 `feishu_tool`，否则是调用方的编排 bug（应先用
    `routable_components` 过滤掉）。

    `extra_tools` 用于把横切工具（写操作审批 `request_write_command`、实时诊断
    `run_diagnostic`）挂到专家上——多/自适应/协调者模式下真正答题、可能提议写操作的是专家，
    护栏得挂在这一层。这些工具与来源正交：飞书来源的组件照样可以诊断它的机器、提议变更。

    `output_type` 非空时专家作为**终端 agent** 产出结构化契约（multi/auto 下 handoff 后
    由专家收尾）——此时叠加契约字段引导、并挂 `output_guardrails`。coordinator 模式下
    专家是被 as_tool 调用、返回文字喂协调者，不传 output_type。
    """
    if c.is_feishu and feishu_tool is None:
        raise ValueError(
            f"组件 {c.name} 的来源是 feishu，但未提供 query_feishu_doc 工具"
            "（需配置 OPS_QA_DOC_QA_BASE_URL，或先用 routable_components 过滤）"
        )
    structured = output_type is not None
    extra = list(extra_tools or [])
    base_tools: list = [feishu_tool] if c.is_feishu else list(DOC_TOOLS)
    names = _tool_names(extra)
    instructions_fn = _feishu_specialist_instructions if c.is_feishu else _specialist_instructions
    return Agent[DocsContext](
        name=f"{c.dir}_specialist",
        handoff_description=f"{c.name} 运维问题（{c.coverage}）",
        instructions=instructions_fn(
            c,
            has_write_tool=_WRITE_TOOL_NAME in names,
            has_diag_tool=DIAG_TOOL_NAME in names,
            structured=structured,
        ),
        tools=base_tools + extra,
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
    feishu_tool: object | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造分诊 agent（持有各组件专家的 handoff）+ 返回可路由到的组件。

    多模型路由（#2）：分诊用 `router.for_role("triage")`，每个专家用
    `router.for_role(component.dir)`——无覆盖时都回退到默认模型（等价单模型）。

    `feishu_tool`（`query_feishu_doc`）非空时，`feishu` 来源的组件也会建出专家（只挂这一个
    工具）；为空时它们被跳过，行为回到"只有本地文档组件"。

    护栏（横切，与编排模式正交）：`input_guardrails` 挂在入口分诊 agent 上（输入护栏
    只在入口 agent 对用户输入跑一次）；`specialist_extra_tools`（如写审批工具）挂到各专家。

    结构化输出（与路由正交）：`output_type` 非空时，各专家（handoff 后终端）与分诊自身
    （自答场景）都产出该契约类型，`output_guardrails`（来源护栏）挂到这些终端 agent。
    """
    structured = output_type is not None
    components = routable_components(parse_index_components(docs_root), feishu_tool)
    specialists = [
        build_specialist_agent(
            c,
            router.for_role(c.dir)[1],
            extra_tools=specialist_extra_tools,
            feishu_tool=feishu_tool,
            output_type=output_type,
            output_guardrails=output_guardrails,
        )
        for c in components
    ]
    instructions = _triage_instructions(components)
    # 分诊台只做路由 + 问候/拒绝自答，唯一会用到的标记是反问轮的 <<CLARIFY>>。
    instructions += STRUCTURED_CONTRACT_SUFFIX if structured else TRIAGE_CLARIFY_NOTE
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
    return triage, components


# ---------------------------------------------------------------------------
# 跨组件协作（coordinator + agents-as-tools）
# ---------------------------------------------------------------------------
#
# 与 triage 的根本区别：triage 用 handoff（转交后控制权不回来），适合"路由到唯一
# 专家"；跨组件诊断需要"问多个专家再综合"，所以用 agents-as-tools——协调者把每个
# 专家当**工具**调用，自己保留控制权，收齐各组件证据后综合出根因链。这是单一巨型
# prompt（原项目那种）很难做干净的场景：每个专家独立作用域、独立上下文，协调者只
# 负责拆解与综合。


def _coordinator_escalate_rule(components: list[Component]) -> str:
    """协调者的升级规则：它跨组件，归属要自己判，所以得把各组件的标记摆出来让它挑一条。

    与专家（标记唯一、照抄即可）不同：协调者可能只在一个组件上找到线索、也可能哪个都不指向。
    归属不明就 `<<ESCALATE:none>>`——不 @ 任何人，好过 @ 错人。
    """
    known = [c for c in components if c.open_id.startswith("ou_")]
    table = "\n".join(f"  - {c.name} → `{escalate_marker(c.open_id, c.dir)}`" for c in known)
    per_component = (
        f"- 归属**明确落在某一个组件**时，用该组件对应的那条（原样照抄）：\n{table}\n"
        if table
        else ""
    )
    return (
        "- 综合各专家的证据后仍然答不出时：先说明你咨询了哪些组件、各自查到什么，"
        "然后在末尾独立一行输出升级标记。\n"
        f"{per_component}"
        "- **跨多个组件 / 归属判断不明**（最常见）→ 输出 `<<ESCALATE:none>>`，不 @ 任何人。\n"
        "- 只要有任一专家给出了足以作答的证据，就正常作答，不要升级。"
    )


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
- **标清每条证据来自哪个组件**：如「（网关）上游实例偶发 unhealthy」「（容器）该实例 OOMKilled」。专家返回里带的来源标注一并保留，原样照抄——本地文档是 `（来源：xxx.md）`，飞书文档来源的组件是 `（来源：飞书文档·<组件>）`，两种都不要改写。
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
    feishu_tool: object | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
    agent_tool_hooks: object | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造跨组件协调者：各组件专家以 `as_tool` 暴露给协调者调用。

    每个专家工具名 `ask_<dir>`；专家用 `router.for_role(dir)` 的模型，协调者用
    `router.for_role("coordinator")`（无覆盖回退默认）。`as_tool` 会把父级的
    DocsContext 透传给专家，专家的 read_doc/grep_docs（或 query_feishu_doc）据此拿到
    docs_root。混合来源在这里最自然地落地：协调者并行问 `ask_redis`（读本地 md）和
    `ask_nginx`（打飞书文档服务），拿回的证据合并成一条根因链，它并不需要知道来源差异。

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
    components = routable_components(parse_index_components(docs_root), feishu_tool)
    specialist_tools = [
        build_specialist_agent(
            c,
            router.for_role(c.dir)[1],
            extra_tools=specialist_extra_tools,
            feishu_tool=feishu_tool,
        ).as_tool(
            tool_name=f"ask_{c.dir}",
            tool_description=(
                f"就某现象咨询 {c.name} 组件专家（覆盖：{c.coverage}）。"
                f"传一个自包含的子问题；返回该组件文档依据下的发现。"
            ),
            max_turns=specialist_max_turns,
            hooks=agent_tool_hooks,  # type: ignore[arg-type]
        )
        for c in components
    ]
    coord_instructions = _coordinator_instructions(components)
    coord_instructions += (
        STRUCTURED_CONTRACT_SUFFIX
        if structured
        else free_text_markers_section(escalate_rule=_coordinator_escalate_rule(components))
    )
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
    return coordinator, components


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
    feishu_tool: object | None = None,
    output_type: object | None = None,
    output_guardrails: list | None = None,
    agent_tool_hooks: object | None = None,
) -> tuple[Agent[DocsContext], list[Component]]:
    """构造自适应分诊 agent：handoffs = 各组件专家 + 跨组件协调者。

    分诊按问题决定转交给单个专家（常见）还是 coordinator（跨组件）。复用
    `build_specialist_agent` / `build_coordinator_agent`，不重写编排。

    护栏：`input_guardrails` 只挂在入口分诊台（输入护栏在入口跑一次即可）；写审批工具
    （`specialist_extra_tools`）挂到两条路径的专家——直接 handoff 的专家、以及 coordinator
    逃生口内 as_tool 的专家。`feishu_tool` 同理透传给两条路径。

    结构化输出：`output_type` 非空时，三类终端都产出契约——直接 handoff 的专家、coordinator
    逃生口（其内部 as_tool 专家仍回文字）、以及分诊自答。
    """
    structured = output_type is not None
    components = routable_components(parse_index_components(docs_root), feishu_tool)
    specialists = [
        build_specialist_agent(
            c,
            router.for_role(c.dir)[1],
            extra_tools=specialist_extra_tools,
            feishu_tool=feishu_tool,
            output_type=output_type,
            output_guardrails=output_guardrails,
        )
        for c in components
    ]
    coordinator, _ = build_coordinator_agent(
        docs_root,
        router,
        specialist_max_turns=specialist_max_turns,
        handoff_description=_AUTO_COORDINATOR_HANDOFF_DESC,
        specialist_extra_tools=specialist_extra_tools,
        feishu_tool=feishu_tool,
        output_type=output_type,
        output_guardrails=output_guardrails,
        agent_tool_hooks=agent_tool_hooks,
    )
    instructions = _auto_triage_instructions(components)
    instructions += STRUCTURED_CONTRACT_SUFFIX if structured else TRIAGE_CLARIFY_NOTE
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
    return triage, components
