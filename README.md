# ops-qa-bot-openai

基于 [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) 的内部运维文档问答机器人。

这是 [`ops-qa-bot`](../ops-qa-bot)（用 **Claude Agent SDK** 构建）的对照实现：**同一个需求、同一份文档、同一套 system prompt 规范**，换成 OpenAI Agents SDK 来搭，用于在真实场景里对比两个 agent SDK 的效果。两个项目的差异分析见 [`COMPARISON.md`](./COMPARISON.md)。

核心思路与参考项目一致：让 agent 通过 `read_doc` / `glob_docs` / `grep_docs` 三个工具按需检索 `docs/` 下的 markdown 文档，用 `docs/INDEX.md` 作为路由表定位组件目录，基于真实文档内容回答问题、标注来源、查不到就说找不到并按规则升级给负责人。

> 关键区别：Claude Agent SDK **自带** `Read`/`Glob`/`Grep` 文件工具，开箱即用；OpenAI Agents SDK **不提供**文件系统工具，所以本项目用 `@function_tool` 自己实现了等价的三个检索工具（含 `docs_root` 沙箱防越权），见 `ops_qa_bot_oai/tools.py`。

## 目录结构

```
ops-qa-bot-openai/
├── docs/                     # 运维文档根目录（从 ops-qa-bot 原样复制，保证对比素材一致）
│   ├── INDEX.md              # 路由表：组件 / 目录 / 负责人 open_id
│   └── redis|mysql|kafka/    # 各组件本地 markdown
├── ops_qa_bot_oai/
│   ├── tools.py              # read_doc / glob_docs / grep_docs（对标 Claude 内置 Read/Glob/Grep）
│   ├── prompt.py             # system prompt（移植自 ops-qa-bot 的核心主线）
│   ├── model.py              # provider 解析：openai / responses / compatible / anthropic / litellm 运行时切换
│   ├── schema.py             # 结构化输出契约 AnswerContract + 来源真实性校验（差异化 #1）
│   ├── bot.py                # OpsQABot：封装 Agent + Runner，ask()/answer()（标记）+ answer_structured()（契约）
│   └── cli.py                # 交互式 REPL + 一次性 --ask + --structured 模式
├── tests/test_tools.py       # 检索工具 + 沙箱 + 标记解析的回归测试（无需 LLM）
├── run.py                    # CLI 入口
└── pyproject.toml
```

## 使用

前置：已安装 [uv](https://docs.astral.sh/uv/)。

```bash
# 同步依赖（首次会创建 .venv）
uv sync                       # 用 openai / responses / compatible provider 时足够
uv sync --extra litellm       # 用 anthropic / litellm provider 时需要

# 配置模型 / provider（见下一节），然后启动交互式问答
uv run python run.py

# 一次性问一个问题就退出（方便做 A/B 对比）
uv run python run.py --ask "Redis 内存告警怎么处理？"

# 指定文档目录 / 隐藏工具调用日志
uv run python run.py --docs /path/to/docs --hide-tools
```

REPL 里：输入问题回车提问；`/reset`（或 `/new`/`新对话`/`重置`）开新会话；空行或 Ctrl+C 退出。

## 模型 / provider 配置（运行时切换）

OpenAI Agents SDK 是 provider 无关的，本项目把"用哪个模型"收敛成几个环境变量（完整模板见 `.env.example`）。**核心原则：按第三方端点暴露的是什么协议，来选 `OPS_QA_PROVIDER`，再配 `OPS_QA_BASE_URL` + `OPS_QA_API_KEY` + `OPS_QA_MODEL`。**

### Provider 一览

| `OPS_QA_PROVIDER` | 端点协议（wire format） | `OPS_QA_BASE_URL` | 装包 |
|---|---|---|---|
| `openai`（默认） | OpenAI 官方，**默认即 Responses**（`/v1/responses`） | 不填（走官方） | `uv sync` |
| `responses` | 第三方 **Responses**（`/v1/responses`） | 必填 | `uv sync` |
| `compatible` | 第三方 **Chat Completions**（`/v1/chat/completions`） | 必填 | `uv sync` |
| `anthropic` | 第三方 **Anthropic Messages**（`/v1/messages`，Claude Agent SDK / Claude Code 那套） | 可选（省略打 Anthropic 官方） | `uv sync --extra litellm` |
| `litellm` | 通用兜底，任意 LiteLLM provider（模型名带前缀如 `anthropic/`、`gemini/`） | 可选 | `uv sync --extra litellm` |

> OpenAI 有 **Responses**（新）和 **Chat Completions**（旧）两套接口；官方 `openai` 默认走 Responses，第三方按它实际暴露哪个端点选 `responses` 或 `compatible`。要和 ops-qa-bot 共用同一个 Claude/代理做"纯框架对比"，通常选 `anthropic`。

### 环境变量一览

| 变量 | 适用 provider | 说明 |
|---|---|---|
| `OPS_QA_PROVIDER` | 全部 | 见上表，缺省 `openai` |
| `OPS_QA_MODEL` | 全部 | 模型名；缺省随 provider 变。`anthropic` 会自动补 `anthropic/` 前缀 |
| `OPS_QA_BASE_URL` | `responses`/`compatible` 必填，`anthropic`/`litellm` 可选 | 第三方端点地址，见下「base_url 怎么填」 |
| `OPS_QA_API_KEY` | 第三方各 provider | `openai` 改走标准 `OPENAI_API_KEY` |
| `OPS_QA_ANTHROPIC_AUTH` | 仅 `anthropic` | `x-api-key`（默认）或 `bearer`，见下「鉴权头」 |

### base_url 怎么填（重要）

底层 client 会**自动追加端点路径**，所以 `OPS_QA_BASE_URL` 只填到**前缀**即可：

- `compatible` → client 追加 `/chat/completions`。填 `https://host/api/v3` → 请求 `https://host/api/v3/chat/completions`。
- `responses` → client 追加 `/responses`。填 `https://host/v1` → 请求 `https://host/v1/responses`。
- `anthropic` → LiteLLM 追加 `/v1/messages`。填 `https://host/api/coding` → 请求 `https://host/api/coding/v1/messages`。

容错：直接粘 provider 给的**完整 URL** 也行——`compatible`/`responses` 会自动剥掉末尾的 `/chat/completions`、`/responses`；`anthropic` 对 `/v1/messages` 后缀幂等。

### 鉴权头：x-api-key vs Bearer（`anthropic` 专属坑）

Anthropic 协议有两种鉴权头，不同第三方网关要求不同：

- **Anthropic 官方、部分代理** → `x-api-key`（默认，不用动）。
- **火山引擎 coding plan 等** → `Authorization: Bearer`。这时设 **`OPS_QA_ANTHROPIC_AUTH=bearer`**，仍用统一的 `OPS_QA_API_KEY`，本项目会自动把它转成 Bearer 发送。

（原理：LiteLLM 传 `api_key` 发 `x-api-key`、改读 `ANTHROPIC_AUTH_TOKEN` 才发 Bearer；`bearer` 开关帮你做了这层转换。）

### 第三方平台实例（已实测拼出的最终 URL）

**智谱 BigModel（GLM）** — 同账号两种格式都可：

```bash
# OpenAI 格式（端点 .../api/paas/v4/chat/completions）
OPS_QA_PROVIDER=compatible \
OPS_QA_BASE_URL=https://open.bigmodel.cn/api/paas/v4 \
OPS_QA_API_KEY=<智谱 key> OPS_QA_MODEL=glm-4.6 \
uv run python run.py --ask "Redis 内存告警怎么处理？"

# Anthropic 格式（端点 .../api/anthropic → /v1/messages）
OPS_QA_PROVIDER=anthropic \
OPS_QA_BASE_URL=https://open.bigmodel.cn/api/anthropic \
OPS_QA_API_KEY=<智谱 key> OPS_QA_MODEL=glm-4.6 \
uv run python run.py --ask "Redis 内存告警怎么处理？"
```

**火山引擎 coding plan** — Anthropic 端点要 Bearer，记得加 `OPS_QA_ANTHROPIC_AUTH=bearer`：

```bash
# OpenAI 格式（推荐，标准 Bearer 鉴权最省心）
OPS_QA_PROVIDER=compatible \
OPS_QA_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3 \
OPS_QA_API_KEY=<火山 ARK key> OPS_QA_MODEL=<模型名或 endpoint-id> \
uv run python run.py --ask "MySQL 主从延迟怎么排查？"

# Anthropic 格式
OPS_QA_PROVIDER=anthropic \
OPS_QA_BASE_URL=https://ark.cn-beijing.volces.com/api/coding \
OPS_QA_API_KEY=<火山 ARK key> OPS_QA_ANTHROPIC_AUTH=bearer OPS_QA_MODEL=<模型名> \
uv run python run.py --ask "MySQL 主从延迟怎么排查？"
```

### 两种对比姿势

- **纯 SDK 框架对比**：用 `anthropic` / `compatible` / `litellm` 让本项目跑**和 ops-qa-bot 同一个模型/代理**，把模型变量固定住，剩下的差异就是两个 agent 框架本身（工具机制、agent loop、prompt 适配）。
- **整套产品对比**：用 `openai` 跑 OpenAI 原生模型，对比"OpenAI SDK + OpenAI 模型" vs "Claude SDK + Claude 模型"的整体效果。

## 结构化输出模式（差异化原型 #1）

OpenAI Agents SDK 支持 `output_type=`：把一轮回答强制成**带 schema 校验的类型对象**，模型必须按字段填、不合法 SDK 会重试。这是 Claude Agent SDK 给不了的一等公民能力（那边只能像 ops-qa-bot 那样在文本里塞 `<<MARKER>>` 再正则解析，模型写错格式就丢信息）。

加 `--structured` 即用这条路径：

```bash
uv run python run.py --ask "Redis 内存告警怎么处理？" --structured
# REPL 也支持：uv run python run.py --structured
```

模型这一轮产出 `AnswerContract`（见 `ops_qa_bot_oai/schema.py`）：

- `decision`：`answer` / `clarify` / `escalate` / `reject`（取代旧的 `<<CLARIFY>>`/`<<ESCALATE>>`）
- `answer`：中文 markdown 正文
- `citations`：答案依据的文档路径列表 —— 拿到后**用代码逐条核对是否真实存在**（把"必须引用真实文档"从 prompt 自律升级成硬校验，编造/越界的来源会被标 ✗ 并告警）
- `escalate_to` / `escalate_dir` / `followups` / `confidence`

跨 provider 用**非严格** schema 下发（`strict_json_schema=False`），兼容 Claude / 智谱 / 火山等不支持 OpenAI strict 结构化输出的端点。自由文本 + 标记的老路径（`answer()` / 不带 `--structured`）保留着，方便并排对比两种产出方式。

## 测试

```bash
uv run pytest            # 检索 / 沙箱 / 标记解析 / base_url 容错 / 结构化契约 + 来源校验（确定性，无需 LLM）
uv run ruff check .      # lint
uv run ruff format .     # 格式化
```

## 范围说明（第一版：核心问答 + CLI）

本项目第一版聚焦"基于文档回答"这条**对比核心主线**，刻意没有移植参考项目里围绕飞书产品形态的扩展：飞书接入（HTTP/长连接）、SSH 实时诊断、数据库只读分析、参数变更审批、定时跟进、反馈日志与问答归档。这些是 ops-qa-bot 的可选/产品层特性，不影响"agentic 文档检索回路"本身的对比。`prompt.py` 保留了与参考项目同款的 `<<ESCALATE>>` / `<<CLARIFY>>` / `<<FOLLOWUPS>>` 标记，因此升级 / 反问 / 追问的**答案塑形行为**可以直接和 Claude 版对照（CLI 会解析这些标记并提示，但没有飞书渲染层）。
