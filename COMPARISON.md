# 两个 SDK 实现的对比笔记

| | `ops-qa-bot` | `ops-qa-bot-openai`（本项目） |
|---|---|---|
| Agent SDK | Claude Agent SDK | OpenAI Agents SDK |
| 文档检索工具 | **内置** `Read` / `Glob` / `Grep` | **自实现** `read_doc` / `glob_docs` / `grep_docs`（`@function_tool`） |
| 会话状态 | 常驻 `ClaudeSDKClient` 子进程持有历史 | 无状态 `Runner` + SDK `Session`（`SQLiteSession`，可落盘持久化） |
| 流式输出 | `receive_response()` 吐 AssistantMessage / ResultMessage | `Runner.run_streamed().stream_events()` |
| 步数保险丝 | `ClaudeAgentOptions(max_turns=...)`，撞上 `ResultMessage.subtype="error_max_turns"` | `Runner.run(max_turns=...)`，撞上抛 `MaxTurnsExceeded` |
| 用量 | `ResultMessage.usage`（含 cache_read/creation、官方 cost_usd） | `result.context_wrapper.usage`（input/output/total、requests、cached、reasoning） |
| 模型 | Claude（SDK 绑定 Anthropic） | provider 无关：OpenAI 原生 / 兼容代理 / LiteLLM 运行时切换 |
| 权限/写防护 | `permission_mode` + PreToolUse hook 拦写命令 | 核心版无 Bash，工具天然只读；沙箱在 `tools.py` 自己做 |

## 关键设计差异

### 1. 文件工具：开箱即用 vs 自己造（含沙箱）

这是两个 SDK 最直观的差异。Claude Agent SDK 给了 agent 一套内置文件工具，`ops-qa-bot` 直接 `tools=["Read","Glob","Grep","Bash"]` 即可。OpenAI Agents SDK 不提供文件系统能力，本项目用 `@function_tool` 实现了语义对齐的三个检索工具，并**自己负责安全边界**：`_resolve_within()` 解析真实路径后做 `docs_root` 子树前缀校验，挡 `..` 越权与符号链接逃逸（对应 ops-qa-bot 里靠 `permission_mode` + hook 管的那层）。

代价是多写了一个 `tools.py`；收益是工具行为完全可控、可单测（`tests/test_tools.py` 不需要 LLM 就能锁住沙箱与检索语义）。

### 2. 会话状态：子进程 vs Session

Claude SDK 用一个常驻子进程维护对话，`interrupt()` 能打断在途答题。OpenAI SDK 的 `Runner` 是无状态的，多轮历史交给一等公民 **Session**（`Runner.run(..., session=...)` 自动读写）：本项目用 `SQLiteSession`，缺省内存态，设 `OPS_QA_SESSION_DB` 即落盘——重启 / 空闲回收后同一会话可恢复上下文，这是子进程持有历史给不了的。`reset()` 即 `clear_session()`。"打断在途"仍需自己用 task cancel 实现（核心版未做）。

### 3. provider 灵活性

ops-qa-bot 绑定 Claude。本项目因为 OpenAI SDK 的 provider 无关性，可以经兼容代理 / LiteLLM 跑**同一个 Claude 模型** —— 这正是做"纯框架对比"的关键：固定模型变量，只比框架。

## A/B 对比怎么做

两个项目用**同一份 `docs/`**（本项目的 docs 从 ops-qa-bot 原样复制），把同一个问题分别喂给两边：

```bash
# Claude 版（在 ops-qa-bot 目录）
cd ../ops-qa-bot && uv run python run.py        # REPL 里问

# OpenAI 版（本目录），一次性问同一个问题
uv run python run.py --ask "Redis 内存告警怎么处理？"
```

建议固定模型变量做"纯框架对比"：本项目用 `OPS_QA_PROVIDER=compatible/litellm` 指向 ops-qa-bot 用的同一个 Claude 模型，再逐题对照。

### 建议的观察维度

- **路由准确性**：是否读对了 `INDEX.md` → 命中正确组件目录。
- **检索效率**：答一题用了几次工具调用 / 几次模型请求（`reqs`），有没有兜圈。
- **来源标注**：每个结论是否带 `（来源：xxx.md）`，路径是否真实。
- **防幻觉**：文档外的问题是否老实说"未找到"+ 升级，而不是编。
- **反问质量**：信息不足时是否恰当反问（`<<CLARIFY>>`）而非硬答 / 滥问。
- **token / 时延**：用量与响应速度。

### 一组可复用的测试问题（覆盖 docs/ 内容）

1. `Redis 内存告警怎么处理？`（应命中 redis/troubleshooting.md，带来源 + ⚠️ FLUSHDB 风险提示）
2. `MySQL 主从延迟怎么排查？`（命中 mysql）
3. `Kafka 消费延迟怎么看？`（命中 kafka）
4. `备份怎么做？`（跨 redis/mysql 多组件，理想是反问 `<<CLARIFY>>`）
5. `Redis 集群跨机房迁移步骤？`（文档没有，理想是"未找到"+ `<<ESCALATE>>`）
6. `帮我写个快排`（运维范围外，应友好拒绝、不编）
