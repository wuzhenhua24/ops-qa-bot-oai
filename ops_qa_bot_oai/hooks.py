"""运行遥测 hooks：精确转交链 + 按 agent 的 token 归账 + 工具调用清单。

此前项目里有两处"手搓推断"：流式路径靠跳过首个 agent_updated_stream_event 识别
handoff；非流式（结构化）用 last_agent != 入口 agent 反推路由。OpenAI Agents SDK 的
**lifecycle hooks**（RunHooks）把这些做成一等公民回调——`on_handoff` 给出精确的
from→to 转交链、`on_llm_end` 把每次 LLM 调用的用量归到发起的 agent 名下、
`on_tool_start` 记录工具调用，流式与非流式统一，不用解析 raw 事件或事后猜。

按 agent 的 token 归账直接服务多模型路由（差异化 #2）的量化：评测台能回答
"分诊的便宜模型花了多少、专家的强模型花了多少"，而不是只有 run 级总量。

生命周期：一个 RunTelemetry 实例挂在 OpsQABot 上、每次答题前 `reset_run()` 清零。
之所以不做成每 run 新建：coordinator 模式的专家是 `as_tool` 子 run，SDK 的
`as_tool(hooks=...)` 是**构建期**参数——构图时注入同一实例，子 run 的 LLM 调用才能
归到对应专家名下。同一 bot 的答题调用天然串行（CLI 顺序、飞书 per-key 锁、评测逐题），
无并发窜账问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse
from agents.tool import Tool


@dataclass
class AgentUsage:
    """某个 agent 在本轮 run 里的累计用量。"""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class RunTelemetry(RunHooks):
    """收集一轮答题的遥测：转交链 / 按 agent 用量 / 工具调用。

    - `handoff_chain`：[(from_agent, to_agent), ...]，精确的转交链路。
    - `route`：最后一次 handoff 的目标（None = 入口 agent 自答），取代 last_agent 推断。
    - `agent_usage()`：{agent 名: {requests, input_tokens, output_tokens}}，
      含 coordinator 模式下 as_tool 专家子 run 的调用（构图时注入同一实例）。
    - `tool_calls`：[(agent 名, 工具名), ...]。
    """

    def __init__(self) -> None:
        self.handoff_chain: list[tuple[str, str]] = []
        self.tool_calls: list[tuple[str, str]] = []
        self._usage: dict[str, AgentUsage] = {}

    def reset_run(self) -> None:
        """新一轮答题前清零（同一实例跨 run 复用，见模块 docstring）。"""
        self.handoff_chain.clear()
        self.tool_calls.clear()
        self._usage.clear()

    @property
    def route(self) -> str | None:
        """最终落点 agent 名（最后一次 handoff 目标）；None 表示入口 agent 自答。"""
        return self.handoff_chain[-1][1] if self.handoff_chain else None

    def agent_usage(self) -> dict[str, dict[str, int]]:
        """按 agent 名的用量快照（插入序 = 首次发言序）。"""
        return {name: u.as_dict() for name, u in self._usage.items()}

    # -- RunHooks 回调 -------------------------------------------------------

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        self.handoff_chain.append((from_agent.name, to_agent.name))

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        u = self._usage.setdefault(agent.name, AgentUsage())
        u.requests += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            u.input_tokens += getattr(usage, "input_tokens", 0) or 0
            u.output_tokens += getattr(usage, "output_tokens", 0) or 0

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
    ) -> None:
        self.tool_calls.append((agent.name, getattr(tool, "name", "?")))
