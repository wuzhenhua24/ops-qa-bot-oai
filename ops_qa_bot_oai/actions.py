"""写操作审批工具（差异化 #4 的另一半）：needs_approval 的 human-in-the-loop。

ops-qa-bot（Claude SDK 版）对"写操作"的处理：prompt 强约束"永不执行"，再加一个
PreToolUse hook 兜底**硬拦**写命令；唯一带审批的路径（DB 参数变更）是靠飞书卡片回调
手工拼出来的 HITL。

OpenAI Agents SDK 把审批做成了一等机制：给工具标 `needs_approval`，模型想调它时
**整个 run 会暂停并抛出 interruption**，由调用方 `state.approve()/reject()` 后
`Runner.run(agent, state)` 续跑——RunState 可序列化，天然支持"挂起等人批"。本工具
`request_write_command` 就是这套：agent 永不直接执行写操作，而是**提议**一条命令，
人工批准后才"登记执行"（本 demo 不真的跑命令，只记录为"已批准、待人工执行"，契合
ops-qa-bot "agent 只提议、人工执行" 的安全姿态）；驳回则什么都不做、agent 回退到
文字建议。

相比 hook：hook 是"事后硬拦 + 退化成文字建议"，这里是"事前挂起 + 人来定夺"，且
approve/reject 的决定权与审计天然落在 RunState 上，不用自己拼回调链路。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import RunContextWrapper, function_tool

from .tools import DocsContext


@dataclass
class WriteRequest:
    """一条被提议的写/变更命令。"""

    command: str
    target: str
    reason: str
    approved: bool = False  # 经人工批准并"登记执行"后置 True


@dataclass
class WriteCommandLog:
    """收集本会话内提议过的写命令（审批结果落在每条的 approved 上）。"""

    requests: list[WriteRequest] = field(default_factory=list)


def make_write_command_tool(log: WriteCommandLog):
    """构造 `request_write_command` 工具：needs_approval=True，批准后才登记。

    用闭包捕获 log，工具被实际执行（= 已通过审批）时往 log 追加一条 approved 记录。
    被驳回时工具根本不执行，log 不会有这条——审批结果如实反映在 log 上。
    """

    @function_tool(needs_approval=True)
    def request_write_command(
        ctx: RunContextWrapper[DocsContext], command: str, target: str, reason: str
    ) -> str:
        """提议一条**写/变更**命令，交人工审批后执行。

        仅在确实需要变更（重启服务 / 改配置 / 删除数据 / 改库参数等）时调用；只读
        诊断或纯知识问答**不要**用本工具。你永远不会直接执行——本工具会挂起等人批准，
        批准后才登记为"待人工执行"。

        Args:
            command: 要执行的具体命令（如 `redis-cli config set maxmemory 8gb`）。
            target: 在哪台机器/实例上执行（如 `10.1.2.3` 或 `mysql-test-01`）。
            reason: 为什么需要这条变更（结合文档/诊断给出依据）。
        """
        # 走到这里说明已通过审批（needs_approval 在执行前拦截）。
        log.requests.append(
            WriteRequest(command=command, target=target, reason=reason, approved=True)
        )
        return (
            f"已登记写操作（已批准，待人工执行）：在 {target} 执行 `{command}`。"
            "请由运维人工执行；agent 不直接对目标机下发命令。"
        )

    return request_write_command
