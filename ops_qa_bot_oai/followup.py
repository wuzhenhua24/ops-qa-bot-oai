"""定时跟进（`schedule_followup` 工具）：让 agent 能"过 N 分钟后自动再查一次"。

典型场景：用户发起一个耗时变更（ALTER / 迁移 / 扩容 / 重启预热），让 bot
"20 分钟后帮我看看执行完没"。单轮答题是同步跑完就返回的——agent 没法真的
sleep 20 分钟，所以以前只能老实回"我没法定时"。本模块把"登记一个未来动作"
包成一个横切工具：agent 只负责**提议**（把"过多久、查什么"登记下来、当轮立刻
回复"好，到点帮你看"），真正的查询发生在**到点的回调里**——届时用存好的 task
跑一轮全新答题（实打实调 query_database 等工具），把结果主动 @ 用户推回群里。

对标 ops-qa-bot（Claude 版）的 scheduled_followup.py，两处结构性改进：

- 参考版要用 `create_sdk_mcp_server` 把工具包成进程内 MCP server 才能挂；这里
  一个 `@function_tool` 就够（OpenAI SDK 的函数工具是一等公民）。
- 参考版的定时器（FollowupScheduler）直接 import 答题入口、住在 feishu_core；
  这里定时器是纯逻辑（`feishu/followup.py`），到点执行什么由接入层注入回调，
  本模块只做"校验入参 + 交给注入的 submitter 登记"——与 db_query / gateway_trace
  的 submitter/Protocol 解耦姿态一致，可脱离飞书单测。

设计要点（与参考版对齐）：
- **工具天生不是写操作**：唯一副作用是"安排一次未来的提问"，到点跑的还是受
  同样只读约束的答题流程，所以不走 needs_approval 审批。
- **MVP 是纯内存定时器**：进程重启丢未触发任务（20 分钟级场景够用）；真有
  "重启丢任务"痛点再补持久化。
- 失败**返回文字而不抛异常**：抛会打断 agent 这一轮；返回引导文字让它如实
  告诉用户（如"超出可登记数量，请精简或过会儿手动再问"）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from agents import RunContextWrapper, function_tool

from .model import env_flag
from .tools import DocsContext

logger = logging.getLogger(__name__)

FOLLOWUP_TOOL_NAME = "schedule_followup"

# task 文本上限：自包含的一句跟进指令远不到这个量级，超长当异常拒。
_MAX_TASK_LEN = 1000


@dataclass
class FollowupConfig:
    """定时跟进配置。缺省 enabled=False（工具不挂载、prompt 不加章节，零感知）。

    纯产品特性、无外部依赖，所以用显式开关 `OPS_QA_FOLLOWUP=1`（不像 doc_qa 从
    base_url 推导）。启用还需接入层注入 submitter（飞书侧定时器链路在位）——CLI
    直用没有定时器，不挂工具。

    `min/max_delay_minutes`：可登记的等待区间，越界工具拒并提示 agent 取合理值
    （防"过 3 天提醒我"这种内存定时器扛不住的诉求）。`max_pending_per_user`：
    单个 (chat, asker) 同时挂起的上限，防误用把后台任务堆爆。
    """

    enabled: bool = False
    min_delay_minutes: int = 1
    max_delay_minutes: int = 120
    max_pending_per_user: int = 5

    @classmethod
    def from_env(cls) -> FollowupConfig:
        """从 `OPS_QA_FOLLOWUP*` 环境变量解析。缺省全关。"""

        def _int(name: str, default: int) -> int:
            raw = (os.environ.get(name) or "").strip()
            try:
                return int(raw) if raw else default
            except ValueError:
                return default

        return cls(
            enabled=env_flag("OPS_QA_FOLLOWUP", default=False),
            min_delay_minutes=_int("OPS_QA_FOLLOWUP_MIN_MINUTES", 1),
            max_delay_minutes=_int("OPS_QA_FOLLOWUP_MAX_MINUTES", 120),
            max_pending_per_user=_int("OPS_QA_FOLLOWUP_MAX_PENDING", 5),
        )


@dataclass
class FollowupRequest:
    """一笔待登记的定时跟进——由 `schedule_followup` 工具校验后组装，交给
    `FollowupSubmitter` 落成"一个 N 分钟后触发的后台任务"。

    `task` 是 agent 写的**自包含**跟进指令（到点时作为新一轮的问题喂回答题流程），
    必须把"查哪个实例/表、查什么、判断标准"都写进去，不能依赖上文。
    """

    delay_minutes: int
    task: str


class FollowupSubmitter(Protocol):
    """把一笔 FollowupRequest 落成"一个到点触发的后台跟进任务"。

    由飞书接入层实现并按 (chat_id, asker) 绑定后注入（见 feishu/followup.py 的
    `make_followup_submitter`）；本模块只依赖这个 Protocol，不 import feishu
    （避免循环）。返回给 agent 看的文字（如"已登记，20 分钟后帮你看"或"超出可
    登记数量"），工具原样转给 LLM。
    """

    async def __call__(self, req: FollowupRequest) -> str: ...


def make_schedule_followup_tool(config: FollowupConfig, submitter: FollowupSubmitter) -> Any:
    """构造 `schedule_followup` 工具：校验 delay 边界 + task 非空/长度后交 submitter 登记。

    失败一律返回引导文字（不抛），让 agent 据此处理（修正参数 / 如实告诉用户登记
    不了）。到点实际执行的链路由 submitter 背后的调度器负责，与本工具无关。
    """

    @function_tool(name_override=FOLLOWUP_TOOL_NAME)
    async def schedule_followup(
        ctx: RunContextWrapper[DocsContext], delay_minutes: int, task: str
    ) -> str:
        """登记一个**定时跟进任务**：过 N 分钟后由系统自动执行一次你指定的检查并推回群里。

        用于用户明确要求「过一会儿 / X 分钟后再帮我看看 Y」这类**需要等待再复查**的
        场景（典型如耗时变更后确认完成没）。**你只是登记，不会也不需要在这一轮等待**：
        工具立刻返回，到点时系统会用你写的 task 跑一轮全新检查。调用成功后把返回的
        确认语如实转达用户即可，不要谎称现在就查到了结果。

        Args:
            delay_minutes: 等待分钟数（整数）。用用户说的时间；没明说就按场景给合理值
                （变更类常见 5~30 分钟）。有上下限，越界会返回提示。
            task: 到点要执行的**自包含**检查指令。必须把复查所需的一切写进去——查哪个
                实例（IP/端口/租户/库/表）、具体查什么、怎么判断完成或异常。**不要依赖
                上文**（到点是全新一轮，没有现在的对话记忆）。
        """
        if delay_minutes < config.min_delay_minutes or delay_minutes > config.max_delay_minutes:
            return (
                f"等待时间需在 {config.min_delay_minutes}~{config.max_delay_minutes} "
                f"分钟之间（你传的是 {delay_minutes}）。请在范围内取值；"
                "需要更久的跟进就告诉用户过会儿手动再问我。"
            )
        task = (task or "").strip()
        if not task:
            return "缺少 task（到点要执行的检查指令）。请写一条自包含的指令再登记。"
        if len(task) > _MAX_TASK_LEN:
            return "task 过长，疑似异常。请精简成一条聚焦的检查指令后重试。"

        try:
            return await submitter(FollowupRequest(delay_minutes=delay_minutes, task=task))
        except Exception:
            logger.exception("submit scheduled followup failed")
            return "登记定时跟进时出错。请如实告诉用户暂时没能登记，让他过会儿手动再问我。"

    return schedule_followup
