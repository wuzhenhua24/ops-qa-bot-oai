"""ops-qa-bot-openai：基于 OpenAI Agents SDK 的运维文档问答机器人。

与 ops-qa-bot（Claude Agent SDK 版）功能对标，用于在真实场景里对比两个 agent SDK。
"""

from .bot import AnswerResult, Markers, OpsQABot, StructuredAnswer, parse_markers
from .model import ModelChoice, resolve_model
from .orchestration import Component, build_triage_agent, parse_index_components
from .schema import AnswerContract, Decision, Followup, validate_citations

__all__ = [
    "OpsQABot",
    "AnswerResult",
    "StructuredAnswer",
    "Markers",
    "parse_markers",
    "ModelChoice",
    "resolve_model",
    "AnswerContract",
    "Decision",
    "Followup",
    "validate_citations",
    "Component",
    "parse_index_components",
    "build_triage_agent",
]
