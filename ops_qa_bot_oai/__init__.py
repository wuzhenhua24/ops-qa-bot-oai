"""ops-qa-bot-openai：基于 OpenAI Agents SDK 的运维文档问答机器人。

与 ops-qa-bot（Claude Agent SDK 版）功能对标，用于在真实场景里对比两个 agent SDK。
"""

from .actions import WriteCommandLog, WriteRequest, make_write_command_tool
from .bot import (
    AnswerResult,
    GuardedAnswer,
    Markers,
    OpsQABot,
    StructuredAnswer,
    parse_markers,
)
from .evaluate import EvalCase, aggregate, load_cases, run_eval, score_case
from .guardrails import (
    citation_output_guardrail,
    detect_injection,
    injection_input_guardrail,
)
from .model import ModelChoice, ModelRouter, build_model_router, resolve_model
from .orchestration import (
    Component,
    build_coordinator_agent,
    build_triage_agent,
    parse_index_components,
)
from .schema import AnswerContract, Decision, Followup, validate_citations

__all__ = [
    "OpsQABot",
    "AnswerResult",
    "StructuredAnswer",
    "GuardedAnswer",
    "Markers",
    "parse_markers",
    "detect_injection",
    "injection_input_guardrail",
    "citation_output_guardrail",
    "WriteCommandLog",
    "WriteRequest",
    "make_write_command_tool",
    "ModelChoice",
    "ModelRouter",
    "build_model_router",
    "resolve_model",
    "AnswerContract",
    "Decision",
    "Followup",
    "validate_citations",
    "Component",
    "parse_index_components",
    "build_triage_agent",
    "build_coordinator_agent",
    "EvalCase",
    "load_cases",
    "score_case",
    "aggregate",
    "run_eval",
]
