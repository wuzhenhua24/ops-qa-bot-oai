"""交互式 REPL。对标 ops-qa-bot 的 cli.py，保持一致的使用手感方便并排对比。

支持的会话指令（纯文本短路，不进答题流程）：
- 空行 / Ctrl+C / Ctrl+D：退出
- /reset、/new、新对话、重置：清空上下文开新会话
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from .bot import GuardedAnswer, OpsQABot, StructuredAnswer, format_tool_call
from .model import MODE_LABELS, MODES, resolve_mode, resolve_model

_RESET_WORDS = {"/reset", "/new", "新对话", "重置"}


def _print_structured(sa: StructuredAnswer) -> None:
    """渲染结构化契约：决策 + 正文 + 来源（带真实性校验）+ 追问 + 置信度。"""
    c = sa.contract
    print(f"bot> [{c.decision.value}]")
    print(c.answer)
    if c.citations:
        invalid = set(sa.invalid_citations)
        print("\n来源：")
        for cite in c.citations:
            mark = "✗ 不存在/越界" if cite in invalid else "✓"
            print(f"  - {cite}  {mark}")
    if c.decision.value == "escalate" and c.escalate_to:
        print(f"\n升级给：{c.escalate_to}（组件目录：{c.escalate_dir or '?'}）")
    if c.followups:
        print("追问建议：" + "、".join(f.value for f in c.followups))
    print(f"置信度：{c.confidence:.2f}")
    if sa.invalid_citations:
        print(f"⚠️ 有 {len(sa.invalid_citations)} 条来源不指向真实文档，答案可能不可靠。")
    if sa.guardrail_blocked:
        print(f"🛡️ 已被输出护栏拦截：{sa.guardrail_blocked}")


def _print_guarded(ga: GuardedAnswer) -> None:
    """渲染带护栏 / 审批的回答。"""
    if ga.blocked:
        print(f"🛡️ bot> 已被输入护栏拦截（{ga.blocked}）：{ga.text}")
        return
    print(f"bot> {ga.text}")
    for req, approved in ga.approvals:
        mark = "✅ 已批准" if approved else "🚫 已驳回"
        cmd = req.arguments.get("command", req.arguments)
        target = req.arguments.get("target", "?")
        print(f"  审批 [{req.tool_name}] 在 {target} 执行 `{cmd}` → {mark}")
    if ga.approved_writes:
        print(f"  （已登记 {len(ga.approved_writes)} 条写操作待人工执行）")


def _make_approver(interactive: bool):
    """构造审批回调。interactive=True 时 y/n 询问；否则一律驳回（安全默认）。"""
    if not interactive:
        return None

    def approver(req) -> bool:
        cmd = req.arguments.get("command", req.arguments)
        target = req.arguments.get("target", "?")
        try:
            ans = input(f"  ⚠️ 批准在 {target} 执行 `{cmd}`？(y/N) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes", "是")

    return approver


def _orchestration_lines(mode: str, bot: OpsQABot) -> list[str]:
    """编排模式的横幅细节（组件专家名单 + 模型路由）；single 模式返回空。"""
    if mode == "single":
        return []
    roster = "、".join(c.name for c in bot.components) or "（无）"
    head = f"组件专家：{roster}"
    if mode == "auto":
        head += "（跨组件问题自动升级协调者）"
    lines = [head]
    if bot.model_router is not None:
        entry = "coordinator" if mode == "coordinator" else "triage"
        roles = [entry] + [c.dir for c in bot.components]
        lines.append(f"模型路由：{bot.model_router.describe(roles)}")
    return lines


async def run_once(
    docs_root: Path,
    question: str,
    show_tools: bool,
    structured: bool = False,
    mode: str = "single",
    guardrails: bool = False,
) -> None:
    """一次性问一个问题就退出（适合脚本调用 / 批量跑题）。"""
    model_choice = resolve_model()
    # 路由(mode) × 输出格式(structured) × 护栏(guardrails) 三者正交，可任意组合。
    bot = OpsQABot(
        docs_root=docs_root,
        model_choice=model_choice,
        mode=mode,
        guardrails=guardrails,
    )
    if mode != "single" and show_tools:
        for ln in _orchestration_lines(mode, bot):
            print(f"[{ln}]")
        print()

    if guardrails and not structured:
        # 一次性模式无人值守：approver=None → 写操作一律驳回（安全默认）。
        if show_tools:
            print("[护栏：输入注入检测 + 写操作审批；一次性模式下写操作默认驳回]\n")
        ga = await bot.answer_guarded(question, approver=None)
        _print_guarded(ga)
        if ga.usage:
            print(
                f"\n[in={ga.usage.get('input_tokens', 0)} "
                f"out={ga.usage.get('output_tokens', 0)} "
                f"reqs={ga.usage.get('requests', 0)}]"
            )
        return

    if structured:
        sa = await bot.answer_structured(question)
        if show_tools and sa.num_turns is not None:
            print(f"[模型 {model_choice.description} · {sa.num_turns} 次模型调用 · 结构化输出]\n")
        _print_structured(sa)
        if sa.usage:
            print(
                f"\n[in={sa.usage.get('input_tokens', 0)} "
                f"out={sa.usage.get('output_tokens', 0)} "
                f"reqs={sa.usage.get('requests', 0)}]"
            )
        return

    result = await bot.answer(question)
    if show_tools and result.num_turns is not None:
        print(f"[模型 {model_choice.description} · {result.num_turns} 次模型调用]\n")
    print(result.text)
    notes = []
    if result.markers.clarify:
        notes.append("这是一轮反问（CLARIFY）")
    if result.markers.escalate:
        notes.append(f"已标记升级：{result.markers.escalate}")
    if result.markers.followups:
        notes.append("追问建议：" + "、".join(result.markers.followups))
    if notes:
        print("\n— " + "；".join(notes))
    if result.usage:
        print(
            f"\n[in={result.usage.get('input_tokens', 0)} "
            f"out={result.usage.get('output_tokens', 0)} "
            f"reqs={result.usage.get('requests', 0)}]"
        )
    if result.subtype == "error_max_turns":
        print("⚠️ 撞到 max_turns 上限，结论可能不完整。")


async def run_repl(
    docs_root: Path,
    show_tools: bool,
    structured: bool = False,
    mode: str = "single",
    guardrails: bool = False,
) -> None:
    model_choice = resolve_model()
    # 路由(mode) × 输出格式(structured) × 护栏(guardrails) 三者正交，可任意组合。
    bot = OpsQABot(
        docs_root=docs_root,
        model_choice=model_choice,
        mode=mode,
        guardrails=guardrails,
    )
    approver = _make_approver(interactive=True) if guardrails else None
    parts = []
    if mode != "single":
        parts.append(MODE_LABELS[mode])
    if structured:
        parts.append("结构化输出")
    if guardrails:
        parts.append("护栏 + 写操作审批")
    print("运维文档问答机器人（OpenAI Agents SDK）")
    print(f"文档根目录：{docs_root}")
    print(f"模型：{model_choice.description}" + (f"（{'；'.join(parts)}）" if parts else ""))
    for ln in _orchestration_lines(mode, bot):
        print(ln)
    print("输入问题后回车提问；/reset 开新会话；空行或 Ctrl+C 退出。\n")

    while True:
        try:
            question = await asyncio.to_thread(input, "你> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        question = question.strip()
        if not question:
            print("再见。")
            return
        if question in _RESET_WORDS:
            bot.reset()
            print("（已开启新会话）\n")
            continue

        print()
        if guardrails and not structured:
            try:
                ga = await bot.answer_guarded(question, approver=approver)
                _print_guarded(ga)
                print()
            except KeyboardInterrupt:
                print("\n（已中断本次回答）\n")
            except Exception as e:  # noqa: BLE001
                print(f"\n[出错] {type(e).__name__}: {e}\n")
            continue

        if structured:
            try:
                sa = await bot.answer_structured(question)
                _print_structured(sa)
                print()
            except KeyboardInterrupt:
                print("\n（已中断本次回答）\n")
            except Exception as e:  # noqa: BLE001
                print(f"\n[出错] {type(e).__name__}: {e}\n")
            continue

        printed_prefix = False
        try:
            async for event in bot.ask(question):
                if event["type"] == "tool":
                    if show_tools:
                        print(f"  → {format_tool_call(event['name'], event['input'])}")
                elif event["type"] == "handoff":
                    if show_tools:
                        print(f"  ⇒ 转交给 {event['agent']}")
                    printed_prefix = False  # 专家接手后重新打印 bot> 前缀
                elif event["type"] == "text":
                    if not printed_prefix:
                        print("bot> ", end="", flush=True)
                        printed_prefix = True
                    print(event["text"], end="", flush=True)
                elif event["type"] == "done":
                    print()
                    usage = event.get("usage")
                    if usage:
                        print(
                            f"  [in={usage.get('input_tokens', 0)} "
                            f"out={usage.get('output_tokens', 0)} "
                            f"reqs={usage.get('requests', 0)}]"
                        )
                    if event.get("subtype") == "error_max_turns":
                        print("  ⚠️ 撞到 max_turns 上限，结论可能不完整。")
                    print()
        except KeyboardInterrupt:
            print("\n（已中断本次回答）\n")
        except Exception as e:  # noqa: BLE001 —— REPL 里不让单次异常掀翻整个会话
            print(f"\n[出错] {type(e).__name__}: {e}\n")


def main() -> None:
    # 加载项目根目录 .env（provider/base_url/key/model 等 OPS_QA_* 都在里面）。
    # 已存在的环境变量优先，不被覆盖。
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(description="内部运维文档问答机器人（OpenAI Agents SDK）")
    parser.add_argument(
        "--docs",
        default=str(Path(__file__).resolve().parent.parent / "docs"),
        help="文档根目录路径（默认：项目自带的 docs/）",
    )
    parser.add_argument(
        "--hide-tools",
        action="store_true",
        help="隐藏 agent 的工具调用日志",
    )
    parser.add_argument(
        "--ask",
        metavar="问题",
        help="一次性问一个问题就退出（不进 REPL），方便做 A/B 对比",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="用结构化输出契约（AnswerContract）替代自由文本 + <<MARKER>>，并校验来源真实性",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=resolve_mode(),
        help="答题编排模式：single（单 agent）/ multi（分诊→专家）/ "
        "coordinator（跨组件协调）/ auto（自适应，默认）。"
        "缺省读环境变量 OPS_QA_MODE，与飞书共用",
    )
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help="开启输入注入护栏 + 写操作审批（HITL），与任何 --mode 并存；"
        "结构化模式下额外加输出来源护栏。开启后走非流式审批路径",
    )
    args = parser.parse_args()
    docs_root = Path(args.docs).resolve()
    show_tools = not args.hide_tools
    if args.ask:
        asyncio.run(
            run_once(
                docs_root,
                args.ask,
                show_tools=show_tools,
                structured=args.structured,
                mode=args.mode,
                guardrails=args.guardrails,
            )
        )
    else:
        asyncio.run(
            run_repl(
                docs_root,
                show_tools=show_tools,
                structured=args.structured,
                mode=args.mode,
                guardrails=args.guardrails,
            )
        )


if __name__ == "__main__":
    main()
