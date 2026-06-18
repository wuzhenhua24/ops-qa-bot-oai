"""交互式 REPL。对标 ops-qa-bot 的 cli.py，保持一致的使用手感方便并排对比。

支持的会话指令（纯文本短路，不进答题流程）：
- 空行 / Ctrl+C / Ctrl+D：退出
- /reset、/new、新对话、重置：清空上下文开新会话
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .bot import OpsQABot, format_tool_call
from .model import resolve_model

_RESET_WORDS = {"/reset", "/new", "新对话", "重置"}


async def run_once(docs_root: Path, question: str, show_tools: bool) -> None:
    """一次性问一个问题就退出。方便把同一问题分别喂给两个项目做 A/B 对比。"""
    model_choice = resolve_model()
    bot = OpsQABot(docs_root=docs_root, model_choice=model_choice)
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


async def run_repl(docs_root: Path, show_tools: bool) -> None:
    model_choice = resolve_model()
    print("运维文档问答机器人（OpenAI Agents SDK）")
    print(f"文档根目录：{docs_root}")
    print(f"模型：{model_choice.description}")
    print("输入问题后回车提问；/reset 开新会话；空行或 Ctrl+C 退出。\n")

    bot = OpsQABot(docs_root=docs_root, model_choice=model_choice)

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
        printed_prefix = False
        try:
            async for event in bot.ask(question):
                if event["type"] == "tool":
                    if show_tools:
                        print(f"  → {format_tool_call(event['name'], event['input'])}")
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
    args = parser.parse_args()
    docs_root = Path(args.docs).resolve()
    show_tools = not args.hide_tools
    if args.ask:
        asyncio.run(run_once(docs_root, args.ask, show_tools=show_tools))
    else:
        asyncio.run(run_repl(docs_root, show_tools=show_tools))


if __name__ == "__main__":
    main()
