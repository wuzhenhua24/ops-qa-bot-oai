#!/usr/bin/env python3
"""离线评测入口（差异化 #5）。

把同一题集跑过多个模式（结构化 / 自由文本 / 多 agent），打分出对比报告。
provider/模型由环境变量决定（见 README 的「模型 / provider 配置」），所以"换 provider
对比"= 换 OPS_QA_* 环境变量再跑一次。

用法：
    uv run python run_eval.py                                  # 默认跑 structured + free + multi
    uv run python run_eval.py --modes structured              # 只跑结构化
    uv run python run_eval.py --modes structured,multi --detail
    uv run python run_eval.py --cases path/to/cases.json --docs path/to/docs
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ops_qa_bot_oai.evaluate import (
    MODES,
    render_case_detail,
    render_report,
    run_eval,
)

_ROOT = Path(__file__).resolve().parent


async def _main(cases: Path, docs: Path, modes: list[str], detail: bool) -> None:
    results = await run_eval(cases_path=cases, docs_root=docs, modes=modes)
    print("\n========== 对比报告 ==========")
    print(render_report(results))
    if detail:
        print("\n========== 逐题明细 ==========")
        print(render_case_detail(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="ops-qa-bot-openai 离线评测")
    parser.add_argument("--cases", default=str(_ROOT / "eval" / "cases.json"), help="题集 JSON")
    parser.add_argument("--docs", default=str(_ROOT / "docs"), help="文档根目录")
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=f"逗号分隔的模式，可选 {'/'.join(MODES)}（默认全跑）",
    )
    parser.add_argument("--detail", action="store_true", help="额外打印逐题对错明细")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        parser.error(f"未知模式 {bad}，可选 {MODES}")

    asyncio.run(_main(Path(args.cases).resolve(), Path(args.docs).resolve(), modes, args.detail))


if __name__ == "__main__":
    main()
