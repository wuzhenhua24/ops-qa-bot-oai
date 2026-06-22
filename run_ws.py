#!/usr/bin/env python3
"""飞书长连接入口：在群里 @机器人 提问，复用 OpenAI Agents SDK 版 OpsQABot 答题。

前置：
    uv sync --extra feishu          # 装 lark-oapi
    export FEISHU_APP_ID=cli_xxx
    export FEISHU_APP_SECRET=xxx
    # 模型 / provider 走同一套 OPS_QA_* 环境变量（见 README）
    uv run python run_ws.py

飞书开放平台：事件订阅选「长连接」，订阅 im.message.receive_v1，开 im:message /
im:message.group_at_msg / im:message:send_as_bot 权限，发版审批通过后把机器人加进群。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="ops-qa-bot-openai 飞书长连接服务")
    parser.add_argument("--docs", default=str(_ROOT / "docs"), help="文档根目录")
    parser.add_argument("--idle-ttl", type=float, default=1800.0, help="会话空闲回收秒数")
    parser.add_argument("--max-turns", type=int, default=30, help="单轮答题步数上限")
    args = parser.parse_args()

    app_id = os.environ.get("FEISHU_APP_ID", "").strip()
    app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise SystemExit("请设置环境变量 FEISHU_APP_ID 和 FEISHU_APP_SECRET")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 延迟导入：未装 lark-oapi 时给出清晰提示而不是 ImportError 栈。
    try:
        from ops_qa_bot_oai.feishu import WsRunner
    except ModuleNotFoundError as e:
        if "lark_oapi" in str(e):
            raise SystemExit("缺少 lark-oapi，请先运行：uv sync --extra feishu") from e
        raise

    runner = WsRunner(
        app_id=app_id,
        app_secret=app_secret,
        docs_root=Path(args.docs).resolve(),
        idle_ttl=args.idle_ttl,
        max_turns=args.max_turns,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
