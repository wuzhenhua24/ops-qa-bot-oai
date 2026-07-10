"""feedback.log 离线统计：一条命令出"满意率 / 被踩问题 / 升级率 / 路由 / 成本"报表。

对齐 ops-qa-bot 的同款工具，并按本项目更丰富的遥测多出三块参考项目没有的维度：

- **路由分布**（route）：各组件专家/协调者各答了多少——分诊是否符合预期。
- **按 agent 用量**（agent_usage）：多模型路由省了多少，便宜分诊 vs 专家强模型各花多少。
- **二次复核**（reviewed/revised/needs_human_review）+ **缓存命中**（cached_input_tokens）。

纯 stdlib、只读日志文件，不依赖服务进程在跑。将来要做"周报推群"，cron + 本模块 +
一次 send_post 即可，聚合逻辑直接复用。

用法：
    uv run python -m ops_qa_bot_oai.feedback_stats                # 近 7 天
    uv run python -m ops_qa_bot_oai.feedback_stats --days 30      # 近 30 天
    uv run python -m ops_qa_bot_oai.feedback_stats --days 0       # 全量
    uv run python -m ops_qa_bot_oai.feedback_stats --log logs/feedback.log

用量口径以**纯 token 数**为主；只有显式传了 --price-input / --price-output
（$/1M tokens）才追加美元估算行（代理/套餐部署没有严格单价，默认不猜）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

DEFAULT_LOG = "logs/feedback.log"


def parse_log(path: Path) -> list[tuple[str, dict]]:
    """逐行解析 feedback.log → [(日期 'YYYY-MM-DD', 事件 dict), ...]。

    每行格式是 "时间戳前缀 + JSON"；前缀取前 10 个字符当日期。脏行（无 JSON /
    解析失败 / 没有 event 字段）静默跳过——统计工具不该因为一行损坏就罢工。
    """
    events: list[tuple[str, dict]] = []
    try:
        f = path.open("r", encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"读不到日志文件 {path}: {e}")
    with f:
        for line in f:
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                obj = json.loads(line[brace:])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "event" not in obj:
                continue
            events.append((line[:10], obj))
    return events


def filter_days(events: list[tuple[str, dict]], days: int, today: date) -> list[tuple[str, dict]]:
    """保留近 N 天（含今天）的事件；days<=0 表示全量。"""
    if days <= 0:
        return events
    cutoff = (today - timedelta(days=days - 1)).isoformat()
    return [(d, e) for d, e in events if d >= cutoff]


def aggregate(events: list[tuple[str, dict]]) -> dict:
    """聚合所有事件 → 报表数据 dict（render 与未来的周报推送共用）。"""
    qa_by_qid: dict[str, dict] = {}
    s: dict = {
        "qa_total": 0,
        "errors": 0,
        "cancelled": 0,
        "clarifications": 0,
        "escalated": 0,
        "max_turns_hit": 0,
        "images_answers": 0,
        "session_expired": 0,
        "reviewed": 0,
        "revised": 0,
        "needs_human_review": 0,
        "approvals_total": 0,
        "blacklist_rejections": 0,
        "route_counter": Counter(),
        "agent_tokens": {},  # agent → Counter(input/output/requests)
        "tokens": Counter(),  # input/output/cached/reasoning/requests
        "up": 0,
        "down": 0,
        "down_items": [],  # [{qid, question, reasons}]
        "reason_counter": Counter(),
        "reason_skipped": 0,
        "feedback_rejected": 0,
        "archives": 0,
        "archive_paths": Counter(),
        "archive_had_draft": 0,
        "archive_edited": 0,
        "active_users": set(),
        "days": Counter(),
    }
    down_reasons: dict[str, list[str]] = {}

    for day, e in events:
        ev = e.get("event")
        if ev == "qa":
            s["qa_total"] += 1
            s["days"][day] += 1
            if qid := e.get("qid"):
                qa_by_qid[qid] = e
            if uid := e.get("user_id"):
                s["active_users"].add(uid)
            if e.get("clarification"):
                s["clarifications"] += 1
            if e.get("escalated_to"):
                s["escalated"] += 1
            if e.get("subtype") == "error_max_turns":
                s["max_turns_hit"] += 1
            if e.get("images_attached"):
                s["images_answers"] += 1
            if e.get("session_expired"):
                s["session_expired"] += 1
            if e.get("reviewed"):
                s["reviewed"] += 1
            if e.get("revised"):
                s["revised"] += 1
            if e.get("needs_human_review"):
                s["needs_human_review"] += 1
            s["approvals_total"] += e.get("approvals") or 0
            s["blacklist_rejections"] += e.get("blacklist_rejections") or 0
            s["route_counter"][e.get("route") or "（入口自答）"] += 1
            usage = e.get("usage") or {}
            s["tokens"]["input"] += usage.get("input_tokens") or 0
            s["tokens"]["output"] += usage.get("output_tokens") or 0
            s["tokens"]["cached"] += usage.get("cached_input_tokens") or 0
            s["tokens"]["reasoning"] += usage.get("reasoning_tokens") or 0
            s["tokens"]["requests"] += usage.get("requests") or 0
            for agent, au in (e.get("agent_usage") or {}).items():
                c = s["agent_tokens"].setdefault(agent, Counter())
                c["input"] += au.get("input_tokens") or 0
                c["output"] += au.get("output_tokens") or 0
                c["requests"] += au.get("requests") or 0
        elif ev == "qa_error":
            s["errors"] += 1
        elif ev == "cancelled":
            s["cancelled"] += 1
        elif ev == "feedback":
            if e.get("rating") == "up":
                s["up"] += 1
            elif e.get("rating") == "down":
                s["down"] += 1
                s["down_items"].append({"qid": e.get("qid"), "question": None, "reasons": []})
        elif ev == "feedback_reason":
            if e.get("skipped"):
                s["reason_skipped"] += 1
            elif not e.get("invalid"):
                for label in e.get("reason_labels") or []:
                    s["reason_counter"][label] += 1
                if qid := e.get("qid"):
                    down_reasons.setdefault(qid, []).extend(e.get("reason_labels") or [])
        elif ev == "feedback_rejected":
            s["feedback_rejected"] += 1
        elif ev == "archive":
            if e.get("duplicate"):
                continue
            s["archives"] += 1
            if p := e.get("path"):
                s["archive_paths"][p] += 1
            if e.get("had_draft"):
                s["archive_had_draft"] += 1
            if e.get("question_edited"):
                s["archive_edited"] += 1

    # 被踩问题回填原题 + 原因（qa 事件按 qid 关联）
    for item in s["down_items"]:
        qa = qa_by_qid.get(item["qid"] or "")
        item["question"] = (qa or {}).get("question") or "（找不到对应 qa 记录）"
        item["reasons"] = down_reasons.get(item["qid"] or "", [])
    return s


def _pct(part: int, total: int) -> str:
    return f"{part / total * 100:.0f}%" if total else "-"


def render(
    s: dict,
    *,
    days: int,
    today: date,
    price_input: float | None = None,
    price_output: float | None = None,
) -> str:
    """聚合结果 → 人读的纯文本报表。传了单价（$/1M tokens）才追加美元估算。"""
    span = f"近 {days} 天" if days > 0 else "全量"
    total = s["qa_total"]
    lines = [f"📊 ops-qa-bot-oai 反馈统计（{span}，截至 {today.isoformat()}）", ""]

    lines.append(
        f"问答 {total} 轮（反问 {s['clarifications']}，升级 {s['escalated']}"
        f" = {_pct(s['escalated'], total)}，撞 max_turns {s['max_turns_hit']}，"
        f"带图 {s['images_answers']}，过期翻篇 {s['session_expired']}）"
    )
    lines.append(
        f"异常：出错 {s['errors']}，/cancel 取消 {s['cancelled']}；"
        f"活跃用户 {len(s['active_users'])} 人"
    )

    rated = s["up"] + s["down"]
    lines.append("")
    lines.append(
        f"反馈：👍 {s['up']} / 👎 {s['down']}（打分率 {_pct(rated, total)}，"
        f"满意率 {_pct(s['up'], rated)}；原因跳过 {s['reason_skipped']}，"
        f"非提问者点击被拒 {s['feedback_rejected']}）"
    )
    if s["reason_counter"]:
        top = "、".join(f"{label}×{n}" for label, n in s["reason_counter"].most_common())
        lines.append(f"被踩原因：{top}")
    for item in s["down_items"]:
        reasons = f"（{'、'.join(item['reasons'])}）" if item["reasons"] else ""
        lines.append(f"  👎 {item['question']}{reasons}")

    if s["route_counter"]:
        lines.append("")
        lines.append("路由分布：")
        for route, n in s["route_counter"].most_common():
            lines.append(f"  {route}: {n}（{_pct(n, total)}）")

    if s["reviewed"]:
        lines.append("")
        lines.append(
            f"二次复核：触发 {s['reviewed']} 轮，重答 {s['revised']}，"
            f"标记需人工复核 {s['needs_human_review']}"
        )
    if s["approvals_total"] or s["blacklist_rejections"]:
        lines.append(
            f"写审批：人工拍板 {s['approvals_total']} 次，"
            f"黑名单自动驳回 {s['blacklist_rejections']} 次"
        )
    if s["archives"]:
        paths = "、".join(f"{p}×{n}" for p, n in s["archive_paths"].most_common())
        lines.append(
            f"问答归档：{s['archives']} 条（{paths}；ARCHIVE_Q 预填 {s['archive_had_draft']}，"
            f"负责人改标题 {s['archive_edited']}）"
        )

    t = s["tokens"]
    lines.append("")
    cache_base = t["input"] + t["cached"]
    lines.append(
        f"用量：input {t['input']:,} + output {t['output']:,} tokens"
        f"（cached {t['cached']:,}，缓存命中 {_pct(t['cached'], cache_base)}；"
        f"reasoning {t['reasoning']:,}；{t['requests']} 次请求）"
    )
    if s["agent_tokens"]:
        lines.append("按 agent 用量（多模型路由的成本拆分）：")
        for agent, c in sorted(s["agent_tokens"].items(), key=lambda kv: -kv[1]["output"]):
            lines.append(
                f"  {agent}: in {c['input']:,} / out {c['output']:,}（{c['requests']} 次）"
            )
    if price_input is not None and price_output is not None:
        cost = (t["input"] * price_input + t["output"] * price_output) / 1_000_000
        lines.append(f"估算成本：${cost:.2f}（按 --price-* 单价，未计缓存折扣）")

    if s["days"]:
        lines.append("")
        lines.append("按日问答量：")
        for day in sorted(s["days"]):
            lines.append(f"  {day}: {s['days'][day]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="feedback.log 离线统计报表")
    ap.add_argument("--days", type=int, default=7, help="近 N 天（0=全量，默认 7）")
    ap.add_argument("--log", type=Path, default=Path(DEFAULT_LOG), help="日志文件路径")
    ap.add_argument("--price-input", type=float, default=None, help="input $/1M tokens")
    ap.add_argument("--price-output", type=float, default=None, help="output $/1M tokens")
    args = ap.parse_args(argv)

    today = date.today()
    events = filter_days(parse_log(args.log), args.days, today)
    s = aggregate(events)
    print(
        render(
            s,
            days=args.days,
            today=today,
            price_input=args.price_input,
            price_output=args.price_output,
        )
    )


if __name__ == "__main__":
    main()
