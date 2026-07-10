"""feedback_stats 离线统计的回归测试（合成日志文件，无需服务在跑）。

覆盖：
- parse_log：时间戳前缀取日期、脏行（无 JSON / 解析失败 / 无 event）静默跳过。
- filter_days：近 N 天窗口 / 0=全量。
- aggregate：满意率口径、被踩问题回填原题+原因、路由分布、按 agent 用量、
  缓存 token、复核计数、归档去重（duplicate 不计）、错误/取消计数。
- render：关键行都在；美元估算行仅在传了单价时出现。

跑法：
    uv run pytest tests/test_feedback_stats.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ops_qa_bot_oai.feedback_stats import aggregate, filter_days, parse_log, render

TODAY = date(2026, 7, 10)


def _line(day: str, obj: dict) -> str:
    return f"{day} 12:00:00,000 {json.dumps(obj, ensure_ascii=False)}"


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "feedback.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _sample_lines() -> list[str]:
    qa1 = {
        "event": "qa",
        "qid": "q1",
        "user_id": "u1",
        "question": "redis 内存告警怎么处理",
        "subtype": "success",
        "route": "redis_specialist",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cached_input_tokens": 600,
            "reasoning_tokens": 50,
            "requests": 3,
        },
        "agent_usage": {
            "triage": {"input_tokens": 100, "output_tokens": 10, "requests": 1},
            "redis_specialist": {"input_tokens": 900, "output_tokens": 190, "requests": 2},
        },
        "reviewed": True,
        "revised": True,
    }
    qa2 = {
        "event": "qa",
        "qid": "q2",
        "user_id": "u2",
        "question": "mysql 主从延迟",
        "subtype": "error_max_turns",
        "escalated_to": "ou_owner",
        "route": "mysql_specialist",
        "usage": {"input_tokens": 500, "output_tokens": 100, "requests": 2},
        "session_expired": True,
    }
    return [
        "2026-07-08 12:00:00 这行没有大括号，应被跳过",
        _line("2026-07-08", {"no_event": True}),
        "2026-07-08 12:00:00 {broken json",
        _line("2026-07-09", qa1),
        _line("2026-07-09", {"event": "feedback", "qid": "q1", "rating": "up"}),
        _line("2026-07-10", qa2),
        _line("2026-07-10", {"event": "feedback", "qid": "q2", "rating": "down"}),
        _line(
            "2026-07-10",
            {"event": "feedback_reason", "qid": "q2", "reason_labels": ["文档过时", "步骤不完整"]},
        ),
        _line("2026-07-10", {"event": "feedback_reason", "qid": "q9", "skipped": True}),
        _line("2026-07-10", {"event": "feedback_rejected", "qid": "q2"}),
        _line("2026-07-10", {"event": "qa_error", "qid": "q3"}),
        _line("2026-07-10", {"event": "cancelled", "qid": "q4"}),
        _line(
            "2026-07-10",
            {
                "event": "archive",
                "qid": "a1",
                "path": "redis/qa-archive.md",
                "had_draft": True,
                "question_edited": True,
            },
        ),
        _line(
            "2026-07-10",
            {"event": "archive", "qid": "a1", "path": "redis/qa-archive.md", "duplicate": True},
        ),
        # 很久以前的事件：--days 7 之外
        _line("2026-06-01", {"event": "qa", "qid": "q0", "user_id": "u0", "subtype": "success"}),
    ]


def test_parse_log_skips_dirty_lines(tmp_path):
    events = parse_log(_write_log(tmp_path, _sample_lines()))
    assert all("event" in e for _, e in events)
    assert len(events) == 12  # 3 条脏行被跳过
    assert events[0][0] == "2026-07-09"  # 日期取前缀前 10 字符


def test_filter_days_window(tmp_path):
    events = parse_log(_write_log(tmp_path, _sample_lines()))
    recent = filter_days(events, 7, TODAY)
    assert all(d >= "2026-07-04" for d, _ in recent)
    assert len(filter_days(events, 0, TODAY)) == 12  # 0=全量


def test_aggregate_core_metrics(tmp_path):
    events = filter_days(parse_log(_write_log(tmp_path, _sample_lines())), 7, TODAY)
    s = aggregate(events)
    assert s["qa_total"] == 2
    assert s["errors"] == 1 and s["cancelled"] == 1
    assert s["escalated"] == 1 and s["max_turns_hit"] == 1 and s["session_expired"] == 1
    assert s["reviewed"] == 1 and s["revised"] == 1
    assert s["active_users"] == {"u1", "u2"}
    assert s["up"] == 1 and s["down"] == 1 and s["reason_skipped"] == 1
    assert s["feedback_rejected"] == 1
    # 被踩问题回填原题 + 原因
    (item,) = s["down_items"]
    assert item["question"] == "mysql 主从延迟"
    assert item["reasons"] == ["文档过时", "步骤不完整"]
    # 路由 / 用量 / 缓存
    assert s["route_counter"]["redis_specialist"] == 1
    assert s["tokens"]["input"] == 1500 and s["tokens"]["cached"] == 600
    assert s["agent_tokens"]["redis_specialist"]["output"] == 190
    # 归档：duplicate 不计
    assert s["archives"] == 1 and s["archive_paths"]["redis/qa-archive.md"] == 1
    assert s["archive_had_draft"] == 1 and s["archive_edited"] == 1


def test_render_report_lines(tmp_path):
    events = filter_days(parse_log(_write_log(tmp_path, _sample_lines())), 7, TODAY)
    s = aggregate(events)
    out = render(s, days=7, today=TODAY)
    assert "问答 2 轮" in out
    assert "👍 1 / 👎 1" in out and "满意率 50%" in out
    assert "mysql 主从延迟" in out and "文档过时" in out
    assert "redis_specialist" in out and "路由分布" in out
    assert "二次复核" in out
    assert "问答归档：1 条" in out
    assert "$" not in out  # 没传单价不出美元行

    priced = render(s, days=7, today=TODAY, price_input=2.5, price_output=10.0)
    assert "估算成本：$" in priced


def test_render_empty_log(tmp_path):
    s = aggregate([])
    out = render(s, days=7, today=TODAY)
    assert "问答 0 轮" in out  # 空日志不崩
