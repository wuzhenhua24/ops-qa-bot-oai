"""飞书消息渲染：问题清洗 + markdown→post + @ 段拼装。

纯逻辑（除 markdown→post 调 lark-oapi 外都无外部依赖），便于单测。markdown 解析
委托给 lark-oapi 的 `markdown_to_post_ast`（覆盖全、跟着 SDK 演进），本模块只负责
把答案正文包成飞书 post，并按需在头部 @ 提问者、在升级时 @ 负责人。
"""

from __future__ import annotations

from typing import Any

# 会话重置关键词（与 CLI 对齐）。
RESET_WORDS = {"/reset", "/new", "新对话", "重置"}
# 跟进任务管理指令：列出自己挂起的定时跟进（带取消按钮）。匹配前先 lower。
TASKS_WORDS = {"/tasks", "跟进任务"}
# 取消在途提问：停掉自己正在处理/排队中的提问（发错了不用干等）。匹配前先 lower。
CANCEL_WORDS = {"/cancel", "取消"}
# 帮助指令：能力清单 + 指令说明。匹配前先 lower（"HELP"/"Help" 也认）。
HELP_WORDS = {"/help", "help", "帮助"}


def clean_question(raw_text: str, mention_keys: list[str]) -> str:
    """从入站文本里剥掉 @bot 占位符（mentions[].key），返回纯问题文本。

    飞书把 @机器人 在原始文本里表示成 `@_user_1` 之类的占位 key；逐个 replace 掉。
    """
    q = raw_text or ""
    for key in mention_keys:
        if key:
            q = q.replace(key, "")
    return q.strip()


def escalate_open_id(payload: str | None) -> str | None:
    """从 ESCALATE marker 负载里抠出负责人 open_id。

    负载形如 `ou_xxx:redis`（owner:目录）或 `none`/空。返回 `ou_xxx` 或 None。
    """
    if not payload:
        return None
    head = payload.split(":", 1)[0].strip()
    return head if head.startswith("ou_") else None


def escalate_dir(payload: str | None) -> str | None:
    """从 ESCALATE marker 负载里抠出组件目录提示（`ou_xxx:redis` 的 `redis`）。

    只做拆分不做校验——目录是 LLM 输出的，落盘前必须过 archive.safe_component_dir
    的白名单校验（防路径穿越 / 不存在的目录）。没带目录或 `none` 返回 None。
    """
    if not payload:
        return None
    parts = payload.split(":", 1)
    if len(parts) < 2:
        return None
    d = parts[1].strip()
    return d or None


def _render_body(markdown: str) -> list[list[dict[str, Any]]]:
    """markdown → 飞书 post 段落列表（委托 lark-oapi）。"""
    from lark_oapi.channel.outbound.markdown import markdown_to_post_ast

    ast = markdown_to_post_ast(markdown or "")
    content = ast.get("zh_cn", {}).get("content") or []
    return content if content else [[{"tag": "text", "text": markdown or ""}]]


def build_answer_post(
    markdown: str,
    *,
    asker_id: str | None = None,
    escalate_to: str | None = None,
    title: str = "",
) -> dict[str, Any]:
    """把答案 markdown 包成飞书 post：头部 @提问者，升级时末尾 @负责人。

    返回结构可直接作为 `channel.send/edit_message` 的 `{"post": ...}` 负载。
    """
    paragraphs: list[list[dict[str, Any]]] = []
    if asker_id:
        # 头部 @ 提问者 + 一个空格，群里多人并行提问时一眼看出归属。
        paragraphs.append([{"tag": "at", "user_id": asker_id}, {"tag": "text", "text": " "}])
    paragraphs.extend(_render_body(markdown))
    if escalate_to:
        paragraphs.append(
            [
                {"tag": "text", "text": "📌 文档未覆盖，已请 "},
                {"tag": "at", "user_id": escalate_to},
                {"tag": "text", "text": " 协助处理。"},
            ]
        )
    return {"zh_cn": {"title": title, "content": paragraphs}}


def placeholder_text(question: str, *, queued: bool = False) -> str:
    """占位消息文案：收到即发，答完编辑替换。

    queued=True 表示同用户前一条问题还没答完、本条在排队等锁，前缀用「🕒 排队中」
    让用户分辨哪条占位真的在跑；拿到锁开始答题时 runner 会再刷一次成「🔍 翻文档中」
    （见 SessionManager.answer 的 on_start 回调）。
    """
    snippet = question.strip().replace("\n", " ")
    if len(snippet) > 24:
        snippet = snippet[:24] + "…"
    icon = "🕒 排队中" if queued else "🔍 翻文档中"
    return f"{icon}：'{snippet}'"


# ---------------------------------------------------------------------------
# 写操作审批卡片（HITL）：提议 → 待批卡片（带按钮）→ 拍板后改成结果卡片
# ---------------------------------------------------------------------------


def _approval_fields(command: str, target: str, reason: str) -> dict[str, Any]:
    md = f"**目标**：{target}\n**命令**：`{command}`\n**理由**：{reason}"
    return {"tag": "div", "text": {"tag": "lark_md", "content": md}}


def build_approval_card(
    *, approval_id: str, command: str, target: str, reason: str, asker_id: str | None = None
) -> dict[str, Any]:
    """待批卡片：命令详情 + 批准/驳回按钮。按钮 value 带 approval_id + decision，
    cardAction 回调据此定位待批项。纯函数，可单测。"""
    elements: list[dict[str, Any]] = []
    if asker_id:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"提问者：<at id={asker_id}></at>"},
            }
        )
    elements.append(_approval_fields(command, target, reason))
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "✅ 批准"},
                    "type": "primary",
                    "value": {"aid": approval_id, "decision": "approve"},
                },
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "🚫 驳回"},
                    "type": "danger",
                    "value": {"aid": approval_id, "decision": "reject"},
                },
            ],
        }
    )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {"tag": "plain_text", "content": "⚠️ 写操作审批：agent 提议执行变更"},
        },
        "elements": elements,
    }


def build_approval_result_card(
    *,
    command: str,
    target: str,
    reason: str,
    approved: bool,
    operator_name: str = "",
    note: str = "",
) -> dict[str, Any]:
    """结果卡片：拍板/超时后替换待批卡片（按钮移除，防重复点击）。纯函数，可单测。"""
    verdict = "✅ 已批准（登记为待人工执行）" if approved else "🚫 已驳回"
    who = f"，由 {operator_name} 拍板" if operator_name else ""
    lines = f"{verdict}{who}。" + (f"\n{note}" if note else "")
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green" if approved else "red",
            "title": {"tag": "plain_text", "content": "写操作审批结果"},
        },
        "elements": [
            _approval_fields(command, target, reason),
            {"tag": "div", "text": {"tag": "lark_md", "content": lines}},
        ],
    }


def _card_value_dict(value: Any) -> dict | None:
    """cardAction 按钮 value 归一成 dict：飞书可能原样回传 dict，也可能是 JSON 字符串。"""
    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def parse_card_action_value(value: Any) -> tuple[str, bool] | None:
    """从 cardAction 的按钮 value 解析 (approval_id, 是否批准)；不是审批按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    aid = value.get("aid")
    decision = value.get("decision")
    if not aid or decision not in ("approve", "reject"):
        return None
    return str(aid), decision == "approve"


# ---------------------------------------------------------------------------
# 定时跟进：/tasks 任务列表卡（每条带取消按钮）
# ---------------------------------------------------------------------------


def _task_excerpt(task: str, limit: int = 80) -> str:
    task = " ".join(task.split())
    return task if len(task) <= limit else task[:limit] + "…"


def build_followup_tasks_card(
    asker_id: str,
    chat_id: str,
    items: list[dict[str, Any]],
    *,
    notice: str | None = None,
) -> dict[str, Any]:
    """/tasks 的跟进任务列表卡：每条挂起跟进一行状态 + 一颗取消按钮。

    `items` 每项：record_id / task / remaining_minutes / firing。执行中（firing）
    的条目只展示状态、不给取消按钮——那一轮已经在跑了。`notice` 是取消回调刷新
    卡片时挂在顶部的结果行（"✅ 已取消…"）；列表为空时整卡收尾成纯文本。
    asker-only 校验在回调侧做，按钮 value 带回 record_id 与归属。纯函数，可单测。
    """
    elements: list[dict[str, Any]] = []
    if notice:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": notice}})
    if not items:
        elements.append(
            {"tag": "div", "text": {"tag": "lark_md", "content": "当前没有挂起的定时跟进了。"}}
        )
    else:
        head = f"⏰ 你有 **{len(items)}** 个挂起的定时跟进："
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": head}})
        for i, item in enumerate(items, 1):
            if item.get("firing"):
                status = "⏳ 正在执行，结果马上发出"
            else:
                m = int(item.get("remaining_minutes", 0))
                status = f"约 {m} 分钟后执行" if m > 0 else "即将执行（不足 1 分钟）"
            elements.append(
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**{i}.** {status}\n{_task_excerpt(str(item.get('task', '')))}",
                    },
                }
            )
            if not item.get("firing"):
                elements.append(
                    {
                        "tag": "action",
                        "actions": [
                            {
                                "tag": "button",
                                "text": {"tag": "plain_text", "content": "🗑 取消这条跟进"},
                                "type": "default",
                                "value": {
                                    "fua": str(item.get("record_id", "")),
                                    "chat": chat_id,
                                    "asker": asker_id,
                                },
                            }
                        ],
                    }
                )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "⏰ 定时跟进任务"},
        },
        "elements": elements,
    }


def parse_followup_cancel_value(value: Any) -> tuple[str, str, str] | None:
    """从 cardAction 按钮 value 解析 (record_id, chat_id, asker_id)；不是取消按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    rid, chat, asker = value.get("fua"), value.get("chat"), value.get("asker")
    if not rid or not chat or not asker:
        return None
    return str(rid), str(chat), str(asker)


# ---------------------------------------------------------------------------
# 问答归档：升级后发给负责人的表单卡 + 提交回执卡 + "已答复"通知 post
# ---------------------------------------------------------------------------


def _excerpt(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def build_archive_form_card(
    qid: str,
    question_default: str,
    owner_id: str,
    archive_path_repr: str,
) -> dict[str, Any]:
    """归档表单卡（card v2 form）：可编辑问题标题 + 多行答案输入框 + 提交按钮。

    question_default：预填进"问题"输入框的标题——优先是答题那轮 LLM 给的归一化
    标题（ARCHIVE_Q），否则是用户原话。负责人可改成更通用的说法再提交；最终写盘
    用框里的值。archive_path_repr：展示给负责人的相对路径（如 "redis/qa-archive.md"），
    让他知道答案会落到哪个文件。提交按钮 value 带 {"aq": qid}，cardAction 回调据此
    识别归档提交（区别于审批的 aid / 跟进取消的 fua）。纯函数，可单测。
    """
    intro = (
        f"<at id={owner_id}></at> 下面的「问题」是系统自动整理的，"
        "可改成更通用的说法（它会作为归档标题和以后的检索关键词）；"
        "把整理过的答案填进答案框，"
        f"提交后会追加进 `{archive_path_repr}`，下次同样的问题 bot 就能直接答。"
    )
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "📝 问答归档"},
            "template": "blue",
        },
        "body": {
            "elements": [
                {"tag": "markdown", "content": intro},
                {
                    "tag": "form",
                    "name": "archive_form",
                    "elements": [
                        {
                            "tag": "input",
                            "name": "question",
                            "default_value": _excerpt(question_default, 100),
                            "max_length": 120,
                            "placeholder": {
                                "tag": "plain_text",
                                "content": "归档用的问题标题（可修订）…",
                            },
                            "required": True,
                        },
                        {
                            "tag": "input",
                            "name": "answer",
                            "input_type": "multiline_text",
                            "rows": 6,
                            "max_length": 1000,
                            "placeholder": {
                                "tag": "plain_text",
                                "content": "粘贴整理后的答案文本（最多 1000 字）…",
                            },
                            "required": True,
                        },
                        {
                            "tag": "button",
                            "name": "submit_btn",
                            "text": {"tag": "plain_text", "content": "提交并归档"},
                            "type": "primary",
                            "form_action_type": "submit",
                            "behaviors": [{"type": "callback", "value": {"aq": qid}}],
                        },
                    ],
                },
            ]
        },
    }


def build_archive_ack_card(icon: str, message: str) -> dict[str, Any]:
    """提交后用来替换原表单卡的提示卡（card v2，纯文本）。"""
    return {
        "schema": "2.0",
        "body": {"elements": [{"tag": "markdown", "content": f"{icon} {message}"}]},
    }


def build_archive_notify_post(
    *,
    asker_id: str,
    owner_id: str,
    question: str,
    answer_markdown: str,
    archive_rel: str,
) -> dict[str, Any]:
    """ "负责人已答复"通知 post：@ 提问者把归档答案推回原群（闭环交付）。

    asker_id 放第一段以 @ 推送（不 @ 的话 asker 永远不知道负责人答了）；答案正文
    走 markdown 渲染保留列表/代码块结构；末尾告知归档路径（bot 已能检索到）。
    """
    paragraphs: list[list[dict[str, Any]]] = [
        [
            {"tag": "at", "user_id": asker_id},
            {"tag": "text", "text": " 📣 你之前的问题负责人已答复："},
        ],
        [{"tag": "text", "text": f"Q: {question}"}],
    ]
    paragraphs.extend(_render_body(answer_markdown))
    paragraphs.append(
        [
            {"tag": "text", "text": f"（已归档进 `{archive_rel}`，答复人 "},
            {"tag": "at", "user_id": owner_id},
            {"tag": "text", "text": "；下次同样的问题可以直接问我）"},
        ]
    )
    return {"zh_cn": {"title": "", "content": paragraphs}}


def parse_archive_submit_value(value: Any) -> str | None:
    """从 cardAction 按钮 value 解析归档提交的 qid；不是归档按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    qid = value.get("aq")
    return str(qid) if qid else None


def extract_form_value(raw: Any) -> dict:
    """从 cardAction 事件的 raw envelope 抽 form_value。

    channel 的 CardActionPayload 只暴露按钮 payload（action.value），form 元素
    （input）的填写结果在 envelope 内部：raw["event"]["action"]["form_value"]。
    """
    try:
        return raw["event"]["action"].get("form_value") or {}
    except (KeyError, TypeError, AttributeError):
        return {}


# ---------------------------------------------------------------------------
# 使用者反馈：答完随答案发 👍/👎 卡；👎 换成原因表单
# ---------------------------------------------------------------------------

# 👎 原因白名单（value → 展示标签）。表单选项和提交过滤共用这一份（过滤在
# feedback.handle_feedback_reason_submit）。放本模块避免 feedback ↔ render 循环导入。
FEEDBACK_REASONS: dict[str, str] = {
    "outdated": "文档过时",
    "incomplete": "步骤不完整",
    "incorrect": "事实错误",
    "verbose": "答案啰嗦 / 没重点",
    "other": "其他",
}


def _feedback_btn(label: str, btn_type: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": btn_type,
        "behaviors": [{"type": "callback", "value": value}],
    }


def build_feedback_card(qid: str, asker_id: str | None) -> dict[str, Any]:
    """答完随答案发的反馈卡：纯 👍/👎 两颗按钮（card v2）。

    用 v2 schema：👎 后要替换成带 form 的原因表单（form 是 v2 才有），原卡和
    替换卡 schema 不一致飞书渲染会失败。v2 没有 `tag:action` 容器，按钮放进
    column_set 并排。按钮 value 键 `fb`（与审批 aid / 归档 aq / 跟进 fua 区分），
    asker_id 随 value 带回做 asker-only 校验，不依赖服务端状态。纯函数，可单测。
    """
    common = {"fb": qid, "asker": asker_id or ""}
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "body": {
            "elements": [
                {
                    "tag": "column_set",
                    "columns": [
                        {
                            "tag": "column",
                            "width": "auto",
                            "elements": [
                                _feedback_btn("👍 有帮助", "primary", {**common, "rating": "up"})
                            ],
                        },
                        {
                            "tag": "column",
                            "width": "auto",
                            "elements": [
                                _feedback_btn("👎 待改进", "default", {**common, "rating": "down"})
                            ],
                        },
                    ],
                }
            ]
        },
    }


def build_feedback_reason_card(qid: str, asker_id: str | None) -> dict[str, Any]:
    """👎 后替换原卡的原因收集表单（card v2 form）。

    multi_select_static 多选原因 + 多行 input 备注（可选）+ 提交按钮；「跳过」放
    form 外（form 内非 submit 按钮行为不明确）。提交回调 value 键 `fbr` +
    kind=submit/skip，字段值从事件的 form_value 里取（见 extract_form_value）。
    """
    options = [
        {"text": {"tag": "plain_text", "content": label}, "value": value}
        for value, label in FEEDBACK_REASONS.items()
    ]
    common = {"fbr": qid, "asker": asker_id or ""}
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": "想了解一下这次回答哪里需要改进，方便我们补文档 / 调优：",
                },
                {
                    "tag": "form",
                    "name": "fb_reason",
                    "elements": [
                        {
                            "tag": "multi_select_static",
                            "name": "reasons",
                            "placeholder": {"tag": "plain_text", "content": "选择原因（可多选）…"},
                            "options": options,
                        },
                        {
                            "tag": "input",
                            "name": "comment",
                            "input_type": "multiline_text",
                            "rows": 2,
                            "max_length": 500,
                            "placeholder": {"tag": "plain_text", "content": "补充说明（可选）…"},
                            "required": False,
                        },
                        {
                            "tag": "button",
                            "name": "fb_submit",
                            "text": {"tag": "plain_text", "content": "提交"},
                            "type": "primary",
                            "form_action_type": "submit",
                            "behaviors": [
                                {"type": "callback", "value": {**common, "kind": "submit"}}
                            ],
                        },
                    ],
                },
                _feedback_btn("跳过", "default", {**common, "kind": "skip"}),
            ]
        },
    }


def build_feedback_ack_card(rating: str) -> dict[str, Any]:
    """点击/提交后的最终 ack 卡（按钮移除，防重复点击）。"""
    msg = "🙏 收到，感谢反馈！" if rating == "up" else "🙏 已记录，我们会据此补文档 / 调优。"
    return {
        "schema": "2.0",
        "body": {"elements": [{"tag": "markdown", "content": msg}]},
    }


def parse_feedback_value(value: Any) -> tuple[str, str, str | None] | None:
    """从按钮 value 解析 (qid, rating, asker_id)；不是反馈按钮返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    qid, rating = value.get("fb"), value.get("rating")
    if not qid or rating not in ("up", "down"):
        return None
    return str(qid), str(rating), str(value.get("asker") or "") or None


def parse_feedback_reason_value(value: Any) -> tuple[str, str, str | None] | None:
    """从按钮 value 解析 (qid, kind, asker_id)，kind ∈ submit/skip；认不出返回 None。"""
    value = _card_value_dict(value)
    if value is None:
        return None
    qid, kind = value.get("fbr"), value.get("kind")
    if not qid or kind not in ("submit", "skip"):
        return None
    return str(qid), str(kind), str(value.get("asker") or "") or None


# ---------------------------------------------------------------------------
# /help：能力清单 + 指令说明
# ---------------------------------------------------------------------------


def build_help_text(
    components: list[str],
    *,
    diag: bool = False,
    gw_trace: bool = False,
    db: bool = False,
    db_change: bool = False,
    followup: bool = False,
    idle_minutes: int = 30,
) -> str:
    """/help 的能力清单文案（markdown）。按实际启用的可选特性动态拼——没启用的
    不出现，避免新人照着帮助试了个被关掉的功能。组件清单来自 INDEX.md（解析
    失败传空列表即可，不列）。纯函数，可单测。
    """
    lines: list[str] = [
        "👋 我是运维问答机器人，**@ 我 + 问题** 即可提问。",
        "",
        "**我能做什么**",
    ]
    comp_suffix = f"（当前覆盖：{'、'.join(components)}）" if components else ""
    lines.append(f"- 📚 文档问答：基于运维文档库回答组件问题{comp_suffix}")
    lines.append("- 🖼️ 看图提问：直接发报错弹窗/监控面板等截图，可附文字说明")
    if diag:
        lines.append(
            "- 🖥️ 实时诊断：带上机器 IP/名字问「现在内存/连接数/load 怎么样」，"
            "我会连到测试环境跑只读命令看实况（永不执行写操作；生产环境不支持）"
        )
    if gw_trace:
        lines.append(
            "- 🔗 网关链路排查：访问失败时把响应头里的 Hi-Trace-Id 发我，"
            "我帮你查链路日志定位哪一跳出的问题"
        )
    if db:
        lines.append(
            "- 🗄️ 数据库实时分析：给出测试库 IP/端口（OceanBase 还需租户），"
            "我用只读账号帮你查 CPU/连接数/慢查询"
        )
    if db_change:
        lines.append("- 🔧 参数变更申请：可帮你发起数据库参数修改，值班人在群里点批准后才会执行")
    if followup:
        lines.append(
            "- ⏰ 定时跟进：「20 分钟后帮我看看 XX 完成没」，到点我自动复查并 @ 你；"
            "发 `/tasks`（或 跟进任务）可查看并取消你挂起的跟进"
        )
    lines.append("")
    lines.append(
        "**答不上来时**：我会自动 @ 对应组件负责人协助；负责人在归档卡里填的答案会"
        "沉淀回文档库，下次同样的问题我就能直接答。"
    )
    lines.append("**答完之后**：可以点答案下方的 👍/👎 给这次回答打分，帮我们持续改进。")
    lines.append("")
    lines.append("**指令**")
    lines.append("- `/reset`（或 `/new`、新对话、重置）：清空你的对话历史，开新会话")
    lines.append("- `/cancel`（或 取消）：停掉你正在处理/排队中的提问（发错了不用干等）")
    if followup:
        lines.append("- `/tasks`（或 跟进任务）：管理你挂起的定时跟进")
    lines.append("- `/help`（或 帮助）：显示本帮助")
    lines.append("")
    lines.append(f"💡 同一会话里可以直接追问；{idle_minutes} 分钟没动静上下文会自动过期。")
    return "\n".join(lines)
