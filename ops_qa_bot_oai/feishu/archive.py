"""问答归档：升级到负责人的问题，负责人填表单把答案沉淀进文档。

闭环（对齐 ops-qa-bot 的同款功能）：

  bot 答不出 → `<<ESCALATE:ou_xxx:dir>>` @ 负责人 + `<<ARCHIVE_Q:标题>>` →
  runner 登记 pending 并发归档表单卡（可编辑标题 + 答案框）→ 负责人填写提交 →
  写入 `docs/<dir>/qa-archive.md` → 原群 @ 提问者推送答案

点睛之处：qa-archive.md 就在组件文档目录里，bot 的文档检索天然能搜到——负责人
答一次 bot 就学会了，下次同样的问题直接命中归档、不再升级。

本模块是存储与业务层（pending 登记表 / 写盘 / 提交处理），卡片渲染在 render.py，
飞书 I/O 在 runner.py。纯逻辑可单测。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .feedback import log_event
from .render import build_archive_ack_card, build_archive_form_card, build_archive_notify_post

logger = logging.getLogger("ops_qa_bot_oai.feishu.archive")

DEFAULT_TTL = 24 * 3600.0  # 表单 24h 没人填就过期（重启也会丢——纯内存，与旧项目一致）
MAX_PENDING = 1000
ANSWER_MAX_LEN = 10_000

# 组件目录白名单模式：单层目录名，不含路径分隔符 / 不以点开头（挡 `..` 穿越）。
_DIR_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9._\-]*$")

# 每个归档文件一把锁，避免并发提交撕裂内容。按解析后的绝对路径共享。
_write_locks: dict[Path, asyncio.Lock] = {}


def safe_component_dir(docs_root: Path, dir_hint: str | None) -> str | None:
    """LLM 给的目录提示 → 校验过的组件目录名；不合法/不存在返回 None（落根目录归档）。

    目录来自 `<<ESCALATE:ou_xxx:dir>>` 标记（模型输出，不可信）：白名单字符 +
    必须是 docs_root 下真实存在的目录，否则宁可归到根目录 qa-archive.md 也不
    照单全收——防路径穿越，也防归进一个检索不到的幻觉目录。
    """
    if not dir_hint:
        return None
    d = dir_hint.strip().strip("/")
    if not d or not _DIR_RE.match(d):
        return None
    if not (docs_root / d).is_dir():
        return None
    return d


@dataclass
class PendingArchive:
    """一张在途归档表单的上下文（qid 定位）。"""

    qid: str
    chat_id: str
    asker_id: str
    question: str  # 用户原话（标题的最终 fallback + 日志对照）
    question_default: str  # 表单预填标题：ARCHIVE_Q 草稿，没有则用户原话
    owner_id: str
    component_dir: str | None  # 已过 safe_component_dir 校验；None 落根目录
    parent_msg_id: str | None  # 通知 asker 时引用回原提问消息
    created_at: float = field(default_factory=time.time)

    @property
    def archive_path_repr(self) -> str:
        return f"{self.component_dir}/qa-archive.md" if self.component_dir else "qa-archive.md"


class ArchiveStore:
    """在途归档表单登记表：qid → PendingArchive。纯内存 + TTL + 容量上限。"""

    def __init__(self, *, ttl: float = DEFAULT_TTL, max_pending: int = MAX_PENDING):
        self.ttl = ttl
        self.max_pending = max(1, max_pending)
        self._pending: dict[str, PendingArchive] = {}

    def register(
        self,
        *,
        chat_id: str,
        asker_id: str,
        question: str,
        question_default: str,
        owner_id: str,
        component_dir: str | None,
        parent_msg_id: str | None,
    ) -> PendingArchive:
        self._prune()
        # 容量保险丝：满了丢最老的（dict 按插入序）。日常体量远到不了。
        while len(self._pending) >= self.max_pending:
            oldest = next(iter(self._pending))
            self._pending.pop(oldest, None)
        rec = PendingArchive(
            qid=uuid.uuid4().hex[:12],
            chat_id=chat_id,
            asker_id=asker_id,
            question=question,
            question_default=question_default,
            owner_id=owner_id,
            component_dir=component_dir,
            parent_msg_id=parent_msg_id,
        )
        self._pending[rec.qid] = rec
        return rec

    def get(self, qid: str | None) -> PendingArchive | None:
        if not qid:
            return None
        rec = self._pending.get(qid)
        if rec is None:
            return None
        if time.time() - rec.created_at > self.ttl:
            self._pending.pop(qid, None)
            return None
        return rec

    def pop(self, qid: str) -> None:
        self._pending.pop(qid, None)

    def pending_count(self) -> int:
        return len(self._pending)

    def _prune(self) -> None:
        now = time.time()
        for k in [k for k, r in self._pending.items() if now - r.created_at > self.ttl]:
            self._pending.pop(k, None)


async def write_qa_archive(
    file_path: Path,
    *,
    qid: str,
    question: str,
    answer: str,
    owner_id: str,
    asker_id: str | None,
) -> bool:
    """append-only 写一条 Q&A。已存在同 qid 跳过返回 False，写入返回 True。

    每个 file_path 一把 asyncio.Lock，并发归档不会撕裂；幂等键是 `qid: <id>`
    字符串在文件里的存在与否，省得维护单独索引。
    """
    file_path = file_path.resolve()
    lock = _write_locks.setdefault(file_path, asyncio.Lock())
    async with lock:
        existing = ""
        if file_path.is_file():
            try:
                existing = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                logger.warning("read qa-archive failed: path=%s err=%s", file_path, e)
        if f"qid: {qid}" in existing:
            return False
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        meta_parts = [f"回答者：<@{owner_id}>", ts, f"qid: {qid}"]
        if asker_id:
            meta_parts.append(f"提问者：<@{asker_id}>")
        # 问题作为 `## ` 标题写一行：折叠掉换行/多余空白，否则会把 markdown 标题撑坏。
        question_line = " ".join(question.split()) or "（无标题）"
        block = (
            f"\n## Q: {question_line}\n\n*{' · '.join(meta_parts)}*\n\n{answer.strip()}\n\n---\n"
        )
        with file_path.open("a", encoding="utf-8") as f:
            f.write(block)
        return True


# 通知任务：(chat_id, post 负载, parent_msg_id)。runner 拿到后发消息。
NotifyTask = tuple[str, dict, "str | None"]


async def handle_archive_submit(
    store: ArchiveStore,
    docs_root: Path,
    *,
    qid: str | None,
    question: str,
    answer: str,
    clicker_id: str | None,
) -> tuple[dict, NotifyTask | None]:
    """处理归档表单提交。返回 (替换原表单卡的卡片, 通知 asker 的任务或 None)。

    `question` 是表单里那个（预填、可编辑）问题框的值——负责人没改就是预填值，
    改了就是改后的；异常路径（API 重放等）为空时按 question_default → 原话兜底。

    多数失败路径（参数缺失、过期、空答案、超长、写盘异常）用 ack 卡告诉点击者，
    原卡片被替换避免重复提交困惑。**唯一例外是"非负责人点击"**：返回重建的表单卡
    保持可见，让真正的负责人还能填——否则别人误点提交会把负责人的表单顶掉。
    """
    if not qid:
        return build_archive_ack_card("⚠️", "归档参数缺失，请联系管理员。"), None
    rec = store.get(qid)
    if rec is None:
        return (
            build_archive_ack_card("⏰", "归档会话已过期或已处理，请联系管理员手动补记。"),
            None,
        )

    if clicker_id and clicker_id != rec.owner_id:
        logger.info(
            "archive submit rejected (not owner): qid=%s by=%s expected=%s",
            qid,
            clicker_id,
            rec.owner_id,
        )
        return (
            build_archive_form_card(
                rec.qid, rec.question_default, rec.owner_id, rec.archive_path_repr
            ),
            None,
        )

    answer_text = (answer or "").strip()
    if not answer_text:
        return build_archive_ack_card("⚠️", "答案不能为空，请填写后再提交。"), None
    if len(answer_text) > ANSWER_MAX_LEN:
        return build_archive_ack_card("⚠️", "答案过长（>10KB），请精简后再提交。"), None

    # 表单"问题"框是 required，正常路径拿到的就是负责人确认/改过的值；为空按
    # question_default → 原话兜底，折叠成一行并截断（最终写成 `## Q: ...`）。
    question_text = " ".join(((question or "").strip() or rec.question_default).split())
    if not question_text:
        question_text = "（无标题）"
    if len(question_text) > 200:
        question_text = question_text[:200].rstrip() + "…"

    if rec.component_dir:
        file_path = docs_root / rec.component_dir / "qa-archive.md"
    else:
        file_path = docs_root / "qa-archive.md"

    try:
        wrote = await write_qa_archive(
            file_path,
            qid=qid,
            question=question_text,
            answer=answer_text,
            owner_id=rec.owner_id,
            asker_id=rec.asker_id,
        )
    except Exception:
        logger.exception("archive write failed: qid=%s path=%s", qid, file_path)
        return build_archive_ack_card("❌", "归档写入失败，请稍后重试或联系管理员。"), None

    # 完成（写入或幂等命中）：从 pending 清掉，避免重复处理。
    store.pop(qid)
    logger.info("archived: qid=%s path=%s wrote=%s", qid, rec.archive_path_repr, wrote)
    # 事件落 feedback.log：had_draft = 表单预填用的是 ARCHIVE_Q 草稿（非用户原话）；
    # question_edited = 负责人改过标题。两者是"LLM 标题质量"的观察抓手——草稿率高
    # 且改动率低说明 ARCHIVE_Q 契约在起作用。
    log_event(
        "archive",
        qid=qid,
        chat_id=rec.chat_id,
        path=rec.archive_path_repr,
        owner_id=rec.owner_id,
        duplicate=True if not wrote else None,
        had_draft=True if rec.question_default != rec.question else None,
        question_edited=(True if question_text != " ".join(rec.question_default.split()) else None),
    )

    # 仅在真写入时通知 asker——幂等命中说明同 qid 已存档过、asker 多半也通知过了。
    notify: NotifyTask | None = None
    if wrote and rec.asker_id:
        notify = (
            rec.chat_id,
            build_archive_notify_post(
                asker_id=rec.asker_id,
                owner_id=rec.owner_id,
                question=question_text,
                answer_markdown=answer_text,
                archive_rel=rec.archive_path_repr,
            ),
            rec.parent_msg_id,
        )
    ack = build_archive_ack_card(
        "✅", f"已归档进 `{rec.archive_path_repr}`，并把答案推送给了提问者。感谢沉淀！"
    )
    return ack, notify
