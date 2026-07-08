# 部署到 Linux 服务器（systemd + 飞书长连接）

适用：飞书**长连接**模式（`run_ws.py`），单机部署，systemd 管控。长连接只出站、
**不需要公网 HTTPS 入口**，内网服务器即可跑，是最省事的上线方式。

与参考项目 `ops-qa-bot`（Claude Agent SDK 版）的差异：本项目基于 **OpenAI Agents SDK**，
配置全走**环境变量**（无 `config.toml`），**不需要 Node.js / claude CLI 登录**，模型鉴权
用 `OPENAI_API_KEY` 或第三方端点的 `OPS_QA_*`。所以「一次性准备」比参考项目更短。

---

## 一次性准备

```bash
# 1. 装 uv（非 root 即可；用于建 venv + 装依赖）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 建专用运行用户（不要用 root 跑）
sudo useradd --system --create-home --shell /bin/bash ops-bot

# 3. 准备代码目录
sudo mkdir -p /opt/ops-qa-bot-oai
sudo chown ops-bot:ops-bot /opt/ops-qa-bot-oai
sudo -u ops-bot git clone <仓库地址> /opt/ops-qa-bot-oai
cd /opt/ops-qa-bot-oai

# 4. 装依赖（用 ops-bot 身份装到项目本地 .venv/）。飞书接入必须带 --extra feishu。
#    若模型走 provider=anthropic 或 litellm，再加 --extra litellm。
sudo -u ops-bot bash -c '
    cd /opt/ops-qa-bot-oai
    ~ops-bot/.local/bin/uv sync --extra feishu
'

# 5. 建运行期可写目录（放会话 SQLite DB；systemd 里 ProtectSystem=strict 其余只读）
sudo -u ops-bot mkdir -p /opt/ops-qa-bot-oai/data
```

> `uv` 装好后在 `~/.local/bin/uv`。若已在全局 PATH，第 4 步直接 `uv sync ...` 即可。

---

## 配置（环境变量文件）

把配置（含 secret）放到 `/etc/`，权限收紧到只有 ops-bot 能读。这里用的是 **systemd
EnvironmentFile 格式**（`KEY=value` 每行一条，**不要** `export`、不要给值加引号）——
和仓库根的 `.env.example` 内容一一对应，照着挑需要的填即可。

```bash
sudo mkdir -p /etc/ops-qa-bot-oai
sudo tee /etc/ops-qa-bot-oai/ops-qa-bot-oai.env >/dev/null <<'EOF'
# ── 飞书应用凭证（必填）──
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx

# ── 模型 / provider（按 .env.example 选一套）──
# 官方 OpenAI：
OPS_QA_PROVIDER=openai
OPENAI_API_KEY=sk-...
# 或第三方 Chat Completions 端点（如智谱 GLM / 火山），改成：
#   OPS_QA_PROVIDER=compatible
#   OPS_QA_BASE_URL=https://open.bigmodel.cn/api/paas/v4
#   OPS_QA_API_KEY=<你的 key>
#   OPS_QA_MODEL=glm-4.6

# ── 答题模式（缺省 auto，一般不用改）──
# OPS_QA_MODE=auto

# ── 会话历史落盘（强烈建议生产设上，重启/回收不丢上下文）──
OPS_QA_SESSION_DB=/opt/ops-qa-bot-oai/data/.sessions.db

# ── 护栏 + 写操作审批（HITL，按需开）──
# OPS_QA_GUARDRAILS=1
# OPS_QA_APPROVERS=ou_xxx,ou_yyy          # 审批人白名单，生产务必设，别留空
# OPS_QA_APPROVAL_TIMEOUT=600
EOF

# 收权限（含 app_secret / api key，不能让其他用户看到）
sudo chown root:ops-bot /etc/ops-qa-bot-oai/ops-qa-bot-oai.env
sudo chmod 640 /etc/ops-qa-bot-oai/ops-qa-bot-oai.env
```

完整变量清单见项目根 `.env.example` 与 README 的「环境变量」表。

> 部署目录里**别再放 `.env`**：配置只留在上面这一处，避免两处不一致时排查困难
> （`run_ws.py` 的 `load_dotenv` 不覆盖已存在的环境变量，systemd 设的值优先，但留个
> 空 `.env` 容易误导人）。

**文档内容**：机器人答的是 `/opt/ops-qa-bot-oai/docs/` 下的文档 + `INDEX.md` 路由表。
仓库自带的是**演示文档**（redis/mysql/kafka/gateway/container），上线前**替换成你自己的
运维文档并维护 `INDEX.md`**，否则答的是示例内容。换目录用 `ExecStart` 加 `--docs`。

---

## 安装并启动 systemd 服务

```bash
sudo cp /opt/ops-qa-bot-oai/deploy/ops-qa-bot-oai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ops-qa-bot-oai

# 检查
sudo systemctl status ops-qa-bot-oai
sudo journalctl -u ops-qa-bot-oai -f      # 实时日志
```

启动成功后日志会回显答题模式、模型、会话历史落盘位置，例如：

```
答题模式：自适应分诊（单专家 / 跨组件协调）（模型 …）
会话历史：落盘 /opt/ops-qa-bot-oai/data/.sessions.db（重启/回收后可恢复上下文）
飞书长连接已启动，等待群内 @机器人 提问 …
```

**飞书开放平台配置**（企业自建应用）：事件订阅方式选「长连接」（不填 Request URL）；
订阅 `im.message.receive_v1`；开 `im:message`、`im:message.group_at_msg`、
`im:message:send_as_bot` 权限；发版审批通过后把机器人加进群。开了写审批（HITL）还需
卡片交互回调能收到，长连接模式下随事件订阅一起走，无需额外配置。

---

## 健康检查与监控

本项目**没有内置 HTTP 健康端点**（不同于参考项目的 `/healthz`）。长连接进程不监听任何
对外端口，监控用 systemd + journald 即可：

```bash
# liveness：进程在不在
sudo systemctl is-active ops-qa-bot-oai        # active / failed，给监控脚本用

# 最近有没有报错
sudo journalctl -u ops-qa-bot-oai -p err --since "10m ago" --no-pager
```

**P0-1 兜底已内建**：单条问题处理时若模型服务异常/超时/网络抖动，机器人会把占位消息
编辑成「⚠️ 出错了，请稍后重试」并记录堆栈到 journald，**不会**让用户对着「🔍 翻文档中」
干等。用 `journalctl -u ops-qa-bot-oai | grep "answer failed"` 可统计这类失败频率。

### 已知局限：检测不到「进程活着但 WS 重连一直失败」

systemd 只知道进程在不在，看不到底层 WS 连接状态。若 lark-oapi 的 WS 断开后陷在重连
失败循环（鉴权过期、网络分区、飞书侧拒绝），进程仍是 active，但实际收不到任何消息——
即「假活」。日志里会持续打 `ws reconnecting ...` 而没有 `ws reconnected`。生产上可以：

- 对 journald 里 `ws reconnecting` 持续出现（且长时间无 `ws reconnected`）做告警；
- 或写个 cron，工作时段内长时间（如 > 2h）无任何 `Q chat=` 提问日志就人工排查。
  **不要**拿「一段时间没提问」直接自动重启，否则夜里没人问就被反复杀。

### 单实例约束（重要）

**同一套 `FEISHU_APP_ID/SECRET` 只能起一个实例**。起两份会双份收事件、**重复回复**
（本项目无 leader 选举）。要多机高可用得自己在上层做互斥/主备，别简单地起两台。

---

## 常用运维操作

```bash
# 状态 / 重启 / 停止
sudo systemctl status ops-qa-bot-oai
sudo systemctl restart ops-qa-bot-oai
sudo systemctl stop ops-qa-bot-oai

# 日志
sudo journalctl -u ops-qa-bot-oai -f                 # 实时
sudo journalctl -u ops-qa-bot-oai --since "1h ago"   # 最近 1 小时
sudo journalctl -u ops-qa-bot-oai -p err             # 仅 ERROR

# 升级代码
sudo -u ops-bot bash -c '
    cd /opt/ops-qa-bot-oai
    git pull
    ~ops-bot/.local/bin/uv sync --extra feishu
'
sudo systemctl restart ops-qa-bot-oai
```

---

## 排错

**服务起不来（`systemctl status` 显示 failed）**
- 看 `journalctl -u ops-qa-bot-oai --no-pager` 最后 50 行
- 常见：EnvironmentFile 路径错、`FEISHU_APP_ID/SECRET` 没填、模型 key/端点填错、
  忘了 `uv sync --extra feishu`（报 `缺少 lark-oapi`）

**反复重启（触发 StartLimit 后停手）**
- `journalctl -u ops-qa-bot-oai -p err --since "10m ago"` 找根因
- 修完：`sudo systemctl reset-failed ops-qa-bot-oai && sudo systemctl restart ops-qa-bot-oai`

**机器人不回话**
- 确认应用已发版审批通过且**已加进群**；权限里有 `im:message.group_at_msg`
- 确认只起了**一个**实例（多实例会互相抢事件，表现为偶发不回或重复回）
- 看日志有没有 `Q chat=`（收到提问）与 `A chat=`（答完）；只有 Q 没有 A 多半是模型调用
  出错，`grep "answer failed"` 看堆栈

**答的是示例内容 / 答不对**
- `docs/` 还是仓库自带演示文档，没换成你的；换完 `--docs` 指过去并 restart

**会话上下文重启就丢**
- 没设 `OPS_QA_SESSION_DB`，或指向的目录不在 `ReadWritePaths` 里导致写不进去

---

## 路径不一致时改哪几行

想换路径（如装到 `/srv/...`），只改 `ops-qa-bot-oai.service` 里这几个字段：

- `User=` / `Group=`（运行身份）
- `WorkingDirectory=`（项目根）
- `EnvironmentFile=`（配置文件路径）
- `ExecStart=`（venv 里的 python + `run_ws.py` 绝对路径；换文档目录加 `--docs`）
- `ReadWritePaths=`（可写目录，至少包含 `OPS_QA_SESSION_DB` 所在目录）

改完 `sudo systemctl daemon-reload && sudo systemctl restart ops-qa-bot-oai`。
