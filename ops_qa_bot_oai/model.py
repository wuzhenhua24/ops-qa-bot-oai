"""模型 / provider 解析：运行时在 OpenAI 原生与兼容代理(含 Claude)之间切换。

OpenAI Agents SDK 是 provider 无关的，本模块把"用哪个模型"收敛成几个环境变量，
让同一套 agent 代码既能跑 OpenAI 原生模型，也能经 OpenAI 兼容代理 / LiteLLM 调
**和 ops-qa-bot 同一个 Claude 模型**——后者是做"纯 SDK 框架差异"对比时的关键：
把模型变量固定住，剩下的差异才是两个 agent 框架本身的。

环境变量：

- `OPS_QA_PROVIDER`：`openai`（默认）| `responses` | `compatible` | `anthropic` | `litellm`
- `OPS_QA_MODEL`：模型名。缺省值随 provider 变（见 _DEFAULT_MODEL）。
- `OPS_QA_BASE_URL`：第三方端点地址。`responses`/`compatible` 必填；`anthropic`/`litellm`
  可选（指向自建代理，省略则打 provider 官方地址）。
- `OPS_QA_API_KEY`：第三方端点的 key；`openai` 走标准 `OPENAI_API_KEY`，`anthropic`
  省略时回退 LiteLLM 读 `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`。
- `OPS_QA_ANTHROPIC_AUTH`：仅 `anthropic` 用。`x-api-key`（默认，Anthropic 官方风格）
  或 `bearer`（发 `Authorization: Bearer`，适配火山引擎等要 Bearer 的第三方网关）。

按"第三方端点说什么协议"选 provider：

- `openai`：OpenAI 官方端点。模型名字符串直接交给 Agent，SDK 用默认 client
  （读 `OPENAI_API_KEY`）。**SDK 默认就走新的 Responses API（`/v1/responses`）**，所以
  官方这条路本身用的就是 Responses 协议。对比的是"OpenAI SDK + OpenAI 模型"整套产品。
- `responses`：**Responses 协议**第三方端点（暴露 `/v1/responses`）。用
  `OpenAIResponsesModel` + 自建 `AsyncOpenAI(base_url, api_key)` 打过去。
- `compatible`：**Chat Completions 协议**第三方端点（暴露 `/v1/chat/completions`）。用
  `OpenAIChatCompletionsModel` + 自建 `AsyncOpenAI(base_url, api_key)` 打过去。
- `anthropic`：**Anthropic 协议**第三方端点（暴露 `/v1/messages`，即 ops-qa-bot 的
  Claude Agent SDK 用的那种代理）。经 SDK 的 `LitellmModel` 走 LiteLLM 的 anthropic
  provider，自动补 `anthropic/` 前缀；`OPS_QA_BASE_URL` 透传成 LiteLLM 的 base_url。
- `litellm`：通用兜底。直连 LiteLLM 支持的任意 provider，模型名要带 provider 前缀
  （如 `anthropic/claude-...`、`openrouter/anthropic/claude-...`、`gemini/...`），
  `OPS_QA_BASE_URL` 可选透传。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agents import Model, OpenAIChatCompletionsModel, OpenAIResponsesModel

_DEFAULT_MODEL = {
    # openai 官方默认就走 Responses API（SDK 的 _use_responses_by_default=True）。
    "openai": "gpt-5",
    # responses：第三方 Responses 协议端点（/v1/responses），默认给个 OpenAI 风格模型名。
    "responses": "gpt-5",
    # 兼容代理 / Anthropic / LiteLLM 默认指向一个 Claude 模型，方便和 ops-qa-bot
    # 对齐。实际值以你的代理/账号支持的为准，用 OPS_QA_MODEL 覆盖。
    # anthropic provider 会自动补 `anthropic/` 前缀，所以这里不带前缀。
    "compatible": "claude-opus-4-8",
    "anthropic": "claude-opus-4-8",
    "litellm": "anthropic/claude-opus-4-8",
}


def normalize_openai_base_url(base_url: str) -> str:
    """容错处理 OpenAI 协议端点的 base_url（Chat Completions 与 Responses 通用）。

    OpenAI client 会自己往 base_url 追加端点路径（Chat Completions 追 `/chat/completions`，
    Responses 追 `/responses`），所以 base_url 应填到路径前缀为止（如智谱的
    `https://open.bigmodel.cn/api/paas/v4`）。但用户往往直接粘贴 provider 给的**完整**
    URL，这里把末尾的 `/chat/completions` 或 `/responses` 和多余斜杠剥掉，避免拼出
    `.../chat/completions/chat/completions` 这种重复路径。
    """
    url = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


@dataclass
class ModelChoice:
    """解析后的模型选择：交给 Agent 的 model（字符串或 Model 实例）+ 给人看的描述。"""

    provider: str
    model_name: str
    model: str | Model  # Agent(model=...) 同时接受字符串与 Model 实例

    @property
    def description(self) -> str:
        return f"{self.provider}:{self.model_name}"


def resolve_model() -> ModelChoice:
    """从环境变量解析出模型选择。无副作用地构造，tracing 开关在这里顺带处理。"""
    provider = (os.environ.get("OPS_QA_PROVIDER") or "openai").strip().lower()
    if provider not in _DEFAULT_MODEL:
        raise ValueError(
            f"未知 OPS_QA_PROVIDER={provider!r}，"
            "可选：openai / responses / compatible / anthropic / litellm"
        )
    model_name = (os.environ.get("OPS_QA_MODEL") or _DEFAULT_MODEL[provider]).strip()

    # 非 OpenAI 原生时通常没有 platform.openai.com 的 tracing key：默认关掉 tracing，
    # 免去 SDK 往 OpenAI backend 传 trace 时报 key 缺失的噪音。这里用 env 开关（惰性、
    # 无副作用）而不是 set_tracing_disabled()——后者会在构造时即时初始化 backend
    # exporter（建 httpx client），在带 SOCKS 代理却没装 socksio 的内网环境会直接抛。
    # 用 setdefault 不覆盖用户显式设置（想看 trace 的可自行设为 false）。
    if provider != "openai":
        os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    if provider == "openai":
        # 字符串直接交给 Agent，SDK 用默认 OpenAI client（读 OPENAI_API_KEY）。
        # 注意：SDK 默认就走新的 Responses API（/v1/responses，_use_responses_by_default=True），
        # 所以官方 OpenAI 这条路本身用的就是 Responses 协议，无需额外配置。
        return ModelChoice(provider=provider, model_name=model_name, model=model_name)

    if provider in ("compatible", "responses"):
        from openai import AsyncOpenAI

        base_url = normalize_openai_base_url((os.environ.get("OPS_QA_BASE_URL") or "").strip())
        api_key = (os.environ.get("OPS_QA_API_KEY") or "").strip()
        if not base_url or not api_key:
            raise ValueError(f"provider={provider} 需要设置 OPS_QA_BASE_URL 和 OPS_QA_API_KEY")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        # compatible → Chat Completions（/v1/chat/completions）；responses → 新的 Responses
        # 协议（/v1/responses）。两者都是 OpenAI wire format，只是端点不同。
        if provider == "responses":
            model: Model = OpenAIResponsesModel(model=model_name, openai_client=client)
        else:
            model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        return ModelChoice(provider=provider, model_name=model_name, model=model)

    # anthropic / litellm 都经 LiteLLM 扩展（需 `--extra litellm`）。
    from agents.extensions.models.litellm_model import LitellmModel

    api_key = (os.environ.get("OPS_QA_API_KEY") or "").strip() or None
    base_url = (os.environ.get("OPS_QA_BASE_URL") or "").strip() or None

    if provider == "anthropic":
        # 走 LiteLLM 的 anthropic provider（Anthropic /v1/messages 协议）；模型名
        # 必须带 `anthropic/` 前缀才会被路由过去，缺省自动补上。base_url 省略时
        # LiteLLM 打官方地址，配了就指向你的 Anthropic 协议代理。
        litellm_model = model_name if "/" in model_name else f"anthropic/{model_name}"

        # 鉴权头风格：Anthropic 官方用 `x-api-key`，但不少第三方 Anthropic 网关
        # （火山引擎 coding plan 等）要 `Authorization: Bearer`。LiteLLM 的规则是：
        # 传了 api_key → 发 x-api-key；api_key 缺省、改读 ANTHROPIC_AUTH_TOKEN → 发 Bearer。
        # 所以想要 Bearer 时，把 key 提升成 ANTHROPIC_AUTH_TOKEN 并不再传 api_key。
        auth_style = (os.environ.get("OPS_QA_ANTHROPIC_AUTH") or "x-api-key").strip().lower()
        if auth_style == "bearer" and api_key:
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
            api_key = None

        model = LitellmModel(model=litellm_model, base_url=base_url, api_key=api_key)
        return ModelChoice(provider=provider, model_name=litellm_model, model=model)

    # litellm 通用兜底：模型名要自带 provider 前缀。
    model = LitellmModel(model=model_name, base_url=base_url, api_key=api_key)
    return ModelChoice(provider=provider, model_name=model_name, model=model)
