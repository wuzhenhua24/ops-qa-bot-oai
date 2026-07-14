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

多模型路由（差异化 #2，仅多 agent 模式下生效，复用同一 provider/client、只换模型名）：

- `OPS_QA_TRIAGE_MODEL`：分诊角色用的模型（建议便宜小模型）。
- `OPS_QA_MODEL_<组件目录大写>`：某组件专家用的模型（如 `OPS_QA_MODEL_REDIS`）。
- 无覆盖时一律回退到 `OPS_QA_MODEL`。

复核者（reviewer）的两档配置（差异化 #7 的"另一个模型"）：

- **只设 `OPS_QA_REVIEWER_MODEL`**：复用主 provider/client（同 URL、同 key），只换模型名。
  端点本身支持多模型（OpenAI 官方、聚合代理等）时这是最省事的姿势。
- **再设 `OPS_QA_REVIEWER_PROVIDER`**：复核者切到**独立端点**——`OPS_QA_REVIEWER_BASE_URL` /
  `OPS_QA_REVIEWER_API_KEY` / `OPS_QA_REVIEWER_ANTHROPIC_AUTH` 语义与主配置的同名变量一致，
  互不影响。主端点只服务一个模型家族（如 Anthropic 协议网关）时，跨家族复核（GPT 审
  Claude）只有这条路。`provider=openai` 时与主配置一样走标准 `OPENAI_API_KEY`。
  注意 anthropic+bearer 的 token 是进程级环境变量（`ANTHROPIC_AUTH_TOKEN`），主/复核
  两个端点都用 bearer 时没法各用各的 token（启动时会 WARNING）。

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

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from agents import Model, ModelSettings, OpenAIChatCompletionsModel, OpenAIResponsesModel

logger = logging.getLogger(__name__)

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


def env_flag(name: str, default: bool = False) -> bool:
    """读环境变量里的布尔开关：`1/true/yes/on`（不分大小写）为真，其余为假。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# 答题编排模式。single/multi/coordinator 为确定性模式（评测台按模式跑分对比用）；
# auto 为自适应默认：分诊台按问题决定转交给单个组件专家（常见）还是跨组件协调者（少数）。
MODES = ("single", "multi", "coordinator", "auto")
MODE_LABELS = {
    "single": "单 agent",
    "multi": "多 agent 编排（分诊 → 专家）",
    "coordinator": "跨组件协调者",
    # "单组件专家"指问题转交给对应组件各自的专家（每组件一个），别写成"单专家"——
    # 会被误读成"所有组件共用一个专家"。
    "auto": "自适应分诊（单组件专家 / 跨组件协调）",
}


def resolve_mode(default: str = "auto") -> str:
    """从环境变量 OPS_QA_MODE 解析编排模式；缺省 auto。终端与飞书共用此变量。"""
    raw = (os.environ.get("OPS_QA_MODE") or "").strip().lower()
    if not raw:
        return default
    if raw not in MODES:
        raise ValueError(f"未知 OPS_QA_MODE={raw!r}，可选：{' / '.join(MODES)}")
    return raw


def resolve_session_db() -> str:
    """从环境变量 OPS_QA_SESSION_DB 解析会话历史的 SQLite 落盘路径。

    缺省 ":memory:"（进程内、重启即丢，与旧行为一致）。设成文件路径（如
    `.sessions.db`）后会话历史持久化：飞书 bot 重启 / 空闲回收后，同一
    (chat, user) 的对话上下文可接着聊。
    """
    return (os.environ.get("OPS_QA_SESSION_DB") or "").strip() or ":memory:"


# 按角色的 temperature 代码默认：分诊是纯路由角色，低温更稳定（同一问题稳定转交同一
# 专家，路由准确率不随机抖）。复核者是判定角色（对证据核对、出 typed verdict），同样低温
# 让"过/需改"的判定稳定、可复现。其余角色不设，沿用 provider 默认。
_ROLE_TEMPERATURE_DEFAULTS: dict[str, float] = {"triage": 0.1, "reviewer": 0.0}


def _env_float(name: str) -> float | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r} 不是合法数字") from e


def role_model_settings(role: str) -> ModelSettings:
    """按角色构造 ModelSettings（SDK 的 per-agent 采样/工具行为配置）。

    多模型路由（#2）此前只按角色换**模型名**；ModelSettings 把"每个 agent 可以有
    不同的调用参数"补齐——分诊和专家不止可以用不同模型，还可以用不同的温度/工具
    并发策略。None 字段不下发（沿用 provider 默认），不影响不支持该参数的端点。

    - **temperature**：`OPS_QA_TEMPERATURE_<角色大写>` > `OPS_QA_TEMPERATURE`（全局）>
      角色代码默认（triage=0.1，路由要稳）> 不设。
    - **parallel_tool_calls**：仅 coordinator 默认 True——它的工作流程就是"对多个相关
      组件**并行**求证"（见 orchestration 的协调者 prompt），不设则模型多半串行逐个
      咨询专家，跨组件排查时延翻倍。`OPS_QA_PARALLEL_TOOLS=0` 可关（个别端点不认
      该参数时用）。
    """
    r = role.strip().lower()
    temp = _env_float(f"OPS_QA_TEMPERATURE_{r.upper()}")
    if temp is None:
        temp = _env_float("OPS_QA_TEMPERATURE")
    if temp is None:
        temp = _ROLE_TEMPERATURE_DEFAULTS.get(r)
    parallel: bool | None = None
    if r == "coordinator":
        parallel = env_flag("OPS_QA_PARALLEL_TOOLS", default=True) or None
    return ModelSettings(temperature=temp, parallel_tool_calls=parallel)


@dataclass
class ModelChoice:
    """解析后的模型选择：交给 Agent 的 model（字符串或 Model 实例）+ 给人看的描述。"""

    provider: str
    model_name: str
    model: str | Model  # Agent(model=...) 同时接受字符串与 Model 实例

    @property
    def description(self) -> str:
        return f"{self.provider}:{self.model_name}"


# 一个"按模型名铸 Model"的工厂：make(raw_name) -> (生效名, Agent 可用的 model)。
# 把 provider/client 的解析与"具体用哪个模型名"解耦——这样多模型路由（#2）能复用
# 同一个 client、只换模型名，按角色/组件铸出不同的 Model。
ModelMaker = Callable[[str], "tuple[str, str | Model]"]


def _build_model_factory(prefix: str = "OPS_QA") -> tuple[str, ModelMaker]:
    """从环境变量解析 provider + client/配置，返回 (provider, make)。

    `make(raw_name)` 复用已解析好的 client/配置，按给定模型名铸出 Model（anthropic 会
    自动补 `anthropic/` 前缀），返回 (生效名, model)。tracing 开关在此顺带处理。

    `prefix` 指定读哪一套环境变量：缺省 `OPS_QA`（主配置）；复核者独立端点传
    `OPS_QA_REVIEWER`，读 `OPS_QA_REVIEWER_PROVIDER` / `_BASE_URL` / `_API_KEY` /
    `_ANTHROPIC_AUTH`，语义与主配置逐一对应。
    """
    provider = (os.environ.get(f"{prefix}_PROVIDER") or "openai").strip().lower()
    if provider not in _DEFAULT_MODEL:
        raise ValueError(
            f"未知 {prefix}_PROVIDER={provider!r}，"
            "可选：openai / responses / compatible / anthropic / litellm"
        )

    # 非 OpenAI 原生时通常没有 platform.openai.com 的 tracing key：默认关掉 tracing，
    # 免去 SDK 往 OpenAI backend 传 trace 时报 key 缺失的噪音。这里用 env 开关（惰性、
    # 无副作用）而不是 set_tracing_disabled()——后者会在构造时即时初始化 backend
    # exporter（建 httpx client），在带 SOCKS 代理却没装 socksio 的内网环境会直接抛。
    # 只有主配置有权决定这个**进程级**开关：主是 openai、复核者独立端点不是时，
    # 不能反过来把主的 tracing 关掉。
    if provider != "openai" and prefix == "OPS_QA":
        os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    if provider == "openai":
        # 字符串直接交给 Agent，SDK 用默认 OpenAI client（读 OPENAI_API_KEY）。
        # SDK 默认就走新的 Responses API（/v1/responses，_use_responses_by_default=True）。
        def make(name: str) -> tuple[str, str | Model]:
            return name, name

        return provider, make

    if provider in ("compatible", "responses"):
        from openai import AsyncOpenAI

        base_url = normalize_openai_base_url((os.environ.get(f"{prefix}_BASE_URL") or "").strip())
        api_key = (os.environ.get(f"{prefix}_API_KEY") or "").strip()
        if not base_url or not api_key:
            raise ValueError(
                f"{prefix}_PROVIDER={provider} 需要设置 {prefix}_BASE_URL 和 {prefix}_API_KEY"
            )
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)  # 解析一次，按名复用
        is_responses = provider == "responses"

        def make(name: str) -> tuple[str, str | Model]:
            # compatible → Chat Completions；responses → Responses 协议。
            if is_responses:
                return name, OpenAIResponsesModel(model=name, openai_client=client)
            return name, OpenAIChatCompletionsModel(model=name, openai_client=client)

        return provider, make

    # anthropic / litellm 都经 LiteLLM 扩展（需 `--extra litellm`）。
    from agents.extensions.models.litellm_model import LitellmModel

    api_key = (os.environ.get(f"{prefix}_API_KEY") or "").strip() or None
    base_url = (os.environ.get(f"{prefix}_BASE_URL") or "").strip() or None
    is_anthropic = provider == "anthropic"

    if is_anthropic:
        # 鉴权头风格：传 api_key → x-api-key；改读 ANTHROPIC_AUTH_TOKEN → Bearer。
        # 火山引擎等第三方 Anthropic 网关要 Bearer，故 bearer 时把 key 提升成 env token。
        auth_style = (os.environ.get(f"{prefix}_ANTHROPIC_AUTH") or "x-api-key").strip().lower()
        if auth_style == "bearer" and api_key:
            # ANTHROPIC_AUTH_TOKEN 是**进程级**的：主/复核两套 anthropic+bearer 配置只能
            # 共用先生效的那个 token。key 不同没法静默"看起来配好了"，必须提醒。
            existing = (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
            if existing and existing != api_key:
                logger.warning(
                    "%s_API_KEY 与已生效的 ANTHROPIC_AUTH_TOKEN 不同：bearer 鉴权走进程级"
                    "环境变量，主/复核两个 anthropic 端点无法各用各的 bearer token，"
                    "本端点将沿用已生效的那个（如需不同 key，请让其中一端改用 x-api-key）。",
                    prefix,
                )
            os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", api_key)
            api_key = None

    def make(name: str) -> tuple[str, str | Model]:
        # anthropic 模型名必须带 `anthropic/` 前缀才会被 LiteLLM 路由过去，缺省自动补。
        eff = name if (not is_anthropic or "/" in name) else f"anthropic/{name}"
        return eff, LitellmModel(model=eff, base_url=base_url, api_key=api_key)

    return provider, make


def _default_model_name(provider: str, prefix: str = "OPS_QA") -> str:
    return (os.environ.get(f"{prefix}_MODEL") or _DEFAULT_MODEL[provider]).strip()


def resolve_model() -> ModelChoice:
    """从环境变量解析出（单一）模型选择。无副作用地构造，tracing 开关顺带处理。"""
    provider, make = _build_model_factory()
    eff_name, model = make(_default_model_name(provider))
    return ModelChoice(provider=provider, model_name=eff_name, model=model)


@dataclass
class RoleFactory:
    """某个角色的独立端点：自己的 provider/client + 缺省模型名（不与主配置共用）。"""

    provider: str
    default_name: str
    make: ModelMaker


@dataclass
class ModelRouter:
    """多模型路由（差异化 #2）：按"角色/组件"铸出不同模型，复用同一 provider/client。

    典型用法是分层：分诊（triage）用便宜小模型、组件专家用强模型，硬核诊断才烧贵的
    token——直接砍成本。也支持按组件单独指定（如敏感组件指到私有/本地模型名）。

    覆盖来自环境变量（均为模型名，复用同一 provider/client）：
    - `OPS_QA_TRIAGE_MODEL`：分诊角色用的模型。
    - `OPS_QA_MODEL_<组件目录大写>`：某组件专家用的模型（如 `OPS_QA_MODEL_REDIS`）。
    无对应覆盖时回退到 `OPS_QA_MODEL`（默认模型）。

    例外是 `role_factories` 里的角色：它们有**自己的 provider/client**（独立端点），
    不走主 client。目前只有 reviewer 开这个口子（设 `OPS_QA_REVIEWER_PROVIDER` 后生效，
    见 build_model_router）——复核的价值来自判断独立性，跨家族（GPT 审 Claude）时主端点
    往往没有目标模型；其余角色对"换端点"没有真实需求，不泛化。
    """

    provider: str
    default_name: str
    overrides: dict[str, str]  # 角色/组件目录（小写）-> 模型名
    _make: ModelMaker
    role_factories: dict[str, RoleFactory] = field(default_factory=dict)  # 角色 -> 独立端点

    def for_role(self, role: str) -> tuple[str, str | Model]:
        """返回某角色（"triage" / "reviewer" / 组件目录名）的 (生效模型名, model)。"""
        r = role.strip().lower()
        rf = self.role_factories.get(r)
        if rf is not None:
            return rf.make(self.overrides.get(r, rf.default_name))
        return self._make(self.overrides.get(r, self.default_name))

    @property
    def default_choice(self) -> ModelChoice:
        eff, model = self._make(self.default_name)
        return ModelChoice(provider=self.provider, model_name=eff, model=model)

    def describe(self, roles: list[str]) -> str:
        """给人看的角色→模型映射（用于 CLI 横幅）。"""
        parts = [f"默认={self.default_name}"]
        for r in roles:
            key = r.strip().lower()
            if key in self.role_factories:
                continue  # 独立端点的角色统一在下面带 provider 列出
            name = self.overrides.get(key)
            if name:
                parts.append(f"{r}={name}")
        for key, rf in self.role_factories.items():
            parts.append(f"{key}={rf.provider}:{self.overrides.get(key, rf.default_name)}")
        return f"{self.provider}（" + "，".join(parts) + "）"


_OVERRIDE_PREFIX = "OPS_QA_MODEL_"


def build_model_router() -> ModelRouter:
    """从环境变量构造多模型路由器。无覆盖时所有角色都用默认模型（等价单模型）。"""
    provider, make = _build_model_factory()
    default_name = _default_model_name(provider)

    overrides: dict[str, str] = {}
    triage = (os.environ.get("OPS_QA_TRIAGE_MODEL") or "").strip()
    if triage:
        overrides["triage"] = triage
    # 二次复核者角色：另一个模型对答案做证据核对（差异化 #7）。建议指到与答题不同的模型，
    # 降低同错同漏；缺省回退默认模型。只设 OPS_QA_REVIEWER_MODEL → 复用主 provider/client
    # 换模型名（端点支持多模型时最省事）；再设 OPS_QA_REVIEWER_PROVIDER → 切独立端点。
    reviewer = (os.environ.get("OPS_QA_REVIEWER_MODEL") or "").strip()
    if reviewer:
        overrides["reviewer"] = reviewer
    # 扫描 OPS_QA_MODEL_<DIR> 形式的按组件覆盖。
    for key, val in os.environ.items():
        if key.startswith(_OVERRIDE_PREFIX) and val.strip():
            comp = key[len(_OVERRIDE_PREFIX) :].strip().lower()
            if comp:
                overrides[comp] = val.strip()

    role_factories: dict[str, RoleFactory] = {}
    if (os.environ.get("OPS_QA_REVIEWER_PROVIDER") or "").strip():
        # 复核者独立端点。配置不合法（缺 URL/key、provider 名打错）在这里就抛——reviewer
        # 运行期是 fail-open 的（挂了视作 approve），配错若拖到运行期只会表现成"复核静默
        # 全过"，必须在启动时响。
        rp, rmake = _build_model_factory(prefix="OPS_QA_REVIEWER")
        rf = RoleFactory(
            provider=rp,
            default_name=_default_model_name(rp, prefix="OPS_QA_REVIEWER"),
            make=rmake,
        )
        role_factories["reviewer"] = rf
        logger.info(
            "复核者使用独立端点：provider=%s，模型=%s（主端点 provider=%s 不受影响）",
            rp,
            overrides.get("reviewer", rf.default_name),
            provider,
        )

    return ModelRouter(
        provider=provider,
        default_name=default_name,
        overrides=overrides,
        _make=make,
        role_factories=role_factories,
    )
