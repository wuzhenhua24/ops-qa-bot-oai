# 评测结果记录

用 `run_eval.py` 跑 `eval/cases.json`（12 题，映射到项目自带 `docs/`）得到的实测快照。指标口径见
`ops_qa_bot_oai/evaluate.py`：决策准确 / 转交准确（仅 multi·auto 计）/ 组件命中 / 来源真实率
都是**确定性**指标，跑一遍 bot 即可算，无需额外 LLM 判分。

> 数字随模型、题集、prompt 变化，仅作某次快照参考；重点看**同一模型下不同配置的相对差异**。

## 2026-07-02 · GLM-5.2（compatible / open.bigmodel.cn）

命令：`uv run python run_eval.py --modes auto,auto+structured`（另附 `multi+structured` 一次单跑）。

| 配置 | 决策准确 | 转交准确 | 组件命中 | 来源真实 | 均tokens | 均轮数 | 均耗时ms |
|------|---------|---------|---------|---------|---------|-------|---------|
| `auto`（自由文本） | 82% | **92%** | **100%** | 100% | **5850** | 4.0 | 34294 |
| `auto+structured` | 82% | 83% | 86% | 100% | 7554 | 3.8 | 32235 |
| `multi+structured` | 82% | 92% | 100% | 100% | 5203 | 2.9 | 23091 |

### 读出来的结论

1. **结构化在 GLM 上有质量代价**：自适应路由下转交准确 92%→83%、组件命中 100%→86%。原因是
   GLM 对编排 agent 的 JSON schema 遵守不稳，部分复杂题产不出合法契约、被降级成 `reject`
   （对应的转交/组件就丢了；降级是 `answer_structured` 捕获 `ModelBehaviorError` 的兜底，
   避免单题掀翻整批）。
2. **结构化更贵**：均 tokens 5850→7554（约 +29%），要额外吐 JSON 结构。
3. **来源真实率两者都 100%**：契约的 `citations` 字段一旦产出就是干净的——印证结构化的核心
   卖点（来源可机器校验），只是 GLM 的合法产出率没到满。
4. **决策准确持平**（82%）。

**一句话**：在 GLM 上，结构化换来了"来源可机器校验"，但牺牲了一点路由/组件准确率、多花约三成
token——是否值得取决于下游是要机器可读契约、还是要最高的路由质量。这种权衡以前靠拍脑袋，现在是
可量化的数字。

### provider 备注

- GLM 的 `json_schema` 输出不规范（裹 ```json 围栏 / 裸换行 / 非法反斜杠 / 前言文字），
  靠 `FenceTolerantOutputSchema`（`schema.py`）容错解析才可用；实测 single/multi/auto 均能出契约。
- 火山 ark 的 deepseek **完全不支持** `json_schema`（直接 400），结构化不可用，需换模型。
