# 评测结果记录

用 `run_eval.py` 跑 `eval/cases.json`（12 题，映射到项目自带 `docs/`）得到的实测快照。指标口径见
`ops_qa_bot_oai/evaluate.py`：决策准确 / 转交准确（仅有分诊路由决策的 multi·auto 计）/ 组件命中 /
来源真实率都是**确定性**指标，跑一遍 bot 即可算，无需额外 LLM 判分。

> 数字随模型、题集、prompt 变化，且 GLM 结构化输出有随机性（见下），仅作某次快照参考；重点看
> **同一模型下不同配置的相对差异**。括号里的 `(N)` 是该指标实际参与打分的题数。
> 评测轴 token：`free`/`structured` = single 路由的自由文本/结构化；其余为 `<routing>[+structured]`。

## 2026-07-02 · GLM-5.2（compatible / open.bigmodel.cn）· 全 8 配置矩阵

命令：`run_eval.py --modes free,structured,multi,multi+structured,auto,auto+structured,coordinator,coordinator+structured`（分几次跑）。

| 配置 | 决策准确 | 转交准确 | 组件命中 | 来源真实 | 均tokens | 均轮数 | 均耗时ms |
|------|---------|---------|---------|---------|---------|-------|---------|
| `free`（single 自由文本） | **100%** | — (0) | 100% | 88% (8) | 10640 | 3.3 | 23744 |
| `structured`（single 结构化） | 73% | — (0) | 86% | 100% (7) | 9611 | 2.6 | 18962 |
| `multi` | 82% | 92% | 100% | 100% (9) | **4174** | 3.2 | 24676 |
| `multi+structured` | 73% | 75% | 71% | 100% (6) | 4626 | 2.5 | 25870 |
| `auto` | 82% | 92% | 100% | 100% (10) | 5850 | 4.0 | 34294 |
| `auto+structured` | 82% | 83% | 86% | 100% (9) | 7554 | 3.8 | 32235 |
| `coordinator` | 73% | — (0) | 100% | 78% (9) | 10896 | 5.9 | 61004 |
| `coordinator+structured` | 55% | — (0) | 57% | 100% (5) | 4786 | 2.6 | 39296 |

### 读出来的结论

1. **`multi` 最省，`single`(free) 决策最准但最贵**：`multi` 靠组件作用域，均 4174 tokens 是全场最低；
   `free` 单 agent 读全库、无作用域，决策准确 100%（全场最高）但 tokens 10640、且自由文本抽取的来源
   没那么可靠（88%，1 题引用不实）。`multi` 在"决策 82% / 组件 100% / 来源 100% / 成本最低"上很均衡。
2. **coordinator 最贵**：free 版 10896 tokens / 61s，是 `multi` 的约 2.6 倍——每题都咨询多个专家。
   量化印证了默认用 `auto`（单组件直接 handoff、只有跨组件才升 coordinator）在成本上的合理性。
3. **转交准确只对 multi·auto 有意义**：`single`（free/structured）与 `coordinator` 都是唯一入口、
   没有分诊路由决策，表里记 `— (0)`（0 题参与该指标）。
4. **结构化在 GLM 上普遍掉档，且不稳**：决策 `free`100→`structured`73、`multi`82→73、`coordinator`73→55；
   组件命中也齐跌。原因是 GLM 对 JSON schema 遵守不稳，复杂题产不出合法契约被降级成 `reject`
   （`answer_structured` 捕获 `ModelBehaviorError` 的兜底，避免单题掀翻整批）。`multi+structured` 两次跑
   82/92/100 与 73/75/71，**波动明显**。
5. **结构化的正收益：来源可机器校验**：各结构化配置来源真实率多为 100%，对比 `free`(88%)/`coordinator`(78%)
   自由文本抽取会混入不实来源。代价是参与打分的题数偏少（如 `coordinator+structured` 仅 5 题有 citations）
   ——"一旦产出契约，来源就干净"，但 GLM 的合法产出率不高。

**一句话**：GLM 这套配置下——要最省用 `multi`、要决策最准用 `single`(free)（但贵）、要"路由自适应 + 兜跨组件"
用默认 `auto`、要跨组件综合用 `coordinator`（最贵）；`+structured` 换来"来源可机器校验"，但普遍牺牲准确率、
更不稳。这些以前靠拍脑袋，现在是可量化的数字。

### provider 备注

- GLM 的 `json_schema` 输出不规范（裹 ```json 围栏 / 裸换行 / 非法反斜杠 / 围栏前带前言），
  靠 `FenceTolerantOutputSchema`（`schema.py`）容错解析才可用；能出契约，但如上所述产出率与稳定性有限。
- 火山 ark 的 deepseek **完全不支持** `json_schema`（直接 400），结构化不可用，需换模型。
