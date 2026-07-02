# 评测结果记录

用 `run_eval.py` 跑 `eval/cases.json`（12 题，映射到项目自带 `docs/`）得到的实测快照。指标口径见
`ops_qa_bot_oai/evaluate.py`：决策准确 / 转交准确（仅有分诊路由决策的 multi·auto 计）/ 组件命中 /
来源真实率都是**确定性**指标，跑一遍 bot 即可算，无需额外 LLM 判分。

> 数字随模型、题集、prompt 变化，且 GLM 结构化输出有随机性（见下），仅作某次快照参考；重点看
> **同一模型下不同配置的相对差异**。括号里的 `(N)` 是该指标实际参与打分的题数。

## 2026-07-02 · GLM-5.2（compatible / open.bigmodel.cn）

命令：`run_eval.py --modes auto,auto+structured` + `--modes multi,multi+structured,coordinator,coordinator+structured`。

| 配置 | 决策准确 | 转交准确 | 组件命中 | 来源真实 | 均tokens | 均轮数 | 均耗时ms |
|------|---------|---------|---------|---------|---------|-------|---------|
| `auto` | 82% | 92% | 100% | 100% (10) | 5850 | 4.0 | 34294 |
| `auto+structured` | 82% | 83% | 86% | 100% (9) | 7554 | 3.8 | 32235 |
| `multi` | 82% | 92% | 100% | 100% (9) | 4174 | 3.2 | 24676 |
| `multi+structured` | 73% | 75% | 71% | 100% (6) | 4626 | 2.5 | 25870 |
| `coordinator` | 73% | — (0) | 100% | 78% (9) | 10896 | 5.9 | 61004 |
| `coordinator+structured` | 55% | — (0) | 57% | 100% (5) | 4786 | 2.6 | 39296 |

### 读出来的结论

1. **coordinator 很贵**：free 版均 10896 tokens / 61s，是 `multi`（4174 / 25s）的约 **2.6 倍**——它对
   每道题都咨询多个专家再综合。这量化印证了默认用 `auto`（大多数单组件问题直接 handoff 给单个专家、
   只有跨组件才升级 coordinator）在成本上的合理性。
2. **coordinator 不计转交准确**（表里 `— (0)`）：它是唯一入口、没有分诊路由决策，每题都归它，故路由
   指标 0 题参与。它的价值在跨组件综合，不在"转交对不对"。
3. **结构化在 GLM 上普遍掉档**：`multi` 82→73（决策）/ 92→75（转交）/ 100→71（组件），`coordinator`
   73→55 / 100→57。原因是 GLM 对 JSON schema 遵守不稳，复杂题产不出合法契约被降级成 `reject`
   （`answer_structured` 捕获 `ModelBehaviorError` 的兜底，避免单题掀翻整批）——降级后转交/组件就丢了。
4. **结构化还更贵/更不稳**：`auto+structured` 均 tokens +29%（5850→7554）；`multi+structured` 两次
   跑分别是 82/92/100 与 73/75/71，**波动明显**——GLM 结构化不仅偏低还不稳定。
5. **来源真实率：产出的契约是干净的，但产出率没到满**：结构化各配置来源真实率多为 100%，但参与打分的
   题数偏少（如 `coordinator+structured` 仅 5 题有 citations）——说明"一旦产出契约，来源就可机器校验"，
   代价是 GLM 的合法产出率不高。`coordinator`（free）78% 则是自由文本里抽取的来源有 2 题不实。

**一句话**：在 GLM 上，结构化换来"来源可机器校验"，但牺牲路由/组件准确率、多花 token、且不稳定；
coordinator 综合能力强但最贵。对 GLM 这套配置，**日常用 `auto`（自由文本）性价比最高**；要机器可读契约
再上 `+structured`，并接受质量与稳定性的折扣。这些以前靠拍脑袋，现在是可量化的数字。

### provider 备注

- GLM 的 `json_schema` 输出不规范（裹 ```json 围栏 / 裸换行 / 非法反斜杠 / 围栏前带前言），
  靠 `FenceTolerantOutputSchema`（`schema.py`）容错解析才可用；能出契约，但如上所述产出率与稳定性有限。
- 火山 ark 的 deepseek **完全不支持** `json_schema`（直接 400），结构化不可用，需换模型。
