[English](REPORT.md) | [简体中文](REPORT.zh-CN.md)

# Jev 类模型 benchmark 调研与实测报告

**被测对象**：本项目实现的决策服务（火山方舟托管 API，当前配置的 8 个底座路由）。

**评测方式**：`POST /v1/systemone` 等价路径；单次前向、贪心解码（`temperature=0`）、只读生成序列位置 0、候选上限 10、不做温度校准。

**数据版本**：第四节数字绑定本轮全量运行产物（8 路由 × 4 组基准，26,184 次调用，0 失败）：`benchmarks/results/matrix-20261003-022427.json`（sha256 `80875133…`）。完整哈希与边界见 6.4。

上一版报告（10-02 运行，含已下线路由名 `doubao-2.0-lite`、`deepseek-v4.1-flash`、`deepseek-v4-flash`）与早期未对齐口径的初测结果已移入[附录 A](#附录-a历史产物已被第四节取代)，正文不再引用。

**准确率口径**：全文均为 **hard-label argmax 准确率**（与业界口径一致），分母是**剔除连接失败后的有效题数**（本轮矩阵 0 失败，分母即题数；唯一例外是 4.4 节的 A/B 对照实验，该表分母含失败题，已在表内标注）。

**目录**

- [一、核心结论](#一核心结论)
- [二、基准与可跑性](#二基准与可跑性)
- [三、评测方法与口径](#三评测方法与口径)
- [四、实测结果](#四实测结果)
- [五、业界对照](#五业界对照)
- [六、复现与可复核性](#六复现与可复核性)
- [附录 A：历史产物（已被第四节取代）](#附录-a历史产物已被第四节取代)

---

## 一、核心结论

1. **路线成立，且同口径下超过业界最好水平。** 用托管 API 的 logprobs 复刻 Jev 形态，`doubao-evolving` 在 JevBench 公开 231 上拿到 **91.8%**、hard 档 **83.8%**，`doubao-2.1-pro` 为 90.9% / 82.0%，高于 Jev 1.13.0（86.58% / 72.97%）、Open-Jev-27B（85.28% / 72.07%）与全部开源权重模型（最高 NeoHorse-Jev-4B 逐样本 75.32%）。
2. **同题同口径进入「90 分档」，hard 档领先。** 在 InternLM 的 [Intern-Decision](https://github.com/InternLM/Intern-Decision) 七项基准上（题目与其官方 bundle **逐 id 一致**），`doubao-evolving` 七项均值 **89.85**、`doubao-2.1-pro` **89.37**，高于 Jev 的 88.74、逼近 Intern-Decision-4B 的 90.02；**hard 档 82.88% 比该表最高值 73.87% 高 9 个点**。落后项集中在 `typed_decisions` 与 ToolACE——对方是把状态转结构化决策的**专项微调模型**（见 5.7）。
3. **差距来自底座模型，不是机制。** 同一套读出逻辑，`doubao-2.0-mini` 只有 81.8%，换成 `doubao-2.1-pro` 升到 90.9%——机制本身没有改过一个字节。
4. **速度与准确率的折中是 `doubao-2.1-lite`**：JevBench 87.9% / Nimble 88.2% / Kev 81.8%，p50 550 ms（约为 pro 档的一半）；要最高准确率就上 `doubao-evolving` 或 `doubao-2.1-pro`（91.8% / 90.9%，Nimble 94.6% / 94.3%）。`doubao-2.0-mini`（355 ms）最快，但 81.8% / Nimble 71.4% 只够极轻量场景。
5. **Kev 是唯一未追平的组**：我们最高 81.80%（六子集等权，`doubao-2.0-pro`）vs Jev 1.13.0 的 85.52，差 3.7 个点；失分集中在 transfer-v9 两个子集（75.9~78.8%），transfer-v4（81.1~86.9%）反而是强项。
6. **模型强 ≠ 适合本机制。** DeepSeek 两个路由整体低于同价位 doubao：`deepseek-flash` 在 VitaminC 事实核验上最高（75.5%），但 Nimble（82.1%）/ Kev（80.4%）明显落后；`deepseek-v4-pro` 四组全部偏低。
7. **格式遵从可兜底**：统一强度的 `logit_bias` 把「模型不吐标签」的读出失败清零，且不改变候选内概率分布，建议生产环境开启（见 4.4）。
8. **confidence 目前不可信**：未做温度校准（`temperature_scale=1.0`）。5.7 的 Brier / ECE 只用于「同题对照」：Brier 更好主要来自准确率更高，而 ECE（hard 0.101、pilot 0.309）说明概率偏激进——要上线概率，先补标注数据跑温度校准。
9. **两个格子空着，不是跑不动就是不可复现**：OpenJev 文本（harness 与 prompt 未公开）；MASSIVE-en（18 类候选超出上限 10）。

---

## 二、基准与可跑性

### 2.1 这类模型的主流 benchmark

| Benchmark | 任务形态 | 规模 | 是否公开 | 许可 |
|---|---|---|---|---|
| **JevBench**（`fstandhartinger/jevbench`） | Choice / Noul / Score 混合，easy / standard / judge / hard 四档 | 534 题，公开 231（72 original + 48 easy + 111 hard） | 公开子集 + sealed 藏题 | MIT |
| **Nimble 公开套件**（Bespoke Labs） | 5 Choice + 5 Noul + 3 Score 子集 | 3,880 条（13 个子集） | 公开（按 manifest 从上游重建） | 随上游 |
| Nimble 内置 holdout | 混合 | 324 条（评测按 compat282 规则取 280 条，见 3.2） | 公开（仓库内置） | 见仓库 |
| **VitaminC** | Choice 3 路（事实核验） | 63,054 条 validation | 公开 | CC BY-SA 3.0 |
| OpenJev text | NLI + 多选重排 + 固定候选 GSM8K | 19 任务 | 公开 | 随上游 |
| Kev | decision-v7 / transfer-v4 / transfer-v9 | 6,436 条（计入评测的 clean 题 5,768 条） | 公开 | Apache-2.0 |
| MASSIVE 1.1 | Choice 18 域意图 | 2,974/语言 | 公开 | CC BY 4.0 |
| BANKING77 | Choice 77 意图 | 13,083 条 | 公开 | CC BY 4.0 |
| TypeSafe 官方 4-workflow | 安全事件 / agent trace / 发票 / 客服 | 私有 | 私有 | — |
| S1 Bench | 6 子集 | 1,999 条 | 仅榜单 | — |

### 2.2 评分口径（业界通行做法）

- **全是 hard-label 比对，不需要 LLM-as-judge**（唯一例外是 TypeSafe 官方私有套件，其参考标签是两个前沿模型的平均）。
- JevBench：按 tier 各占 1/3 取平均；`score` 题看 argmax，期望值单独作为 MAE 报告。
- Nimble：accuracy + Wilson CI + ECE(10 bins) + Brier；`score` 另报 MAE，`multinli` / `civil_comments` 另报与人工分布的一致性（JSD / TVD）。
- Kev：六个子集 clean accuracy 等权平均。
- **共同点：单次前向取 argmax，不允许多次采样投票。** 这正好与本项目的实现方式同构。

### 2.3 候选数上限下的可跑性

本服务候选上限为 10（`top_logprobs` 硬上限 20，扣掉 EOS、全角变体、标点后安全线是 10）。

| 能跑 | 跑不了（原因） |
|---|---|
| JevBench 公开 231（实测最多 6 个候选） | MASSIVE 18 域、BANKING77 77 意图（候选超限） |
| Nimble 内置 holdout（评测取 280 条，实测 2~6 个候选） | 16 选项的 decision-models-under-pressure |
| VitaminC（3 路） | TypeSafe 官方 4-workflow（私有 + 双模型共识参考） |
| Kev 六子集（候选 >10 的题跳过并计数） | S1 Bench（题集未公布） |
| Nimble 13 子集中候选 ≤10 的项 | OpenJev 文本（harness 与 prompt 未公开） |

---

## 三、评测方法与口径

### 3.1 读出机制与 Jev 的等价性

Jev / vLLM 的做法是**停在答案位置做一次前向，直接读分布**（prefill 打分）。方舟不开放 prefill 打分接口，因此本实现采用等价方案：

> prompt 停在答案位置 → 贪心解码生成 **1 个** token → 只读**生成序列位置 0** 的 `top_logprobs` → 在候选标签内做归一化。

位置 0 的语义就是「答案位置的下一个 token 分布」，与 Jev 的一次前向等价。代价是多了一层对模型配合的依赖（模型必须真的把标签作为第一个 token 吐出来）：关闭 `logit_bias` 的对照实验里出现过 3/231 次读出失败，开启后为 0（见 4.4）；**本轮全量矩阵 26,184 次调用 0 次读出失败**。

相关的机制硬约束（详见 [docs/design.md](../docs/design.md)）：

| 约束 | 值 | 原因 |
|---|---|---|
| `temperature` | `0`（贪心） | 保证答案位置确定性地落在生成序列第 0 位 |
| `max_tokens` | `1` | 避免生成解释性文本，否则位置 0 不是答案分布 |
| `top_logprobs` | `20` | 方舟硬上限；要容纳 EOS、全角变体、标点等非候选 token |
| 候选上限 | `10` | 20 条 top_logprobs 扣除干扰项后的安全线，超出直接 422 |
| `disable_thinking` | `true` | 思考模型带 logprobs 会直接 400 |

### 3.2 计分口径与失败处理

| 项 | 规则 |
|---|---|
| 计分对象 | 题级 hard-label argmax，无 LLM-as-judge |
| **准确率分母** | **剔除连接失败后的有效题数 `n_valid`**，不是总题数 |
| 连接类错误 | 单独统计 `n_errors`；重试最多 3 次（退避 1s/3s/9s），不做事后重采样 |
| 单元格有效性 | 失败率 > 20% 标记 `valid=false`，不计入准确率结论 |
| 模型拒绝作答 | （首 token 不是候选标签）记为**答错**，不做特殊处理 |
| 候选超限的题 | Kev 中候选 >10 的题**跳过并计数**，不记为答错 |

第四节的 4.2 单独列出每个单元格的失败题数，便于读者自行换算分母。

子集口径（对齐业界对照，规则与哈希见 6.3）：

- **Nimble**：不跑 324 条全量，按 compat282 规则（state ≤384 tokens、请求体 ≤2048 tokens、候选 ≤26、整族保留）实测得 **280 条**。与业界标注的 282 条差 2 条，是 384/386 tokens 的边界分歧，**如实记录不凑数**。
- **VitaminC**：按 Nimble manifest 的 599 个 `unique_id` 精确取行，**匹配 599/599**。
- **Kev**：六子集全量，仅 `_meta.variant == "clean"` 的题计入，得 **5,768** 条。

---

## 四、实测结果

数据来源：单份全量矩阵 `benchmarks/results/matrix-20261003-022427.json`（8 路由 × 4 组基准，26,184 次调用，0 失败）。产物哈希见 6.4。

### 4.1 全量矩阵（主结果）

| 路由 | JevBench 231 | Hard 111 | Nimble 280 | VitaminC 599 | Kev 六子集等权 | p50¹ |
|---|---:|---:|---:|---:|---:|---:|
| **doubao-evolving** | **91.8%** | **83.8%** | **94.6%** | 73.1% | — | 1152 ms |
| doubao-2.1-pro | 90.9% | 82.0% | 94.3% | 73.6% | — | 1110 ms |
| doubao-2.0-pro | 90.0% | 81.1% | 89.3% | 72.0% | **81.80%** | 1611 ms |
| doubao-2.1-turbo | 88.7% | 79.3% | 90.4% | 74.6% | — | 1593 ms |
| **doubao-2.1-lite** | 87.9% | 76.6% | 88.2% | 72.5% | 81.77% | 550 ms |
| deepseek-flash | 83.1% | 67.6% | 82.1% | **75.5%** | 80.40% | 421 ms |
| doubao-2.0-mini | 81.8% | 64.0% | 71.4% | 74.8% | — | **355 ms** |
| deepseek-v4-pro | 78.4% | 63.1% | 81.4% | 67.1% | — | 735 ms |

¹ p50 取 JevBench 231 的中位延迟。排序按 JevBench 降序，同分按 hard 降序。Kev 列只在 3 个代表性路由上跑了全部六子集（`doubao-2.0-pro` / `doubao-2.1-lite` / `deepseek-flash`，5,768 题/路由）；该列是**六子集等权平均**，与业界 Kev 口径一致；若改用题级 pooled 口径，三者为 81.1% / 81.0% / 80.0%。

² **本表与上一版报告（10-02 运行）不可逐格对比**：两次运行之间，同路由同基准的正确数波动 ±2~7 题（贪心解码下 MoE 路由 / 批处理的非确定性；4.4 节同配置重复运行即差 7 题）。上一版里的 `doubao-2.0-lite`、`deepseek-v4.1-flash`、`deepseek-v4-flash` 已不在当前配置，其当时的数字见[附录 A](#附录-a历史产物已被第四节取代)。

**分题型（JevBench 231）**：`choice` 最高 92.8%、`noul` 最高 91.9%、`score` 最高 88.9%（均在 `doubao-evolving` / `doubao-2.1-pro`）；Nimble 的 `score` 子集（54 题）最高 94.4%（`doubao-evolving`）、中位约 87%，最低 57.4%（`doubao-2.0-mini`）。

### 4.2 有效样本与读出失败

**本轮 26,184 次调用 0 次失败**（连接类与读出类均为 0），8 路由 × 四组基准的每个单元格都 `valid=true`，准确率分母即题数（JevBench 231 / Nimble 280 / VitaminC 599 / Kev 5,768）。旧版的逐单元格失败表因此不再需要；如需逐项核对，运行产物与 `provenance.json` 里有每题的原始记录。

这与当前配置一致：`doubao` 全系已开启 `logit_bias`；DeepSeek 两条路由未开启，本轮也没有出现读出失败。读出失败的历史反例与兜底效果见 4.4。

### 4.3 Kev 六子集明细

| 路由 | decision-v7-dev | decision-v7-test | transfer-v4-dev | transfer-v4-test | transfer-v9-dev | transfer-v9-test | 等权 |
|---|---:|---:|---:|---:|---:|---:|---:|
| doubao-2.0-pro | 81.8% | 81.1% | 84.5% | 86.9% | 77.8% | 78.8% | **81.80%** |
| doubao-2.1-lite | 82.1% | 80.5% | 85.5% | 86.7% | 77.9% | 77.9% | 81.77% |
| deepseek-flash | 82.7% | 81.9% | 81.1% | 83.5% | 75.9% | 77.3% | 80.40% |

未追平 Jev 1.13.0 的 85.52（差 3.7 个点）：差距集中在 **transfer-v9** 两个子集（75.9~78.8%）；**transfer-v4** 反而是强项（81.1~86.9%，高于 Kev-4B 的 81.47）。`deepseek-flash` 在 decision-v7 上最高（82.7 / 81.9%），但两个 transfer 组最低——三条路由各有短板，都没追平。三路由共 17,304 题、0 失败。Jev 1.13.0 未公布分子集明细，只能在六子集等权上比。

### 4.4 logit_bias 兜底实验

同一批 231 题、同一模型（`doubao-2.0-mini`）、只切换 `logit_bias` 开关。为把「开关效应」和「运行间波动」分开，开启臂跑两次：

| 配置 | 运行产物（同目录） | 总体 | Hard | 读出失败 |
|---|---|---:|---:|---:|
| bias 关闭 | `jevbench-20261003-093725-nobias` | 189/231 = 81.82% | 73/111 = 65.77% | **3 题** |
| bias 开启 | `jevbench-20261003-033600-withbias` | 186/231 = 80.52% | 71/111 = 63.96% | **0 题** |
| bias 开启（重复） | `jevbench-20261003-033544-withbias2` | 193/231 = 83.55% | 76/111 = 68.47% | **0 题** |

3 个失败都是「模型首 token 没吐标签」（实际输出「请」「首先」「首先」）。开启统一偏置后**格式失败清零**。准确率的升降不应归于开关：同配置两次开启臂自身就差 7 题（186 vs 193），关闭臂的 189 落在两者之间——差异被运行间波动（MoE 路由 / 批处理非确定性）淹没。

**口径说明（必读）**：本表来自 `run_jevbench.py` 的单基准产物，分母是 231（读出失败计为答错）；4.1 的 `doubao-2.0-mini` 行（189/231、0 失败）来自矩阵产物、是另一次运行。同模型同基准在四次独立运行间的正确数落在 186~193，两套数字都保留、各自标注来源，不做合并。

结论：`logit_bias` 对**格式失败**是确定性的兜底（3 → 0），对准确率无可辨别的系统性影响；统一强度偏置在候选内归一化时会自动抵消，不改变概率分布，无需撤销。

---

## 五、业界对照

### 5.1 六基准对照（含 ModelScope `NeoHorse-Jev-4B` 模型卡的表）

NeoHorse-Jev-4B 模型卡（TokenRhythm，2026-09-24）给出了六个基准组的横向表，与我们跑的组**部分重叠**。合并后如下（0–100，越高越好）：

| 模型 | JevBench | Kev | OpenJev 文本 | Nimble | VitaminC | MASSIVE | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| **本实现 · doubao-evolving** | **91.8** | — | — | **94.6** | 73.1 | 跑不了¹ | — |
| **本实现 · doubao-2.1-pro** | 90.9 | — | — | 94.3 | 73.6 | 跑不了¹ | — |
| **本实现 · doubao-2.0-pro** | 90.0 | 81.80 | — | 89.3 | 72.0 | 跑不了¹ | 83.3⁴ |
| **本实现 · doubao-2.1-lite** | 87.9 | 81.77 | — | 88.2 | 72.5 | 跑不了¹ | 82.6⁴ |
| **本实现 · deepseek-flash** | 83.1 | 80.40 | — | 82.1 | **75.5** | 跑不了¹ | 80.3⁴ |
| JEV-27B（AutoTrust，Apache-2.0） | 88.70 | 83.75 | 73.89 | **92.91** | 77.46 | **87.71** | **84.07** |
| Jev 1.13.0（TypeSafe 托管） | 87.18 | **85.52** | 72.96 | 91.84 | — | — | 83.85² |
| Open-Jev-9B（开源权重） | 77.13 | 77.87 | 65.39 | 80.50 | 68.28 | 84.86 | 75.67 |
| NeoHorse-Jev-4B（开源权重） | 75.73³ | 81.92 | 58.74 | 87.23 | 77.13 | 85.43 | 77.70 |
| Kev-4B（开源权重） | 73.71 | 81.47 | 54.75 | 73.40 | 76.46 | 85.71 | 74.25 |
| Laya English（开源权重） | 55.82 | 61.30 | 40.07 | 45.04 | 78.63 | 68.57 | 58.24 |
| Laya Typed Decisions | — | — | — | 48.94 | 78.30 | 65.43 | — |
| NeoHorse-1-4B（基线参照） | — | — | — | 69.15 | 63.27 | 82.86 | — |

¹ MASSIVE-en 是 18 类分类，超出本服务候选上限 10，改不了。

² Jev 1.13.0 的 AVG 是 AutoTrust 代跑结果，VitaminC / MASSIVE 两项缺，均值按其余组折算。

³ NeoHorse 模型卡标注：JevBench 的 **75.73 是按任务族宏平均**，同文件里给出的**逐样本准确率是 75.32**——我们的数字是逐样本准确率，应与 75.32 比较。

⁴ 我们的 AVG 是 JevBench / Kev / Nimble / VitaminC **四组均值**，缺 MASSIVE 与 OpenJev 两组，**不可与 6 组均值直接横比**。

### 5.2 同口径：JevBench 公开 231（逐样本准确率）

| 模型 | 公开 231 | Hard 111 | 来源 |
|---|---:|---:|---|
| **本实现 · doubao-evolving** | **91.8%** | **83.8%** | 本次实测 |
| **本实现 · doubao-2.1-pro** | 90.9% | 82.0% | 本次实测 |
| Jev 1.13.0（TypeSafe 托管） | 86.58% | 72.97% | Open-Jev-27B-v1.1 模型卡 |
| Open-Jev-27B-v1.1 | 85.28% | 72.07% | 同上 |
| Open-Jev 9B | 77.49% | 59.46% | 同上 |
| Open-Jev 2B | 64.94% | 41.44% | 同上 |
| NeoHorse-Jev-4B | 75.32% | — | NeoHorse 模型卡（逐样本） |

### 5.3 横比前必看的口径差异

1. **JevBench 指标定义**：NeoHorse 表用任务族宏平均（75.73），我们用逐样本准确率。他们文件里同样给出逐样本 75.32，**可比的是 91.8% vs 75.32%**。
2. **Nimble 子集**：已对齐。双方都用 compat282 规则从 324 条里筛（对方标注 282 条，我们实测 280 条）。
3. **VitaminC 抽样**：已对齐。双方都用 Nimble manifest 的 599 条（seed 20260918）。
4. **Kev**：我们用六子集 clean 等权，与业界口径一致；但 Jev 1.13.0 未公布分子集明细，无法逐子集核对。
5. **不是 prefill 打分**：我们多了一层「模型需配合吐出标签」的依赖（关闭 `logit_bias` 时失败率 1.3%，开启后 0；本轮矩阵 0），业界是直接读分布。
6. **未做温度校准**：只比对 argmax，不比对概率质量（ECE / Brier 未参与），因此**本报告的任何概率都不要当置信度用**。

### 5.4 差距归因

| 观察 | 归因 |
|---|---|
| JevBench / Nimble 全面超过业界最好水平 | 底座模型更强，机制等价（见结论 2） |
| Kev 落后 3.7 个点，失分集中在 transfer-v9 | 该子集偏「跨域迁移」，需更强泛化；本轮把 Kev 扩到 3 条路由（含 `deepseek-flash`）仍未追平 |
| DeepSeek 两个路由整体偏低 | 与本机制（单 token 标签读出）适配性差，非模型能力问题 |
| 开源权重模型 Kev 普遍 81~83% 但我方 JevBench 更高 | 两类基准考察的迁移能力不同，不宜用单一基准下结论 |
| `doubao-2.0-mini` 在 Nimble `score` 上只有 57.4% | 低端路由在有序评分上不稳定；高端路由同一子集可达 92.6~94.4% |

### 5.5 跨厂商同题集对照（火山方舟 vs 阿里云百炼）

本服务现在支持多厂商（`provider` 配置项，见 [README 四、配置](../README.zh-CN.md#配置)）。两个厂商的机制差异不影响读出逻辑，但**槽位预算差一倍以上**，所以在**同一批题**上做一次对照。

题集：JevBench choice 分层抽样 60 题，**两批的题号列表已逐项校验完全一致**。为做等约束比较，两边都钉在 `top_logprobs=5`——注意这对火山是**人为压制**（它实际能用 20，线上也是按 20 跑的）。

| 端点 / 模型 | 准确率 | p50 | 读出失败 |
|---|---:|---:|---:|
| **火山 `doubao-seed-2-1-lite`**（压到 5 槽位） | **57/60 = 95.0%** | 853 ms | 0 |
| 百炼 `qwen3.7-plus` | 54/60 = 90.0% | 967 ms | 0 |
| 百炼 `qwen3.8-flash` | 54/60 = 90.0% | 913 ms | 0 |
| 百炼 `qwen3.5-plus` | 53/60 = 88.3% | 1107 ms | 0 |

逐题对比（火山 vs 百炼最强的 `qwen3.7-plus`）：

```
对错结论一致   57/60
两边都错        3 题
火山错/百炼对   0 题
火山对/百炼错   3 题   ← 全部是 hard 档（2 个概率推理 + 1 个多跳）
```

**火山严格占优（多对 3 题、少对 0 题），但准确率差距不显著**——3 题之差在 60 题样本上约 1.3 个标准误。**真正的差异是结构性的槽位预算：**

| 候选数 | 火山 全覆盖 | 百炼最好（`qwen3.5-plus`） |
|---|---:|---:|
| 3 | 4/4 | 4/4 |
| 4 | 3/4 | **4/4** |
| 5 | 2/4 | 3/4 |
| **6** | 0/4 | **0/4** |

即：**火山能稳定跑 10 个候选，百炼只能跑 4 个**（6 候选在百炼全部 6 个测试模型上都是 0/4 全覆盖）。槽位是被噪声吃掉的——百炼 top-5 里混着 `<|im_end|>`、`The`、`Let`、`A/B/C/D`，按模型不同平均每题占 0.88（`qwen3.5-plus`）~ 2.44（`qwen3.8-27b`）个槽位，**越新的旗舰噪声越多**。

选型结论：**百炼上应选 `qwen3.5-plus` 或 `qwen3.7-plus`，不要选最新的 3.8 旗舰。**

还有一个与文档相悖的实测事实：**百炼的 `logprobs` 支持名单和官方文档是反的**——文档白名单里列的 `qwen-plus-2025-04-28`、`qwen3-32b` 实测返回 `null`；文档没列的 qwen3.5/3.6/3.7/3.8 全系实测都支持（前提是显式关思考）。两个厂商的共同硬前提是：**思考模式与 `logprobs` 互斥**，百炼上不传关思考参数会让 `logprobs` 静默变 `null`（火山是直接 400）。

产物绑定：百炼三模型 `benchmarks/results/probe-aliyun-20261002-174305.json`（sha256 `eec18b67c1411968…`）；火山对照 `benchmarks/results/probe-aliyun-20261002-174832.json`（sha256 `d694d9b872641601…`）。复现命令见 [benchmarks/README.zh-CN.md](README.zh-CN.md)。

### 5.6 备选读出策略（逐候选 yes/no）的实测结论：不采用

参考开源项目 LLM2Jev 的思路评估过第二条读出路径（记为 **B**）：**每个候选拆成一次独立的 yes/no 询问**，每个候选一次调用，读位置 0 的 `yes`/`no` 两个 token，`q = P(yes)/(P(yes)+P(no))` 作为该候选分数，N 个分数再归一化。

它唯一不可替代的优势是**每次只占 2 个 token，因此候选数不受 `top_logprobs` 槽位约束**——正好对准 §5.5 里百炼只有 5 个槽位这个短板。探针 `benchmarks/probe_yesno.py` 在两个厂商各跑了一轮，结论是不采用：

| | 火山方舟（20 槽位） | 阿里云百炼（5 槽位） |
|---|---|---|
| B 的读出命中率 | **140/140 = 100%**（靠等量 `logit_bias`） | **99/140 = 70.7%**（无分词接口 → 只能裸跑） |
| 同子集准确率 vs A | 打平（各 28/30，**错题完全相同**） | 打平（各 4/6） |
| B / A 调用量 | 4.67× | 4.67× |
| 延迟 p50（B vs A） | 9777 ms vs 2598 ms | 9573 ms vs 1601 ms |
| 结论 | **能跑但零增益** | **根本跑不了** |

**百炼上的失败归因（41 次未命中，全部同一个原因）**：

```
模型实际输出的首 token：'no' × 41
失败原因：top_logprobs at position 0 are missing ['yes'] × 41
```

即 5 个槽位被 `no` 和噪声占满，`P(yes)/(P(yes)+P(no))` 的分母半边拿不到。**模型答得完全正确，是槽位不够**——这不是改 prompt 能修的，需要 `logit_bias` 把 `yes`/`no` 同时推进 top-k，而百炼没有在线分词接口拿不到 token id。

**一个必须注意的口径陷阱**：B 在百炼上的全量准确率 86.7% 看起来"打败"A 的 66.7%，但那是假象——A 在同一批 30 题里**只有 6 题能跑**（24 题因候选超限 422），分母完全不同。按 A 能跑的 6 题同子集一拉平，**两者都是 4/6 = 66.7%**。

这轮同时量化了 A 在百炼上的短板：**80% 的题因候选超限跑不了**。B 想解决的正是这个，但它自己在该厂商上不成立。

产物绑定：火山 `benchmarks/results/probe-yesno-20261002-164300.json`（sha256 `f66b3cc407d7234a…`）；百炼 `benchmarks/results/probe-yesno-20261002-182144.json`（sha256 `d1e230829e88e0ff…`）。重启条件与不采用的完整理由见 [docs/design.md §7.1](../docs/design.md)。

### 5.7 同题同口径：Intern-Decision 七项基准对照（InternLM）

[Intern-Decision](https://github.com/InternLM/Intern-Decision) 是同一赛道的开源项目（Qwen3.5 微调的 4B/2B/0.8B 权重 + 96 条已知分布校准基准），它把**七项准确率测试集和校准 pilot 一起放在仓库里**——因此这是本报告唯一一次「同题同口径」的横向比较，与 5.1 / 5.2 的「引用公布值」有本质区别。

**题目对齐是先验证过的**：JevBench 231 的 id 列表哈希与我们主矩阵完全一致（`04399f09d6b39036…`），其余四套件直接使用其 bundle（四个 `test.jsonl` 的 sha256 与上游 `manifest.json` 记录**逐个核对一致** ✅）；决策总数 12,351 与其 manifest 记录（10,751 行 / 12,351 个决策）逐项吻合。评分口径见 6.3。

| 模型 | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **本实现 · doubao-evolving** | 100.00 | **98.61** | **82.88** | 73.45 | 89.35 | 89.58 | 95.07 | **89.85** | **0.2500** | 0.1009 |
| **本实现 · doubao-2.1-pro** | 100.00 | **98.61** | 81.08 | 73.50 | 87.74 | 89.55 | 95.11 | 89.37 | 0.2641 | 0.1054 |
| **本实现 · doubao-2.1-lite** | 100.00 | 97.22 | 77.48 | 73.75 | 83.23 | 87.92 | 94.21 | 87.69 | 0.3086 | 0.1283 |
| **本实现 · deepseek-flash** | 100.00 | 97.22 | 66.67 | 70.65 | 88.06 | 88.49 | **96.61** | 86.81 | 0.5466² | 0.2289² |
| Jev（TypeSafe 托管） | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.3584 | 0.0947 |
| Intern-Decision-4B（开源权重） | 100.00 | 98.61 | 73.87 | **80.55** | **96.45** | 90.82 | 89.86 | **90.02** | 0.3468¹ | 0.0653¹ |
| Intern-Decision-2B（开源权重） | 100.00 | 84.72 | 63.96 | 79.35 | **96.45** | 89.96 | 78.33 | 84.68 | 0.4373¹ | 0.1002¹ |
| Intern-Decision-0.8B（开源权重） | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.5295¹ | 0.0657¹ |
| JevK5 | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.3662 | **0.0467** |
| SemIf | 100.00 | 98.61 | 61.26 | 62.80 | 85.16 | 89.22 | 92.53 | 84.23 | 0.4980 | 0.1122 |
| Kev | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.7383 | 0.2622 |
| Laya | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | **92.84** | 14.84 | 57.77 | 0.8042 | 0.2465 |

均值（Average）为七项准确率的**算术平均**，与该表定义一致；Brier / ECE 取自 **111 条 JevBench-Hard**，置信度取 max(P)，ECE 用 10 等宽区间——全部照搬其文档定义。**加粗 = 本实现在该列的最优值，或公布值中的最优**（Easy 列多家并列满分，不加粗）。

**怎么读这张表：**

- **均值**：`doubao-evolving` 89.85、`doubao-2.1-pro` 89.37，介于 Jev（88.74）与 Intern-Decision-4B（90.02）之间；四条路由均高于其余全部公布值（JevK5 85.16、SemIf 84.23、Kev 79.56、Laya 57.77）。
- **强项是 hard 档**：82.88 / 81.08，比公布最高值 73.87 高 7~9 个点；Easy 与 Original 与最好的模型同档（100 / 98.61）；WildJailBreak 94.2~96.6，其中 `deepseek-flash` 的 **96.61 是全表最高**。
- **弱项是 `typed_decisions` 与 ToolACE**：前者 70.7~73.8 vs ID-4B 的 80.55，后者 83.2~89.4 vs ID-2B/4B 的 96.45。归因清楚——Intern-Decision 是把「状态 → 结构化决策」当**训练目标微调**出来的专项模型，我们是通用底座 + 单 token 标签读出、**未做任何微调**；反过来说，通用模型在同题上进入 90 分档、hard 档还反超 9 个点，才是机制可迁移性的证据。
- **分布列要分开看**：Brier 同时受准确率影响（hard 档高 9 个点会机械地压低 Brier），ECE 才是纯校准列——我们 0.1009 与 Jev 的 0.0947 接近，但不如 ID-4B 校准后的 0.0653。

¹ Intern-Decision 三行的 Brier / ECE 用其**在独立划分上拟合的温度**（T = 1.99 / 2.10 / 2.75，见其表 T 列）之后的概率计算；我们与本表其它基线都是**未校准**的原始概率——这两列因此不能当「校准能力」横比。

² DeepSeek 只返回 1 个真实概率，其余 `top_logprobs` 是占位符（-9999）；占位符按「最低观测值 −3 nats」兜底，coverage 仅 0.17~0.34（hard 111 题里只有 38 题达到 reliable 阈值），这两格是**下限近似**。它的准确率不受影响（argmax 仍取真实观测值）。

### 5.8 同题同口径：96 条已知分布校准 pilot

上游 pilot 给出每条题的**精确参考分布**，用期望多类别 Brier / 期望 ECE 评分（越低越好，口径与其 `docs/CALIBRATION_BENCHMARK.md` 一致）：

| 模型 | 期望 Brier ↓ | 期望 ECE ↓ | 样本 |
|---|---:|---:|---:|
| **本实现 · doubao-evolving** | **0.258** | 0.309 | 94 |
| **本实现 · doubao-2.1-pro** | 0.262 | 0.308 | 94 |
| **本实现 · doubao-2.1-lite** | 0.305 | 0.342 | 94 |
| **本实现 · deepseek-flash** | 0.351² | 0.369² | 94 |
| Intern-Decision-4B，未校准 | 0.628 | 0.213 | 96 |
| Intern-Decision-4B，已校准（T=1.99） | 0.550 | **0.089** | 96 |
| Jev（`jev-1.13.0`） | 0.595 | 0.130 | 96 |

**Brier 我们明显更低（0.258 vs 0.550 / 0.595），ECE 明显更差（0.309 vs 0.089 / 0.130）——两者合起来说明同一件事：概率的分辨力够，但置信度过于激进。** 最直接的例证是 pilot 首题「公正骰子」（参考分布为均匀 1/6 = 0.1667）：我们给出 `1` 的概率 **0.642**、其选项顺序反转后给出 `6` 的概率 **0.973**。这不是读出错误（argmax 与参考一致），而是把「确定答案」的自信直接搬到了「已知均匀」的分布上。

这也正是温度校准要解决的问题：4B 的「已校准」列把 ECE 从 0.213 压到 0.089，说明该任务上校准收益显著。**在补标注数据完成校准之前，5.7 / 5.8 的概率不要当置信度用**（与结论 8 一致）。

两条口径说明：① 我们的样本是 **94 而非 96**——pilot 中 `sum_of_dice/02` 的正反两个变体是 13 候选，超出本服务候选上限 10，四条路由上均被拒（同一对题，与路由无关）；② 上游还给出六个类别的明细，本轮只做总体对照（类别字段在其 `references.jsonl` 的 `category` 里，需要时可拆）。

产物绑定：`benchmarks/results/intern-20261003-125308-full.json`（sha256 `7acb2dbf884846d4…`）——4 路由 × 6 套件、43,388 次调用、1 小时 56 分。**无连接层重试**（10,847 条/路由全部一次成功），读出失败共 10 例：8 例是上述 pilot 超限题，另 2 例是 `doubao-2.1-pro` 在 toolace / wildjailbreak 各遇到一次服务端错误（Ark 500 / 请求失败，按口径记为答错，各占该套件 0.3% / 0.05%）。本节的题目哈希、数据哈希与题集 id 哈希均记录在 `provenance.json`，复现命令见 6.1。

---

## 六、复现与可复核性

本报告每个数字都能沿「数据集 → 子集规则 → 评测协议 → 运行产物」这条链复核。机器可读版本见 `provenance.json`。

### 6.1 复现命令

```bash
# 1) 数据（上游仓库，各自许可）
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench
git clone --depth 1 https://github.com/bespokelabsai/nimble   benchmarks/nimble
git clone --depth 1 https://github.com/jaredpalmer/kev        benchmarks/kev
# VitaminC 通过 HF datasets 在线拉取（tals/vitaminc validation）

# 2) 全量矩阵（第四节主结果，约 2.6 万次调用）
python3 benchmarks/run_matrix.py                                  # 8 路由 × 3 组基准
python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite,deepseek-flash
python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-<ts>.json   # 断点续跑

# 3) 单基准 / 单路由（第 4.4 节实验用这个）
python3 benchmarks/run_jevbench.py --tag nobias
python3 benchmarks/run_jevbench.py --tag withbias --logit-bias
python3 benchmarks/run_nimble.py   --model doubao-2.1-lite
python3 benchmarks/run_vitaminc.py --model doubao-2.1-lite

# 4) Intern-Decision 七项对照（5.7 / 5.8 节，约 4.3 万次调用）
#    数据不随仓库分发，按 benchmarks/intern_decision.py 顶部 DATA_HINT 从上游抓取
python3 benchmarks/run_intern_suite.py --routes doubao-evolving,doubao-2.1-pro,doubao-2.1-lite,deepseek-flash

# 5) 刷新可复核信息（每次评测后重跑；matrix 与 intern 两类产物一起传入，provenance.json 会同时记录）
python3 benchmarks/make_provenance.py --run benchmarks/results/matrix-<ts>.json benchmarks/results/intern-<ts>-full.json
```

### 6.2 实现指纹

| 项 | 值 |
|---|---|
| 实现指纹（`src/llm2decision/` + `benchmarks/` 源码，26 个文件内容哈希） | `d5efd27f85aed25f…` |
| 引擎 | `POST /v1/systemone`（label-logit readout） |
| 解码 | `temperature=0`（贪心）、`max_tokens=1`、**只读生成序列位置 0** |
| 候选上限 | 10（超出直接 422，不参与评测） |
| 输出切口 | hard-label argmax；无 LLM-as-judge |
| 校准 | 未做温度校准（`temperature_scale=1.0`）；5.7 / 5.8 的 Brier / ECE 是**未校准的原始概率**，不得当置信度使用 |
| 失败口径 | 连接失败单独统计并在失败率 >20% 时标记单元格无效；模型拒绝作答记为答错 |

上表的指纹标识的是本仓库当前这棵树。相对第四节那次运行（`matrix-20261003-022427.json`），本轮之后代码有三处变化：① `src/llm2decision/core/readout.py` 新增 `SENTINEL_LOGPROB`，把厂商占位符（-9999）判为「未观测」——它只改变 `coverage` / `reliable` 两个字段与兜底后的分布形状，**不进入 argmax，因此第四节的准确率数字不受影响**（既有报告也从未发布过依赖 coverage 的数值）；② `src/llm2decision/api/debug.py` 的调试页中英文切换（UI 层）；③ 新增两个基准脚本 `benchmarks/intern_decision.py`、`benchmarks/run_intern_suite.py`（5.7 / 5.8 用）。第四节的数字因此仍可归因到当前实现，5.7 / 5.8 的分布列按新的哨兵规则计算。运行产物随 Release [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003) 发布。

### 6.3 数据集与子集规则

| 基准 | 来源 | 子集规则 | 评分口径 | 本地数据哈希 |
|---|---|---|---|---|
| JevBench | `fstandhartinger/jevbench`（MIT） | 公开 231 = original 72 + easy 48 + hard 111，全量不抽样 | 题级 hard-label argmax | `5c2414edb3006b8b…`（与上游 manifest 记录一致 ✅） |
| Nimble | `bespokelabsai/nimble` 的 `data/eval.jsonl` | compat282：state ≤384 tokens、请求体 ≤2048 tokens、候选 ≤26、整族保留（实测 **280** 条） | 逐样本完全匹配 | `8e9e48b8de520659…` |
| VitaminC | HF `tals/vitaminc` validation（CC BY-SA 3.0） | nimble599：按 Nimble manifest 的 599 个 `unique_id` 精确取行（匹配 599/599） | 3 路 choice argmax | `30bd72d23b3e58ae…`（manifest 文件） |
| Kev | `jaredpalmer/kev`（Apache-2.0）6 个子集 | 全量，仅 `_meta.variant=="clean"`；候选 >10 的题跳过并计数 | 题级 argmax，六项等权平均 | `8d5765d7aec4d08c…`（6 个文件） |
| Intern-Decision 七项 | `InternLM/Intern-Decision` 的 `benchmarks/accuracy-v1`（仓库 Apache-2.0；**AG News 的上游 license 标注为 unknown**，本项目不重分发数据） | 其 bundle 全量不抽样：agnews 7,600 / toolace 310 / typed_decisions 400 条（2,000 决策）/ wildjailbreak 2,210；JevBench 用本表首行同一批题（id 逐项一致） | 一行多问 = 一次请求（typed_decisions 一行产 5 个决策）；逐决策 hard-label argmax；七项为算术平均；hard 档另报未校准 Brier / ECE | `4ff1cad1531f29cb…`（5 个文件；4 个 `test.jsonl` 与上游 manifest 逐个核对一致 ✅） |
| Intern-Decision 校准 pilot | 同上仓库 `benchmarks/known-distribution-pilot-v1` | 全量 96 条（其中 2 条是 13 候选、超候选上限 10，实测覆盖 94） | 期望多类别 Brier / 期望 ECE 对精确参考分布；准确率列按「与参考 argmax 一致」计 | `54a2c97dffd972aa…`（3 个文件，与上游 manifest 一致 ✅） |

### 6.4 运行产物绑定

| 基准 | 实际题数 | 题集 id 列表哈希 |
|---|---:|---|
| JevBench | 231 | `04399f09d6b39036…` |
| Nimble | 280 | `f0584784646c74f6…` |
| VitaminC | 599 | `2154d495d6b8d7bd…` |
| Kev（题级） | 5,768 | `73fd023d385cb87b…` |
| Intern-Decision：typed_decisions | 400 行 / 2,000 决策 | `7bb7caf4aebac8d0…` |
| Intern-Decision：toolace | 310 | `a6aae10f092815a4…` |
| Intern-Decision：agnews | 7,600 | `8e83621e743626bb…` |
| Intern-Decision：wildjailbreak | 2,210 | `ba4c08831294c81b…` |
| Intern-Decision：pilot | 96（有效 94） | `6d79b66cbdb7929b…` |

上表第一行 JevBench（231，`04399f09d6b39036…`）在 5.7 那次 Intern-Decision 运行中算出的 id 哈希完全相同——这是 5.7「同题」说法的机器可验证证明，也是两次运行能放进同一张表的前提。

**第四节数字绑定本轮运行产物：**
- 主体矩阵：`benchmarks/results/matrix-20261003-022427.json`（sha256 `80875133be0562ea…`，8 路由 × 4 组基准，26,184 次调用）
- 5.7 / 5.8 对照：`benchmarks/results/intern-20261003-125308-full.json`（sha256 `7acb2dbf884846d4…`，4 路由 × 6 套件，43,388 次调用，无连接层重试）
- 4.4 节实验（同目录）：`jevbench-20261003-093725-nobias.summary.json`、`jevbench-20261003-033600-withbias.summary.json`、`jevbench-20261003-033544-withbias2.summary.json`
- 机器可读 provenance：`benchmarks/provenance.json`（实现指纹、数据集哈希、题集 id 哈希，以及上述**两次运行**的文件 sha256 与题集哈希）

这些产物不随仓库提交（体积原因），统一发布于 Release [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003)；5.5 / 5.6 的跨厂商探针与附录 A 的历史产物也一并收录，便于一处核验。下载后可按上列 sha256 校验。

**哈希随每次运行变化**——分数必须与具体哈希绑定，这正是本节存在的意义。

### 6.5 已知不可复核项（如实列出）

| 项 | 为什么不可复核 |
|---|---|
| OpenJev 文本（19 项） | 该系的 harness 与 prompt 模板未公开，且 `control`(805)、`chess`(500) 两个数据源找不到 —— 因此本报告不填这一格 |
| `zhihz/openjev` 的 89.8% / 97.7% | 它的评测数据未随发布包提供，只能引用其公布值。注意它**与本项目机制同构**（只读下一 token 的候选字母分布、不生成文本），是独立 Web 应用而非模型权重 |
| 开源模型的 JevBench 分数 | 27B / 9B 权重需 GPU 与 Open-Jev loader，本机无法运行，表中为其公布值 |
| 业界表的 Kev / Nimble / VitaminC 分数 | 来自各模型卡公布值，无法逐题复现，只能对齐子集规则后引用 |
| Intern-Decision 三行与其 Brier / ECE | 其权重需 GPU 与 XTuner 后端，本机无法运行，只能引用公布值；这两列还叠加了各自拟合的温度，无法逐题复现 |
| Intern-Decision 的 Hard TVD（10 条公开记录） | 本轮未计算——对照目标是它的两张结果表（七项准确率、校准 pilot），TVD 不在其中；我们报的是自己的 Brier / ECE |

---

## 附录 A：历史产物（已被第四节取代）

### A.1 上一版全量矩阵（2026-10-02，10 路由）

上一版报告第四节绑定 `benchmarks/results/matrix-20261002-112301.json`（10 路由）与 `matrix-20261003-005505.json`（`deepseek-flash` 的全量补跑）。因两次运行间存在 ±2~7 题的波动（4.1 脚注 ²），新版正文不与旧数字逐格合并；旧版报告全文保留在 git 历史，产物在 Release [`eval-20261002`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261002)。三个已下线路由名当时的数字：

| 旧路由名（10-02 运行） | JevBench 231 | Hard 111 | Nimble 280 | VitaminC 599 | Kev 六子集等权 |
|---|---:|---:|---:|---:|---:|
| doubao-2.0-lite | 90.0% | 80.2% | 86.4% | 73.1% | — |
| deepseek-v4.1-flash | 87.0% | 75.7% | 83.2% | 75.5% | 80.7% |
| deepseek-v4-flash | 73.5% | 53.1% | 60.4% | 63.4% | — |

### A.2 早期单基准产物

以下数字来自**早期单独跑的单基准产物**，口径与第四节不一致（Nimble 用 324 条全量、VitaminC 用 300 条抽样、分母含失败题），**已不再作为结论依据**。保留仅为追溯「换底座 + 对齐口径前后」的差异。

| 基准 | 早期结果（`doubao-2.0-mini`） | 第四节同模型口径 | 差异来源 |
|---|---|---|---|
| JevBench 公开 231 | 187/231 = 80.95%；hard 71/111 = 63.96% | 189/231 = 81.8%；hard 71/111 = 64.0% | 不同运行（±2 题）；早期运行 bias 关闭、含 3 题读出失败（计为答错） |
| JevBench 分档 | easy 48/48 = 100%、original 68/72 = 94.4%、hard 71/111 = 63.96% | — | 同一批题，不同运行 |
| JevBench 分题型 | choice 112/139 = 80.6%、noul 61/74 = 82.4%、score 14/18 = 77.8% | 见 4.1 脚注 | 同上 |
| Nimble 内置 holdout | 221/324 = 68.21%（choice 61.6% / noul 84.2% / score 54.7%） | 200/280 = 71.4% | **子集不同**：324 条全量 vs compat282 的 280 条 |
| VitaminC | 231/300 = 77.00%（seed 20260918 自抽 300） | 448/599 = 74.8% | **抽样不同**：自抽 300 vs manifest 的 599 |
| 延迟 | p50 ≈ 399 ms、p95 ≈ 952 ms（并发 6） | 355 ms | 并发与批处理条件不同 |

**一条被本节数据推翻的旧结论**：早期报告曾写「有序评分（score）是全行业短板，本实现 54.69%」。那个 54.69% 来自上表 Nimble **324 条**口径的 64 道 score 题；在对齐后的 **280 条**口径下（54 道 score 题）同模型本轮是 57.4%，而最好的路由达 **94.4%**。因此该结论**不成立**，第四节已改为按路由分别陈述（见 4.1 脚注与 5.4）。

早期产物文件（均收录于 Release [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003)）：

- `benchmarks/results/jevbench-20261002-074113-nobias.summary.json`（187/231）
- `benchmarks/results/jevbench-20261002-074204-withbias.summary.json`（186/231）
- `benchmarks/results/nimble-20261002-074243-nobias.summary.json`（324 条）
- `benchmarks/results/vitaminc-20261002-074608-nobias.summary.json`（300 条）
