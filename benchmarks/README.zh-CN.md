[English](README.md) | [简体中文](README.zh-CN.md)

# benchmarks 使用指南

本目录是**评测台**：把 `src/llm2decision/` 里的决策服务（`POST /v1/systemone`）拉到 Jev 类 benchmark 上跑，回答「准不准、贵不贵、跟业界差多少」。

- **本文只讲怎么跑。** 跑出来的数字、横向对照与口径分析在 [REPORT.md](REPORT.md)。
- 服务本身怎么用、怎么部署，见 [项目根 README](../README.md)。
- 机制原理（只读位置 0、标签句柄、温度校准），见 [docs/design.md](../docs/design.md)。

## 前置条件

| 项 | 说明 |
|---|---|
| Python | 3.11+；服务依赖 `httpx` / `pyyaml` / `pydantic` / `fastapi`；VitaminC 另需 `datasets` |
| 方舟 Key | 项目根目录的 `llm2decision.yaml` 要已配好可用路由（`defaults.api_key`） |
| 工作目录 | **所有命令都在项目根目录执行**（脚本会自行把根目录插入 `sys.path`） |
| 网络 | 需能访问 `ark.cn-beijing.volces.com`；VitaminC 还要能访问 HuggingFace |

## 一、脚本一览

| 脚本 | 用途 | 需要的数据 |
|---|---|---|
| [run_jevbench.py](run_jevbench.py) | JevBench 公开 231 题（72 original + 48 easy + 111 hard） | `jevbench/`（手动 clone） |
| [run_nimble.py](run_nimble.py) | Nimble 内置 holdout：`compat282`（280 条，对齐业界口径）/ `full324`（324 条全量） | `nimble/`（手动 clone） |
| [run_vitaminc.py](run_vitaminc.py) | VitaminC 事实核验：`nimble599`（599 条，对齐业界口径）/ `sample300`（自抽 300） | HF `tals/vitaminc` + `nimble/` 的 manifest |
| [run_kev.py](run_kev.py) | Kev 六子集（decision-v7 / transfer-v4 / transfer-v9），题级计分 | 自动下载 |
| [run_matrix.py](run_matrix.py) | **全量矩阵**：多路由 × 多基准，结果落一份 JSON，支持断点续跑 | 上述各项 |
| [run_intern_suite.py](run_intern_suite.py) | **Intern-Decision 同题对照**：七项准确率（含 JevBench 三档）+ 96 条校准 pilot，结果落一份 JSON，逐套件落盘 | `intern-decision/`（手动抓取，见二）；转换逻辑在 [intern_decision.py](intern_decision.py) |
| [probe_yesno.py](probe_yesno.py) | **备选读出策略探针**：验证「逐候选 yes/no 独立打分」在任意厂商上是否可行。**已实测结论：不采用**（见 [REPORT 5.6](REPORT.md)） | 无（跑 JevBench 题） |
| [probe_provider.py](probe_provider.py) | **跨厂商兼容性探针**：验证任意 OpenAI 兼容端点能否承载本机制（logprobs 可读性 / 关思考写法 / 槽位容量 / 准确率） | 无 + 需 API Key（可从 llm2decision.yaml 匹配路由自动取） |
| [make_provenance.py](make_provenance.py) | 刷新 `provenance.json`：实现指纹 / 数据集哈希 / 题集哈希 / 运行产物哈希 | 无 |

`probe_provider.py` 起初只服务百炼，现已改成通用端点探针，**换厂商之前先跑它，别照着文档猜**（各家 top_logprobs 上限、关思考参数、logprobs 白名单都不一样）：

```bash
# 百炼（key 缺省时按 base_url 从 llm2decision.yaml 里取匹配路由的 key）
python3 benchmarks/probe_provider.py --models qwen3.7-plus --limit 60

# 火山做同题集对照（关思考参数名不同，要显式指定）
python3 benchmarks/probe_provider.py \
  --base-url https://ark.cn-beijing.volces.com/api/v3 \
  --api-key "<ARK key>" --thinking-param thinking \
  --models doubao-seed-2-1-lite-260915 --limit 60

# OpenRouter（一个端点覆盖多家模型；上限缺省按 20，关思考通常是 none）
python3 benchmarks/probe_provider.py \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "<OpenRouter key>" --thinking-param none \
  --models openai/gpt-4o-mini --limit 60
```

`--max-top-logprobs` 缺省按 `--base-url` 自动判定（含 `dashscope` 用 5，其余用 20），可显式覆盖——**别拿 5 去测一个支持 20 的端点**，会把槽位容量测低。

`--limit` 控制准确率题数；槽位容量测试固定按候选数分桶（3/4/5/6 各 4 题），与 `--limit` 无关。

### 常用命令

```bash
# JevBench 全量 231（约 40s）
python3 benchmarks/run_jevbench.py --tag nobias
python3 benchmarks/run_jevbench.py --logit-bias --tag withbias   # 开 logit_bias 兜底

# 冒烟：只跑前 20 题、指定路由
python3 benchmarks/run_jevbench.py --limit 20 --model doubao-2.1-lite

# Nimble / VitaminC
python3 benchmarks/run_nimble.py   --subset compat282
python3 benchmarks/run_vitaminc.py --subset nimble599
python3 benchmarks/run_vitaminc.py --subset sample300 --n 300

# Kev（注意：这个脚本用 --route，且必填）
python3 benchmarks/run_kev.py --route doubao-2.0-pro
python3 benchmarks/run_kev.py --route doubao-2.0-pro --limit 8   # 每子集抽 8 条

# 全量矩阵（8 路由 × 4 组基准，约 2.6 万次调用）
python3 benchmarks/run_matrix.py
python3 benchmarks/run_matrix.py --routes doubao-2.1-lite,doubao-2.1-pro --benches jevbench,nimble
python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite,deepseek-flash   # Kev 题量大，默认只跑指定路由
python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-<时间戳>.json  # 断点续跑

# Intern-Decision 同题对照（七项 + pilot，4 路由约 4.3 万次调用、约 2 小时）
python3 benchmarks/run_intern_suite.py --routes doubao-evolving,doubao-2.1-pro,doubao-2.1-lite,deepseek-flash
python3 benchmarks/run_intern_suite.py --routes doubao-2.1-lite --suites jevbench,pilot --limit 20   # 冒烟

# 刷新可复核信息（每次评测后都应重跑；matrix 与 intern 两类产物一起传入，provenance.json 同时记录）
python3 benchmarks/make_provenance.py --run benchmarks/results/matrix-<时间戳>.json benchmarks/results/intern-<时间戳>-full.json
```

**参数约定上的两个不一致点**（照现状记录，避免踩坑）：`run_kev.py` 用 `--route`（必填），其余 runner 用 `--model`（可选，缺省走 `llm2decision.yaml` 的 `default_model`）；`--concurrency` 各脚本默认 6~8 不等。

## 二、数据准备

四组基准里只有 Kev 是**自动下载**的，其余需要手动准备：

```bash
# 1) JevBench（MIT）—— run_jevbench.py 读 datasets/public/*.jsonl
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench

# 2) Nimble（Bespoke Labs）—— run_nimble.py 读 data/eval.jsonl；
#    同时提供 VitaminC nimble599 子集所需的 manifest（docs/assets/public-benchmarks/subsets/）
git clone --depth 1 https://github.com/bespokelabsai/nimble benchmarks/nimble

# 3) Kev（Apache-2.0）—— run_kev.py 的 ensure_data() 会自动拉到 benchmarks/kev/*.jsonl，无需手动
# 4) VitaminC —— 数据不在仓库里：run_vitaminc.py 通过 HF datasets 在线拉 tals/vitaminc
pip install datasets

# 5) Intern-Decision（仓库 Apache-2.0；AG News 的上游 license 标注为 unknown，本项目只评测、不重分发）
#    JevBench 三档复用 1) 的同一批题（id 逐项一致），无需另抓
g='repos/InternLM/Intern-Decision/contents/benchmarks'
mkdir -p benchmarks/intern-decision/accuracy-v1/{agnews,toolace,typed_decisions,wildjailbreak} \
         benchmarks/intern-decision/calibration-pilot-v1
for s in agnews toolace typed_decisions wildjailbreak; do
  gh api -H 'Accept: application/vnd.github.raw' "$g/accuracy-v1/$s/test.jsonl" \
    > "benchmarks/intern-decision/accuracy-v1/$s/test.jsonl"
done
gh api -H 'Accept: application/vnd.github.raw' "$g/accuracy-v1/manifest.json" \
  > benchmarks/intern-decision/accuracy-v1/manifest.json
for f in inputs.jsonl references.jsonl manifest.json; do
  gh api -H 'Accept: application/vnd.github.raw' "$g/known-distribution-pilot-v1/$f" \
    > "benchmarks/intern-decision/calibration-pilot-v1/$f"
done
# 抓完用 accuracy-v1/manifest.json 里的 sha256 校验；这台机器上 sha256 已核对一致
```

自检：

```bash
python3 benchmarks/run_matrix.py --limit 5
# 预期：打印前三组各 5 条，并提示「未指定 --kev-routes：跳过 Kev（其余三组照跑）」
python3 benchmarks/run_matrix.py --limit 5 --kev-routes <任一路由名>   # 再验 Kev 的自动下载
```

## 三、产物与命名

全部落在 `results/`：

| 脚本 | 文件名 | 内容 |
|---|---|---|
| `run_jevbench.py` / `run_nimble.py` / `run_vitaminc.py` | `<基准>-<时间戳>-<tag>.jsonl` + `.summary.json` | 逐题记录 + 汇总 |
| `run_kev.py` | `kev-<时间戳>-<tag>.json` | 六子集汇总 + 跳题统计 |
| `run_matrix.py` | `matrix-<时间戳>.json` | 路由 × 基准的矩阵，每跑完一个路由落盘一次 |
| `run_intern_suite.py` | `intern-<时间戳>-<tag>.json` | Intern-Decision 六套件 + pilot 的逐条记录与汇总，每跑完一个套件落盘一次 |
| `probe_yesno.py` | `probe-yesno-<时间戳>.json` | 四个假设的原始证据 |
| `probe_provider.py` | `probe-provider-<时间戳>.json` | 四个假设的原始证据（含每题的 top tokens） |
| `make_provenance.py` | `provenance.json` | 机器可读溯源（固定文件名，每次覆盖） |

`--tag` 只用在前四个脚本上，用来区分同一基准的不同配置（如 `nobias` / `withbias`）。

## 四、口径约定（读数字前必看）

这些约定决定了数字怎么算，改动它们等于改结论：

1. **准确率是 hard-label argmax**，与业界口径一致，无 LLM-as-judge。
2. **分母是剔除连接失败后的有效题数 `n_valid`**，不是总题数；失败数单独计在 `n_errors`。唯一例外是 `run_jevbench.py --logit-bias` 那组 A/B 实验（分母含失败题），REPORT.md 4.4 节有说明。
3. **连接类错误重试 3 次**（退避 1s/3s/9s），不做事后重采样。
4. **失败率 > 20% 的单元格标记 `valid=false`**，不计入准确率结论。
5. **候选上限 10**：超过 10 个候选的题会被 `run_kev.py` 跳过并计数（`skipped_too_many_candidates`）。这也是 MASSIVE（18 类）、BANKING77（77 类）跑不了的原因。
6. **子集口径**：Nimble 用 `compat282`（实测 280 条）、VitaminC 用 `nimble599`（匹配 599/599）、Kev 只计 `_meta.variant == "clean"` 的题。完整规则与哈希见 REPORT.md 第六节。
7. **Intern-Decision 的两张表照搬其口径**：七项均值是七项准确率的**算术平均**；pilot 用**期望**多类别 Brier / ECE（对精确参考分布，不是对采样标签）；hard 档的 Brier / ECE 是**未校准**原始概率（对方三行是拟合温度后的值），不要当校准结论。见 REPORT.md 5.7 / 5.8。

## 五、想加一个新 benchmark

1. 新建 `run_<名字>.py`，提供三个函数：`to_payload(题) -> dict`（转成 `SystemOneRequest` 的形状）、`predicted_label(题, decision)`、`is_correct(gold, predicted)` —— 可参照 [run_nimble.py](run_nimble.py) 的写法。
2. 在 [run_matrix.py](run_matrix.py) 里 `import` 它、加进 `BENCHMARKS` 与 `load_items()` / `check()` 的分支，矩阵就能带它一起跑。
3. 把数据来源、子集规则、评分口径补进 [make_provenance.py](make_provenance.py) 的 `PROTOCOLS`，否则可复核链会断。
4. 跑完在 [REPORT.md](REPORT.md) 补一节，数字要绑定到具体的 `results/` 产物哈希。

## 六、已知限制

| 限制 | 说明 |
|---|---|
| MASSIVE-en / BANKING77 跑不了 | 18 / 77 个候选，超出服务候选上限 10。绕过路径见 [probe_yesno.py](probe_yesno.py) |
| OpenJev 文本 19 项不可复现 | 上游 harness 与 prompt 未公开，且 `control`、`chess` 两个数据源找不到 |
| 开源模型的 JevBench 分数 | 需 GPU 与 Open-Jev loader，本机跑不了，REPORT.md 里引用的是其公布值 |
| `probe_yesno.py` 不改 `src/llm2decision/` | 它是可行性探针，不参与服务运行；结论见 REPORT.md 与 docs/design.md |
| Intern-Decision pilot 只覆盖 94/96 | 其中 `sum_of_dice/02` 的正反两题是 13 候选，超出候选上限 10，四条路由上均被 422 拒（与路由无关），对照表里我们那几行分母是 94 |
| Intern-Decision 的 AG News 许可未明 | 其上游元数据把 license 标为 unknown，因此本仓库只抓取、不提交、不重分发数据（`benchmarks/intern-decision/` 已在 `.gitignore`） |
