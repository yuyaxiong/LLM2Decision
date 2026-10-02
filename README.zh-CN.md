# LLM2Decision

[English](README.md) | [简体中文](README.zh-CN.md)

[![tests](https://github.com/yuyaxiong/LLM2Decision/actions/workflows/test.yml/badge.svg)](https://github.com/yuyaxiong/LLM2Decision/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**把任意托管 LLM 变成类型化决策模型。** 给它一个 state 和几道带类型的问题，拿回来的是**你的**候选项上的概率分布——而不是生成的文本。

```python
# Instead of asking for text and parsing it...
# "Classify this ticket as billing/technical/other. Reply with one word."

# ...read the probabilities of the options themselves.
POST /v1/decide
{"state": "I was charged twice for the same order.",
 "questions": {"intent": {"type": "choice", "instructions": "Which team should handle this?",
                          "criteria": {"billing": "Charges, invoices, refunds",
                                       "technical": "Problems using the product",
                                       "other": "Neither fits"}}}}
```

```json
{"decisions": {"intent": {"label": "billing",
                          "distribution": {"billing": 0.946, "technical": 0.0476, "other": 0.0064},
                          "confidence": 0.946, "coverage": 1.0, "reliable": true}},
 "model": "doubao-seed-2-1-lite-260915", "route": "doubao-2.1-lite",
 "latency_ms": 1427.24, "calls": 1}
```

不生成任何文本，所以没有幻觉，也没有东西要解析。返回的答案是一个分布，你的代码可以直接拿它做阈值判断。

---

## 为什么要有这个项目

用聊天 LLM 做一个小而有限的决策，绕的弯出乎意料地多：

| 通常的做法 | 问题 |
|---|---|
| 提示它"用一个词回答" | 它还是回你一整段；你只好写解析器，再写兜底逻辑 |
| 让它给出 `"confidence": 0.9` | 那是**生成的文本**，不是概率。它从根上就是未校准的 |
| 用 schema 约束输出 | 只保证了输出的*形状*，不保证*置信度*。本质上仍是在生成 token |
| 微调一个小分类器 | 有效，但需要标注数据，而你现在多半还没有 |

本项目走的是读出这条路：让 prompt 停在答案位置，让模型产出**恰好一个 token**，然后读取它分配给每个候选答案的概率。这正是分类头会给出的信号，只不过是从你已有的 API 里拿到的。

**不需要 GPU。不需要微调。不绑定厂商。**

## 你能得到什么

- **`choice`** —— 在你的选项里选一个，并给出所有选项上的分布
- **`noul`** —— 对某个陈述给出是/否概率（`true_probability`，取值在 `[0,1]`）
- **`score`** —— 有序量表上的位置，并提供 `expected` 值用于排序
- **一次请求**里可以对同一个 state 问好几个问题
- 一个诚实的失败模式：如果读出不可信，它会**报错**，而不是编造一个分布

**不确定你的步骤是否适用？** [`docs/when-to-migrate.md`](docs/when-to-migrate.md) 是配套的实用指南：哪些步骤值得改造、每种问题类型的模板、如何把*无法*去掉的生成过程包在决策护栏里、面向改写和摘要的"抽取优先"模式，以及你应该给自己设定的验收标准。

## 安装

```bash
pip install llm2decision            # once published; for now, from source:
git clone https://github.com/yuyaxiong/LLM2Decision && cd LLM2Decision
pip install -e ".[dev]"
```

环境要求：Python 3.10+，以及任意 OpenAI 兼容端点的 API key。

## 配置

复制示例文件并填入你的密钥：

```bash
cp llm2decision.yaml.example llm2decision.yaml
```

```yaml
default_model: doubao-2.1-lite

defaults:
  provider: ark                                   # ark | dashscope | deepseek | openai_compatible
  base_url: https://ark.cn-beijing.volces.com/api/v3
  api_key: "<your-key>"

models:
  doubao-2.1-lite:
    model: doubao-seed-2-1-lite-260915
```

然后启动它：

```bash
uvicorn llm2decision.api.main:app --port 8000
# open http://127.0.0.1:8000/debug for a debug UI
```

**你只需要配一样东西：`llm2decision.yaml`**，就是上面 `cp` 出来的那份——所有选项都在里面。优先级：`route > defaults > environment > built-in default`，环境变量只是兜底、绝不覆盖配置。`.env.example` 是给容器 / CI 场景的**替代**方案（用 `LLM2DECISION_API_KEY` 注入而不写文件），两条路选一条。

### 提供方

不同厂商的差异，决定了这套机制能不能用。所有相关逻辑都在一个文件里：[`src/llm2decision/core/providers.py`](src/llm2decision/core/providers.py)。

| Provider | `top_logprobs` 上限 | 候选上限 | 关闭思考 | 在线 tokenizer |
|---|---:|---:|---|---|
| `ark`（火山方舟） | 20 | **10** | `thinking: {"type":"disabled"}` | 是 |
| `dashscope`（阿里云百炼） | 5 | **4** | `enable_thinking: false` | 否 |
| `deepseek`（DeepSeek） | 20 | **10** | `thinking: {"type":"disabled"}` | 否 |
| `openai_compatible`（通用） | 20 | 8 *（未验证）* | — | 否 |

在接入新厂商前，有三件事值得知道：

1. **`top_logprobs` 不是你的候选预算。** 这些槽位还要分给 EOS token、全角变体、标点，以及模型想开口解释时会去够的那些词（`The`、`Let`、`<|im_end|>`）。实测：20 个槽位能可靠容纳 10 个候选；5 个槽位能容纳 4 个。
2. **思考模型在这里不管用——除非能关掉思考：** 在 `max_tokens=1` 下，推理链会把唯一那个 token 吃掉，位置 0 为空、`logprobs.content` 返回 `null`。厂商在这方面的文档写得很差——我们实测到宣称支持 `logprobs` 的模型实际返回 `null`，也遇到过文档没写却工作正常的。各厂商的具体写法（如 DeepSeek 只认 `thinking: {"type":"disabled"}`，其他写法 HTTP 200 静默失败）见 [`docs/design.md`](docs/design.md)。
3. **所以要实测，别只读文档。** 在把本项目指向新厂商之前，先跑一遍探测脚本：

```bash
python3 benchmarks/probe_provider.py \
  --base-url https://openrouter.ai/api/v1 --api-key '<key>' \
  --thinking-param none --models openai/gpt-4o-mini --limit 60
```

它用大约 140 次调用回答四个问题，并且如果 `logprobs` 不支持，跑 1 次就会停下：`logprobs` 是否透传、有没有办法关闭思考、能容纳多少候选、以及你实际能拿到多少准确率。**如果它报告 `logprobs` 返回 `null`，那这套机制在该端点上就是行不通的**——那是死路，不是配置问题。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/v1/decide` | 原生端点 |
| `POST` | `/v1/systemone` | 请求与响应完全相同，命名是为了兼容 [Jev](https://typesafe.ai/) |
| `GET` | `/v1/models` | 列出路由及其 provider 和真实模型 ID |
| `GET` | `/health` | 存活检查 + 密钥是否已配置 |
| `GET` | `/debug` | 单页 UI：构造请求，查看概率条形图 |

### 请求

```jsonc
{
  "state": "free text or serialized JSON — the context shared by all questions",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "Charges, invoices, refunds", "other": "Neither fits" }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "The customer explicitly asks for a refund."
    },
    "satisfaction": {
      "type": "score",
      "instructions": "How satisfied is the customer?",
      "scale": ["Very unhappy", "Unhappy", "Neutral", "Happy"],
      "values": [0, 1, 2, 3]        // optional; defaults to each label's own numeric
                                    // value when it parses as a number, else its index
    }
  },
  "model": "doubao-2.1-lite",       // optional route name; defaults to default_model
  "debug": false                    // true also returns raw_candidates
}
```

`choice` 需要 `criteria`；`noul` 需要非空的 `instructions`；`score` 需要至少 2 级的 `scale`。

### 响应

| 字段 | 说明 |
|---|---|
| `decisions.<name>.label` | 分布的 argmax（`choice` / `score`） |
| `decisions.<name>.distribution` | 仅在你的候选项上重新归一化 |
| `decisions.<name>.true_probability` | 仅 `noul` |
| `decisions.<name>.expected` | 仅 `score`，使用 `values` 计算 |
| `decisions.<name>.coverage` | 落在你的句柄上的概率质量占比 |
| `decisions.<name>.reliable` | `coverage >= 0.5`——**低于这个值，就不要相信该分布** |
| `decisions.<name>.generated_token` | 模型在位置 0 实际产出的内容 |
| `decisions.<name>.calls` | 该问题消耗的模型调用次数（当前策略下为 1） |
| `decisions.<name>.timing` | `prepare_ms` / `call_ms` / `readout_ms`，**不含排队等待** |
| `decisions.<name>.raw_candidates` | 仅在 `debug: true` 时返回 |
| `latency_ms` | 端到端墙钟时间，**包含服务内并发排队** |
| `calls` | 整个请求消耗的模型调用次数 |

`latency_ms` 减去某问题各段 `timing` 之和就是排队/调度开销——区分"模型慢"和"队列长"的办法。实测：`prepare_ms + readout_ms` 不到 1.5 ms，**几乎所有延迟都来自上游往返**。

### 错误

| 状态码 | 含义 |
|---|---|
| `422` | 错误请求：候选数量超过 provider 上限、未知路由、schema 违规 |
| `502` | 读出失败。详情里会带上**模型实际产出的内容** |

`502` 是刻意设计的。如果位置 0 上不是你的某个候选句柄，本服务拒绝猜测，会把原始输出交给你，而不是编造一个分布。

## 工作原理

```
state + typed questions
        │
        ▼  prompt ends exactly where the answer goes
   ┌─────────────┐
   │  LLM API    │   temperature=0 (greedy)  ·  max_tokens=1  ·  logprobs=true
   └─────────────┘
        │
        ▼  read position 0 only
   candidate handles → logprobs → softmax within candidates
        │
        ▼
   distribution + label + coverage/reliable
```

这个设计带来四条硬约束，且都不可配置：`temperature = 0`（贪心，位置 0 是确定性的，不是抽奖）；`max_tokens = 1`（一个 token 即停，再长读到的就是模型自己写的文字）；只读位置 0（位置 *k > 0* 是写完前缀*之后*的分布——另一个量）；句柄必须是单个 token（多 token 句柄在 `top_logprobs` 里没有条目，读出会漏掉它——`coverage` 下降、`reliable` 变 false，但请求不会失败）。句柄由系统分配（`1`–`9`，再往上 `A`–`K`；本身是 `0-9A-Z` 单字符的 label 直接使用）。

一个可选项：`logit_bias_enabled: true`（需要在线 tokenizer）给每个句柄施加**相同**偏置——均匀偏置在重新归一化时相互抵消，不扭曲相对概率，只是把句柄推进 top-k 以减少读出失败（实测：失败减少，准确率不变）。

更完整的说明，包括实测到的失败模式以及为什么某些东西不被支持，见 [`docs/design.md`](docs/design.md)。

## 校准

`confidence` 是模型给出的概率，**不是经过校准的正确率概率**。如果你想拿它做阈值，先做校准：

```bash
python3 -m llm2decision.calibrate --data labeled.jsonl --cache responses.json --write
```

它通过在 log 空间最小化 NLL 拟合 temperature，并写入你的配置。注意点：几十个样本远远不够——如果几乎全都答对，拟合会凭空造出一个假的 temperature；需要几百个**包含困难样本**的例子（问题欠定时它会警告）。

## 基准测试

`benchmarks/` 目录里放着测试框架、结果，以及一份关于哪些结果无法复现的诚实说明。亮点：一次 10 条路由 × 4 个基准的运行（28,404 次调用，路由名取自当次配置），外加 `deepseek-flash` 的三组全量补跑（1,110 次调用）：

| 路由 | JevBench 231 | hard 111 | Nimble 280 | VitaminC 599 | p50 |
|---|---:|---:|---:|---:|---:|
| `doubao-2.1-pro` | **91.2%** | **82.4%** | 94.2% | 72.9% | 1174 ms |
| `doubao-2.1-lite` | 89.6% | 79.3% | 89.6% | 72.0% | **870 ms** |
| `deepseek-v4.1-flash`¹ | 87.0% | 75.7% | 83.2% | **75.5%** | 859 ms |
| `deepseek-flash`² | 85.7% | 72.1% | 83.2% | 75.1% | 477 ms |

¹ 当次运行配置里的路由名；DeepSeek 此后把该别名更名为 `deepseek-flash`，旧数字保留不修正。
² 2026-10-03 用当前代码与配置做的三组全量补跑（1,110 次调用，0 失败）；别名无版本号，无法证明与 ¹ 是同一构建，延迟也来自另一日运行、不可直接比较。产物绑定见 [`benchmarks/REPORT.md`](benchmarks/REPORT.md) 第 6.4 节。

### JevBench 231 与 Jev 生态对照

同一口径（公开 JevBench 231、逐样本准确率）。本项目的数字来自仓库内实测，其余来自各模型自己的模型卡——闭源托管与开源权重并列，按分数降序：

| 模型 | 运行方式 | JevBench 231 | hard 111 |
|---|---|---:|---:|
| `doubao-2.1-pro`（本项目） | 本仓库实测 | **91.2%** | **82.4%** |
| `doubao-2.1-lite`（本项目默认路由） | 本仓库实测 | 89.6% | 79.3% |
| Jev 1.13.0 | TypeSafe 托管（闭源） | 86.58% | 72.97% |
| Open-Jev-27B-v1.1 | 开源权重 | 85.28% | 72.07% |
| Open-Jev 9B | 开源权重 | 77.49% | 59.46% |
| NeoHorse-Jev-4B | 开源权重 | 75.32%¹ | — |
| Open-Jev 2B | 开源权重 | 64.94% | 41.44% |

¹ 模型卡公布值。NeoHorse 卡里另有 75.73（按题型 family 的宏平均）；逐样本口径是 75.32，这里只能和它比。各来源的子集规则与指标定义有差异——并列引用前先读 [`benchmarks/REPORT.md`](benchmarks/REPORT.md) 第 5.3 节。

每个数字都绑定了数据集哈希、子集规则和运行产物哈希——见 [`benchmarks/REPORT.md`](benchmarks/REPORT.md)，并请注意，这份报告记录的是它自己的缺口（哪些无法复现、为什么），而不是把空白填上。绑定的运行产物以 Release [`eval-20261002`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261002) 发布；数据集不进仓库，从上游获取。

```bash
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench
python3 benchmarks/run_matrix.py --benches jevbench,nimble
```

数据集**不**随仓库附带：它们体积大，应当在其各自的许可下从上游获取。

## 测试

```bash
pytest        # 94 tests, fully offline, no API key needed
```

## 项目结构

```
src/llm2decision/
  core/       config · schema · providers · readout · labels
  llm/        client (OpenAI-compatible transport) · service (orchestration)
  prompts/    versioned templates + loader
  api/        main · debug UI
  calibrate.py
benchmarks/   benchmark harness, probe scripts, REPORT.md
tests/        94 offline tests
```

## 现状与局限

- **与 TypeSafe AI 及其 Jev 产品没有任何关联。** `/v1/systemone` 路由的存在只是为了让已有调用方能够迁移，关系仅此而已。
- 候选数受 provider 上限约束（Ark/DeepSeek 10、百炼 4，见"提供方"）；超过就拆问题，或读 [`docs/design.md`](docs/design.md) 第 7.1 节里我们实测并放弃的按候选逐个处理策略。
- 语言：本仓库里的代码、注释、配置和文档**全部是英文**；主要文档另有中文版，以 `*.zh-CN.md` 形式并存（`README`、`docs/design`、`docs/when-to-migrate`、`benchmarks/README`、`benchmarks/REPORT`），靠人工保持同步。
- 只支持文本输入。没有图像通道。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
