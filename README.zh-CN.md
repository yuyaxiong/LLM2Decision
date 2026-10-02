# LLM2Decision

[English](README.md) | [简体中文](README.zh-CN.md)

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

**你只需要配一样东西：`llm2decision.yaml`**，就是上面 `cp` 出来的那份。所有选项都在里面，别的什么都不用碰。

优先级顺序是 `route > defaults > environment > built-in default`——环境变量只是**兜底**，绝不会覆盖配置文件里写死的值。`.env.example` **不是**第二份要你填的配置：它只是给纯容器 / CI 场景的一个可选替代方案，那种场景宁可用 `LLM2DECISION_API_KEY` 注入 Key 而不写文件，于是可以完全不用 `llm2decision.yaml`。两条路选一条，不要都做。

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
2. **思考模型在这里不管用。** 如果模型先吐出一段推理链，位置 0 上放的就是推理文本，而不是你的答案——而且在 `max_tokens=1` 下，这条链会把唯一那个 token 吃掉，`logprobs.content` 直接返回 `null`，这道题根本读不出来。有些厂商允许关闭，有些不允许——而且厂商在这方面的文档写得很差。我们实测发现，有一个宣称支持 `logprobs` 的模型实际返回 `null`，另一个文档里根本没写的却工作正常。DeepSeek 的推理优先模型（`deepseek-flash`、`deepseek-v4-pro`）**可以**关，但只认 `thinking: {"type":"disabled"}` 这一种写法——`enable_thinking:false` 和 `chat_template_kwargs` 都会静默失败（HTTP 200 但读不到），这也是 DeepSeek 需要独立档、而不能用通用档的原因。
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

用 `latency_ms` 减去某个问题各段 `timing` 之和，得到的就是排队/调度开销——这是区分"模型慢"和"队列长"的唯一办法。在本代码库上实测，`prepare_ms + readout_ms` 不到 1.5 ms，所以**基本上所有延迟都来自上游往返**。

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

这个设计带来四条硬约束，且都不可配置：

- **`temperature = 0`。** 贪心解码让答案位置是确定性的。一旦开启采样，"位置 0"就成了一次抽奖，而不是一个分布。
- **`max_tokens = 1`。** 一个 token，然后停下。再长一点，模型就开始解释，而你读到的就成了它自己那堆文字上的条件分布。
- **只读位置 0。** 位置 *k > 0* 是模型写完一段前缀*之后*的分布——那是另一个（而且错误的）量。
- **句柄必须是单个 token。** 一个会被切成好几段的句柄，在 `top_logprobs` 里没有单独的条目，所以它的概率读不出来。句柄由系统替你分配——`1`–`9` 对应最多九个候选，再往上用字母（候选上限 20 时为 `A`–`K`）；如果你的所有 label 本身就已经是 `0-9A-Z` 里的单个字符，则直接使用这些 label。多 token 的句柄**不会**让请求失败：读出只是漏掉它（`coverage` 下降，`reliable` 变为 false），而且整个请求会放弃 `logit_bias`，而不是只给部分候选加偏置、另一些不加。

一个可选项：开启 `logit_bias_enabled: true`（需要 provider 支持在线 tokenizer）后，每个候选句柄都会获得**相同**的偏置强度。在候选集合内部重新归一化时，均匀偏置会相互抵消，所以它不会扭曲相对概率——它只是把句柄推进 top-k，让读出不再失败。实测：这清除了格式失败的案例，同时准确率没有变化。

更完整的说明，包括实测到的失败模式以及为什么某些东西不被支持，见 [`docs/design.md`](docs/design.md)。

## 校准

`confidence` 是模型给出的概率，**不是经过校准的正确率概率**。如果你想拿它做阈值，先做校准：

```bash
python3 -m llm2decision.calibrate --data labeled.jsonl --cache responses.json --write
```

它通过在 log 空间最小化 NLL 来拟合一个 temperature，并把 `temperature_scale` 写入你的配置。我们亲身体会到的注意点：几十个样本远远不够——如果几乎全都答对，拟合会凭空造出一个假的 temperature。你需要几百个**包含困难样本**的例子，而且当问题欠定时它会给出警告。

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

在 JevBench 231 上同协议已发布的基线：Jev 1.13.0 为 86.58%，Open-Jev-27B 为 85.28%。

每个数字都绑定了数据集哈希、子集规则和运行产物哈希——见 [`benchmarks/REPORT.md`](benchmarks/REPORT.md)，并请注意，这份报告记录的是它自己的缺口（哪些无法复现、为什么），而不是把空白填上。

```bash
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench
python3 benchmarks/run_matrix.py --benches jevbench,nimble
```

数据集**不**随仓库附带：它们体积大，应当在其各自的许可下从上游获取。

## 测试

```bash
pytest        # 90 tests, fully offline, no API key needed
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
tests/        90 offline tests
```

## 现状与局限

- **与 TypeSafe AI 及其 Jev 产品没有任何关联。** `/v1/systemone` 路由的存在只是为了让已有调用方能够迁移，关系仅此而已。
- 候选数量受 provider 上限约束（Ark 与 DeepSeek 上 10 个，百炼上 4 个）。超过这个数，就拆问题——或者读 [`docs/design.md`](docs/design.md) 第 7.1 节，了解我们实测并放弃的那种按候选逐个处理的替代策略。
- 语言：本仓库里的代码、注释、配置和文档**全部是英文**；主要文档另有中文版，以 `*.zh-CN.md` 形式并存（`README`、`docs/design`、`docs/when-to-migrate`、`benchmarks/README`、`benchmarks/REPORT`），靠人工保持同步。
- 只支持文本输入。没有图像通道。
- `openai_compatible` 的限制取自 OpenAI 规范，**未经实测**。在信任它们之前，先用探测脚本验证。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
