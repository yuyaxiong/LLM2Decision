# Design notes

[English](design.md) | [简体中文](design.zh-CN.md)

面向**想理解 / 想修改本项目**的人。只想把它跑起来、看接口和配置，请回到 [README.md](../README.zh-CN.md)。

本文覆盖：原理（logit readout）、读出机制（位置 0 与标签句柄）、prompt 与版本化、温度校准、多模型路由、实测边界与失效模式，以及**刻意不做的事**。

---

## 一、原理

### 1.1 核心前提

Jev 这类模型不是新架构。Transformer 每次前向本来就会对下一个 token 输出一个完整概率分布，所谓"决策模型"只是把这个分布**读出来**，而不是采样成文本再解析。原理三步：

1. 把题目组织成 prompt，让"答案"成为模型下一个要生成的 token；
2. 只取该位置的 logits/logprob，按候选标签筛出需要的分量；
3. 归一化 + 温度缩放，得到候选上的概率分布。

任何能返回 next-token 对数概率的接口都能实现这件事。

**与 Jev 的差别在"怎么拿到那个位置"**：Jev / vLLM 的 `logprob_token_ids` / SGLang 的 `/v1/score` 是 prefill 打分——把 prompt 停在答案该开始的地方，一次前向直接读该位置的分布，**不生成任何 token**。火山方舟 Chat API 只对**模型实际生成的 token** 返回 logprobs，没有 echo/prefill 打分，所以本项目的做法是：让模型真的吐出第一个 token，再读该位置的 top-k。配合"贪心解码 + prompt 停在答案位置"，这个位置就是答案位置，读出的分布与 Jev 语义一致（见 [2.2](#22-只读位置-0)）。

### 1.2 请求生命周期

```text
state + questions
       │
       ▼
┌────────────────────────────────────────────────┐
│ ① prompt 构造（src/llm2decision/prompts/loader.py +          │
│    src/llm2decision/prompts/templates/*_v1.txt）             │
│    system: 角色约束 + few-shot（静态，可命中前缀缓存）│
│    user:   [输入] [任务] [候选] [输出要求]           │
└────────────────────────────────────────────────┘
       │
       ▼  POST {base_url}/chat/completions
          logprobs=true, top_logprobs=20,
          temperature=0（贪心）, top_p=1.0, max_tokens=1
┌────────────────────────────────────────────────┐
│ ② 一次前向，只生成 1 个 token（实测模型输出一个 │
│    字符就 EOS），目的是拿到该位置的 logprobs     │
└────────────────────────────────────────────────┘
       │
       ▼  choices[0].logprobs.content[] : [{token, logprob, top_logprobs:[...]}]
┌────────────────────────────────────────────────┐
│ ③ 读答案位置（src/llm2decision/core/readout.py）               │
│    只读位置 0：它的 token 必须是候选句柄            │
│    （贪心解码下位置 0 恒为 argmax，就是这个位置的   │
│     完整分布，与 Jev 的一次前向读答案位置等价）      │
│    取该位置 top-k → 筛出候选句柄 → renormalize      │
│    未落进 top-k 的候选用 floor 兜底，算 coverage     │
│    位置 0 不是候选句柄 → 报错（绝不返回假分布）      │
└────────────────────────────────────────────────┘
       │
       ▼  p_i ∝ exp(logprob_i / T)   T = temperature_scale
┌────────────────────────────────────────────────┐
│ ④ 组装 typed decision（src/llm2decision/llm/service.py）       │
│    choice → distribution / label / confidence   │
│    noul   → true_probability / decision         │
│    score  → distribution / expected             │
└────────────────────────────────────────────────┘
```

一道题一次调用；同一请求内的多道题并发执行（受 `max_concurrency` 限制），只读、不落状态。

---

## 二、读出机制

### 2.1 标签句柄（为什么候选要用编号）

概率只能按 **token** 读，所以每个候选必须能映射到一个**单 token**。因此：

| 场景 | 句柄 | 说明 |
|---|---|---|
| choice，候选数 1~9 | `1`~`9` | 数字在主流分词器里都是单 token |
| choice，候选数 10~20 | `A`~`T` | 同理 |
| score，档位标签本身是唯一单字符 | 直接复用（如 `0`,`1`,…,`5`） | 模型看到的档位编号就是它要输出的字符 |
| noul | `1`=是，`2`=否 | 额外接受 `Y/YES/TRUE/是`、`N/NO/FALSE/否` 等别名 |

多字符或中文标签（如 `billing`、`退货退款`）会被切成多个 token，无法直接读概率，所以一律映射成编号，返回时再换回原始标签。候选说明文本照常展示给模型看，只是不参与概率读出。

token 文本比较前会做归一化：NFKC 折成半角（`１`→`1`）、去掉首尾空白与标点、转大写。实测 top-k 里真的会出现全角 `１`/`０`，不做这步会漏读。

### 2.2 只读位置 0

prompt 已经停在"答案本该开始"的地方，所以**模型生成的第一个 token 就是答案位置**，它的 `top_logprobs` 就是模型对候选的原始判断。这里有三条硬性约定：

1. **只读位置 0，不往后扫。** 位置 k>0 的分布是"模型先写了一段前缀之后的**条件分布**"，含义已经变了；如果前缀里已经泄漏答案（例如模型先写"答案是 B"），后面读出来的概率完全失真。向后扫描是容错，不是正确性机制，所以本项目直接不做。
2. **位置 0 的 token 必须是候选句柄。** 模型的第一个 token 不是候选句柄，说明它在写解释（实测出现过首 token 是「由于」「各个」）——此时**直接报错**，不猜。
3. **贪心解码保证确定性**：采样温度固定为 0，位置 0 恒为 argmax。实测方舟返回的 `logprobs`/`top_logprobs` 是**采样前**计算的，不受该值影响，所以分布依然完整可读（同一条 prompt 在 `temperature=0` 与 `1` 下，位置 0 都返回 20 条候选、14~15 个不同 logprob 值）。

为什么第 1、2 条重要——如果放宽成"哪个位置命中候选就取哪个"，解释性文本的 top-k 里偶然出现的一个字母 token 就会被当成答案，其余候选再用 floor 补值，最终输出一个 `confidence=0.65` 的**看起来完全合理的假分布**。这是决策服务最危险的失败模式，所以宁可报错。

补齐规则：未出现在位置 0 top-k 里的候选，取 `min(已观测 logprob) - 3.0` 兜底，使分布仍可归一化，并记录 `coverage`；`coverage < 0.5` → `reliable=false`，调用方可据此降级或丢弃。失败时报 `ReadoutError`，HTTP 502，错误信息里带上模型的实际输出，便于定位 prompt 问题。

**兜底开关（`logit_bias_enabled`）**：prompt 只能提高"首 token 是标签"的概率，不能硬约束。打开后会先经分词接口取每个候选句柄的 token id，再给**所有候选**施加同一强度的 `logit_bias`，把它们整体推高。因为强度相同，候选内 renormalize 后偏置自动抵消（已实测：加 `+20` 与不加，候选内分布一致），所以读出的相对概率不变。只要有任一候选拿不到单 token id，就整体放弃施加而不做"部分偏置"——那才会真的破坏分布。默认关闭。

### 2.3 三种题型的映射

| 题型 | 输入 | 读出 | 输出字段 |
|---|---|---|---|
| `choice` | `criteria`: 标签 → 说明（≤20） | 候选句柄上的分布 | `distribution`、`label`(argmax)、`confidence` |
| `noul` | `instructions`: 待判断陈述 | 句柄 `1`(是) / `2`(否) | `true_probability = P(是)`、`decision = P(是) ≥ 0.5` |
| `score` | `scale`: 有序档位（2~20）、可选 `values` | 各档位上的分布 | `distribution`、`expected = Σ value_i · p_i`、`label`(argmax) |

`score` 的 `values` 缺省时：标签能转成数字就用数字（`"3"`→3），否则用下标（0,1,2,…）。（接口层最终把候选数收窄到 10，见 [README 三、接口](../README.zh-CN.md#api)。）

`noul` 返回的候选 `label` 固定是线上的 `yes` / `no`，与 prompt 语言无关。中文场景（模板 `v1`）里 prompt 让模型输出 `1` / `2`，句柄也仍是 `1` / `2`——`label` 是响应字段，不是 prompt 文本。

---

## 三、prompt 与版本化

prompt 模板**不在代码里**，而是按版本号放在 `src/llm2decision/prompts/templates/` 下，由 `src/llm2decision/prompts/loader.py` 加载渲染。

### 3.1 模板与占位符

一个版本包含四份模板：

| 文件 | 作用 |
|---|---|
| `system_v1.txt` | system 消息：角色约束 + few-shot 示例（静态，可命中前缀缓存） |
| `choice_v1.txt` | choice 的 user 消息：`[输入] [任务] [候选] [输出要求]` |
| `noul_v1.txt` | noul 的 user 消息：`[输入] [待判断陈述] [输出要求]` |
| `score_v1.txt` | score 的 user 消息：`[输入] [任务] [档位] [输出要求]` |

模板用 `str.format` 的 `{占位符}` 语法标注运行时插值位置：`{state}`、`{instructions}`、`{options}`（choice 候选）/ `{levels}`（score 档位）、`{example}`（输出要求里引用的示例编号）。动态材料放 user 消息，system 保持静态，方便复用前缀缓存。

初期版本维持与原实现逐字一致（有单测逐字锁定 choice 的渲染结果），保证模板化本身不改变行为。

### 3.2 加载与配置

- `load_prompt_set(version)` 按版本号加载四份模板，版本不存在时抛 `UnknownPromptVersionError`，错误信息里**列出全部可用版本**（`available_versions()` 以 `system_<version>.txt` 为准）；命中进程内缓存时不再读盘，同一版本只从磁盘读一次。
- 配置项 **`prompt_version`**：默认 `v1`，可以配在 `defaults` 段全局生效，也可以按模型档覆盖（同一套服务里可以让不同路由跑不同 prompt 版本）；环境变量 `LLM2DECISION_PROMPT_VERSION`。目前附带两个版本：**`v1`**（中文模板，已发布的全部实测数字都是在它上面跑出来的）与 **`v2`**（英文模板，块结构与占位符完全一致）。切换版本会改变模型看到的 prompt，任何数字都必须重新测过才算。
- 版本号贯穿到 `build_messages(..., version=...)`，由 `src/llm2decision/llm/service.py` 在每次判定时按命中路由的 `ModelSettings.prompt_version` 取值。

### 3.3 为什么这么做

- **换 prompt 不改代码**：调整措辞、加约束、做 A/B，只新增一个 `*_v2.txt` 版本并把 `prompt_version` 指过去，不碰任何 Python；回滚同样只是改一个配置值。
- **可灰度**：`prompt_version` 支持按路由覆盖，可以先让一个路由跑新版本、其余保持旧版本，用真实流量对比。
- **可与评测绑定**：模板改动会直接改变模型输出的首 token 分布，**评测结果必须与跑出它的 prompt 版本绑定**。这就是 `benchmarks/provenance.json` 存在的意义——机器可读地记录实现指纹与运行产物哈希，版本一变，成绩就得重新绑定。未知版本宁可报错，也不静默回落，避免"以为跑的是 v2、实际是 v1"。

---

## 四、温度校准

### 4.1 原理

模型给出的原始 logprob 是"下一个 token 的分布"，**不等于"答案正确的概率"**。实测干净用例上直接给到 `1.0` / `0.9999`，当置信度用会严重误导。

校准只引入一个标量温度 `T`：

```text
p_i = softmax(logprob_i / T)
```

`T` 在标注样本上最小化负对数似然（NLL）拟合（`python3 -m llm2decision.calibrate`）：

- 两分类存在闭式解：`T* = Δ / logit(p)`，其中 `Δ` 是 logit 间隔、`p` 是正例比例（代码里有对应单测校验）；
- 一般情形在 log 空间做黄金分割搜索（NLL 关于 T 通常单峰）；
- 报告拟合前后的准确率 / NLL / Brier / ECE，并按题型给出各自的最优温度。

**不可辨识的判定**：若搜索压到边界，或拟合后 NLL 已经归零，说明这批样本里模型几乎全对，NLL 会一路朝"更尖"的方向走，得到的是**假温度**。此时 CLI 会打 `<-- unidentifiable` 标记与警告，不要把它写进配置。实测 16 条样本时，按题型最优温度分别是 choice=0.05（顶到下界）、noul=0.17、score=2.50，量级完全不一致，正是样本不足的表现。

### 4.2 数据格式

JSONL，一行一个样本，`gold` 必须是该题候选里的标签：

```json
{"state": "我的订单被扣了两次钱。", "question": {"type": "choice", "instructions": "选出客户诉求对应的意图。", "criteria": {"billing": "支付、退款、账单相关", "technical": "故障、报错、功能异常", "other": "其他咨询"}}, "gold": "billing"}
{"state": "请把我的钱退回来。", "question": {"type": "noul", "instructions": "客户明确要求退款。"}, "gold": "yes"}
{"state": "服务非常好，问题很快就解决了。", "question": {"type": "score", "instructions": "评估客户满意度。", "scale": ["0","1","2","3","4","5"]}, "gold": "5"}
```

### 4.3 运行

```bash
python3 -m llm2decision.calibrate --data data/labels.jsonl --cache data/raw_logprobs.jsonl
# 确认无误后写回该路由档（默认路由；其他路由加 --model <路由名>）
python3 -m llm2decision.calibrate --data data/labels.jsonl --cache data/raw_logprobs.jsonl --write
```

| 参数 | 说明 |
|---|---|
| `--data` | 标注样本 JSONL（必填） |
| `--model` | 模型路由名，默认用 `default_model` |
| `--cache` | 原始 logprobs 缓存文件；已存在则直接复用，不再调用接口（调参不重复计费） |
| `--config` | 配置文件路径，默认 `llm2decision.yaml` |
| `--write` | 把拟合出的温度写进该路由档（`models.<route>`），注释与其它内容保留 |

### 4.4 输出解读

```text
Samples: 16 (choice=9, noul=4, score=3)
Temp        Accuracy  NLL       Brier     ECE
Raw T=1.0000 0.9375    0.1574    0.1072    0.0713
Fitted T=1.9971 0.9375    0.1214    0.0850    0.0753
Best temperature per question type: choice=0.0500  <-- unidentifiable: too few samples, or this subset is almost all-correct
Best temperature per question type: noul=0.1701  <-- unidentifiable: too few samples, or this subset is almost all-correct
Best temperature per question type: score=2.5005
```

（上面是 16 条手标样本的真实输出，**只用于演示流程，不要当可用校准值**。）

看三件事：

1. **NLL / Brier 是否下降**——这是拟合目标，通常一定下降；
2. **ECE 是否下降**——这才是"概率能不能当置信度用"的指标；样本里模型几乎全对时 ECE 不会改善（上例 0.0713 → 0.0753，基本没动）；
3. **是否出现 `unidentifiable`**——出现就说明样本不可用。

可用校准的样本要求：几百条量级，且**必须包含模型会答错的难样本**，最好按题型分别评估后再决定是否分别拟合。

当前服务只支持一个**全局温度**（`temperature_scale`，按路由独立），尚未做按题型的分别校准。

---

## 五、多模型路由

### 5.1 命名规范

路由名格式 **`<厂商>-<主版本>-<档位>`**，例如：

- `doubao-2.0-mini` / `doubao-2.1-lite` / `doubao-2.1-pro`
- `deepseek-v4-flash` / `deepseek-v4.1-flash`
- 无固定版本号的移动别名例外，按 `<厂商>-<别名>` 命名（`doubao-evolving`）

**主版本是模型身份的一部分，必须进路由名**，才能区分未来的 `doubao-3.0-*`；**精确构建日期留在 `model` 字段**（如 `doubao-seed-2-0-mini-260215`）。**换大版本时新增一个路由名并保留旧名**（旧名继续指向旧模型），这样既不打穿调用方，又能灰度对比两版。

### 5.2 厂商档案与差异

厂商差异**不散落在业务代码里**，而是集中在 `src/llm2decision/core/providers.py` 的 `ProviderProfile`，配置项 `provider` 决定用哪一份档案（不写默认 `ark`）。放 `core/` 而不是 `llm/`，是因为 `core/config.py` 要按 provider 做校验，而 core 不该反向依赖 llm。

| 字段 | 含义 |
|---|---|
| `max_top_logprobs` | 该厂商的槽位硬上限（ark 20 / dashscope 5），`config` 加载时按它校验 |
| `max_candidates` | 候选数上限（ark 10 / dashscope 4），在 `service` 层按生效路由校验 |
| `supports_tokenize` | 是否有在线分词接口；没有就只能放弃 `logit_bias` 兜底 |
| `thinking_field` / `thinking_disabled_value` | 关思考参数放进请求体顶层的字段名与取值 |

三个落点：

1. **配置校验**（`config.py::_build_model`）——`provider` 必须先解析，因为 `top_logprobs` 的内置默认值依赖它（火山 20 / DeepSeek 20 / 百炼 5）；未知 provider 报错并列出可用值。
2. **请求组装**（`llm/client.py`）——`payload.update(profile.thinking_payload(disable_thinking))`：火山与 DeepSeek 得到 `thinking={"type":"disabled"}`，百炼得到 `enable_thinking=false`。这也正是 DeepSeek 推理优先模型能用的前提：不关思考，它唯一那个生成 token 会落进推理链，`logprobs.content` 直接返回 `null`。
3. **能力校验**（`llm/service.py::decide`）——候选数超限在调用模型**之前**抛 `TooManyCandidatesError`（422）。放在服务层而不是 Pydantic 层，是因为只有解析出路由后才能知道用的是哪家厂商；`schema.py` 只保留句柄方案的物理上限 20。

**跨厂商路由必须显式写 `base_url` 与 `api_key`**，否则会继承 `defaults` 段里别家的地址。配置加载期会拿 `base_url` 和各厂商的 `default_base_url` 比对，命中不一致直接报错——这是实测中最容易踩的坑（打错端点后报错很难懂）。

### 5.3 配置分层与优先级

配置分三层，优先级：**档内配置 > `defaults` 段 > 环境变量 > 内置默认值**。即环境变量不会覆盖已写入配置文件的值；档内没写的项继承 `defaults`，`defaults` 也没有才看环境变量，最后落内置默认值。`llm2decision.yaml.example` 给出四家的配置样例（火山方舟 / 阿里云百炼 / DeepSeek / 通用 OpenAI 兼容端点）；字段清单见 [README 的 Configure 一节](../README.md#configure)。

### 5.4 每个路由相互隔离

**同厂商内加一个模型 = 加一个 `models.<名字>` 段，不需要改代码**（跨厂商要先有对应的 `provider`）：`GET /v1/models` 会立刻把它暴露给上游，`POST /v1/decide` 带上 `"model": "<名字>"` 即可路由过去。每个路由有**独立的一套超参**（`temperature_scale`、`max_concurrency`、`prompt_version`、`api_key`、`base_url` 等）、**独立的 httpx 连接池**与**独立的并发闸门**（`asyncio.Semaphore`）。传了不存在的路由名直接 422，不会静默回落到默认路由。

### 5.5 无配置文件的容器场景

没有配置文件时，可以用 `LLM2DECISION_MODEL` 等环境变量起一个名为 `default` 的单路由；此时 `LLM2DECISION_MODEL` 必填，否则启动即报错。

---

## 六、实测边界与失效模式

以下事实来自真实调用：

| 事实 | 实测值 | 影响 |
|---|---|---|
| `top_logprobs` 硬上限 20 | 请求 5 → 返回 5 条；请求 20 → 返回 20 条；请求 21 → 400 拒绝 | 这是接口协议限制，与具体模型无关；20 条里混着全角变体、EOS、标点、英文片段，按归一化去重后约 16 条，候选句柄只占其中几个 |
| 无法按 token id 取 logprob | `logprob_token_ids`、`echo`、`prompt_logprobs` 传入后均被静默忽略（HTTP 200 但不返回对应 token 的 logprob）；旧版 `POST /completions` 返回 404 | 「对任意候选逐个取 logprob」这条路在方舟上走不通，只能自部署（vLLM `logprob_token_ids` / SGLang `/v1/score`） |
| 分词 API 可用 | `POST /tokenization`：`"1 2 3 billing"` → `[144,348,145,348,146,58341]`，数字是单 token | 可用它拿 token id（供 `logit_bias` 使用）或校验标签是否单 token |
| `logit_bias` 生效，但统一偏置不影响读出 | 给候选 id 加 `+20`：该候选 logprob 变 `0.0`、其余 token 的 logprob 整体下移 20；而**对所有候选取同一强度时，候选内归一化的结果与不加偏置完全一致**（三组实测值一致） | 它作用在采样前的 logits 上并被计入返回结果。统一偏置在候选内会自动抵消，所以可以安全地用它提高「首 token 落在候选上」的遵从率；只有对不同候选施加**不同**偏置时才需要显式撤销 |
| 思考模式与 logprobs 互斥 | 未关闭思考时请求带 logprobs 直接 400：`Reasoning model does not support n > 1, logit_bias, logprobs, top_logprobs` | 必须 `disable_thinking: true`（本项目默认） |
| 候选数与 coverage | 3 候选 → 1.0；12 候选（含糊输入）→ 0.833（10/12）；20 候选 → 0.35 | 这就是把火山的上限设成 10 的依据：超过之后尾部候选开始落空、`reliable` 可能变 false（argmax 通常仍正确） |
| **百炼的槽位上限是 5** | 请求 `top_logprobs=5` → 恒返回 5 条；文档写「取值范围 [0,5]」 | 候选上限因此降到 **4**。实测按候选数分桶：3 候选 4/4 全覆盖、4 候选 2/4（qwen3.5-plus 与 qwen3.7-plus 能到 4/4）、5 候选 1/4、**6 候选在全部 6 个模型上 0/4** |
| **百炼槽位被噪声吃掉 0.9~2.4 个/题** | top-5 里混着 `<|im_end|>`、`The`、`Let`、`A/B/C/D`、`[`、`"`；按模型不同平均每题 0.88（qwen3.5-plus）~2.44（qwen3.8-27b） | 反直觉：**越新的旗舰噪声越多**。选型偏好 qwen3.5-plus / qwen3.7-plus，而不是 3.8 旗舰 |
| **百炼不传关思考参数时 `logprobs` 直接为 `null`** | `enable_thinking=false` ✓、`thinking={"type":"disabled"}` ✓、**不传 ✗**（HTTP 200 但 `logprobs: null`） | 关思考不是可选项而是硬前提；百炼上失败表现为静默 `null`，不像火山那样 400 报错 |
| **百炼的 logprobs 白名单与文档不符** | 文档列的 `qwen-plus-2025-04-28` / `qwen3-32b` 实测 `null`；文档没列的 qwen3.5/3.6/3.7/3.8 全系实测都支持 | 新增厂商时**必须实测**，不能靠文档白名单下结论 |
| **百炼没有在线分词接口** | 官方给的是 `logit_bias_id` 映射表文件 | 该厂商路由无法施加 `logit_bias`，`_logit_bias_for` 直接返回 None 并记 warning；实测 60 题不施加偏置也是 0 读出失败 |
| 采样温度不影响读出的分布 | 同一条 prompt 在 `temperature=0` 与 `1` 下，位置 0 都返回 20 条候选、14~15 个不同 logprob 值 | 可以用 `temperature=0` 贪心，让答案位置确定落在位置 0 |
| 生成长度 | `completion_tokens=1`（三题合计 3） | 模型输出一个字符就 EOS，`max_tokens` 只是硬上限 |
| 原始概率过度自信 | 干净用例 1.0 / 0.9999 | 必须校准后才能当置信度 |
| 位置 0 未必是标签 | 退化输入（候选无语义 + state 含糊）时输出解释 | 硬报错并附实际输出，不会编造分布 |
| 单请求延迟 | 3 题并发约 1030 ms；单题约 395 ms（doubao-seed-2-0-mini） | 每题一次调用 |
| **耗时几乎全在上游往返** | 单题三段实测：`prepare_ms` 0.02~1.05、`call_ms` 800~1426、`readout_ms` 0.03~0.13（火山与百炼各测） | prompt 组装与 logprob 读出**合计不到 1.5 ms**，在它们身上做优化纯属噪声；要提速只能换更快的底座、减少调用次数，或调大 `max_concurrency` |
| **并发排队可由「`latency_ms` − 分段之和」捕获** | 4 题并发时，最快那题的未归因部分 162.87 ms，其余 0.49~39.56 ms；单题时仅 0.37~1.04 ms | 这是区分「模型慢」与「排队久」的唯一办法。注意实现细节：分段的 `call_ms` 必须量在信号量**内部**，量在外面会把排队时间混进来——探针早期正是这么错的，把 926 ms 的真实调用耗时量成了 8823 ms |
| token 消耗 | 3 题约 622 prompt tokens | 主要是候选清单与说明 |

结论性建议：**choice 的候选数控制在 10 个以内**分布才可信；`coverage` / `reliable` 要透传给下游，不要无视。

**主要失效模式**：位置 0 输出的是解释而不是候选句柄（报 `ReadoutError` → 502，`detail` 附模型实际输出）；候选数过多时尾部候选落空、`coverage` 下降（`reliable=false`）。两种都设计成**明确失败**而不是给出一份看似合理的分布。各底座的实测准确率、延迟与失败率见 [benchmarks/REPORT.md](../benchmarks/REPORT.md)。

---

## 七、不做的事与原因

| 不做 | 原因 |
|---|---|
| **不训练**：没有 LoRA、没有标量决策头 | 直接读托管 API 的 `top_logprobs`，省掉数据与 GPU；能力上限主要取决于底座模型，换底座即可提分 |
| **不做 prefill 打分** | 方舟 Chat API 不支持 echo / `prompt_logprobs`，只能靠"生成 1 个 token + 读位置 0"等价替代；要真正的 prefill 打分只能自部署 vLLM / SGLang |
| **不向后扫描位置 k>0** | 那些位置的分布是"写了前缀之后"的条件分布，语义已变；把扫描当容错会产出假分布（见 [2.2](#22-只读位置-0)） |
| **位置 0 不是候选就报错，不猜** | 宁可失败也不能返回一份"看起来合理"的假概率——这是决策服务最危险、也最需要避免的失败模式 |
| **不做"部分 logit_bias"** | 只给部分候选加偏置会真的扭曲候选内分布；拿不到全部单 token id 时整体放弃 |
| **不做「逐候选 yes/no」独立打分（策略 B）** | **已实测，两处都不成立**：火山能跑但零增益，百炼根本跑不了。见下方 7.1 |
| **不做多次采样投票** | 单次前向、贪心解码即可确定读出；重复采样既提高成本也破坏与 Jev 的同构对齐 |
| **不做动作、不落状态** | 服务只返回判定结果，不执行任何副作用操作；同一请求内的多道题只读 |
| **温度不按题型分别校准** | 当前只支持每路由一个全局 `temperature_scale`；按题型校准需要足够的、含难样本的标注数据 |

### 7.1 备选读出策略（逐候选 yes/no）：已实测，不采用

参考开源项目 LLM2Jev 的思路，曾评估过第二条读出路径（记为 B）：**把每个候选项拆成一次独立的 yes/no 询问**，每个候选一次调用，读位置 0 的 `yes`/`no` 两个 token 概率，`q = P(yes)/(P(yes)+P(no))` 作为该候选分数，N 个分数再归一化。

它理论上有一个 A 给不了的优势：**每次只占 2 个 token，因此候选数不受 `top_logprobs` 槽位约束**——这恰好对准百炼只有 5 个槽位这个短板。探针（`benchmarks/probe_yesno.py`，两轮共 4 组实测）的结论是**不采用**：

| | 火山方舟（20 槽位） | 阿里云百炼（5 槽位） |
|---|---|---|
| B 的读出命中率 | **140/140 = 100%**（靠等量 `logit_bias`） | **99/140 = 70.7%**（无分词接口 → 只能裸跑） |
| 同子集准确率 vs A | 打平（各 28/30，错题完全相同） | 打平（各 4/6） |
| B / A 调用量 | 4.67× | 4.67× |
| 结论 | **能跑但零增益** | **根本跑不了** |

**百炼上的失败根因很干净，且不是 prompt 能修的**：41 次未命中**全部**是「模型输出 `no` 时，`yes` 掉出了 top-5」，失败原因全部是 `top_logprobs at position 0 are missing ['yes']`（探针输出的原始英文串）。也就是 5 个槽位被 `no` 加噪声占满，`P(yes)/(P(yes)+P(no))` 的分母半边拿不到。模型答得完全正确，是槽位不够。

反过来这轮实测也量化了 A 在百炼上的短板：**同一批 30 题里，A 只有 6 题能跑，24 题（80%）因候选超限被 422 拒绝**——B 想解决的正是这个，但它自己在这个厂商上不成立。

**要重启这条路的前置条件**（任一满足才值得再做）：

1. **百炼的 `logit_bias_id` 映射表接入**——这是唯一能救 B 的路径（把 `yes`/`no` 同时推进 top-5），而且顺带能给 A 在百炼上补上格式兜底。代价是映射表与模型版本绑定、需要随版本维护。
2. **出现候选数 >4 的真实业务步骤**，且无法拆成两个 ≤4 的阶段化 `question`（拆题更省，应先尝试）。

**不采用的理由不止"数据不好"**：即便百炼那关通了，B 相对 A 也只是"用 4.67 倍调用换一个候选上限"，准确率上实测打平。引入它意味着第二套 prompt 模板、第二套测试、以及报告口径的分叉——这笔维护成本需要一个真实的 >4 候选场景来支撑。

---

## 八、目录结构（模块地图）

| 路径 | 职责 |
|---|---|
| `src/llm2decision/api/main.py` | FastAPI 应用：`POST /v1/decide`（原生）、`POST /v1/systemone`（Jev 兼容）、`GET /v1/models`、`GET /health`，异常到 HTTP 状态码的映射；`create_app()` 支持注入配置 |
| `src/llm2decision/api/debug.py` | `GET /debug`：单页 HTML 调试界面（内联 CSS/JS，无构建、无外部依赖） |
| `src/llm2decision/core/config.py` | 多模型路由配置加载：档内配置 > `defaults` > 环境变量 > 默认值；启动校验 |
| `src/llm2decision/core/schema.py` | 请求/响应模型与**结构性**校验（句柄方案的物理上限 20）；按厂商的候选上限在 service 层 |
| `src/llm2decision/core/labels.py` | 标签 → 单 token 句柄的生成与归一化匹配 |
| `src/llm2decision/core/readout.py` | **核心**：读位置 0 的 logprobs，校验、筛出候选句柄、归一化并给出 `coverage` / `reliable` |
| `src/llm2decision/prompts/loader.py` | prompt 模板的版本化加载与渲染（`load_prompt_set` / `available_versions` / `build_messages`） |
| `src/llm2decision/prompts/templates/*_v1.txt` | 模板正文：`system` / `choice` / `noul` / `score` |
| `src/llm2decision/core/providers.py` | **厂商档案**：各家的槽位上限、候选上限、关思考参数、有无分词接口 |
| `src/llm2decision/llm/client.py` | OpenAI 兼容客户端：chat（`logprobs`、`logit_bias`）、分词接口、超时与错误封装；厂商差异按 profile 适配 |
| `src/llm2decision/llm/service.py` | 决策编排：按 `model` 路由名分派、按厂商校验候选上限、多题并发、三种 decision 组装、usage 汇总 |
| `src/llm2decision/calibrate.py` | 温度校准 CLI：采集 logprobs、拟合 T、输出指标、按路由写回配置 |
| `llm2decision.yaml` | 运行配置（12 个模型路由，横跨火山方舟与阿里云百炼） |
| `tests/` | 单元与接口测试（离线，不调用真实接口） |
| `benchmarks/` | 评测脚本、`REPORT.md` 实测报告与 `provenance.json` 溯源 |
| `pytest.ini` | pytest 配置（`pythonpath = .`） |

---

## 延伸阅读

- [README.md](../README.zh-CN.md) —— 快速开始、接口说明、配置与已知限制。
- [when-to-migrate.md](when-to-migrate.zh-CN.md) —— **改造方法论**：判断一个 LLM 步骤能否决策化、三类题型的改造模板、生成步骤的守卫用法、「抽取优先」范式与验收标准。
- [benchmarks/REPORT.md](../benchmarks/REPORT.md) —— 实测数字与业界对照。
- `benchmarks/provenance.json` —— 每次评测的机器可读溯源。
