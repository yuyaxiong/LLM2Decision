# Design notes

[English](design.md) | [简体中文](design.zh-CN.md)

For people who **want to understand / want to modify this project**. If you just want to get it running and look at the API and configuration, go back to [README.md](../README.md).

This document covers: the principle (logit readout), the readout mechanism (position 0 and label handles), prompts and versioning, temperature calibration, multi-model routing, measured boundaries and failure modes, and the things we **deliberately don't do**.

---

## 1. Principles

### 1.1 The core premise

Models like Jev are not a new architecture. Every Transformer forward pass already outputs a full probability distribution over the next token; a "decision model" merely **reads that distribution out** instead of sampling it into text and parsing it. Three steps:

1. Organize the question into a prompt so that the "answer" becomes the next token the model is about to generate;
2. Take the logits/logprob at that position only, and filter out the components you need according to the candidate labels;
3. Normalize + temperature-scale to get a probability distribution over the candidates.

Any interface that can return next-token log-probabilities can do this.

**The difference from Jev lies in "how you get to that position"**: Jev / vLLM's `logprob_token_ids` / SGLang's `/v1/score` do prefill scoring — stop the prompt where the answer should begin, read the distribution at that position in a single forward pass, and **generate no token at all**. The Volcengine Ark Chat API only returns logprobs for **tokens the model actually generated**; it has no echo/prefill scoring, so this project's approach is: make the model actually emit the first token, then read the top-k at that position. Combined with "greedy decoding + prompt stopped at the answer position", that position *is* the answer position, and the distribution read out is semantically the same as Jev's (see [2.2](#22-read-position-0-only)).

### 1.2 Request lifecycle

```text
state + questions
       │
       ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1. Prompt construction (src/llm2decision/prompts/loader.py +        │
│    src/llm2decision/prompts/templates/*_v1.txt)                    │
│    system: role constraints + few-shot (static, prefix-cacheable)  │
│    user:   [input] [task] [candidates] [output requirements]       │
└────────────────────────────────────────────────────────────────────┘
       │
       ▼  POST {base_url}/chat/completions
          logprobs=true, top_logprobs=20,
          temperature=0 (greedy), top_p=1.0, max_tokens=1
┌────────────────────────────────────────────────────────────────────┐
│ 2. One forward pass, generating exactly 1 token (measured: the      │
│    model emits one character and then EOS); the point is to get    │
│    the logprobs at that position                                   │
└────────────────────────────────────────────────────────────────────┘
       │
       ▼  choices[0].logprobs.content[] : [{token, logprob, top_logprobs:[...]}]
┌────────────────────────────────────────────────────────────────────┐
│ 3. Read the answer position (src/llm2decision/core/readout.py)      │
│    Read position 0 only: its token must be a candidate handle      │
│    (greedy decoding makes position 0 the argmax, i.e. the full     │
│     distribution there — equivalent to Jev's one-pass read)        │
│    Take the top-k → filter candidate handles → renormalize         │
│    Candidates outside the top-k fall back to a floor; coverage     │
│     is computed                                                    │
│    Position 0 is not a candidate handle → error (never return a    │
│     fake distribution)                                             │
└────────────────────────────────────────────────────────────────────┘
       │
       ▼  p_i ∝ exp(logprob_i / T)   T = temperature_scale
┌────────────────────────────────────────────────────────────────────┐
│ 4. Assemble the typed decision (src/llm2decision/llm/service.py)    │
│    choice → distribution / label / confidence                      │
│    noul   → true_probability / decision                            │
│    score  → distribution / expected                                │
└────────────────────────────────────────────────────────────────────┘
```

One call per question; multiple questions within the same request run concurrently (bounded by `max_concurrency`). Reads only, no state is persisted.

---

## 2. Readout mechanism

### 2.1 Label handles (why candidates must be numbered)

Probabilities can only be read by **token**, so every candidate must map to a **single token**. Hence:

| Situation | Handle | Notes |
|---|---|---|
| choice, 1~9 candidates | `1`~`9` | digits are single tokens in all mainstream tokenizers |
| choice, 10~20 candidates | `A`~`T` | same |
| score, where the level label is itself a unique single character | reused as-is (e.g. `0`,`1`,…,`5`) | the level number the model sees is exactly the character it must output |
| noul | `1`=yes, `2`=no | additionally accepts aliases such as `Y/YES/TRUE/是` and `N/NO/FALSE/否` |

Multi-character or Chinese labels (such as `billing` or `退货退款`) get split into multiple tokens and can't be read directly as probabilities, so they're all mapped to numbers and swapped back to the original labels on the way out. The candidate description text is still shown to the model as usual; it just doesn't participate in the probability readout.

Token text is normalized before comparison: NFKC folding to half-width (`１`→`1`), stripping leading/trailing whitespace and punctuation, and uppercasing. We've measured full-width `１`/`０` actually showing up in the top-k, and skipping this step causes missed reads.

### 2.2 Read position 0 only

The prompt is already stopped where the answer should begin, so **the first token the model generates is the answer position**, and its `top_logprobs` are the model's raw judgment over the candidates. There are three hard rules here:

1. **Read position 0 only; don't scan further.** The distribution at position k>0 is the **conditional distribution** given the prefix the model has already written, so its meaning has already changed; if the prefix leaks the answer (for example the model first writes "the answer is B"), the probabilities read afterwards are completely distorted. Backward scanning is a fault-tolerance measure, not a correctness mechanism, so this project simply doesn't do it.
2. **The token at position 0 must be a candidate handle.** If the model's first token isn't a candidate handle, it means the model is writing an explanation (we've measured first tokens such as 「由于」 and 「各个」) — in that case it **errors out immediately**, with no guessing.
3. **Greedy decoding guarantees determinism**: the sampling temperature is fixed at 0, so position 0 is always the argmax. Measured on Ark, the returned `logprobs`/`top_logprobs` are computed **before** sampling and are unaffected by that value, so the distribution is still fully readable (for the same prompt at `temperature=0` and `1`, position 0 returns 20 candidates with 14~15 distinct logprob values in both cases).

Why rules 1 and 2 matter — if you relaxed them to "whichever position hits a candidate is the one to take", then a letter token that happens to turn up in the top-k of explanatory text would be treated as the answer, the remaining candidates would be topped up with the floor value, and the service would end up emitting a **completely plausible-looking fake distribution** with `confidence=0.65`. This is the most dangerous failure mode for a decision service, so it errors rather than handing back a fabricated distribution.

The topping-up rule: candidates that don't appear in the position-0 top-k are assigned a floor of `min(observed logprob) - 3.0`, so the distribution can still be normalized, and `coverage` is recorded; `coverage < 0.5` → `reliable=false`, which lets the caller downgrade or discard the result. On failure it raises `ReadoutError` with HTTP 502, and the error message carries the model's actual output to make prompt problems easier to localize.

**The safety switch (`logit_bias_enabled`)**: the prompt can only raise the probability that the first token is a label; it can't hard-constrain it. When enabled, the service first looks up each candidate handle's token id via the tokenizer endpoint, then applies a `logit_bias` of the same strength to **all candidates**, pushing them up together. Because the strength is identical, the bias cancels out after renormalization within the candidates (measured: with `+20` and without it, the within-candidate distribution is identical), so the relative probabilities read out don't change. If even one candidate can't get a single-token id, the whole thing is abandoned rather than applying a "partial bias" — that is what would actually corrupt the distribution. Off by default.

### 2.3 Mapping of the three question types

| Question type | Input | Readout | Output fields |
|---|---|---|---|
| `choice` | `criteria`: label → description (≤20) | distribution over candidate handles | `distribution`, `label` (argmax), `confidence` |
| `noul` | `instructions`: the statement to judge | handles `1` (yes) / `2` (no) | `true_probability = P(yes)`, `decision = P(yes) ≥ 0.5` |
| `score` | `scale`: ordered levels (2~20), optional `values` | distribution over the levels | `distribution`, `expected = Σ value_i · p_i`, `label` (argmax) |

When `score`'s `values` are omitted: if a label can be converted to a number, use the number (`"3"`→3), otherwise use the index (0,1,2,…). (The API layer ultimately narrows the candidate count to 10; see [the API section of the README](../README.md#api).)

`noul` questions always return the candidates' `label` as the wire-format strings `yes` / `no`, independent of the prompt language. In the Chinese scenario (template `v1`) the prompt asks the model for `1` / `2`, and the handles stay `1` / `2` — the label is a response field, not prompt text.

---

## 3. Prompts and versioning

Prompt templates are **not in the code**; they live by version number under `src/llm2decision/prompts/templates/`, and are loaded and rendered by `src/llm2decision/prompts/loader.py`.

### 3.1 Templates and placeholders

One version consists of four templates:

| File | Purpose |
|---|---|
| `system_v1.txt` | the system message: role constraints + few-shot examples (static, prefix-cacheable) |
| `choice_v1.txt` | the user message for choice: `[input] [task] [candidates] [output requirements]` |
| `noul_v1.txt` | the user message for noul: `[input] [statement to judge] [output requirements]` |
| `score_v1.txt` | the user message for score: `[input] [task] [levels] [output requirements]` |

Templates use `str.format`'s `{placeholder}` syntax to mark the runtime interpolation points: `{state}`, `{instructions}`, `{options}` (choice candidates) / `{levels}` (score levels), `{example}` (the example index referenced in the output requirements). Dynamic material goes in the user message and the system message stays static, which makes prefix caching reusable.

The initial version stays byte-for-byte identical to the original implementation (a unit test locks down the rendered result for choice character by character), guaranteeing that templating itself doesn't change behavior.

### 3.2 Loading and configuration

- `load_prompt_set(version)` loads the four templates by version number; if the version doesn't exist it raises `UnknownPromptVersionError`, and the error message **lists all available versions** (`available_versions()` keys off `system_<version>.txt`); an in-process cache hit skips disk I/O, so each version is read from disk only once.
- Config key **`prompt_version`**: defaults to `v1`; it can be set in the `defaults` section to apply globally, or overridden per model tier (so different routes in the same service can run different prompt versions); environment variable `LLM2DECISION_PROMPT_VERSION`. Two versions ship today: **`v1`** (Chinese templates — the version every published benchmark number was measured on) and **`v2`** (English templates, identical block structure and placeholders). Switching versions changes the prompt the model sees, so re-measure before trusting any number.
- The version number flows through to `build_messages(..., version=...)`, and `src/llm2decision/llm/service.py` reads it on each decision from the `ModelSettings.prompt_version` of the matched route.

### 3.3 Why this design

- **Change the prompt without changing code**: to adjust wording, add constraints, or run an A/B, just add a new `*_v2.txt` version and point `prompt_version` at it — no Python touched; rolling back is likewise a single config value.
- **Gray-releaseable**: `prompt_version` can be overridden per route, so you can put one route on the new version while the rest stay on the old one and compare on real traffic.
- **Bindable to evaluation**: a template change directly changes the first-token distribution the model produces, so **evaluation results must be bound to the prompt version that produced them**. That is the point of `benchmarks/provenance.json` — it records the implementation fingerprint and the hashes of run artifacts machine-readably, so whenever the version changes the results have to be re-bound. An unknown version errors rather than silently falling back, avoiding the case where "you think you ran v2 but actually ran v1".

---

## 4. Temperature calibration

### 4.1 Principle

The raw logprob the model gives is a "distribution over the next token", **not "the probability that the answer is correct"**. On clean examples we measured values right up at `1.0` / `0.9999`, and using that as a confidence would be seriously misleading.

Calibration introduces just one scalar temperature `T`:

```text
p_i = softmax(logprob_i / T)
```

`T` is fitted on labeled samples by minimizing negative log-likelihood (NLL) (`python3 -m llm2decision.calibrate`):

- The binary case has a closed-form solution: `T* = Δ / logit(p)`, where `Δ` is the logit gap and `p` is the positive-class ratio (a unit test verifies this in code);
- In the general case it runs a golden-section search in log space (NLL is usually unimodal in T);
- It reports accuracy / NLL / Brier / ECE before and after fitting, and gives the optimal temperature per question type.

**Detecting non-identifiability**: if the search hits a boundary, or the fitted NLL has already gone to zero, it means the model is almost always right on this sample set, so NLL heads steadily toward "sharper" and what you get is a **fake temperature**. In that case the CLI prints a `<-- unidentifiable` marker and a warning; don't write it into the config. On a measured set of 16 samples the per-type optimal temperatures were choice=0.05 (pinned to the lower bound), noul=0.17, score=2.50 — wildly inconsistent in magnitude, exactly the signature of too little data.

### 4.2 Data format

JSONL, one sample per line; `gold` must be a label from that question's candidates:

```json
{"state": "I was charged twice for the same order.", "question": {"type": "choice", "instructions": "Pick the intent that matches what the customer wants.", "criteria": {"billing": "Charges, refunds, invoices", "technical": "Faults, errors, broken features", "other": "Anything else"}}, "gold": "billing"}
{"state": "Please give me my money back.", "question": {"type": "noul", "instructions": "The customer explicitly asks for a refund."}, "gold": "yes"}
{"state": "Great service, the problem was fixed quickly.", "question": {"type": "score", "instructions": "Rate customer satisfaction.", "scale": ["0","1","2","3","4","5"]}, "gold": "5"}
```

### 4.3 Running

```bash
python3 -m llm2decision.calibrate --data data/labels.jsonl --cache data/raw_logprobs.jsonl
# once you've confirmed it, write back to that route tier (the default route; add --model <route name> for others)
python3 -m llm2decision.calibrate --data data/labels.jsonl --cache data/raw_logprobs.jsonl --write
```

| Argument | Description |
|---|---|
| `--data` | labeled samples JSONL (required) |
| `--model` | model route name; defaults to `default_model` |
| `--cache` | raw logprobs cache file; if it already exists it's reused and the API isn't called again (no repeated billing while tuning) |
| `--config` | config file path; defaults to `llm2decision.yaml` |
| `--write` | write the fitted temperature into that route tier (`models.<route>`), preserving comments and everything else |

### 4.4 Reading the output

```text
Samples: 16 (choice=9, noul=4, score=3)
Temp        Accuracy  NLL       Brier     ECE
Raw T=1.0000 0.9375    0.1574    0.1072    0.0713
Fitted T=1.9971 0.9375    0.1214    0.0850    0.0753
Best temperature per question type: choice=0.0500  <-- unidentifiable: too few samples, or this subset is almost all-correct
Best temperature per question type: noul=0.1701  <-- unidentifiable: too few samples, or this subset is almost all-correct
Best temperature per question type: score=2.5005
```

(The above is the real output for 16 hand-labeled samples, **shown only to demonstrate the flow — don't treat it as a usable calibration value**.)

Three things to look at:

1. **Whether NLL / Brier went down** — this is the fitting objective, and usually it does;
2. **Whether ECE went down** — this is the metric for "can the probability be used as a confidence"; when the model is almost always right on the samples, ECE won't improve (in the example above, 0.0713 → 0.0753, essentially unchanged);
3. **Whether `unidentifiable` appears** — if it does, the samples are unusable.

For a usable calibration you need on the order of a few hundred samples, and they **must include hard samples the model gets wrong**; best to evaluate per question type and only then decide whether to fit separately per type.

The service currently supports only one **global temperature** (`temperature_scale`, independent per route); per-type separate calibration isn't implemented yet.

---

## 5. Multi-model routing

### 5.1 Naming convention

Route names take the form **`<vendor>-<major version>-<tier>`**, for example:

- `doubao-2.0-mini` / `doubao-2.1-lite` / `doubao-2.1-pro`
- `deepseek-v4-flash` / `deepseek-v4.1-flash`
- Moving aliases without a fixed version number are the exception and are named `<vendor>-<alias>` (`doubao-evolving`)

**The major version is part of the model's identity and must go into the route name**, so that future `doubao-3.0-*` can be told apart; **the exact build date stays in the `model` field** (e.g. `doubao-seed-2-0-mini-260215`). **When you change a major version, add a new route name and keep the old one** (the old name keeps pointing at the old model), so you neither break callers nor lose the ability to gray-compare the two versions.

### 5.2 Provider profiles and differences

Provider differences are **not scattered through the business code**; they're centralized in `ProviderProfile` in `src/llm2decision/core/providers.py`, and the `provider` config key decides which profile to use (defaults to `ark` when unset). It lives in `core/` rather than `llm/` because `core/config.py` needs to validate by provider, and core must not depend on llm in the reverse direction.

| Field | Meaning |
|---|---|
| `max_top_logprobs` | the provider's hard slot cap (ark 20 / dashscope 5); `config` validates against it at load time |
| `max_candidates` | candidate-count cap (ark 10 / dashscope 4); validated in the `service` layer against the effective route |
| `supports_tokenize` | whether there's an online tokenizer endpoint; without one you can only give up the `logit_bias` safety net |
| `thinking_field` / `thinking_disabled_value` | the field name and value for the disable-thinking parameter, placed at the top level of the request body |

Three places it lands:

1. **Config validation** (`config.py::_build_model`) — `provider` must be resolved first, because the built-in default for `top_logprobs` depends on it (Volcengine Ark 20 / DeepSeek 20 / DashScope 5); an unknown provider errors and lists the available values.
2. **Request assembly** (`llm/client.py`) — `payload.update(profile.thinking_payload(disable_thinking))`: Ark and DeepSeek get `thinking={"type":"disabled"}`, DashScope gets `enable_thinking=false`. This is also what makes DeepSeek's reasoning-first models usable at all: without it their single generated token goes into the reasoning chain and `logprobs.content` comes back `null`.
3. **Capability validation** (`llm/service.py::decide`) — exceeding the candidate cap raises `TooManyCandidatesError` (422) **before** the model is called. It sits in the service layer rather than the Pydantic layer because only after the route is resolved do you know which provider you're on; `schema.py` keeps only the handle scheme's physical limit of 20.

**Cross-provider routes must set `base_url` and `api_key` explicitly**, otherwise they inherit another provider's address from the `defaults` section. At config load time `base_url` is compared against each provider's `default_base_url`, and a mismatch errors immediately — this is the easiest pitfall to hit in practice (after pointing at the wrong endpoint, the error is very hard to understand).

### 5.3 Config layering and precedence

Config has three layers, with precedence **tier config > `defaults` section > environment variables > built-in defaults**. That is, an environment variable will not override a value already written to the config file; an item not set in the tier inherits from `defaults`, and only if `defaults` doesn't have it either does it fall to the environment variable, and finally to the built-in default. `llm2decision.yaml.example` gives sample configs for four providers (Volcengine Ark / Alibaba Cloud DashScope / DeepSeek / a generic OpenAI-compatible endpoint); for the field list see [the Configure section of the README](../README.md#configure).

### 5.4 Each route is isolated

**Adding a model within the same provider = adding a `models.<name>` section, with no code change needed** (cross-provider requires the corresponding `provider` to exist first): `GET /v1/models` immediately exposes it upstream, and `POST /v1/decide` with `"model": "<name>"` routes to it. Every route has **its own set of hyperparameters** (`temperature_scale`, `max_concurrency`, `prompt_version`, `api_key`, `base_url`, etc.), **its own httpx connection pool**, and **its own concurrency gate** (`asyncio.Semaphore`). Passing a route name that doesn't exist gives a straight 422 and never silently falls back to the default route.

### 5.5 Container scenario with no config file

With no config file, you can boot a single route named `default` from environment variables such as `LLM2DECISION_MODEL`; in that case `LLM2DECISION_MODEL` is required, and startup fails immediately without it.

---

## 6. Measured boundaries and failure modes

The following facts come from real calls:

| Fact | Measured value | Impact |
|---|---|---|
| `top_logprobs` hard cap of 20 | request 5 → 5 returned; request 20 → 20 returned; request 21 → rejected with 400 | This is an API protocol limit, independent of the specific model; among the 20 are full-width variants, EOS, punctuation, and English fragments, and after normalization and dedup about 16 remain, of which only a few are candidate handles |
| Can't fetch logprobs by token id | `logprob_token_ids`, `echo`, `prompt_logprobs` are all silently ignored when passed (HTTP 200 but no logprob for the corresponding token); the legacy `POST /completions` returns 404 | The "fetch logprobs one candidate at a time" path doesn't work on Ark; you'd have to self-host (vLLM `logprob_token_ids` / SGLang `/v1/score`) |
| Tokenizer API is available | `POST /tokenization`: `"1 2 3 billing"` → `[144,348,145,348,146,58341]`, digits are single tokens | Can be used to get token ids (for `logit_bias`) or to check whether a label is a single token |
| `logit_bias` takes effect, but a uniform bias doesn't affect the readout | adding `+20` to a candidate id: that candidate's logprob becomes `0.0` and the other tokens' logprobs all shift down by 20; yet **when the same strength is applied to all candidates, the within-candidate normalized result is identical to applying no bias at all** (three sets of measured values agree) | It acts on the pre-sampling logits and is counted in the returned result. A uniform bias cancels out within the candidates automatically, so it can safely be used to raise the compliance rate of "the first token lands on a candidate"; only when applying **different** biases to different candidates do you need to explicitly undo it |
| Thinking mode and logprobs are mutually exclusive | with thinking not disabled, a request carrying logprobs is rejected outright with 400: `Reasoning model does not support n > 1, logit_bias, logprobs, top_logprobs` | `disable_thinking: true` is mandatory (the default in this project) |
| Candidate count and coverage | 3 candidates → 1.0; 12 candidates (ambiguous input) → 0.833 (10/12); 20 candidates → 0.35 | This is the basis for setting Volcengine Ark's cap to 10: beyond that, tail candidates start falling through and `reliable` can become false (the argmax is usually still correct) |
| **DashScope's slot cap is 5** | requesting `top_logprobs=5` → always returns 5; the docs say "range [0,5]" | The candidate cap therefore drops to **4**. Measured, bucketed by candidate count: 3 candidates 4/4 fully covered, 4 candidates 2/4 (qwen3.5-plus and qwen3.7-plus can reach 4/4), 5 candidates 1/4, **6 candidates 0/4 across all 6 models** |
| **DashScope's slots are eaten by noise at 0.9~2.4 per question** | the top-5 mixes in `<|im_end|>`, `The`, `Let`, `A/B/C/D`, `[`, `"`; depending on the model, an average of 0.88 (qwen3.5-plus) to 2.44 (qwen3.8-27b) slots per question | Counterintuitive: **the newer the flagship, the more noise**. Model selection prefers qwen3.5-plus / qwen3.7-plus over the 3.8 flagship |
| **On DashScope, if you don't pass the disable-thinking parameter, `logprobs` is simply `null`** | `enable_thinking=false` ✓, `thinking={"type":"disabled"}` ✓, **not passing it ✗** (HTTP 200 but `logprobs: null`) | Disabling thinking is not optional but a hard prerequisite; on DashScope the failure shows up as a silent `null`, not a 400 like Ark |
| **DashScope's logprobs allowlist doesn't match the docs** | the docs' `qwen-plus-2025-04-28` / `qwen3-32b` measured `null`; the whole qwen3.5/3.6/3.7/3.8 family, which the docs don't list, all measured as supported | When adding a provider you **must measure**; you can't draw conclusions from a documentation allowlist |
| **DashScope has no online tokenizer endpoint** | what the vendor provides is a `logit_bias_id` mapping-table file | This provider's routes can't apply `logit_bias`; `_logit_bias_for` returns None directly and logs a warning; measured on 60 questions, 0 readout failures even without the bias |
| Sampling temperature doesn't affect the read-out distribution | for the same prompt at `temperature=0` and `1`, position 0 returns 20 candidates with 14~15 distinct logprob values in both cases | You can use `temperature=0` greedy so the answer position reliably lands at position 0 |
| Generation length | `completion_tokens=1` (3 total across three questions) | The model emits one character and then EOS; `max_tokens` is only a hard upper bound |
| Raw probabilities are overconfident | clean examples 1.0 / 0.9999 | Must be calibrated before use as a confidence |
| Position 0 isn't necessarily a label | on degenerate input (semantically empty candidates + ambiguous state) it outputs an explanation | It hard-errors and attaches the actual output; it won't fabricate a distribution |
| Single-request latency | 3 questions concurrently ≈ 1030 ms; a single question ≈ 395 ms (doubao-seed-2-0-mini) | One call per question |
| **Almost all the time is upstream round-trip** | single question, three measured segments: `prepare_ms` 0.02~1.05, `call_ms` 800~1426, `readout_ms` 0.03~0.13 (measured on both Ark and DashScope) | prompt assembly and logprob readout **together are under 1.5 ms**; optimizing them is pure noise. To go faster you can only switch to a faster backbone, cut the number of calls, or raise `max_concurrency` |
| **Concurrency queuing can be captured by "`latency_ms` − sum of segments"** | with 4 questions concurrent, the fastest question's unattributed portion is 162.87 ms and the rest 0.49~39.56 ms; for a single question only 0.37~1.04 ms | This is the only way to tell "the model is slow" from "queuing is long". Note the implementation detail: the segmented `call_ms` must be measured **inside** the semaphore; measuring it outside mixes in queue time — the probe got exactly this wrong early on, measuring a real 926 ms call as 8823 ms |
| Token consumption | ~622 prompt tokens for 3 questions | Mostly the candidate list and descriptions |

Concluding advice: **keep the candidate count for choice within 10** for the distribution to be trustworthy; pass `coverage` / `reliable` through to downstream and don't ignore them.

**Main failure modes**: position 0 outputs an explanation instead of a candidate handle (raises `ReadoutError` → 502, with the model's actual output in `detail`); with too many candidates the tail ones fall through and `coverage` drops (`reliable=false`). Both are designed as **explicit failures** rather than handing back a plausible-looking distribution. See [benchmarks/REPORT.md](../benchmarks/REPORT.md) for each backbone's measured accuracy, latency, and failure rate.

---

## 7. Things we don't do, and why

| Not done | Reason |
|---|---|
| **No training**: no LoRA, no scalar decision head | We read `top_logprobs` straight off a hosted API, saving the data and the GPU; the capability ceiling is mostly set by the backbone model, so switching backbones is enough to improve scores |
| **No prefill scoring** | the Ark Chat API doesn't support echo / `prompt_logprobs`, so we can only substitute the equivalent "generate 1 token + read position 0"; real prefill scoring would require self-hosting vLLM / SGLang |
| **No backward scan over positions k>0** | the distributions at those positions are conditional on the written prefix and have changed meaning; treating scanning as fault tolerance produces fake distributions (see [2.2](#22-read-position-0-only)) |
| **If position 0 isn't a candidate, error — don't guess** | better to fail than to return a "plausible-looking" fake probability — this is the most dangerous failure mode for a decision service, and the one most worth avoiding |
| **No "partial logit_bias"** | biasing only some candidates really would distort the within-candidate distribution; if we can't get single-token ids for all of them we abandon it entirely |
| **No independent per-candidate yes/no scoring (strategy B)** | **measured; it fails on both counts**: it runs on Ark but with zero gain, and doesn't run at all on DashScope. See 7.1 below |
| **No multi-sample voting** | a single forward pass with greedy decoding already determines the readout; repeated sampling both costs more and breaks the isomorphic alignment with Jev |
| **No actions, no state** | the service only returns decision results and performs no side-effecting operations; multiple questions within one request are read-only |
| **No per-question-type temperature calibration** | only one global `temperature_scale` per route is supported; per-type calibration needs enough labeled data containing hard samples |

### 7.1 Alternative readout strategy (per-candidate yes/no): measured, not adopted

Drawing on the ideas of the open-source project LLM2Jev, we once evaluated a second readout path (denoted B): **break each candidate option into an independent yes/no question**, one call per candidate, read the two token probabilities `yes`/`no` at position 0, use `q = P(yes)/(P(yes)+P(no))` as that candidate's score, then normalize the N scores.

It has one theoretical advantage that A can't provide: **it occupies only 2 tokens per call, so the candidate count isn't constrained by `top_logprobs` slots** — which lines up exactly with DashScope's weakness of having only 5 slots. The probe (`benchmarks/probe_yesno.py`, 4 measured groups across two rounds) concluded **not adopted**:

| | Volcengine Ark (20 slots) | Alibaba Cloud DashScope (5 slots) |
|---|---|---|
| B's readout hit rate | **140/140 = 100%** (via an equal-amount `logit_bias`) | **99/140 = 70.7%** (no tokenizer endpoint → can only run bare) |
| Accuracy on the same subset vs A | tied (28/30 each, the exact same wrong items) | tied (4/6 each) |
| B / A call volume | 4.67× | 4.67× |
| Conclusion | **runs, but zero gain** | **doesn't run at all** |

**The root cause on DashScope is clean, and it's not something a prompt can fix**: all 41 misses are "when the model outputs `no`, `yes` drops out of the top-5", and every failure reason is `top_logprobs at position 0 are missing ['yes']`. That is, the 5 slots are filled by `no` plus noise, and the denominator half of `P(yes)/(P(yes)+P(no))` can't be obtained. The model answers completely correctly — there just aren't enough slots.

Conversely, this round of measurement also quantified A's weakness on DashScope: **of the same batch of 30 questions, only 6 could run with A; 24 (80%) were rejected with 422 for exceeding the candidate cap** — exactly what B set out to solve, yet B itself doesn't hold up on this provider.

**Preconditions for restarting this path** (at least one must hold for it to be worth doing again):

1. **Integrating DashScope's `logit_bias_id` mapping table** — the only path that can save B (pushing `yes`/`no` into the top-5 together), and it would incidentally give A a format safety net on DashScope too. The cost is that the mapping table is bound to model versions and needs to be maintained as they change.
2. **A real business step with >4 candidates appears**, and it can't be split into two staged `question`s of ≤4 each (splitting is cheaper, so try that first).

**The reason for not adopting it is more than "the data looks bad"**: even if the DashScope hurdle were cleared, B relative to A would only be "4.67× the calls for one extra candidate-cap ceiling", with accuracy measured as tied. Introducing it means a second set of prompt templates, a second set of tests, and a fork in reporting conventions — that maintenance cost needs a real >4-candidate scenario to justify it.

---

## 8. Directory structure (module map)

| Path | Responsibility |
|---|---|
| `src/llm2decision/api/main.py` | FastAPI app: `POST /v1/decide` (native), `POST /v1/systemone` (Jev-compatible), `GET /v1/models`, `GET /health`, and the mapping from exceptions to HTTP status codes; `create_app()` supports injecting config |
| `src/llm2decision/api/debug.py` | `GET /debug`: single-page HTML debug UI (inline CSS/JS, no build, no external dependencies) |
| `src/llm2decision/core/config.py` | multi-model route config loading: tier config > `defaults` > env vars > defaults; startup validation |
| `src/llm2decision/core/schema.py` | request/response models and **structural** validation (the handle scheme's physical limit of 20); per-provider candidate caps live in the service layer |
| `src/llm2decision/core/labels.py` | generation of label → single-token handles, and normalized matching |
| `src/llm2decision/core/readout.py` | **core**: read the logprobs at position 0, validate, filter the candidate handles, normalize, and produce `coverage` / `reliable` |
| `src/llm2decision/prompts/loader.py` | versioned loading and rendering of prompt templates (`load_prompt_set` / `available_versions` / `build_messages`) |
| `src/llm2decision/prompts/templates/*_v1.txt` | the template bodies: `system` / `choice` / `noul` / `score` |
| `src/llm2decision/core/providers.py` | **provider profiles**: each vendor's slot cap, candidate cap, disable-thinking parameter, and whether a tokenizer endpoint exists |
| `src/llm2decision/llm/client.py` | OpenAI-compatible client: chat (`logprobs`, `logit_bias`), the tokenizer endpoint, timeout and error wrapping; provider differences adapted by profile |
| `src/llm2decision/llm/service.py` | decision orchestration: dispatch by `model` route name, validate candidate caps per provider, run questions concurrently, assemble the three decision kinds, aggregate usage |
| `src/llm2decision/calibrate.py` | temperature calibration CLI: collect logprobs, fit T, print metrics, write back to config per route |
| `llm2decision.yaml` | runtime config (12 model routes spanning Volcengine Ark and Alibaba Cloud DashScope) |
| `tests/` | unit and API tests (offline, no real API calls) |
| `benchmarks/` | benchmark scripts, the `REPORT.md` measurement report, and `provenance.json` provenance |
| `pytest.ini` | pytest config (`pythonpath = .`) |

---

## Further reading

- [README.md](../README.md) — quick start, API reference, configuration, and known limitations.
- [when-to-migrate.md](when-to-migrate.md) — **the migration methodology**: judging whether an LLM step can be turned into a decision, migration templates for the three question types, guard usage for generation steps, the "extraction-first" paradigm, and acceptance criteria.
- [benchmarks/REPORT.md](../benchmarks/REPORT.md) — measured numbers and industry comparisons.
- `benchmarks/provenance.json` — machine-readable provenance for every evaluation run.
