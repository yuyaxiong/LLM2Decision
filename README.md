# LLM2Decision

[English](README.md) | [简体中文](README.zh-CN.md)

[![tests](https://github.com/yuyaxiong/LLM2Decision/actions/workflows/test.yml/badge.svg)](https://github.com/yuyaxiong/LLM2Decision/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Turn any hosted LLM into a typed decision model.** Give it a state and a few typed questions; get back a probability distribution over *your* options — not generated text.

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

No text is generated, so there is nothing to hallucinate and nothing to parse. The answer is a distribution your code can threshold.

---

## Why this exists

Using a chat LLM for a small, bounded decision is surprisingly indirect:

| The usual way | The problem |
|---|---|
| Prompt it to "reply with one word" | It replies with a paragraph anyway; you write a parser and a fallback |
| Ask it for `"confidence": 0.9` | That is **generated text**, not a probability. It is uncalibrated by construction |
| Constrain the output with a schema | Guarantees the *shape*, not the *confidence*. Still token generation |
| Fine-tune a small classifier | Works, but needs labels you probably don't have yet |

This project takes the readout route: stop the prompt at the answer position, let the model produce **exactly one token**, and read the probability it assigned to each of your candidate answers. That is the same signal a classification head would give you, obtained from an API you already have.

**No GPU. No fine-tuning. No vendor lock-in.**

## What you get

- **`choice`** — pick one of your options, with a distribution over all of them
- **`noul`** — yes/no probability for a statement (`true_probability` in `[0,1]`)
- **`score`** — position on an ordered scale, with `expected` value for ranking
- Several questions about the same state in **one request**
- An honest failure mode: if the readout isn't trustworthy, it **errors** instead of inventing a distribution

**Not sure whether your step qualifies?** [`docs/when-to-migrate.md`](docs/when-to-migrate.md) is the practical companion: which steps are worth converting, templates for each question type, how to wrap generation you *can't* remove in decision guards, an "extractive-first" pattern for rewriting and summarization, and the acceptance bar to hold yourself to.

## Install

```bash
pip install llm2decision            # once published; for now, from source:
git clone https://github.com/yuyaxiong/LLM2Decision && cd LLM2Decision
pip install -e ".[dev]"
```

Requirements: Python 3.10+, and an API key for any OpenAI-compatible endpoint.

## Configure

Copy the example and fill in your key:

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

Then start it:

```bash
uvicorn llm2decision.api.main:app --port 8000
```

**Debug UI** — open `http://127.0.0.1:8000/debug` in a browser: a single-page console to hand-build `/v1/decide` requests, pick a model route, and inspect the full readout — candidate distribution, raw token/logprobs, timing breakdown. Bilingual: EN / 中文 toggle in the top-right, following your browser language by default.

**You only need one thing to configure: `llm2decision.yaml`,** created by the `cp` above — every option lives there. Precedence: `route > defaults > environment > built-in default`; environment variables are a fallback, never an override. `.env.example` is the *alternative* for container/CI setups that inject `LLM2DECISION_API_KEY` instead of writing a file — pick one path, not both.

### Providers

Different vendors differ in ways that decide whether this mechanism works at all. All of it lives in one file: [`src/llm2decision/core/providers.py`](src/llm2decision/core/providers.py).

| Provider | `top_logprobs` cap | Candidate cap | Disable thinking | Online tokenizer |
|---|---:|---:|---|---|
| `ark` (Volcengine Ark) | 20 | **10** | `thinking: {"type":"disabled"}` | yes |
| `dashscope` (Alibaba Cloud DashScope) | 5 | **4** | `enable_thinking: false` | no |
| `deepseek` (DeepSeek) | 20 | **10** | `thinking: {"type":"disabled"}` | no |
| `openai_compatible` (generic) | 20 | 8 *(unverified)* | — | no |

Three things worth knowing before adding a vendor:

1. **`top_logprobs` is not your candidate budget.** Those slots are shared with the EOS token, full-width variants, punctuation, and the words a model reaches for when it wants to start explaining itself (`The`, `Let`, `<|im_end|>`). Measured: 20 slots carry 10 candidates reliably; 5 slots carry 4.
2. **Thinking models don't work here unless the chain can be disabled:** with `max_tokens=1` the reasoning eats the only token, so position 0 is empty and `logprobs.content` comes back `null`. Vendors document this badly — we measured one model advertised as supporting `logprobs` that returns `null`, and another absent from the docs that works fine. The exact per-vendor spellings (DeepSeek needs `thinking: {"type":"disabled"}`; other spellings fail silently with HTTP 200) are in [`docs/design.md`](docs/design.md).
3. **So measure, don't read the docs.** Before pointing this at a new vendor, run the probe:

```bash
python3 benchmarks/probe_provider.py \
  --base-url https://openrouter.ai/api/v1 --api-key '<key>' \
  --thinking-param none --models openai/gpt-4o-mini --limit 60
```

It answers four things in ~140 calls and stops after 1 if `logprobs` is unsupported: is `logprobs` passed through, is there a way to disable thinking, how many candidates fit, and what accuracy you actually get. **If it reports `logprobs` returned `null`, this mechanism cannot work on that endpoint** — that's a dead end, not a config problem.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/decide` | Native endpoint |
| `POST` | `/v1/systemone` | Same body and response, named for [Jev](https://typesafe.ai/) compatibility |
| `GET` | `/v1/models` | List routes with provider and real model IDs |
| `GET` | `/health` | Liveness + whether keys are configured |
| `GET` | `/debug` | Single-page UI: build a request, see probability bars |

### Request

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

`choice` requires `criteria`; `noul` requires non-empty `instructions`; `score` requires `scale` with ≥2 levels.

### Response

| Field | Notes |
|---|---|
| `decisions.<name>.label` | argmax of the distribution (`choice` / `score`) |
| `decisions.<name>.distribution` | renormalized over your candidates only |
| `decisions.<name>.true_probability` | `noul` only |
| `decisions.<name>.expected` | `score` only, using `values` |
| `decisions.<name>.coverage` | share of probability mass that landed on your handles |
| `decisions.<name>.reliable` | `coverage >= 0.5` — **below this, distrust the distribution** |
| `decisions.<name>.generated_token` | what the model actually produced at position 0 |
| `decisions.<name>.calls` | model calls for this question (1 with the current strategy) |
| `decisions.<name>.timing` | `prepare_ms` / `call_ms` / `readout_ms`, **excluding queue wait** |
| `decisions.<name>.raw_candidates` | only with `debug: true` |
| `latency_ms` | end-to-end wall time, **including in-service concurrency queueing** |
| `calls` | model calls for the whole request |

`latency_ms` minus a question's `timing` sum is queue/scheduling overhead — the way to tell "slow model" from "long queue". Measured: `prepare_ms + readout_ms` < 1.5 ms, so **essentially all latency is the upstream round trip**.

### Errors

| Status | Meaning |
|---|---|
| `422` | Bad request: candidate count over the provider's cap, unknown route, schema violation |
| `502` | The readout failed. The detail carries **what the model actually produced** |

`502` is deliberate. If position 0 doesn't hold one of your candidate handles, this service refuses to guess and hands you the raw output instead of a fabricated distribution.

## How it works

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

Four hard constraints fall out of this design, and none are configurable: `temperature = 0` (greedy, so position 0 is deterministic, not a lottery draw); `max_tokens = 1` (one token, then stop — any longer and you're reading the model's own prose); read position 0 only (position *k > 0* is the distribution *after* a written prefix — a different quantity); and single-token handles (a multi-token handle has no `top_logprobs` entry, so the readout misses it — `coverage` drops and `reliable` turns false, without failing the request). Handles are assigned for you (`1`–`9`, then `A`–`K`; labels that are already single `0-9A-Z` characters are used as-is).

An optional extra: `logit_bias_enabled: true` (needs an online tokenizer) applies the **same** bias to every handle — uniform bias cancels out during renormalization, so it only pushes handles into the top-k to stop readout failures (measured: fewer failures, accuracy unchanged).

Fuller write-up, including the measured failure modes and why certain things aren't supported: [`docs/design.md`](docs/design.md).

## Calibration

`confidence` is the model's probability, **not a calibrated correctness probability**. If you want to threshold on it, calibrate first:

```bash
python3 -m llm2decision.calibrate --data labeled.jsonl --cache responses.json --write
```

It fits a temperature in log space by minimizing NLL and writes `temperature_scale` into your config. Caveat: a few dozen samples is not enough — if nearly everything is correct the fit invents a fake temperature. You need a few hundred examples **including hard ones** (it warns when the problem is under-identified).

## Benchmarks

`benchmarks/` holds the harness, the results, and an honest account of what doesn't reproduce. Highlights: one full run of the currently configured 8 routes × 4 benchmark groups (26,184 calls, 0 failures):

| Route | JevBench 231 | hard 111 | Nimble 280 | VitaminC 599 | Kev six-subset | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `doubao-evolving` | **91.8%** | **83.8%** | **94.6%** | 73.1% | — | 1152 ms |
| `doubao-2.1-pro` | 90.9% | 82.0% | 94.3% | 73.6% | — | 1110 ms |
| `doubao-2.1-lite` | 87.9% | 76.6% | 88.2% | 72.5% | 81.8% | 550 ms |
| `deepseek-flash` | 83.1% | 67.6% | 82.1% | **75.5%** | 80.4% | 421 ms |
| `doubao-2.0-mini` | 81.8% | 64.0% | 71.4% | 74.8% | — | **355 ms** |

¹ Kev was run on three routes only (`doubao-2.0-pro` 81.80%, `doubao-2.1-lite` 81.77%, `deepseek-flash` 80.40%, six-subset equal-weighted); p50 is the JevBench median. Same-route numbers move by ±2~7 items between runs, so don't compare them cell by cell with earlier editions (the retired route names are listed in the report's Appendix A). Artifact binding: see [`benchmarks/REPORT.md`](benchmarks/REPORT.md) section 6.4.

### JevBench 231 vs the Jev field

Same-protocol comparison on the public JevBench 231 (per-sample accuracy). Our rows are measured with this codebase; everything else is a published value from that model's own card — hosted/closed and open-weight models side by side, strongest first:

| Model | Where it runs | JevBench 231 | hard 111 |
|---|---|---:|---:|
| `doubao-evolving` (this project) | here, measured | **91.8%** | **83.8%** |
| `doubao-2.1-lite` (this project, example-config default) | here, measured | 87.9% | 76.6% |
| Jev 1.13.0 | TypeSafe, hosted (closed) | 86.58% | 72.97% |
| Open-Jev-27B-v1.1 | open weights | 85.28% | 72.07% |
| Open-Jev 9B | open weights | 77.49% | 59.46% |
| NeoHorse-Jev-4B | open weights | 75.32%¹ | — |
| Open-Jev 2B | open weights | 64.94% | 41.44% |

¹ Published value. NeoHorse's card also lists 75.73 as a task-family macro average; 75.32 is its per-sample figure, the one comparable here. Subset rules and metric definitions differ between sources — read [`benchmarks/REPORT.md`](benchmarks/REPORT.md) section 5.3 before quoting these side by side.

### Same items, same protocol: the Intern-Decision seven suites

[Intern-Decision](https://github.com/InternLM/Intern-Decision) (InternLM) ships its seven accuracy suites and the 96-case calibration pilot inside the repository, making it the only benchmark family we can compare against on **identical items** — verified **id by id** against its official bundles (id-list hash `04399f09d6b39036…`). The seven-suite average is the arithmetic mean of the seven accuracies:

| Model | Average | hard 111 | typed_decisions | ToolACE | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `doubao-evolving` (this project) | **89.85** | **82.88** | 73.45 | 89.35 | **0.2500** | 0.1009 |
| `doubao-2.1-pro` (this project) | 89.37 | 81.08 | 73.50 | 87.74 | 0.2641 | 0.1054 |
| `doubao-2.1-lite` (this project) | 87.69 | 77.48 | 73.75 | 83.23 | 0.3086 | 0.1283 |
| `deepseek-flash` (this project) | 86.81 | 66.67 | 70.65 | 88.06 | 0.5466¹ | 0.2289¹ |
| Intern-Decision-4B | **90.02** | 73.87 | **80.55** | **96.45** | 0.3468² | 0.0653² |
| Jev 1.13.0 | 88.74 | 72.07 | 73.35 | 91.29 | 0.3584 | 0.0947 |
| JevK5 / SemIf / Kev | 85.16 / 84.23 / 79.56 | 73.87 / 61.26 / 45.05 | 64.50 / 62.80 / 65.60 | 80.97 / 85.16 / 87.42 | — | — |

¹ DeepSeek's `top_logprobs` carry only one real value (the rest are -9999 placeholders), so these two cells are a **lower-bound approximation**. ² That model's three rows compute Brier / ECE after **fitted temperatures**, while ours are **uncalibrated** raw probabilities and cannot be read as calibration conclusions — likewise on the pilot our Brier is lower (0.258 vs 0.550) but our ECE is worse (0.309 vs 0.089). The full 12-row table (including Intern-Decision-0.8B/2B and Laya) and every protocol note are in [`benchmarks/REPORT.md`](benchmarks/REPORT.md) 5.7 / 5.8.

Every number is bound to a dataset hash, a subset rule, and a run-artifact hash — see [`benchmarks/REPORT.md`](benchmarks/REPORT.md), and note that the report documents its own gaps (what couldn't be reproduced, and why) rather than filling the blanks. The bound run artifacts ship as the [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003) release; datasets stay out of the repository and come from upstream.

```bash
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench
python3 benchmarks/run_matrix.py --benches jevbench,nimble
```

Datasets are **not** vendored: they're large and should come from upstream under their own licenses.

## Tests

```bash
pytest        # 94 tests, fully offline, no API key needed
```

## Project layout

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

## Status and limitations

- **Not affiliated with TypeSafe AI or their Jev product.** The `/v1/systemone` route exists so existing callers can migrate, and that's the whole of the relationship.
- Candidate counts are capped (10 on Ark and DeepSeek, 4 on DashScope — see Providers); beyond that, split the question, or read section 7.1 of [`docs/design.md`](docs/design.md) for the per-candidate strategy we measured and rejected.
- Language: everything here — code, comments, config, and docs — is English. Chinese editions of the main documents ship alongside as `*.zh-CN.md` (`README`, `docs/design`, `docs/when-to-migrate`, `benchmarks/README`, `benchmarks/REPORT`); they are kept in sync by hand.
- Only text input. No image channel.

## License

MIT — see [LICENSE](LICENSE).
