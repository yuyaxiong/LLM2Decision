[English](README.md) | [简体中文](README.zh-CN.md)

# benchmarks Guide

This directory is the **evaluation harness**: it takes the decision service in `src/llm2decision/` (`POST /v1/systemone`) and runs it against Jev-class benchmarks to answer "how accurate, how expensive, and how far behind the industry we are".

- **This document only covers how to run things.** The resulting numbers, cross-comparisons, and protocol analysis are in [REPORT.md](REPORT.md).
- For how to use and deploy the service itself, see the [project root README](../README.md).
- For the mechanism (reading position 0 only, label handles, temperature calibration), see [docs/design.md](../docs/design.md).

## Prerequisites

| Item | Description |
|---|---|
| Python | 3.11+; the service depends on `httpx` / `pyyaml` / `pydantic` / `fastapi`; VitaminC additionally needs `datasets` |
| Ark key | `llm2decision.yaml` in the project root must already have a working route configured (`defaults.api_key`) |
| Working directory | **All commands are run from the project root** (the scripts insert the root into `sys.path` themselves) |
| Network | Must be able to reach `ark.cn-beijing.volces.com`; VitaminC also needs HuggingFace access |

## 1. Script overview

| Script | Purpose | Required data |
|---|---|---|
| [run_jevbench.py](run_jevbench.py) | JevBench public 231 items (72 original + 48 easy + 111 hard) | `jevbench/` (manual clone) |
| [run_nimble.py](run_nimble.py) | Nimble built-in holdout: `compat282` (280 items, aligned with the industry protocol) / `full324` (all 324 items) | `nimble/` (manual clone) |
| [run_vitaminc.py](run_vitaminc.py) | VitaminC fact verification: `nimble599` (599 items, aligned with the industry protocol) / `sample300` (self-sampled 300) | HF `tals/vitaminc` + the `nimble/` manifest |
| [run_kev.py](run_kev.py) | The six Kev subsets (decision-v7 / transfer-v4 / transfer-v9), per-item scoring | auto-download |
| [run_matrix.py](run_matrix.py) | **Full matrix**: multiple routes × multiple benchmarks, results written to a single JSON, supports resume | all of the above |
| [probe_yesno.py](probe_yesno.py) | **Alternative readout strategy probe**: verifies whether "independent per-candidate yes/no scoring" is feasible on an arbitrary vendor. **Measured conclusion: not adopted** (see [REPORT 5.6](REPORT.md)) | none (runs JevBench items) |
| [probe_provider.py](probe_provider.py) | **Cross-vendor compatibility probe**: verifies whether an arbitrary OpenAI-compatible endpoint can carry this mechanism (logprobs readability / thinking-disable syntax / slot capacity / accuracy) | none + needs an API key (can be auto-matched to a route from llm2decision.yaml) |
| [make_provenance.py](make_provenance.py) | Refreshes `provenance.json`: implementation fingerprint / dataset hashes / item-set hashes / run-artifact hashes | none |

`probe_provider.py` initially only served DashScope; it has now been turned into a generic endpoint probe, and **you should run it before switching vendors rather than guessing from the docs** (each vendor differs in its top_logprobs cap, thinking-disable parameter, and logprobs whitelist):

```bash
# DashScope (when the key is omitted, the key of a matching route is taken from llm2decision.yaml by base_url)
python3 benchmarks/probe_provider.py --models qwen3.7-plus --limit 60

# Ark for a same-item-set comparison (the thinking-disable parameter has a different name and must be specified explicitly)
python3 benchmarks/probe_provider.py \
  --base-url https://ark.cn-beijing.volces.com/api/v3 \
  --api-key "<ARK key>" --thinking-param thinking \
  --models doubao-seed-2-1-lite-260915 --limit 60

# OpenRouter (one endpoint covers multiple vendors; the cap defaults to 20, and thinking is usually disabled with none)
python3 benchmarks/probe_provider.py \
  --base-url https://openrouter.ai/api/v1 \
  --api-key "<OpenRouter key>" --thinking-param none \
  --models openai/gpt-4o-mini --limit 60
```

`--max-top-logprobs` defaults to auto-detection from `--base-url` (5 when it contains `dashscope`, otherwise 20) and can be overridden explicitly — **do not test an endpoint that supports 20 with 5**, as that will under-measure its slot capacity.

`--limit` controls the number of accuracy items; the slot-capacity test always buckets by candidate count (4 items each for 3/4/5/6) and is independent of `--limit`.

### Common commands

```bash
# JevBench full 231 (about 40s)
python3 benchmarks/run_jevbench.py --tag nobias
python3 benchmarks/run_jevbench.py --logit-bias --tag withbias   # enable the logit_bias fallback

# Smoke test: run only the first 20 items on a specified route
python3 benchmarks/run_jevbench.py --limit 20 --model doubao-2.1-lite

# Nimble / VitaminC
python3 benchmarks/run_nimble.py   --subset compat282
python3 benchmarks/run_vitaminc.py --subset nimble599
python3 benchmarks/run_vitaminc.py --subset sample300 --n 300

# Kev (note: this script uses --route, and it is required)
python3 benchmarks/run_kev.py --route doubao-2.0-pro
python3 benchmarks/run_kev.py --route doubao-2.0-pro --limit 8   # sample 8 items per subset

# Full matrix (10 routes × 4 benchmark groups, about 28k calls)
python3 benchmarks/run_matrix.py
python3 benchmarks/run_matrix.py --routes doubao-2.1-lite,doubao-2.1-pro --benches jevbench,nimble
python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite   # Kev has a large item count, so by default only the specified routes are run
python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-<timestamp>.json  # resume

# Refresh reproducibility info (should be re-run after every evaluation)
python3 benchmarks/make_provenance.py --run benchmarks/results/matrix-<timestamp>.json
```

**Two inconsistencies in argument conventions** (recorded as-is to avoid pitfalls): `run_kev.py` uses `--route` (required), while the other runners use `--model` (optional, defaulting to `default_model` in `llm2decision.yaml`); `--concurrency` defaults to anything from 6 to 8 depending on the script.

## 2. Data preparation

Of the four benchmark groups, only Kev is **auto-downloaded**; the rest must be prepared manually:

```bash
# 1) JevBench (MIT) — run_jevbench.py reads datasets/public/*.jsonl
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench

# 2) Nimble (Bespoke Labs) — run_nimble.py reads data/eval.jsonl;
#    it also provides the manifest needed by the VitaminC nimble599 subset (docs/assets/public-benchmarks/subsets/)
git clone --depth 1 https://github.com/bespokelabsai/nimble benchmarks/nimble

# 3) Kev (Apache-2.0) — ensure_data() in run_kev.py auto-fetches to benchmarks/kev/*.jsonl, no manual step
# 4) VitaminC — the data is not in the repo: run_vitaminc.py pulls tals/vitaminc online via HF datasets
pip install datasets
```

Self-check:

```bash
python3 benchmarks/run_matrix.py --limit 5
# Expected: prints 5 items for each of the first three groups, and notes "no --kev-routes specified: skipping Kev (the other three groups still run)"
python3 benchmarks/run_matrix.py --limit 5 --kev-routes <any route name>   # then verify Kev's auto-download
```

## 3. Artifacts and naming

Everything lands in `results/`:

| Script | Filename | Content |
|---|---|---|
| `run_jevbench.py` / `run_nimble.py` / `run_vitaminc.py` | `<bench>-<timestamp>-<tag>.jsonl` + `.summary.json` | per-item records + summary |
| `run_kev.py` | `kev-<timestamp>-<tag>.json` | six-subset summary + skip statistics |
| `run_matrix.py` | `matrix-<timestamp>.json` | route × benchmark matrix, flushed once per completed route |
| `probe_yesno.py` | `probe-yesno-<timestamp>.json` | raw evidence for the four hypotheses |
| `probe_provider.py` | `probe-provider-<timestamp>.json` | raw evidence for the four hypotheses (including per-item top tokens) |
| `make_provenance.py` | `provenance.json` | machine-readable provenance (fixed filename, overwritten each time) |

`--tag` is only used by the first four scripts, to distinguish different configurations of the same benchmark (such as `nobias` / `withbias`).

## 4. Protocol conventions (must read before interpreting the numbers)

These conventions determine how the numbers are computed; changing them is equivalent to changing the conclusions:

1. **Accuracy is hard-label argmax**, consistent with the industry protocol, with no LLM-as-judge.
2. **The denominator is the number of valid items after excluding connection failures, `n_valid`**, not the total item count; failures are counted separately in `n_errors`. The only exception is the A/B experiment group from `run_jevbench.py --logit-bias` (whose denominator includes failed items), explained in REPORT.md section 4.4.
3. **Connection-type errors are retried 3 times** (backoff 1s/3s/9s), with no post-hoc resampling.
4. **Cells with a failure rate > 20% are marked `valid=false`** and are not included in accuracy conclusions.
5. **Candidate cap 10**: items with more than 10 candidates are skipped and counted by `run_kev.py` (`skipped_too_many_candidates`). This is also why MASSIVE (18 classes) and BANKING77 (77 classes) cannot be run.
6. **Subset protocol**: Nimble uses `compat282` (measured 280 items), VitaminC uses `nimble599` (matched 599/599), and Kev counts only items with `_meta.variant == "clean"`. The full rules and hashes are in REPORT.md section 6.

## 5. Adding a new benchmark

1. Create `run_<name>.py` providing three functions: `to_payload(item) -> dict` (converts an item into the shape of a `SystemOneRequest`), `predicted_label(item, decision)`, and `is_correct(gold, predicted)` — follow the style of [run_nimble.py](run_nimble.py).
2. `import` it in [run_matrix.py](run_matrix.py), add it to `BENCHMARKS` and to the branches in `load_items()` / `check()`, and the matrix will run it along with the rest.
3. Add the data source, subset rules, and scoring protocol to `PROTOCOLS` in [make_provenance.py](make_provenance.py), otherwise the reproducibility chain breaks.
4. After running, add a section to [REPORT.md](REPORT.md); the numbers must be bound to the specific `results/` artifact hashes.

## 6. Known limitations

| Limitation | Description |
|---|---|
| MASSIVE-en / BANKING77 cannot be run | 18 / 77 candidates, exceeding the service's candidate cap of 10. For a workaround see [probe_yesno.py](probe_yesno.py) |
| OpenJev text 19 items not reproducible | The upstream harness and prompts are not public, and the `control` and `chess` data sources cannot be found |
| JevBench scores of open-source models | Require a GPU and the Open-Jev loader and cannot be run on this machine; REPORT.md cites their published values |
| `probe_yesno.py` does not modify `src/llm2decision/` | It is a feasibility probe and does not participate in service operation; for the conclusions see REPORT.md and docs/design.md |
