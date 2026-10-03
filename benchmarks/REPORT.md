[English](REPORT.md) | [简体中文](REPORT.zh-CN.md)

# Survey and Empirical Report on Jev-class Model Benchmarks

**Subject under test**: the decision service implemented in this project (Volcengine Ark-hosted API, the 8 base-model routes currently configured).

**Evaluation method**: the `POST /v1/systemone` equivalent path; a single forward pass, greedy decoding (`temperature=0`), reading only generation-sequence position 0, candidate cap 10, no temperature calibration.

**Data version**: section 4 is bound to this round's full run artifact (8 routes × 4 benchmark groups, 26,184 calls, 0 failures): `benchmarks/results/matrix-20261003-022427.json` (sha256 `80875133…`). Full hashes and boundaries are in section 6.4.

The previous edition of this report (the 10-02 run, including the retired route names `doubao-2.0-lite`, `deepseek-v4.1-flash`, `deepseek-v4-flash`) and the early initial results that were not protocol-aligned have been moved to [Appendix A](#appendix-a-historical-artifacts-superseded-by-section-4), and the main text no longer cites them.

**Accuracy protocol**: throughout, all figures are **hard-label argmax accuracy** (consistent with the industry protocol), and the denominator is the **number of valid items after excluding connection failures** (this round's matrix had 0 failures, so the denominator is simply the item count; the only exception is the A/B comparison experiment in section 4.4, whose table denominator includes failed items and is annotated as such inside the table).

**Contents**

- [1. Core conclusions](#1-core-conclusions)
- [2. Benchmarks and runnability](#2-benchmarks-and-runnability)
- [3. Evaluation method and protocol](#3-evaluation-method-and-protocol)
- [4. Measured results](#4-measured-results)
- [5. Industry comparison](#5-industry-comparison)
- [6. Reproduction and reproducibility](#6-reproduction-and-reproducibility)
- [Appendix A: historical artifacts (superseded by section 4)](#appendix-a-historical-artifacts-superseded-by-section-4)

---

## 1. Core conclusions

1. **The approach holds, and under the same protocol it surpasses the industry's best.** Using hosted-API logprobs to replicate the Jev form, `doubao-evolving` reaches **91.8%** on the JevBench public 231 and **83.8%** on the hard tier, with `doubao-2.1-pro` at 90.9% / 82.0% — both above Jev 1.13.0 (86.58% / 72.97%), Open-Jev-27B (85.28% / 72.07%), and all open-weight models (the highest being NeoHorse-Jev-4B at 75.32% per-sample).
2. **Same items, same protocol: we land in the "90 band" and lead on the hard tier.** On InternLM's [Intern-Decision](https://github.com/InternLM/Intern-Decision) seven-suite benchmark (items **verified identical, id by id**, to its official bundles), `doubao-evolving` averages **89.85** and `doubao-2.1-pro` **89.37** across the seven suites — above Jev's 88.74 and close to Intern-Decision-4B's 90.02; the **hard tier at 82.88% is 9 points above that table's best (73.87%)**. The weak spots are `typed_decisions` and ToolACE — the counterpart is a **specialist fine-tune** trained to turn state into structured decisions (see 5.7).
3. **The gap comes from the base model, not the mechanism.** With the same readout logic, `doubao-2.0-mini` scores only 81.8%, while switching to `doubao-2.1-pro` raises it to 90.9% — not a single byte of the mechanism changed.
4. **The speed/accuracy compromise is `doubao-2.1-lite`**: JevBench 87.9% / Nimble 88.2% / Kev 81.8% at a p50 of 550 ms (about half of the pro tier); for maximum accuracy go to `doubao-evolving` or `doubao-2.1-pro` (91.8% / 90.9%, Nimble 94.6% / 94.3%). `doubao-2.0-mini` (355 ms) is the fastest, but its 81.8% / Nimble 71.4% is only enough for very lightweight scenarios.
5. **Kev is the only group that has not caught up**: our best is 81.80% (six subsets, equal-weighted, `doubao-2.0-pro`) vs Jev 1.13.0's 85.52, a gap of 3.7 points; the losses concentrate in the two transfer-v9 subsets (75.9~78.8%), while transfer-v4 (81.1~86.9%) is actually a strength.
6. **A stronger model ≠ a better fit for this mechanism.** Both DeepSeek routes are broadly below same-price doubao: `deepseek-flash` is the highest on VitaminC fact verification (75.5%), but lags clearly on Nimble (82.1%) / Kev (80.4%); `deepseek-v4-pro` is low across all four groups.
7. **Format compliance can be backstopped**: a uniform-strength `logit_bias` zeroes out the readout failures caused by "the model not emitting a label", without changing the within-candidate probability distribution; enabling it in production is recommended (see 4.4).
8. **confidence is currently untrustworthy**: no temperature calibration has been done (`temperature_scale=1.0`). The Brier / ECE values in 5.7 / 5.8 serve the same-item comparison only: the better Brier comes mostly from higher accuracy, while the ECE (hard 0.101, pilot 0.309) shows the probabilities are over-confident — before shipping probabilities, gather labeled data and run temperature calibration.
9. **Two cells are blank — either unrunnable or irreproducible**: OpenJev text (harness and prompts not public); MASSIVE-en (18 candidate classes exceeds the cap of 10).

---

## 2. Benchmarks and runnability

### 2.1 Mainstream benchmarks for this kind of model

| Benchmark | Task form | Scale | Public? | License |
|---|---|---|---|---|
| **JevBench** (`fstandhartinger/jevbench`) | Mixed Choice / Noul / Score, four tiers easy / standard / judge / hard | 534 items, 231 public (72 original + 48 easy + 111 hard) | Public subset + sealed held-out items | MIT |
| **Nimble public suite** (Bespoke Labs) | 5 Choice + 5 Noul + 3 Score subsets | 3,880 items (13 subsets) | Public (rebuilt from upstream per manifest) | Follows upstream |
| Nimble built-in holdout | Mixed | 324 items (evaluation takes 280 by the compat282 rule, see 3.2) | Public (bundled in the repo) | See repo |
| **VitaminC** | Choice 3-way (fact verification) | 63,054 validation items | Public | CC BY-SA 3.0 |
| OpenJev text | NLI + multi-choice reranking + fixed-candidate GSM8K | 19 tasks | Public | Follows upstream |
| Kev | decision-v7 / transfer-v4 / transfer-v9 | 6,436 items (5,768 clean items counted in evaluation) | Public | Apache-2.0 |
| MASSIVE 1.1 | Choice over 18 intent domains | 2,974/language | Public | CC BY 4.0 |
| BANKING77 | Choice over 77 intents | 13,083 items | Public | CC BY 4.0 |
| TypeSafe official 4-workflow | Security incidents / agent trace / invoices / customer service | Private | Private | — |
| S1 Bench | 6 subsets | 1,999 items | Leaderboard only | — |

### 2.2 Scoring protocol (common industry practice)

- **It is all hard-label comparison; no LLM-as-judge is needed** (the only exception is the TypeSafe official private suite, whose reference labels are the average of two frontier models).
- JevBench: averaged over tiers with each tier weighted 1/3; for `score` items the argmax is used, and the expectation is reported separately as MAE.
- Nimble: accuracy + Wilson CI + ECE(10 bins) + Brier; `score` additionally reports MAE, and `multinli` / `civil_comments` additionally report agreement with the human distribution (JSD / TVD).
- Kev: equal-weighted average of the clean accuracy of the six subsets.
- **Common point: a single forward pass taking the argmax; multiple-sample voting is not allowed.** This is exactly isomorphic to how this project is implemented.

### 2.3 Runnability under the candidate cap

The candidate cap of this service is 10 (`top_logprobs` has a hard cap of 20; after deducting EOS, full-width variants, and punctuation, the safe line is 10).

| Runnables | Not runnable (reason) |
|---|---|
| JevBench public 231 (at most 6 candidates measured) | MASSIVE 18 domains, BANKING77 77 intents (candidates exceed the cap) |
| Nimble built-in holdout (280 items taken for evaluation, 2~6 candidates measured) | the 16-option decision-models-under-pressure |
| VitaminC (3-way) | TypeSafe official 4-workflow (private + two-model consensus reference) |
| Kev six subsets (items with >10 candidates are skipped and counted) | S1 Bench (item set not published) |
| entries among Nimble's 13 subsets with ≤10 candidates | OpenJev text (harness and prompts not public) |

---

## 3. Evaluation method and protocol

### 3.1 Readout mechanism and its equivalence to Jev

Jev / vLLM's approach is to **stop at the answer position and do a single forward pass, reading the distribution directly** (prefill scoring). Ark does not expose a prefill scoring interface, so this implementation uses an equivalent scheme:

> the prompt stops at the answer position → greedy decoding generates **1** token → read only the `top_logprobs` at **generation-sequence position 0** → normalize within the candidate labels.

The semantics of position 0 are exactly "the next-token distribution at the answer position", equivalent to Jev's single forward pass. The cost is an extra dependency on model cooperation (the model must actually emit the label as its first token): with `logit_bias` disabled, the control experiment saw 3/231 readout failures, dropping to 0 once it is enabled (see 4.4); **this round's full matrix had 0 readout failures across 26,184 calls**.

The relevant hard mechanism constraints (see [docs/design.md](../docs/design.md) for details):

| Constraint | Value | Reason |
|---|---|---|
| `temperature` | `0` (greedy) | Guarantees the answer position deterministically lands at generation-sequence index 0 |
| `max_tokens` | `1` | Avoids generating explanatory text, which would make position 0 not the answer distribution |
| `top_logprobs` | `20` | Ark's hard cap; needed to accommodate non-candidate tokens such as EOS, full-width variants, and punctuation |
| Candidate cap | `10` | The safe line after deducting distractors from the 20 top_logprobs entries; exceeding it returns 422 directly |
| `disable_thinking` | `true` | Thinking models return a direct 400 when combined with logprobs |

### 3.2 Scoring protocol and failure handling

| Item | Rule |
|---|---|
| Scoring target | per-item hard-label argmax, no LLM-as-judge |
| **Accuracy denominator** | **the number of valid items after excluding connection failures, `n_valid`**, not the total item count |
| Connection-type errors | counted separately in `n_errors`; retried at most 3 times (backoff 1s/3s/9s), with no post-hoc resampling |
| Cell validity | a failure rate > 20% marks the cell `valid=false`, and it is excluded from accuracy conclusions |
| Model refuses to answer | (first token is not a candidate label) recorded as **incorrect**, with no special handling |
| Items exceeding the candidate cap | in Kev, items with >10 candidates are **skipped and counted**, not recorded as incorrect |

Section 4.2 lists the failure count for each cell separately, so readers can recompute the denominators themselves.

Subset protocol (aligned with the industry comparison; rules and hashes in 6.3):

- **Nimble**: the full 324 items are not run; by the compat282 rule (state ≤384 tokens, request body ≤2048 tokens, candidates ≤26, whole families retained) the measured result is **280 items**. The 2-item difference from the industry-annotated 282 is a boundary disagreement at 384/386 tokens, **recorded honestly without padding the numbers**.
- **VitaminC**: rows are taken exactly by the 599 `unique_id`s in the Nimble manifest, **matching 599/599**.
- **Kev**: all six subsets in full, counting only items with `_meta.variant == "clean"`, yielding **5,768** items.

---

## 4. Measured results

Data source: a single full matrix, `benchmarks/results/matrix-20261003-022427.json` (8 routes × 4 benchmark groups, 26,184 calls, 0 failures). Artifact hashes are in section 6.4.

### 4.1 Full matrix (main results)

| Route | JevBench 231 | Hard 111 | Nimble 280 | VitaminC 599 | Kev six-subset equal-weighted | p50¹ |
|---|---:|---:|---:|---:|---:|---:|
| **doubao-evolving** | **91.8%** | **83.8%** | **94.6%** | 73.1% | — | 1152 ms |
| doubao-2.1-pro | 90.9% | 82.0% | 94.3% | 73.6% | — | 1110 ms |
| doubao-2.0-pro | 90.0% | 81.1% | 89.3% | 72.0% | **81.80%** | 1611 ms |
| doubao-2.1-turbo | 88.7% | 79.3% | 90.4% | 74.6% | — | 1593 ms |
| **doubao-2.1-lite** | 87.9% | 76.6% | 88.2% | 72.5% | 81.77% | 550 ms |
| deepseek-flash | 83.1% | 67.6% | 82.1% | **75.5%** | 80.40% | 421 ms |
| doubao-2.0-mini | 81.8% | 64.0% | 71.4% | 74.8% | — | **355 ms** |
| deepseek-v4-pro | 78.4% | 63.1% | 81.4% | 67.1% | — | 735 ms |

¹ p50 is the median latency over JevBench 231. Sorted by JevBench descending, ties broken by hard descending. The Kev column was only run for all six subsets on 3 representative routes (`doubao-2.0-pro` / `doubao-2.1-lite` / `deepseek-flash`, 5,768 items/route); that column is the **equal-weighted average of the six subsets**, consistent with the industry Kev protocol; under a per-item pooled protocol instead, the three would be 81.1% / 81.0% / 80.0%.

² **This table must not be compared cell by cell with the previous edition (the 10-02 run)**: between the two runs, the same route on the same benchmark moves by ±2~7 items (MoE routing / batching nondeterminism under greedy decoding; in 4.4, two runs of the same configuration differ by 7 items). The routes `doubao-2.0-lite`, `deepseek-v4.1-flash`, and `deepseek-v4-flash` from the previous edition are no longer in the current configuration; their numbers at the time are in [Appendix A](#appendix-a-historical-artifacts-superseded-by-section-4).

**By question type (JevBench 231)**: `choice` peaks at 92.8%, `noul` at 91.9%, `score` at 88.9% (all in `doubao-evolving` / `doubao-2.1-pro`); the Nimble `score` subset (54 items) peaks at 94.4% (`doubao-evolving`) with a median around 87%, and bottoms out at 57.4% (`doubao-2.0-mini`).

### 4.2 Valid samples and readout failures

**This round had 0 failures across 26,184 calls** (both connection-type and readout-type), every cell of the 8 routes × 4 benchmark groups is `valid=true`, and the accuracy denominator is simply the item count (JevBench 231 / Nimble 280 / VitaminC 599 / Kev 5,768). The previous per-cell failure table is therefore no longer needed; for item-level verification, the run artifact and `provenance.json` carry the raw records.

This is consistent with the current configuration: the entire `doubao` line has `logit_bias` enabled; the two DeepSeek routes do not, and this round saw no readout failures on them either. Historical counterexamples of readout failure and the fallback's effect are in 4.4.

### 4.3 Kev six-subset detail

| Route | decision-v7-dev | decision-v7-test | transfer-v4-dev | transfer-v4-test | transfer-v9-dev | transfer-v9-test | equal-weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| doubao-2.0-pro | 81.8% | 81.1% | 84.5% | 86.9% | 77.8% | 78.8% | **81.80%** |
| doubao-2.1-lite | 82.1% | 80.5% | 85.5% | 86.7% | 77.9% | 77.9% | 81.77% |
| deepseek-flash | 82.7% | 81.9% | 81.1% | 83.5% | 75.9% | 77.3% | 80.40% |

We have not caught up with Jev 1.13.0's 85.52 (3.7 points behind): the gap is concentrated in the two **transfer-v9** subsets (75.9~78.8%); **transfer-v4** is instead a strength (81.1~86.9%, above Kev-4B's 81.47). `deepseek-flash` is the highest on decision-v7 (82.7 / 81.9%) but the lowest on both transfer groups — each of the three routes has its weak spot, and none has caught up. 17,304 items across the three routes, 0 failures. Jev 1.13.0 does not publish per-subset detail, so comparison is only possible on the six-subset equal-weighted figure.

### 4.4 logit_bias fallback experiment

The same batch of 231 items, the same model (`doubao-2.0-mini`), toggling only the `logit_bias` switch. To separate the "switch effect" from "run-to-run variance", the enabled arm was run twice:

| Configuration | Run artifact (same directory) | Overall | Hard | Readout failures |
|---|---|---:|---:|---:|
| bias off | `jevbench-20261003-093725-nobias` | 189/231 = 81.82% | 73/111 = 65.77% | **3 items** |
| bias on | `jevbench-20261003-033600-withbias` | 186/231 = 80.52% | 71/111 = 63.96% | **0 items** |
| bias on (repeat) | `jevbench-20261003-033544-withbias2` | 193/231 = 83.55% | 76/111 = 68.47% | **0 items** |

All 3 failures were "the model's first token was not a label" (it actually output "请" ("please"), "首先" ("first") ×2). Enabling the uniform bias **zeroes out the format failures**. The accuracy movement should not be attributed to the switch: two runs of the same enabled configuration differ by 7 items (186 vs 193), and the off arm's 189 sits between them — the difference is drowned by run-to-run variance (MoE routing / batching nondeterminism).

**Protocol note (must read)**: this table comes from single-benchmark artifacts of `run_jevbench.py`, with a denominator of 231 (readout failures counted as wrong); the `doubao-2.0-mini` row in 4.1 (189/231, 0 failures) comes from the matrix artifact and is a different run. The correct count for this model on this benchmark lands in the 186~193 range across four independent runs; both sets of numbers are retained, each labeled with its source, and not merged.

Conclusion: `logit_bias` is a deterministic fallback for **format failures** (3 → 0) with no discernible systematic effect on accuracy; the uniform-strength bias cancels itself out during within-candidate normalization, leaving the probability distribution unchanged, so it need not be undone.

---

## 5. Industry comparison

### 5.1 Six-benchmark comparison (including the table from the ModelScope `NeoHorse-Jev-4B` model card)

The NeoHorse-Jev-4B model card (TokenRhythm, 2026-09-24) provides a cross-comparison table of six benchmark groups that **partially overlaps** with the groups we ran. Merged, it is as follows (0–100, higher is better):

| Model | JevBench | Kev | OpenJev text | Nimble | VitaminC | MASSIVE | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| **this implementation · doubao-evolving** | **91.8** | — | — | **94.6** | 73.1 | cannot run¹ | — |
| **this implementation · doubao-2.1-pro** | 90.9 | — | — | 94.3 | 73.6 | cannot run¹ | — |
| **this implementation · doubao-2.0-pro** | 90.0 | 81.80 | — | 89.3 | 72.0 | cannot run¹ | 83.3⁴ |
| **this implementation · doubao-2.1-lite** | 87.9 | 81.77 | — | 88.2 | 72.5 | cannot run¹ | 82.6⁴ |
| **this implementation · deepseek-flash** | 83.1 | 80.40 | — | 82.1 | **75.5** | cannot run¹ | 80.3⁴ |
| JEV-27B (AutoTrust, Apache-2.0) | 88.70 | 83.75 | 73.89 | **92.91** | 77.46 | **87.71** | **84.07** |
| Jev 1.13.0 (TypeSafe hosted) | 87.18 | **85.52** | 72.96 | 91.84 | — | — | 83.85² |
| Open-Jev-9B (open weights) | 77.13 | 77.87 | 65.39 | 80.50 | 68.28 | 84.86 | 75.67 |
| NeoHorse-Jev-4B (open weights) | 75.73³ | 81.92 | 58.74 | 87.23 | 77.13 | 85.43 | 77.70 |
| Kev-4B (open weights) | 73.71 | 81.47 | 54.75 | 73.40 | 76.46 | 85.71 | 74.25 |
| Laya English (open weights) | 55.82 | 61.30 | 40.07 | 45.04 | 78.63 | 68.57 | 58.24 |
| Laya Typed Decisions | — | — | — | 48.94 | 78.30 | 65.43 | — |
| NeoHorse-1-4B (baseline reference) | — | — | — | 69.15 | 63.27 | 82.86 | — |

¹ MASSIVE-en is an 18-class classification, which exceeds this service's candidate cap of 10 and cannot be changed.

² Jev 1.13.0's AVG is an AutoTrust run-on-behalf result, with VitaminC / MASSIVE missing; the mean is computed over the remaining groups.

³ The NeoHorse model card notes: for JevBench, **75.73 is a task-family macro average**, while the **per-sample accuracy given in the same file is 75.32** — our number is per-sample accuracy, so it should be compared with 75.32.

⁴ Our AVG is the **four-group mean of JevBench / Kev / Nimble / VitaminC**, missing the MASSIVE and OpenJev groups, and **must not be directly compared with a 6-group mean**.

### 5.2 Same protocol: JevBench public 231 (per-sample accuracy)

| Model | public 231 | Hard 111 | Source |
|---|---:|---:|---|
| **this implementation · doubao-evolving** | **91.8%** | **83.8%** | this measurement |
| **this implementation · doubao-2.1-pro** | 90.9% | 82.0% | this measurement |
| Jev 1.13.0 (TypeSafe hosted) | 86.58% | 72.97% | Open-Jev-27B-v1.1 model card |
| Open-Jev-27B-v1.1 | 85.28% | 72.07% | same as above |
| Open-Jev 9B | 77.49% | 59.46% | same as above |
| Open-Jev 2B | 64.94% | 41.44% | same as above |
| NeoHorse-Jev-4B | 75.32% | — | NeoHorse model card (per-sample) |

### 5.3 Protocol differences you must read before cross-comparison

1. **JevBench metric definition**: the NeoHorse table uses a task-family macro average (75.73), while we use per-sample accuracy. Their file also gives the per-sample 75.32, and **the comparable pair is 91.8% vs 75.32%**.
2. **Nimble subset**: already aligned. Both sides filter from 324 items by the compat282 rule (they annotate 282 items; we measure 280).
3. **VitaminC sampling**: already aligned. Both sides use the 599 items from the Nimble manifest (seed 20260918).
4. **Kev**: we use the six-subset clean equal-weighted average, consistent with the industry protocol; but Jev 1.13.0 does not publish per-subset detail, so item-by-item verification is impossible.
5. **Not prefill scoring**: we have an extra dependency on "the model cooperating to emit the label" (failure rate 1.3% with `logit_bias` disabled, 0 once enabled, 0 in this round's matrix), whereas the industry reads the distribution directly.
6. **No temperature calibration**: we compare only argmax, not probability mass (ECE / Brier are not involved), so **none of the probabilities in this report should be used as confidence**.

### 5.4 Attributing the gaps

| Observation | Attribution |
|---|---|
| JevBench / Nimble broadly surpass the industry's best | stronger base models, equivalent mechanism (see conclusion 2) |
| Kev lags by 3.7 points, with losses concentrated in transfer-v9 | that subset leans toward "cross-domain transfer" and needs stronger generalization; this round extended Kev to 3 routes (including `deepseek-flash`) and still has not caught up |
| Both DeepSeek routes are broadly low | poor fit with this mechanism (single-token label readout), not a question of model capability |
| Open-weight models are generally 81~83% on Kev yet we are higher on JevBench | the two benchmark types test different transfer abilities; it is unwise to draw conclusions from a single benchmark |
| `doubao-2.0-mini` manages only 57.4% on Nimble `score` | low-end routes are unstable on ordered scoring; high-end routes reach 92.6~94.4% on the same subset |

### 5.5 Cross-vendor comparison on the same item set (Volcengine Ark vs Alibaba Cloud DashScope)

The service now supports multiple vendors (the `provider` config item, see [README section 4, configuration](../README.md#configure)). The mechanism differences between the two vendors do not affect the readout logic, but **the slot budget differs by more than a factor of two**, so we ran one comparison on the **same batch of items**.

Item set: a stratified sample of 60 JevBench choice items, and **the item-id lists of both batches have been verified item by item to be fully identical**. For an equal-constraint comparison, both sides are pinned to `top_logprobs=5` — note that this is an **artificial constraint** for Ark (which can actually use 20 and runs at 20 in production).

| Endpoint / model | Accuracy | p50 | Readout failures |
|---|---:|---:|---:|
| **Ark `doubao-seed-2-1-lite`** (pinned to 5 slots) | **57/60 = 95.0%** | 853 ms | 0 |
| DashScope `qwen3.7-plus` | 54/60 = 90.0% | 967 ms | 0 |
| DashScope `qwen3.8-flash` | 54/60 = 90.0% | 913 ms | 0 |
| DashScope `qwen3.5-plus` | 53/60 = 88.3% | 1107 ms | 0 |

Item-by-item comparison (Ark vs DashScope's strongest, `qwen3.7-plus`):

```
verdict agreement   57/60
both wrong           3 items
Ark wrong/DashScope right  0 items
Ark right/DashScope wrong  3 items   ← all hard tier (2 probabilistic reasoning + 1 multi-hop)
```

**Ark strictly dominates (3 more correct, 0 fewer correct), but the accuracy gap is not significant** — a 3-item difference on a 60-item sample is about 1.3 standard errors. **The real difference is the structural slot budget:**

| Candidate count | Ark full coverage | DashScope best (`qwen3.5-plus`) |
|---|---:|---:|
| 3 | 4/4 | 4/4 |
| 4 | 3/4 | **4/4** |
| 5 | 2/4 | 3/4 |
| **6** | 0/4 | **0/4** |

That is: **Ark can reliably run 10 candidates, while DashScope can only run 4** (6 candidates is 0/4 full coverage across all 6 DashScope test models). The slots are eaten by noise — DashScope's top-5 is mixed with `<|im_end|>`, `The`, `Let`, `A/B/C/D`, occupying on average 0.88 (`qwen3.5-plus`) ~ 2.44 (`qwen3.8-27b`) slots per item depending on the model, and **the newer the flagship, the more noise**.

Selection conclusion: **on DashScope, choose `qwen3.5-plus` or `qwen3.7-plus`, not the newest 3.8 flagship.**

There is also a measured fact that contradicts the documentation: **DashScope's `logprobs` support list is the reverse of the official documentation** — `qwen-plus-2025-04-28` and `qwen3-32b`, which the documentation whitelist lists, actually return `null`; the entire qwen3.5/3.6/3.7/3.8 line, which the documentation does not list, is measured to be supported (provided thinking is explicitly disabled). The common hard precondition for both vendors is: **thinking mode and `logprobs` are mutually exclusive**; on DashScope, not passing the thinking-disable parameter makes `logprobs` silently become `null` (whereas Ark returns a direct 400).

Artifact bindings: for the three DashScope models, `benchmarks/results/probe-aliyun-20261002-174305.json` (sha256 `eec18b67c1411968…`); for the Ark comparison, `benchmarks/results/probe-aliyun-20261002-174832.json` (sha256 `d694d9b872641601…`). Reproduction commands are in [benchmarks/README.md](README.md).

### 5.6 Measured conclusion on the alternative readout strategy (per-candidate yes/no): not adopted

Following the idea of the open-source project LLM2Jev, we evaluated a second readout path (denoted **B**): **split each candidate into an independent yes/no query**, one call per candidate, reading the two tokens `yes`/`no` at position 0, using `q = P(yes)/(P(yes)+P(no))` as that candidate's score, then normalizing the N scores.

Its one irreplaceable advantage is that **each call occupies only 2 tokens, so the candidate count is not constrained by the `top_logprobs` slot budget** — which targets exactly the shortcoming in §5.5 that DashScope has only 5 slots. The probe `benchmarks/probe_yesno.py` was run for one round on each vendor, and the conclusion is not to adopt it:

| | Volcengine Ark (20 slots) | Alibaba Cloud DashScope (5 slots) |
|---|---|---|
| B's readout hit rate | **140/140 = 100%** (relying on an equal-amount `logit_bias`) | **99/140 = 70.7%** (no tokenizer interface → can only run bare) |
| Same-subset accuracy vs A | tied (28/30 each, **identical wrong items**) | tied (4/6 each) |
| B / A call volume | 4.67× | 4.67× |
| Latency p50 (B vs A) | 9777 ms vs 2598 ms | 9573 ms vs 1601 ms |
| Conclusion | **runs but gains nothing** | **cannot run at all** |

**Attribution of the failures on DashScope (41 misses, all for the same reason)**:

```
actual first token the model output: 'no' × 41
failure reason: the top_logprobs at position 0 lack ['yes'] × 41
```

That is, the 5 slots are filled by `no` and noise, so the denominator half of `P(yes)/(P(yes)+P(no))` is unobtainable. **The model answers perfectly correctly; there simply were not enough slots** — this is not fixable by changing the prompt; it requires `logit_bias` to push both `yes`/`no` into the top-k, and DashScope has no online tokenizer interface to obtain token ids.

**One protocol trap that must be noted**: B's full-set accuracy on DashScope, 86.7%, appears to "beat" A's 66.7%, but that is an illusion — A **can only run 6 items** out of the same batch of 30 (24 items return 422 due to candidate overflow), so the denominators are completely different. Narrowed to the same 6 items that A can run, **both are 4/6 = 66.7%**.

This round also quantified A's shortcoming on DashScope: **80% of items cannot be run due to candidate overflow**. That is exactly what B wanted to solve, but B itself does not hold up on that vendor.

Artifact bindings: for Ark, `benchmarks/results/probe-yesno-20261002-164300.json` (sha256 `f66b3cc407d7234a…`); for DashScope, `benchmarks/results/probe-yesno-20261002-182144.json` (sha256 `d1e230829e88e0ff…`). Restart conditions and the full rationale for not adopting it are in [docs/design.md §7.1](../docs/design.md).

### 5.7 Same items, same protocol: the Intern-Decision seven suites (InternLM)

[Intern-Decision](https://github.com/InternLM/Intern-Decision) is an open-source project in the same lane (Qwen3.5-based 4B/2B/0.8B fine-tunes plus a 96-case known-distribution calibration benchmark) that ships **its seven accuracy suites and the calibration pilot inside the repository** — which makes this the report's only **same-item, same-protocol** horizontal comparison, fundamentally different from the published-value citations in 5.1 / 5.2.

**Item alignment was verified first**: the JevBench 231 id list hash matches our main matrix exactly (`04399f09d6b39036…`), the other four suites come straight from its bundles (the sha256 of all four `test.jsonl` files **verified one by one** against its `manifest.json` ✅), and the decision total (12,351) matches its manifest record (10,751 rows / 12,351 decisions) item by item. Scoring protocol: see 6.3.

| Model | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **this implementation · doubao-evolving** | 100.00 | **98.61** | **82.88** | 73.45 | 89.35 | 89.58 | 95.07 | **89.85** | **0.2500** | 0.1009 |
| **this implementation · doubao-2.1-pro** | 100.00 | **98.61** | 81.08 | 73.50 | 87.74 | 89.55 | 95.11 | 89.37 | 0.2641 | 0.1054 |
| **this implementation · doubao-2.1-lite** | 100.00 | 97.22 | 77.48 | 73.75 | 83.23 | 87.92 | 94.21 | 87.69 | 0.3086 | 0.1283 |
| **this implementation · deepseek-flash** | 100.00 | 97.22 | 66.67 | 70.65 | 88.06 | 88.49 | **96.61** | 86.81 | 0.5466² | 0.2289² |
| Jev (TypeSafe hosted) | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.3584 | 0.0947 |
| Intern-Decision-4B (open weights) | 100.00 | 98.61 | 73.87 | **80.55** | **96.45** | 90.82 | 89.86 | **90.02** | 0.3468¹ | 0.0653¹ |
| Intern-Decision-2B (open weights) | 100.00 | 84.72 | 63.96 | 79.35 | **96.45** | 89.96 | 78.33 | 84.68 | 0.4373¹ | 0.1002¹ |
| Intern-Decision-0.8B (open weights) | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.5295¹ | 0.0657¹ |
| JevK5 | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.3662 | **0.0467** |
| SemIf | 100.00 | 98.61 | 61.26 | 62.80 | 85.16 | 89.22 | 92.53 | 84.23 | 0.4980 | 0.1122 |
| Kev | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.7383 | 0.2622 |
| Laya | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | **92.84** | 14.84 | 57.77 | 0.8042 | 0.2465 |

Average is the **arithmetic mean of the seven accuracy columns**, matching that table's definition; Brier / ECE come from the **111 JevBench-Hard items**, confidence taken as max(P), ECE over 10 equal-width bins — its documented rules adopted as-is. **Bold = this implementation's best value in that column, or the best among published values** (the Easy column has several tied perfect scores and is left unbolded).

**How to read this table:**

- **Average**: `doubao-evolving` 89.85 and `doubao-2.1-pro` 89.37 sit between Jev (88.74) and Intern-Decision-4B (90.02); all four routes beat every other published value (JevK5 85.16, SemIf 84.23, Kev 79.56, Laya 57.77).
- **The hard tier is the strength**: 82.88 / 81.08, 7~9 points above the published best (73.87); Easy and Original are on par with the best models (100 / 98.61); WildJailBreak is 94.2~96.6, where `deepseek-flash`'s **96.61 is the highest in the whole table**.
- **The weak spots are `typed_decisions` and ToolACE**: 70.7~73.8 vs ID-4B's 80.55, and 83.2~89.4 vs ID-2B/4B's 96.45. The attribution is clear — Intern-Decision is a specialist fine-tune whose **training objective** is exactly "state → structured decisions", while we are a general-purpose base model with single-token label readout and **no fine-tuning at all**; conversely, that a general model reaches the 90 band on the same items and beats the field by 9 points on the hard tier is the real evidence for the mechanism's transferability.
- **Read the two distribution columns separately**: Brier is also driven by accuracy (being 9 points up on hard mechanically lowers it), whereas ECE is the pure calibration column — our 0.1009 is close to Jev's 0.0947 but behind ID-4B's calibrated 0.0653.

¹ The three Intern-Decision rows compute Brier / ECE from probabilities **after their individually fitted temperatures** (T = 1.99 / 2.10 / 2.75, the T column in that table); we and every other baseline in this table use **uncalibrated** raw probabilities — so these two columns cannot be read as a calibration comparison.

² DeepSeek returns only one real probability; its remaining `top_logprobs` are placeholders (-9999), which are floored at "lowest observed − 3 nats". Coverage is only 0.17~0.34 (just 38 of the 111 hard items reach the reliability threshold), so these two cells are a **lower-bound approximation**. Its accuracy is unaffected (the argmax still comes from real observations).

### 5.8 Same items, same protocol: the 96-case known-distribution pilot

The upstream pilot ships an **exact reference distribution** for every case and scores expected multiclass Brier / expected ECE (lower is better, matching its `docs/CALIBRATION_BENCHMARK.md`):

| Model | Expected Brier ↓ | Expected ECE ↓ | Cases |
|---|---:|---:|---:|
| **this implementation · doubao-evolving** | **0.258** | 0.309 | 94 |
| **this implementation · doubao-2.1-pro** | 0.262 | 0.308 | 94 |
| **this implementation · doubao-2.1-lite** | 0.305 | 0.342 | 94 |
| **this implementation · deepseek-flash** | 0.351² | 0.369² | 94 |
| Intern-Decision-4B, uncalibrated | 0.628 | 0.213 | 96 |
| Intern-Decision-4B, calibrated (T=1.99) | 0.550 | **0.089** | 96 |
| Jev (`jev-1.13.0`) | 0.595 | 0.130 | 96 |

**Our Brier is clearly lower (0.258 vs 0.550 / 0.595) while our ECE is clearly worse (0.309 vs 0.089 / 0.130) — together they say one thing: the probabilities resolve well, but the confidence is far too aggressive.** The most direct evidence is the pilot's first case, a fair die (reference distribution uniform at 1/6 = 0.1667): we assign probability **0.642** to `1`, and after the option order is reversed, **0.973** to `6`. This is not a readout error (the argmax matches the reference) — it is carrying "certainty about the answer" straight over onto a distribution that is known to be uniform.

That is precisely what temperature calibration fixes: the "calibrated" 4B column drops ECE from 0.213 to 0.089, so the payoff on this task is significant. **Until labeled data is gathered and calibration is done, do not treat the probabilities in 5.7 / 5.8 as confidence** (consistent with conclusion 8).

Two protocol notes: ① our case count is **94, not 96** — the `sum_of_dice/02` pair (canonical / reversed) has 13 candidates, over this service's cap of 10, and is rejected on all four routes (the same two cases regardless of route); ② upstream also publishes a six-category breakdown; this round compares the pooled totals only (the category field lives in its `references.jsonl`, and can be split later if needed).

Artifact binding: `benchmarks/results/intern-20261003-125308-full.json` (sha256 `7acb2dbf884846d4…`) — 4 routes × 6 suites, 43,388 calls, 1 h 56 min. **No connection-layer retries** (all 10,847 records per route succeeded on the first attempt); 10 readout failures in total: 8 are the over-cap pilot cases above, and 2 are server-side errors (Ark 500 / request failure) hit once each by `doubao-2.1-pro` on toolace / wildjailbreak, recorded as wrong per the protocol (0.3% / 0.05% of those suites). The item hashes, dataset hashes, and item-set id hashes for this section are recorded in `provenance.json`; reproduction commands are in 6.1.

---

## 6. Reproduction and reproducibility

Every number in this report can be verified along the chain "dataset → subset rules → evaluation protocol → run artifacts". A machine-readable version is in `provenance.json`.

### 6.1 Reproduction commands

```bash
# 1) Data (upstream repos, each with its own license)
git clone --depth 1 https://github.com/fstandhartinger/jevbench benchmarks/jevbench
git clone --depth 1 https://github.com/bespokelabsai/nimble   benchmarks/nimble
git clone --depth 1 https://github.com/jaredpalmer/kev        benchmarks/kev
# VitaminC is pulled online via HF datasets (tals/vitaminc validation)

# 2) Full matrix (main results of section 4, about 26k calls)
python3 benchmarks/run_matrix.py                                  # 8 routes × 3 benchmark groups
python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite,deepseek-flash
python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-<ts>.json   # resume

# 3) single benchmark / single route (used for the section 4.4 experiment)
python3 benchmarks/run_jevbench.py --tag nobias
python3 benchmarks/run_jevbench.py --tag withbias --logit-bias
python3 benchmarks/run_nimble.py   --model doubao-2.1-lite
python3 benchmarks/run_vitaminc.py --model doubao-2.1-lite

# 4) Intern-Decision seven-suite comparison (section 5.7 / 5.8, about 43k calls)
#    The data is not distributed with this repository; fetch it from upstream per the
#    DATA_HINT at the top of benchmarks/intern_decision.py
python3 benchmarks/run_intern_suite.py --routes doubao-evolving,doubao-2.1-pro,doubao-2.1-lite,deepseek-flash

# 5) Refresh reproducibility info (re-run after every evaluation; pass both artifact kinds together,
#    provenance.json records all of them)
python3 benchmarks/make_provenance.py --run benchmarks/results/matrix-<ts>.json benchmarks/results/intern-<ts>-full.json
```

### 6.2 Implementation fingerprint

| Item | Value |
|---|---|
| Implementation fingerprint (content hashes of the 26 source files under `src/llm2decision/` + `benchmarks/`) | `d5efd27f85aed25f…` |
| Engine | `POST /v1/systemone` (label-logit readout) |
| Decoding | `temperature=0` (greedy), `max_tokens=1`, **reading only generation-sequence position 0** |
| Candidate cap | 10 (exceeding it returns 422 and is excluded from evaluation) |
| Output interface | hard-label argmax; no LLM-as-judge |
| Calibration | no temperature calibration (`temperature_scale=1.0`); the Brier / ECE in 5.7 / 5.8 are **uncalibrated raw probabilities** and must not be used as confidence |
| Failure protocol | connection failures counted separately and the cell marked invalid when the failure rate is >20%; a model refusing to answer is recorded as error |

The fingerprint above identifies this repository's current tree. Relative to the section 4 run (`matrix-20261003-022427.json`), three changes landed afterwards: ① `src/llm2decision/core/readout.py` gained `SENTINEL_LOGPROB`, treating vendor placeholders (-9999) as "not observed" — this only changes the `coverage` / `reliable` fields and the shape of the floored distribution and **never enters the argmax, so section 4's accuracy numbers are unaffected** (no published number ever depended on coverage); ② the debug page's EN/ZH toggle in `src/llm2decision/api/debug.py` (UI layer); ③ two new benchmark scripts, `benchmarks/intern_decision.py` and `benchmarks/run_intern_suite.py` (used by 5.7 / 5.8). Section 4's numbers therefore remain attributable to the current implementation, and the distribution columns of 5.7 / 5.8 are computed under the new sentinel rule. The run artifacts are published in the [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003) release.

### 6.3 Datasets and subset rules

| Benchmark | Source | Subset rule | Scoring protocol | Local data hash |
|---|---|---|---|---|
| JevBench | `fstandhartinger/jevbench` (MIT) | public 231 = original 72 + easy 48 + hard 111, full set with no sampling | per-item hard-label argmax | `5c2414edb3006b8b…` (consistent with the upstream manifest record ✅) |
| Nimble | `data/eval.jsonl` of `bespokelabsai/nimble` | compat282: state ≤384 tokens, request body ≤2048 tokens, candidates ≤26, whole families retained (measured **280** items) | exact per-sample match | `8e9e48b8de520659…` |
| VitaminC | HF `tals/vitaminc` validation (CC BY-SA 3.0) | nimble599: rows taken exactly by the 599 `unique_id`s in the Nimble manifest (matching 599/599) | 3-way choice argmax | `30bd72d23b3e58ae…` (manifest file) |
| Kev | the 6 subsets of `jaredpalmer/kev` (Apache-2.0) | full set, only `_meta.variant=="clean"`; items with >10 candidates skipped and counted | per-item argmax, equal-weighted average of the six | `8d5765d7aec4d08c…` (6 files) |
| Intern-Decision seven suites | `benchmarks/accuracy-v1` of `InternLM/Intern-Decision` (repository Apache-2.0; **AG News upstream metadata reports an unknown license**, and this project does not redistribute the data) | its bundles in full, no sampling: agnews 7,600 / toolace 310 / typed_decisions 400 records (2,000 decisions) / wildjailbreak 2,210; JevBench reuses the same items as the first row of this table (ids verified identical) | one request per record with all its questions (a typed_decisions record yields five decisions); per-decision hard-label argmax; the seven-suite mean is arithmetic; the hard tier also reports uncalibrated Brier / ECE | `4ff1cad1531f29cb…` (5 files; all four `test.jsonl` verified one by one against the upstream manifest ✅) |
| Intern-Decision calibration pilot | `benchmarks/known-distribution-pilot-v1` of the same repository | all 96 cases (two of them have 13 candidates, over the cap of 10, so 94 are actually covered) | expected multiclass Brier / expected ECE against the exact reference distribution; the accuracy column counts a decision correct when it matches the reference argmax | `54a2c97dffd972aa…` (3 files, consistent with the upstream manifest ✅) |

### 6.4 Run artifact bindings

| Benchmark | Actual item count | Item-set id list hash |
|---|---:|---|
| JevBench | 231 | `04399f09d6b39036…` |
| Nimble | 280 | `f0584784646c74f6…` |
| VitaminC | 599 | `2154d495d6b8d7bd…` |
| Kev (per-item) | 5,768 | `73fd023d385cb87b…` |
| Intern-Decision: typed_decisions | 400 records / 2,000 decisions | `7bb7caf4aebac8d0…` |
| Intern-Decision: toolace | 310 | `a6aae10f092815a4…` |
| Intern-Decision: agnews | 7,600 | `8e83621e743626bb…` |
| Intern-Decision: wildjailbreak | 2,210 | `ba4c08831294c81b…` |
| Intern-Decision: pilot | 96 (94 scored) | `6d79b66cbdb7929b…` |

The JevBench row (231, `04399f09d6b39036…`) comes out **id-for-id identical** in the section 5.7 Intern-Decision run — the machine-checkable proof behind the "same items" claim, and the precondition for putting both runs in one table.

**Section 4 is bound to this round's run artifacts:**
- Main matrix: `benchmarks/results/matrix-20261003-022427.json` (sha256 `80875133be0562ea…`, 8 routes × 4 benchmark groups, 26,184 calls)
- Section 5.7 / 5.8 comparison: `benchmarks/results/intern-20261003-125308-full.json` (sha256 `7acb2dbf884846d4…`, 4 routes × 6 suites, 43,388 calls, no connection-layer retries)
- Section 4.4 experiment (same directory): `jevbench-20261003-093725-nobias.summary.json`, `jevbench-20261003-033600-withbias.summary.json`, `jevbench-20261003-033544-withbias2.summary.json`
- Machine-readable provenance: `benchmarks/provenance.json` (implementation fingerprint, dataset hashes, item-set id hashes, and the file sha256 / item-set hashes of **both runs** above)

These artifacts are not committed to the repository (size); they are published in the [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003) release, which also collects the section 5.5 / 5.6 cross-vendor probes and the Appendix A historical artifacts, so everything can be verified in one place. Each download can be checked against the sha256 values above.

**Hashes change with every run** — scores must be bound to a specific hash, and that is precisely why this section exists.

### 6.5 Known non-reproducible items (listed honestly)

| Item | Why it is not reproducible |
|---|---|
| OpenJev text (19 items) | The harness and prompt templates of that family are not public, and the two data sources `control`(805) and `chess`(500) cannot be found — so this report leaves that cell blank |
| `zhihz/openjev`'s 89.8% / 97.7% | Its evaluation data is not provided with the release package, so only its published values can be cited. Note that it is **isomorphic to this project's mechanism** (reading only the candidate-letter distribution of the next token, without generating text) and is an independent web application rather than model weights |
| JevBench scores of open-source models | The 27B / 9B weights require a GPU and the Open-Jev loader, which cannot be run on this machine; the table shows their published values |
| Kev / Nimble / VitaminC scores in the industry table | Come from each model card's published values, cannot be reproduced item by item, and can only be cited after aligning the subset rules |
| The three Intern-Decision rows and their Brier / ECE | Their weights need a GPU and the XTuner backend, so they cannot be run on this machine and only the published values can be cited; those two columns are also computed after each model's fitted temperature and cannot be reproduced item by item |
| Intern-Decision's Hard TVD (10 public records) | Not computed this round — the comparison targets its two result tables (seven-suite accuracy, calibration pilot), and TVD is not among them; we report our own Brier / ECE instead |

---

## Appendix A: historical artifacts (superseded by section 4)

### A.1 The previous full matrix (2026-10-02, 10 routes)

The previous edition bound section 4 to `benchmarks/results/matrix-20261002-112301.json` (10 routes) and `matrix-20261003-005505.json` (the `deepseek-flash` full re-run). Because of the ±2~7-item movement between runs (footnote ² in 4.1), the new main text does not merge old numbers cell by cell; the full previous edition is preserved in git history, and its artifacts are in the [`eval-20261002`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261002) release. The numbers of the three retired route names at the time:

| Retired route name (10-02 run) | JevBench 231 | Hard 111 | Nimble 280 | VitaminC 599 | Kev six-subset equal-weighted |
|---|---:|---:|---:|---:|---:|
| doubao-2.0-lite | 90.0% | 80.2% | 86.4% | 73.1% | — |
| deepseek-v4.1-flash | 87.0% | 75.7% | 83.2% | 75.5% | 80.7% |
| deepseek-v4-flash | 73.5% | 53.1% | 60.4% | 63.4% | — |

### A.2 Early single-benchmark artifacts

The following numbers come from **single-benchmark artifacts run separately at an earlier stage**, whose protocol is inconsistent with section 4 (Nimble used all 324 items, VitaminC used a 300-item sample, and denominators included failed items), and **they are no longer used as a basis for conclusions**. They are retained only to trace the difference "before and after switching base models + aligning protocols".

| Benchmark | Early result (`doubao-2.0-mini`) | Same-model protocol in section 4 | Source of difference |
|---|---|---|---|
| JevBench public 231 | 187/231 = 80.95%; hard 71/111 = 63.96% | 189/231 = 81.8%; hard 71/111 = 64.0% | different runs (±2 items); the early run had bias off and 3 readout failures (counted as wrong) |
| JevBench by tier | easy 48/48 = 100%, original 68/72 = 94.4%, hard 71/111 = 63.96% | — | same item batch, different runs |
| JevBench by question type | choice 112/139 = 80.6%, noul 61/74 = 82.4%, score 14/18 = 77.8% | see the footnote in 4.1 | same as above |
| Nimble built-in holdout | 221/324 = 68.21% (choice 61.6% / noul 84.2% / score 54.7%) | 200/280 = 71.4% | **different subsets**: the full 324 vs the 280 of compat282 |
| VitaminC | 231/300 = 77.00% (self-sampled 300, seed 20260918) | 448/599 = 74.8% | **different sampling**: self-sampled 300 vs the manifest's 599 |
| Latency | p50 ≈ 399 ms, p95 ≈ 952 ms (concurrency 6) | 355 ms | different concurrency and batching conditions |

**An old conclusion overturned by this section's data**: the early report once stated that "ordered scoring (score) is an industry-wide blind spot, at 54.69% for this implementation". That 54.69% came from the 64 score items under the Nimble **324-item** protocol in the table above; under the aligned **280-item** protocol (54 score items) the same model scores 57.4% this round, while the best route reaches **94.4%**. That conclusion therefore **does not hold**, and section 4 has been changed to state it per route (see the footnote in 4.1 and 5.4).

Early artifact files (all collected in the [`eval-20261003`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261003) release):

- `benchmarks/results/jevbench-20261002-074113-nobias.summary.json` (187/231)
- `benchmarks/results/jevbench-20261002-074204-withbias.summary.json` (186/231)
- `benchmarks/results/nimble-20261002-074243-nobias.summary.json` (324 items)
- `benchmarks/results/vitaminc-20261002-074608-nobias.summary.json` (300 items)
