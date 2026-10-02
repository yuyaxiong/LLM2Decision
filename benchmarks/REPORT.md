[English](REPORT.md) | [简体中文](REPORT.zh-CN.md)

# Survey and Empirical Report on Jev-class Model Benchmarks

**Subject under test**: the decision service implemented in this project (Volcengine Ark-hosted API, 10 base-model routes; plus a full re-run of `deepseek-flash` on 2026-10-03).
**Evaluation method**: the `POST /v1/systemone` equivalent path; a single forward pass, greedy decoding (`temperature=0`), reading only generation-sequence position 0, candidate cap 10, no temperature calibration.
**Data version**: section 4 is bound to two run artifacts; full hashes and boundaries are in section 6.4.
Main matrix (10 routes × 4 benchmark groups, 28,404 calls): `benchmarks/results/matrix-20261002-112301.json` (sha256 `974c543d…`).
`deepseek-flash` re-run (1,110 calls): `benchmarks/results/matrix-20261003-005505.json` (sha256 `ef115979…`).
Early initial results that were not protocol-aligned have been moved to [Appendix A](#appendix-a-historical-artifacts-superseded-by-section-4), and the main text no longer cites them.
**Accuracy protocol**: throughout, all figures are **hard-label argmax accuracy** (consistent with the industry protocol), and the denominator is the **number of valid items after excluding connection failures** (the only exception is the A/B comparison experiment in section 4.4, whose table denominator includes failed items and is annotated as such inside the table).

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

1. **The approach holds, and under the same protocol it surpasses the industry's best.** Using hosted-API logprobs to replicate the Jev form, `doubao-2.1-pro` reaches **91.2%** on the JevBench public 231 and **82.4%** on the hard tier, above Jev 1.13.0 (86.58% / 72.97%), Open-Jev-27B (85.28% / 72.07%), and all open-weight models (the highest being NeoHorse-Jev-4B at 75.32% per-sample).
2. **The gap comes from the base model, not the mechanism.** With the same readout logic, the initial `doubao-2.0-mini` scored only 81.1%, while switching to `doubao-2.1-pro` raised it to 91.2% — not a single byte of the mechanism changed.
3. **The default route is `doubao-2.1-lite`**: JevBench 89.6% / Nimble 89.6% / Kev 81.5%, accuracy on par with the pro tier, at a p50 of only 870 ms (about half of `doubao-2.0-pro`). `doubao-2.0-mini` (497 ms) is the fastest, but its 81.1% is only enough for very lightweight scenarios.
4. **Kev is the only group that has not caught up**: our best is 81.8% (six subsets, equal-weighted, `doubao-2.0-pro`) vs Jev 1.13.0's 85.52, a gap of 3.7 points; the losses concentrate in the two transfer-v9 subsets (76~79%), while transfer-v4 (82~87%) is actually a strength.
5. **A stronger model ≠ a better fit for this mechanism.** All three DeepSeek routes are broadly below same-price doubao. `deepseek-v4.1-flash` is the highest on VitaminC fact verification (75.5%), but leads by only 0.2 points, and lags clearly on Nimble / Kev.
6. **Format compliance can be backstopped**: a uniform-strength `logit_bias` zeroes out the readout failures caused by "the model not emitting a label", without changing the within-candidate probability distribution; enabling it in production is recommended (see 4.4).
7. **confidence is currently untrustworthy**: no temperature calibration has been done (`temperature_scale=1.0`), so this report uses only argmax accuracy and does not report ECE / Brier. To ship probabilities, first gather labeled data and run temperature calibration.
8. **Two cells are blank — either unrunnable or irreproducible**: OpenJev text (harness and prompts not public); MASSIVE-en (18 candidate classes exceeds the cap of 10).

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

The semantics of position 0 are exactly "the next-token distribution at the answer position", equivalent to Jev's single forward pass. The cost is an extra dependency on model cooperation (the model must actually emit the label as its first token), with a **measured failure rate of 1.3%, dropping to 0 once `logit_bias` is enabled** (see 4.4).

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

Data source: the main matrix `benchmarks/results/matrix-20261002-112301.json` (28,404 calls), plus the `deepseek-flash`² re-run. Artifact hashes are in section 6.4.

### 4.1 Full matrix (main results)

| Route | JevBench 231 | Hard 111 | Nimble 280 | VitaminC 599 | Kev six-subset equal-weighted | p50¹ |
|---|---:|---:|---:|---:|---:|---:|
| **doubao-2.1-pro** | **91.2%** | **82.4%** | **94.2%** | 72.9% | — | 1174 ms |
| doubao-evolving | 91.1% | 82.1% | **94.2%** | 73.3% | — | 1062 ms |
| doubao-2.0-pro | 90.0% | 81.1% | 89.3% | 71.8% | **81.8%** | 1686 ms |
| doubao-2.0-lite | 90.0% | 80.2% | 86.4% | 73.1% | — | 1163 ms |
| **doubao-2.1-lite** | 89.6% | 79.3% | 89.6% | 72.0% | 81.5% | **870 ms** |
| doubao-2.1-turbo | 88.7% | 79.3% | 90.4% | 75.2% | — | 2078 ms |
| deepseek-v4.1-flash | 87.0% | 75.7% | 83.2% | **75.5%** | 80.7% | 859 ms |
| deepseek-flash² | 85.7% | 72.1% | 83.2% | 75.1% | — | 477 ms |
| doubao-2.0-mini | 81.1% | 65.1% | 72.5% | 75.3% | — | **497 ms** |
| deepseek-v4-pro | 78.3% | 64.0% | 79.6% | 67.3% | — | 1432 ms |
| deepseek-v4-flash | 73.5% | 53.1% | 60.4% | 63.4% | — | 919 ms |

¹ p50 is the median latency over JevBench 231. Sorted by JevBench descending, ties broken by hard descending. The Kev column was only run for all six subsets on 3 representative routes (about 5,768 items/route); that column is the **equal-weighted average of the six subsets**, consistent with the industry Kev protocol; under a per-item pooled protocol instead, `doubao-2.0-pro` would be 81.2%, `doubao-2.1-lite` 80.9%, and `deepseek-v4.1-flash` 80.3%.
² The `deepseek-flash` row is a full re-run on 2026-10-03 (all 231/280/599 items, no sampling; 1,110 calls, 0 failures); its artifact binding is in section 6.4.
Route names follow the config of each run: `deepseek-v4.1-flash` was the route name during the 10-02 run, `deepseek-flash` is the current alias.
The alias carries no version, so it cannot be proven to be the same build — the two rows are kept side by side rather than merged.
That row's latency comes from a different day's run and is not directly comparable with the others in this table.

**By question type (JevBench 231)**: `choice` peaks at 91.4%, `noul` at 93.0%, `score` at 88.9%; the Nimble `score` subset (54 items) peaks at 94.4% (`doubao-evolving`), with a median around 87%.

### 4.2 Valid samples and readout failures

Entries are `n_errors/n`; the accuracy denominator is `n - n_errors`. The failure rate of every cell is far below the 20% threshold.

| Route | JevBench | Hard | Nimble | VitaminC |
|---|---:|---:|---:|---:|
| doubao-2.1-pro | 3/231 | 3/111 | 2/280 | 2/599 |
| doubao-evolving | 5/231 | 5/111 | 3/280 | 0/599 |
| doubao-2.0-pro | 0/231 | 0/111 | 0/280 | 0/599 |
| doubao-2.0-lite | 0/231 | 0/111 | 0/280 | 0/599 |
| doubao-2.1-lite | 0/231 | 0/111 | 0/280 | 0/599 |
| doubao-2.1-turbo | 0/231 | 0/111 | 0/280 | 1/599 |
| deepseek-v4.1-flash | 0/231 | 0/111 | 0/280 | 0/599 |
| deepseek-flash² | 0/231 | 0/111 | 0/280 | 0/599 |
| doubao-2.0-mini | 3/231 | 2/111 | 0/280 | 0/599 |
| deepseek-v4-pro | 0/231 | 0/111 | 0/280 | 0/599 |
| deepseek-v4-flash | 5/231 | 0/111 | 0/280 | 0/599 |

² `deepseek-flash` — the run behind this row, and why it is not merged with `deepseek-v4.1-flash`, are described in the footnote to section 4.1.

Kev six-subset failure counts: `doubao-2.0-pro` 5/5,768, `doubao-2.1-lite` 3/5,765, `deepseek-v4.1-flash` 0/5,768.

### 4.3 Kev six-subset detail

| Route | decision-v7-dev | decision-v7-test | transfer-v4-dev | transfer-v4-test | transfer-v9-dev | transfer-v9-test | equal-weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| doubao-2.0-pro | 81.9% | 81.1% | 84.5% | 86.9% | 77.8% | 78.9% | **81.8%** |
| doubao-2.1-lite | 81.2% | 80.9% | 84.9% | 86.4% | 77.3% | 78.5% | 81.5% |
| deepseek-v4.1-flash | 83.2% | 81.8% | 81.9% | 83.4% | 76.3% | 77.7% | 80.7% |

We have not caught up with Jev 1.13.0's 85.52; the gap is concentrated in the two **transfer-v9** subsets (76~79%); **transfer-v4** is instead a strength (82~87%, above Kev-4B's 81.47). Jev 1.13.0 does not publish per-subset detail, so comparison is only possible on the six-subset equal-weighted figure.

### 4.4 logit_bias fallback experiment

The same batch of 231 items, the same model (`doubao-2.0-mini`), toggling only the `logit_bias` switch:

| Configuration | Overall | Hard | Readout failures |
|---|---:|---:|---:|
| bias off | 187/231 = 80.95% | 71/111 = 63.96% | **3 items** |
| bias on | 186/231 = 80.52% | 71/111 = 63.96% | **0 items** |

All 3 failures were "the model's first token was not a label" (it actually output "请" ("please") and "首先" ("first")). Enabling the uniform bias **zeroes out the format failures**.

**Protocol note (must read)**: this table comes from a single-benchmark artifact of `run_jevbench.py`, with a denominator of 231 (including failed items). It is **not the same run** as the 185 (denominator 228) for `doubao-2.0-mini` in 4.1 — the correct counts differ by 2 items between the two runs, attributable to MoE routing / batching nondeterminism under greedy decoding; the 231 vs 228 denominators reflect two different failure-counting protocols. This is the only place in this report where the same model on the same benchmark shows two sets of numbers; both are retained, each labeled with its source, and not merged.

Conclusion: `logit_bias` is an effective fallback for **format failures** (1.3% → 0), and the uniform-strength bias cancels itself out during within-candidate normalization, leaving the probability distribution unchanged, so it need not be undone.

---

## 5. Industry comparison

### 5.1 Six-benchmark comparison (including the table from the ModelScope `NeoHorse-Jev-4B` model card)

The NeoHorse-Jev-4B model card (TokenRhythm, 2026-09-24) provides a cross-comparison table of six benchmark groups that **partially overlaps** with the groups we ran. Merged, it is as follows (0–100, higher is better):

| Model | JevBench | Kev | OpenJev text | Nimble | VitaminC | MASSIVE | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| **this implementation · doubao-2.1-pro** | **91.2** | — | — | **94.2** | 72.9 | cannot run¹ | — |
| **this implementation · doubao-evolving** | 91.1 | — | — | **94.2** | 73.3 | cannot run¹ | — |
| **this implementation · doubao-2.0-pro** | 90.0 | 81.8 | — | 89.3 | 71.8 | cannot run¹ | 83.2⁴ |
| **this implementation · doubao-2.1-lite** | 89.6 | 81.5 | — | 89.6 | 72.0 | cannot run¹ | 83.2⁴ |
| **this implementation · deepseek-v4.1-flash** | 87.0 | 80.7 | — | 83.2 | **75.5** | cannot run¹ | 81.6⁴ |
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
| **this implementation · doubao-2.1-pro** | **91.2%** | **82.4%** | this measurement |
| **this implementation · doubao-2.0-pro** | 90.0% | 81.1% | this measurement |
| Jev 1.13.0 (TypeSafe hosted) | 86.58% | 72.97% | Open-Jev-27B-v1.1 model card |
| Open-Jev-27B-v1.1 | 85.28% | 72.07% | same as above |
| Open-Jev 9B | 77.49% | 59.46% | same as above |
| Open-Jev 2B | 64.94% | 41.44% | same as above |
| NeoHorse-Jev-4B | 75.32% | — | NeoHorse model card (per-sample) |

### 5.3 Protocol differences you must read before cross-comparison

1. **JevBench metric definition**: the NeoHorse table uses a task-family macro average (75.73), while we use per-sample accuracy. Their file also gives the per-sample 75.32, and **the comparable pair is 91.2% vs 75.32%**.
2. **Nimble subset**: already aligned. Both sides filter from 324 items by the compat282 rule (they annotate 282 items; we measure 280).
3. **VitaminC sampling**: already aligned. Both sides use the 599 items from the Nimble manifest (seed 20260918).
4. **Kev**: we use the six-subset clean equal-weighted average, consistent with the industry protocol; but Jev 1.13.0 does not publish per-subset detail, so item-by-item verification is impossible.
5. **Not prefill scoring**: we have an extra dependency on "the model cooperating to emit the label" (failure rate 1.3%, 0 once `logit_bias` is enabled), whereas the industry reads the distribution directly.
6. **No temperature calibration**: we compare only argmax, not probability mass (ECE / Brier are not involved), so **none of the probabilities in this report should be used as confidence**.

### 5.4 Attributing the gaps

| Observation | Attribution |
|---|---|
| JevBench / Nimble broadly surpass the industry's best | stronger base models, equivalent mechanism (see conclusion 2) |
| Kev lags by 3.7 points, with losses concentrated in transfer-v9 | that subset leans toward "cross-domain transfer" and needs stronger generalization; we can optionally run `doubao-2.1-pro` to verify |
| All three DeepSeek routes are broadly low | poor fit with this mechanism (single-token label readout), not a question of model capability |
| Open-weight models are generally 81~83% on Kev yet we are higher on JevBench | the two benchmark types test different transfer abilities; it is unwise to draw conclusions from a single benchmark |
| `deepseek-v4-flash` collapses to 48.2% on Nimble `score` | low-end routes are unstable on ordered scoring; high-end routes reach 87~94% on the same subset |

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

# 2) Full matrix (main results of section 4, about 28k calls)
python3 benchmarks/run_matrix.py                                  # 10 routes × 3 benchmark groups
python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite,deepseek-v4.1-flash
python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-<ts>.json   # resume

# 3) single benchmark / single route (used for the section 4.4 experiment)
python3 benchmarks/run_jevbench.py --tag nobias
python3 benchmarks/run_jevbench.py --tag withbias --logit-bias
python3 benchmarks/run_nimble.py   --model doubao-2.1-lite
python3 benchmarks/run_vitaminc.py --model doubao-2.1-lite

# 4) Refresh reproducibility info (re-run after every evaluation)
python3 benchmarks/make_provenance.py --run benchmarks/results/matrix-<ts>.json
```

### 6.2 Implementation fingerprint

| Item | Value |
|---|---|
| Implementation fingerprint (content hashes of the 24 source files under `src/llm2decision/` + `benchmarks/`) | `c3bab240c103cc4e…` |
| Engine | `POST /v1/systemone` (label-logit readout) |
| Decoding | `temperature=0` (greedy), `max_tokens=1`, **reading only generation-sequence position 0** |
| Candidate cap | 10 (exceeding it returns 422 and is excluded from evaluation) |
| Output interface | hard-label argmax; no LLM-as-judge |
| Calibration | no temperature calibration (`temperature_scale=1.0`), hence accuracy only, no ECE / Brier |
| Failure protocol | connection failures counted separately and the cell marked invalid when the failure rate is >20%; a model refusing to answer is recorded as an error |

The fingerprint above identifies this repository's current tree.
The published runs predate the package rename (`app/` → `src/llm2decision/`) and the comment-only English pass.
After those steps we verified equivalence: the v1 prompt templates and the rendered messages for all three question types are byte-identical, and re-running JevBench 231 with the current code (`doubao-2.1-lite`, `doubao-2.1-pro`) reproduces the old implementation's artifacts (±2 questions, the greedy-decoding non-determinism recorded in section 4.4).
So the numbers below remain attributable to this implementation; the spot-check artifacts are published in the [`eval-20261002`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261002) release.

### 6.3 Datasets and subset rules

| Benchmark | Source | Subset rule | Scoring protocol | Local data hash |
|---|---|---|---|---|
| JevBench | `fstandhartinger/jevbench` (MIT) | public 231 = original 72 + easy 48 + hard 111, full set with no sampling | per-item hard-label argmax | `5c2414edb3006b8b…` (consistent with the upstream manifest record ✅) |
| Nimble | `data/eval.jsonl` of `bespokelabsai/nimble` | compat282: state ≤384 tokens, request body ≤2048 tokens, candidates ≤26, whole families retained (measured **280** items) | exact per-sample match | `8e9e48b8de520659…` |
| VitaminC | HF `tals/vitaminc` validation (CC BY-SA 3.0) | nimble599: rows taken exactly by the 599 `unique_id`s in the Nimble manifest (matching 599/599) | 3-way choice argmax | `30bd72d23b3e58ae…` (manifest file) |
| Kev | the 6 subsets of `jaredpalmer/kev` (Apache-2.0) | full set, only `_meta.variant=="clean"`; items with >10 candidates skipped and counted | per-item argmax, equal-weighted average of the six | `8d5765d7aec4d08c…` (6 files) |

### 6.4 Run artifact bindings

| Benchmark | Actual item count | Item-set id list hash |
|---|---:|---|
| JevBench | 231 | `04399f09d6b39036…` |
| Nimble | 280 | `f0584784646c74f6…` |
| VitaminC | 599 | `2154d495d6b8d7bd…` |
| Kev (per-item) | 5,768 | `73fd023d385cb87b…` |

**Section 4 is bound to two run artifacts:**
- Main matrix: `benchmarks/results/matrix-20261002-112301.json` (sha256 `974c543dc6637fe1…`)
- `deepseek-flash` re-run row: `benchmarks/results/matrix-20261003-005505.json` (sha256 `ef115979aa40c49c…`)
- The section 4.4 experiment: `jevbench-20261002-074113-nobias.summary.json` and `jevbench-20261002-074204-withbias.summary.json` in the same directory

These artifacts are not committed to the repository (size); they are published in the [`eval-20261002`](https://github.com/yuyaxiong/LLM2Decision/releases/tag/eval-20261002) release, where each download can be checked against the sha256 values above.
**Hashes change with every run** — scores must be bound to a specific hash, and that is precisely why this section exists.

### 6.5 Known non-reproducible items (listed honestly)

| Item | Why it is not reproducible |
|---|---|
| OpenJev text (19 items) | The harness and prompt templates of that family are not public, and the two data sources `control`(805) and `chess`(500) cannot be found — so this report leaves that cell blank |
| `zhihz/openjev`'s 89.8% / 97.7% | Its evaluation data is not provided with the release package, so only its published values can be cited. Note that it is **isomorphic to this project's mechanism** (reading only the candidate-letter distribution of the next token, without generating text) and is an independent web application rather than model weights |
| JevBench scores of open-source models | The 27B / 9B weights require a GPU and the Open-Jev loader, which cannot be run on this machine; the table shows their published values |
| Kev / Nimble / VitaminC scores in the industry table | Come from each model card's published values, cannot be reproduced item by item, and can only be cited after aligning the subset rules |

---

## Appendix A: historical artifacts (superseded by section 4)

The following numbers come from **single-benchmark artifacts run separately at an earlier stage**, whose protocol is inconsistent with section 4 (Nimble used all 324 items, VitaminC used a 300-item sample, and denominators included failed items), and **they are no longer used as a basis for conclusions**. They are retained only to trace the difference "before and after switching base models + aligning protocols".

| Benchmark | Early result (`doubao-2.0-mini`) | Same-model protocol in section 4 | Source of difference |
|---|---|---|---|
| JevBench public 231 | 187/231 = 80.95%; hard 71/111 = 63.96% | 185/228 = 81.1%; hard 71/109 = 65.1% | denominator 231 (including failures) vs 228 (excluding failures); another 2 items from greedy-decoding nondeterminism |
| JevBench by tier | easy 48/48 = 100%, original 68/72 = 94.4%, hard 71/111 = 63.96% | — | same item batch, different denominator protocol |
| JevBench by question type | choice 112/139 = 80.6%, noul 61/74 = 82.4%, score 14/18 = 77.8% | see the footnote in 4.1 | same as above |
| Nimble built-in holdout | 221/324 = 68.21% (choice 61.6% / noul 84.2% / score 54.7%) | 203/280 = 72.5% | **different subsets**: the full 324 vs the 280 of compat282 |
| VitaminC | 231/300 = 77.00% (self-sampled 300, seed 20260918) | 451/599 = 75.3% | **different sampling**: self-sampled 300 vs the manifest's 599 |
| Latency | p50 ≈ 399 ms, p95 ≈ 952 ms (concurrency 6) | 497 ms | different concurrency and batching conditions |

**An old conclusion overturned by this section's data**: the early report once stated that "ordered scoring (score) is an industry-wide blind spot, at 54.69% for this implementation". That 54.69% came from the 64 score items under the Nimble **324-item** protocol in the table above; under the aligned **280-item** protocol (54 score items) the same model scores 62.96%, while the best route reaches **94.4%**. That conclusion therefore **does not hold**, and section 4 has been changed to state it per route (see the footnote in 4.1 and 5.4).

Early artifact files:

- `benchmarks/results/jevbench-20261002-074113-nobias.summary.json` (187/231)
- `benchmarks/results/jevbench-20261002-074204-withbias.summary.json` (186/231, still used in section 4.4)
- `benchmarks/results/nimble-20261002-074243-nobias.summary.json` (324 items)
- `benchmarks/results/vitaminc-20261002-074608-nobias.summary.json` (300 items)
