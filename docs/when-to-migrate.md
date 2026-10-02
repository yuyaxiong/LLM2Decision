[English](when-to-migrate.md) | [简体中文](when-to-migrate.zh-CN.md)

# When to Replace Hand-Written Prompts with a Decision Model

Convert "let the model generate a chunk of text, then parse it with string operations" into "let the model assign probabilities directly over your candidates, and let code take the argmax".

This document answers two questions: **which steps are worth converting**, and **how to convert them without breaking things**.

---

## 1. Criteria

A conversion is worth doing only when all three conditions hold **at the same time**:

| Condition | Description | Counterexample |
|---|---|---|
| 1. The output is a **finite set** | Candidates are enumerable, from a few to a dozen or so | Final answers, open-ended text rewriting |
| 2. The judgment **depends on the given input** rather than world knowledge | The answer can be derived from the material | Requires external facts, commonsense reasoning |
| 3. The downstream consumer is a **code branch**, not a human reader | The result feeds an if / switch / routing | Summaries, copy, text shown to users |

If 1.2.3. all hold → convert. If only one holds → keep generation, but you can use decisions as a **guard** (Section 4).

A common misjudgment: **"the model answers this correctly today, so it must be convertible" is wrong.** The criteria are about the **form** of the output, not the current accuracy. The converse also holds—a step that currently falls back on `strings.Contains(resp, "true")` and has mediocre accuracy can still be converted as long as it satisfies 1.2.3., and converting it usually improves both accuracy and stability.

## 2. Interface

Every decision step goes through the same interface—**do not build a separate API for each step**:

```jsonc
POST /v1/decide
{
  "model": "doubao-2.1-lite",        // optional; route hard steps to a stronger backend
  "state": "<text assembled from the raw material>",
  "questions": {
    "<step name>": { "type": "choice | noul | score", ... }
  },
  "debug": false
}
```

Conventions:

- **The step name is the question key**—lowercase with underscores, identical to the telemetry field name, so you can compare before and after.
- One request can carry multiple questions. Note: the model makes one independent call per question (concurrent inside the service), so **multiple questions ≠ a single forward pass**; the `calls` in the response tells you how many were actually issued.
- The response always carries `label` / `confidence` / `coverage` / `reliable` / `generated_token`.
- **The candidate cap is decided by the provider**, not a fixed 10: Volcengine Ark's 20 slots can reliably carry 10 candidates, DashScope has only 5 slots so its cap is 4, and the generic OpenAI-compatible tier conservatively takes 8. Exceeding it returns 422 immediately. **First run `benchmarks/probe_provider.py` to measure your vendor's real cap**—don't guess from the docs.
- Fixed on the model side, and **deliberately not exposed as a configuration option**: greedy decoding (temperature=0), `max_tokens=1`, and reading only the first token at the answer position.

## 3. Three Conversion Templates

### Type A: Finite candidates (classification / routing)

| Before | After |
|---|---|
| The prompt asks for a number or keyword → parse it into an integer → out of range falls back to a default | `choice`, candidates passed in as `criteria`, take `label` |

```json
{"intent": {"type": "choice", "instructions": "which capability should this user utterance go to",
  "criteria": {"billing": "billing, refund, and payment issues", "technical": "outages, errors, malfunctioning features",
               "account": "login, permissions, and account information", "other": "none of the above"}}}
```

**Threshold policy**: when `confidence < θ`, don't force the argmax—take the existing fallback branch, or route to a stronger backend and re-run. Calibrate θ from historical data.

### Type B: Yes / no judgments

| Before | After |
|---|---|
| The prompt asks for True/False → string matching | `noul`, take `true_probability`, compare it against a threshold |

```json
{"needs_retrieval": {"type": "noul", "instructions": "Is this user utterance asking to look up material / read a document?"}}
```

Whether to use `decision` (≥0.5) or a threshold (say ≥0.8) depends on how much the business tolerates false negatives versus false positives—**you must state which one you picked in the conversion notes**, otherwise future maintainers can't tell whether the behavior changed.

### Type C: Ordinal scoring

| Before | After |
|---|---|
| Relevance scored by rules (e.g. BM25) | `score` grades each candidate span on its own, then sort by expected value |

```json
{"doc_relevance": {"type": "score", "instructions": "How relevant this material is to the user's question",
  "scale": ["0","1","2","3","4","5"], "values": [0,1,2,3,4,5]}}
```

⚠️ **`score` is the least accurate of the types in this mechanism**—both our measurements and industry data show it's the weak spot. It's fine for ranking, **don't use it as a hard gate**.

### Type D: Must generate

Final answers, copy, open-ended rewriting, summaries and the like are still produced by a generation model. But they don't have to be left untouched—see Sections 5 and 6.

## 4. Decision Guards on Generation Steps

A generation step can't be converted into a decision model, but you can insert decision guards at critical points to block hallucinations at the exit:

| Guard | Question type | Description |
|---|---|---|
| Citation check | `choice` | Can each factual point in the answer be grounded in the retrieved material (SUPPORTS / REFUTES / NOT_ENOUGH) |
| Rewrite protection | `noul` | Did the rewritten query introduce entities, model numbers, or codes that appear in neither the original sentence nor the material |
| Answer compliance | `noul` | Did the answer cross a policy or scope boundary |
| Should it ask back | `noul` | Should it currently ask a clarifying question instead of answering |

Guard failure → **refuse, regenerate, or downgrade the copy**, rather than letting the model "check itself again".

## 5. Extractive-First: Split "Generation" into "Selection + Assembly"

Section 4 said generation steps don't change. But their goals (faster, no quality drop, no hallucination) are still within reach if you switch methodology:

> **Extractive-first**: split "generate a new chunk of text" into "**select content + assemble**".
> As long as every word in the output has a source—an original span, a closed vocabulary, a template slot—**hallucination is impossible**.
> Decisions handle "what to pick"; code handles "how to stitch it together". Only when the material genuinely has nothing usable do you fall back to generation, and that must pass the guards.

### 5.1 Rewriting

| Step | Approach | Question type |
|---|---|---|
| 1. Intent classification | Decide which kind of rewrite this sentence is | `choice` |
| 2. Span segmentation | Code tokenizes (a dictionary or an off-the-shelf NER) | Code |
| 3. Per-span keep/drop | Decide "keep / drop" for each span | `noul` |
| 4. Synonym substitution | Replacements **may only be picked from a domain dictionary**; inventing words is not allowed | `choice` |
| 5. Assembly | Code stitches them together by template | Code |
| 6. Fallback rewrite | Generate only when "nothing in the dictionary matches and a rewrite is mandatory" | Generation |
| 7. Guard | Does the result introduce entities absent from both the original sentence and the material | `noul` |

Benefit: rewriting drops from 1–3 s / several hundred tokens to about 0.4 s / 1 token, and entities can't appear out of thin air.

### 5.2 Structured extraction

Don't leave field values as free text—that's the same as leaving hallucination inside the structure:

| Field type | Approach |
|---|---|
| Category | `choice` |
| Entities, codes | If a dictionary exists → match in code; only let the model adjudicate with `noul` when it's ambiguous |
| Hierarchical values (major / minor category) | A two-level dictionary: first `choice` picks the major category, then `choice` picks the specific item |
| Fusion (extraction result + original text → search query) | **No generation needed**; just assemble by a fixed template |

### 5.3 Summarization

First decide which kind you need:

- **Extractive summarization (preferred)**: run `score` on every sentence of the material (should this sentence go into the summary, on a 0–5 scale), and once it clears the threshold, **copy the original text verbatim** and concatenate. Zero hallucination (every sentence is source text), and judging N sentences concurrently is faster than generating a whole passage. Applies to: key-point digests, multi-document summaries.
- **Generative summarization (when wording must be organized)**: keep generation + three gates
  1. **Before generation**: feed only whitelisted material, and state in the prompt "you may only use the material below";
  2. **After generation**: check each sentence with `noul`—"can this sentence be grounded in the material?"—and delete or rewrite any sentence that fails;
  3. **Exit**: string-validate numbers, model numbers, and codes (see 5.4).

### 5.4 The anti-hallucination trio (applies to any scenario that must generate)

| # | Measure | Cost |
|---|---|---|
| 1. | **Constrain the input before generation**: whitelisted material + no external knowledge | 0 |
| 2. | **Verify sentence by sentence after generation**: each sentence judged by `noul`—"can it be grounded in the material?" | One decision call, can be async, doesn't block the first byte |
| 3. | **String validation at the exit**: numbers / model numbers / codes must hit in the material, otherwise reject | **0 cost, 0 latency** |

Do 3. first: it blocks the vast majority of content that "looks plausible but is actually fabricated", and it's done entirely in code.

### 5.5 Speedups that don't change quality

| Measure | Description |
|---|---|
| Prefix caching | Reuse the system prompt and the fixed material prefix; saves repeated prefill on repeated calls over long material |
| Structured output instead of long text | One token for a decision vs hundreds of tokens for generation |
| Tiered routing | Route simple samples to the small tier and hard samples to the strong tier (see Section 7) |
| Streaming | Keep generation tasks streaming; perceived latency is first-token time |
| Result reuse | Cache rewrite / summary results for the same material + similar inputs |

## 6. Prompt Template Standard

**The key difference**: a generative prompt teaches the model "how to answer", while a decision prompt does only two things—lay the candidates out clearly and pin the output to the first character.

```
[Input]
<raw material, pasted in verbatim; do not summarize here>

[Task]
<one sentence stating what to judge>

[Candidates]
1. billing: payment, refund, and billing related
2. technical: outages, errors, malfunctioning features

[Output requirement]
Output only the number of the chosen candidate (e.g. 1); output nothing else.
Answer number (output exactly one character, nothing else):
```

Four hard requirements:

1. **Write clear candidate descriptions.** The model tells candidates apart solely by their descriptions, and vague descriptions cause **systematic misclassification**—this is the most common failure cause after conversion, and it's not a model-capability problem.
2. **Put static content in the system prompt** (role constraints + few-shot) and put changing material in the user message—that's how you hit the prefix cache.
3. **Never allow "explain first, then answer".** This mechanism reads only the first token; any prefix will break the judgment. There are no exceptions to this rule.
4. **Split into multiple questions when candidates exceed the cap**, for example judge the major category first and then the minor category. Splitting is far cheaper than introducing a second readout path.

## 7. Model Selection and Routing

Based on a measured matrix over 10 backends (see [`benchmarks/REPORT.md`](../benchmarks/REPORT.md)):

| Use | Basis |
|---|---|
| Default (lots of lightweight judgments) | Value tier: accuracy on par with the strong tier, at about half the latency |
| Hard samples / require multi-hop and numerical reasoning | Strong tier. In our measurements, the hard tier is where the backends diverge |
| Fact-verification type (SUPPORTS/REFUTES) | Not necessarily the same as the "strong tier"—**choose by task family, not by model size** |

Two lessons:

- **Don't default to the strongest tier.** The bottleneck is often not reasoning difficulty but something else (structured-output stability, recognition ability at a given tier). Do controlled experiments for model selection, not intuition.
- **A stronger model ≠ a better fit for this mechanism.** Within the same vendor, the large tier isn't necessarily better than the small tier, because the distribution of noise tokens differs—in our measurements, some vendors' newer flagships mix more noise into each slot, giving them a smaller candidate capacity instead.

## 8. Acceptance Criteria

Every step you convert must clear these five:

| Dimension | Metric | Pass line |
|---|---|---|
| Effectiveness | Agreement rate with the existing generative output on **the same batch of real samples** | Downstream is a branch decision: ≥98%; threshold-based: misjudgment rate no worse than status quo |
| Hallucination | Share of outputs falling inside the candidate set | **100%** (otherwise it's an error and excluded from the stats—this is the essential difference between this mechanism and generation) |
| Latency | Per-step P50 / P95 | P95 at least 50% lower than "generation + string parsing" |
| Stability | Rejection rate (share of cases where the model didn't answer in the expected format) | ≤2%; if above, enable `logit_bias` as a fallback or change the prompt |
| Cost | Tokens per step | No higher than status quo |

**Sample requirement**: sampling must **include the hard samples that currently take the fallback branch**—those are the ones that determine whether the swap is viable. Sampling only easy samples yields an inflated pass rate; this is the easiest place to fool yourself.

## 9. Checklist

- [ ] Is the candidate set closed, and within this provider's candidate cap? If not, split into multiple questions
- [ ] Do the candidate descriptions let the model tell them apart? Self-test with 5 real borderline samples
- [ ] Has the prompt been changed to the "output exactly one character" structure?
- [ ] Is the threshold θ calibrated from historical data? Is it stated whether you take `decision` or `true_probability ≥ θ`?
- [ ] Fallback policy: which branch does it take when the model doesn't answer in the expected format?
- [ ] Routing choice: is the small tier enough for this step? Does it need routing to the strong tier?
- [ ] Does the sample set include hard samples? Is the agreement rate up to standard?
- [ ] Are the telemetry fields aligned with existing metrics, so you can compare before and after?
- [ ] Rollout: shadow-run and record first without affecting production, then shift traffic

## 10. Boundaries: What Not to Try to Convert

Scenarios that require **creating language**—copy generation, brand writing, multi-turn tone, complex technical explanations—are generation by nature. Don't convert them into decision models; accept that they may hallucinate, and merely use the three gates in 5.4 to block the **facts** inside them.

Rule of thumb: **the downstream is "for humans to read" (condition 3. doesn't hold), and the content can't be extracted from the input → keep generation.**
