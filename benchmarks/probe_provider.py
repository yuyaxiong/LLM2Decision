"""Probe: verify whether **any** OpenAI-compatible endpoint can support this project's readout mechanism.

The mechanism relies on "top_logprobs at position 0", and hosted platforms differ a lot here:
    - Different top_logprobs caps (Ark 20, DashScope 5, OpenAI/OpenRouter 20)
    - Some models enable thinking mode by default; the reasoning chain displaces position 0, and the parameter to disable thinking differs per vendor
    - logprobs is often gated by a model allowlist and the docs are not always accurate (measured: qwen3.5~3.8 support it, while qwen3-32b, which the docs list as supported, does not)
So before switching vendors, run this probe to answer the four questions rather than guessing from the docs.

This script does not modify any code under src/llm2decision/; it only verifies four things:
    Assumption 1  Are logprobs / top_logprobs accepted (rejected with 400, or silently returned as null?)
    Assumption 2  Can thinking be disabled so that position 0 lands on the answer position rather than the reasoning chain
    Assumption 3  With top_logprobs=5, how many actual business candidates fit into the 5 slots (including EOS / full-width / punctuation noise)
    Assumption 4  On a model that works, what is the accuracy of strategy A (compared against Ark on the same question set)

The readout logic **reuses the project's real implementation** (build_messages / question_items /
read_distribution / _build_decision) from src/llm2decision/, so that this measures the real mechanism
rather than a separate approximation.

Usage:
    # 1) First configure a route in llm2decision.yaml whose base_url matches; the probe reuses its key automatically
    python3 benchmarks/probe_provider.py --models qwen3.7-plus --limit 60
    # 2) Or pass the key and endpoint directly
    export LLM2DECISION_PROBE_KEY=sk-xxx
    python3 benchmarks/probe_provider.py --base-url https://openrouter.ai/api/v1 \
        --thinking-param none --models openai/gpt-4o-mini --limit 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

import httpx  # noqa: E402

from llm2decision.core.labels import build_lookup, normalize_token  # noqa: E402
from llm2decision.core.readout import ReadoutError, extract_content, read_distribution, softmax  # noqa: E402
from llm2decision.core.schema import Question  # noqa: E402
from llm2decision.llm.service import _build_decision, question_items  # noqa: E402
from llm2decision.prompts import build_messages  # noqa: E402

import run_jevbench as jev  # noqa: E402
import run_matrix as matrix  # noqa: E402

RESULT_DIR = BENCH_DIR / "results"

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODELS = ["qwen-plus-2025-04-28", "qwen3-32b"]
DEFAULT_LIMIT = 20
# top_logprobs cap: DashScope hard-caps at 5; other OpenAI-compatible endpoints use 20 (Ark / OpenAI / OpenRouter are all 20).
# Override explicitly with --max-top-logprobs; pinning this value matters because you must not test a 20-capable endpoint with 5.
DEFAULT_MAX_TOP_LOGPROBS = 20
DASHSCOPE_MAX_TOP_LOGPROBS = 5


def default_top_logprobs(base_url: str) -> int:
    return DASHSCOPE_MAX_TOP_LOGPROBS if "dashscope" in base_url else DEFAULT_MAX_TOP_LOGPROBS

# Three ways to disable thinking, tried one by one to find which works for this model (an empty dict value = attach no parameter)
THINKING_MODES: Dict[str, dict] = {
    "enable_thinking=false": {"enable_thinking": False},
    "thinking=disabled": {"thinking": {"type": "disabled"}},
    "omitted (model default)": {},
}

# Default disable-thinking parameter for cross-vendor comparison: DashScope uses enable_thinking, Ark uses thinking
THINKING_PARAMS: Dict[str, dict] = {
    "enable_thinking": {"enable_thinking": False},
    "thinking": {"thinking": {"type": "disabled"}},
    "none": {},
}

# Explicitly means "use the client-configured disable-thinking parameter", distinct from "attach no parameters"
USE_CLIENT_THINKING = object()


class DashScopeError(RuntimeError):
    pass


class OpenAICompatClient:
    """OpenAI-compatible client (DashScope / Ark / any compatible endpoint).

    Serves the probe only; not part of src/llm2decision/. The default disable-thinking parameter is determined per vendor by the constructor argument.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        thinking: Optional[dict] = None,
        timeout: float = 60.0,
        max_top_logprobs: Optional[int] = None,
    ) -> None:
        self._thinking = {"enable_thinking": False} if thinking is None else dict(thinking)
        self._max_top_logprobs = max_top_logprobs or DEFAULT_MAX_TOP_LOGPROBS
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_completions(
        self,
        model: str,
        messages: Sequence[dict],
        top_logprobs: Optional[int] = None,
        extra: object = USE_CLIENT_THINKING,
        logprobs: bool = True,
    ) -> dict:
        """Return a dict: body on success, status/error on failure. Never raises; the probe records the result."""
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = (
                self._max_top_logprobs if top_logprobs is None else top_logprobs
            )
        # Disable-thinking parameters are not OpenAI-standard, so put them at the top level
        merged = dict(self._thinking) if extra is USE_CLIENT_THINKING else dict(extra or {})
        if merged:
            payload.update(merged)

        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            return {"status": 0, "error": f"{type(exc).__name__}: {exc}"}
        if response.status_code >= 400:
            return {"status": response.status_code, "error": response.text[:400]}
        try:
            return {"status": response.status_code, "body": response.json()}
        except ValueError:
            return {"status": response.status_code, "error": "response is not valid JSON"}


def inspect_response(result: dict) -> dict:
    """Break one call down into the fields the probe cares about: whether logprobs is present, the token at position 0, and how many candidates were returned."""
    if "body" not in result:
        return {"ok": False, "status": result.get("status"), "reason": result.get("error")}
    body = result["body"]
    choices = body.get("choices") or []
    if not choices:
        return {"ok": False, "status": result["status"], "reason": "response has no choices"}

    choice = choices[0]
    logprobs = choice.get("logprobs")
    if logprobs is None:
        return {
            "ok": False,
            "status": result["status"],
            "reason": "logprobs returned null: this model does not support it or it was silently ignored",
            "content": (choice.get("message") or {}).get("content"),
            "finish_reason": choice.get("finish_reason"),
        }
    content = logprobs.get("content") or []
    if not content:
        return {"ok": False, "status": result["status"], "reason": "logprobs.content is empty"}
    entry = content[0]
    top = entry.get("top_logprobs") or []
    return {
        "ok": True,
        "status": result["status"],
        "generated_token": entry.get("token"),
        "generated_logprob": entry.get("logprob"),
        "top_size": len(top),
        "top_tokens": [item.get("token") for item in top],
        "content": (choice.get("message") or {}).get("content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


def choice_tasks(limit: int) -> List[dict]:
    """Take JevBench choice questions and **stratify by tier** when sampling.

    Used for assumption 4's accuracy: only tier stratification makes the results comparable with other benchmarks.
    """
    tasks = [task for task in jev.load_tasks() if task["question"]["type"] == "choice"]
    return matrix._stratified(tasks, limit)


def slot_probe_tasks(per_bucket: int = 4) -> List[dict]:
    """Sample bucketed by **candidate count** so the slot-capacity test has a gradient.

    JevBench choice questions have 3/4/5/6 candidates (tier-stratified sampling would draw only one kind,
    failing to test "how many candidates fit into 5 slots"). Here we take a few questions per candidate count,
    balancing by tier within each bucket.
    """
    tasks = [task for task in jev.load_tasks() if task["question"]["type"] == "choice"]
    buckets: Dict[int, List[dict]] = {}
    for task in tasks:
        buckets.setdefault(len(candidates_of(task)), []).append(task)
    picked: List[dict] = []
    for count in sorted(buckets):
        ordered = sorted(buckets[count], key=lambda item: item["_tier"])
        stride = max(1, len(ordered) // per_bucket)
        picked.extend(ordered[::stride][:per_bucket])
    return picked


def candidates_of(task: dict) -> List[str]:
    return list((task["question"].get("criteria") or {}).keys())


async def probe_thinking_mode(
    client: OpenAICompatClient, model: str, task: dict
) -> Dict[str, dict]:
    """Assumption 2: try each thinking mode once and see where position 0 lands and whether it reads out correctly."""
    payload = jev.to_payload(task)
    question = Question(**payload["questions"]["q"])
    items = question_items(question)
    handles = [handle for handle, _, _ in items]
    messages = build_messages(payload["state"], question, items, version="v1")

    results: Dict[str, dict] = {}
    for label, extra in THINKING_MODES.items():
        result = await client.chat_completions(model, messages, extra=extra)
        info = inspect_response(result)
        if info.get("ok"):
            # Whether the token at position 0 hits a candidate handle -- a necessary condition for the mechanism to hold
            lookup = build_lookup(handles)
            token = info.get("generated_token") or ""
            info["position0_is_handle"] = normalize_token(token) in lookup
        results[label] = info
    return results


async def probe_slot_capacity(
    client: OpenAICompatClient, model: str, tasks: Sequence[dict]
) -> dict:
    """Assumption 3: how many business candidates fit within the top_logprobs cap.

    Per question, record: total candidates, actual hits, coverage, and number of top entries returned.
    """
    rows: List[dict] = []
    for task in tasks:
        payload = jev.to_payload(task)
        question = Question(**payload["questions"]["q"])
        items = question_items(question)
        handles = [handle for handle, _, _ in items]
        labels = candidates_of(task)
        lookup = build_lookup(handles)
        messages = build_messages(payload["state"], question, items, version="v1")
        result = await client.chat_completions(model, messages)
        info = inspect_response(result)

        row = {
            "id": task["id"],
            "tier": task["_tier"],
            "n_candidates": len(labels),
            "read_ok": bool(info.get("ok")),
            "reason": info.get("reason"),
            "top_size": info.get("top_size"),
            "generated_token": info.get("generated_token"),
        }
        if info.get("ok"):
            try:
                content = extract_content(result["body"])
                readout = read_distribution(content, handles, lookup)
                row["hit"] = int(round(readout.coverage * len(handles)))
                row["coverage"] = round(readout.coverage, 4)
                row["reliable"] = readout.reliable
                row["top_tokens"] = info.get("top_tokens")
            except ReadoutError as exc:
                row["read_ok"] = False
                row["reason"] = str(exc)
        rows.append(row)

    readable = [row for row in rows if row["read_ok"]]
    by_candidates: Dict[int, List[dict]] = {}
    for row in rows:
        by_candidates.setdefault(row["n_candidates"], []).append(row)
    return {
        "n": len(rows),
        "n_readable": len(readable),
        "top_size_values": sorted({row["top_size"] for row in rows if row.get("top_size")}),
        "by_candidate_count": {
            str(count): {
                "n": len(group),
                "readable": sum(1 for row in group if row["read_ok"]),
                "avg_hit": round(
                    statistics.mean([row["hit"] for row in group if row["read_ok"]]), 2
                ) if any(row["read_ok"] for row in group) else None,
                "full_coverage": sum(1 for row in group if row.get("coverage") == 1.0),
            }
            for count, group in sorted(by_candidates.items())
        },
        "rows": rows,
    }


async def probe_accuracy(
    client: OpenAICompatClient, model: str, tasks: Sequence[dict]
) -> dict:
    """Assumption 4: run the full strategy A end to end (build_messages -> readout -> _build_decision) and report accuracy."""
    records: List[dict] = []
    for task in tasks:
        payload = jev.to_payload(task)
        question = Question(**payload["questions"]["q"])
        items = question_items(question)
        handles = [handle for handle, _, _ in items]
        lookup = build_lookup(handles)
        messages = build_messages(payload["state"], question, items, version="v1")
        started = time.perf_counter()
        result = await client.chat_completions(model, messages)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        record = {"id": task["id"], "tier": task["_tier"], "expected": task["expected"],
                  "latency_ms": latency_ms, "error": None}
        if "body" not in result:
            record["error"] = f"HTTP {result.get('status')}: {result.get('error')}"
        else:
            try:
                content = extract_content(result["body"])
                readout = read_distribution(content, handles, lookup)
                probabilities = softmax(readout.logprobs, 1.0)
                decision = _build_decision(question, items, probabilities, readout, debug=True)
                predicted = jev.predicted_label(task, decision)
                record.update({
                    "predicted": predicted,
                    "correct": jev.is_correct(task["expected"], predicted),
                    "coverage": round(readout.coverage, 4),
                    "generated_token": readout.generated_token,
                })
            except ReadoutError as exc:
                record["error"] = f"ReadoutError: {exc}"
        records.append(record)

    valid = [r for r in records if not r["error"]]
    correct = sum(1 for r in valid if r["correct"])
    latencies = sorted(r["latency_ms"] for r in valid) or [0.0]
    return {
        "n": len(records),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "correct": correct,
        "accuracy": round(correct / len(valid), 4) if valid else None,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "error_samples": sorted({r["error"] for r in records if r["error"]})[:4],
        "records": records,
    }


async def run_model(
    client: OpenAICompatClient, model: str, tasks: Sequence[dict], logprobs_probe: bool
) -> dict:
    print(f"\n{'=' * 62}\nModel: {model}\n{'=' * 62}")
    report: Dict[str, object] = {"model": model}

    # ---- Assumption 1: are logprobs / top_logprobs accepted ----
    if logprobs_probe:
        payload = jev.to_payload(tasks[0])
        question = Question(**payload["questions"]["q"])
        items = question_items(question)
        messages = build_messages(payload["state"], question, items, version="v1")
        result = await client.chat_completions(model, messages)
        info = inspect_response(result)
        report["assumption1_logprobs"] = info

        # "Preliminary errors" like auth / an unknown model name must be distinguished from "model does not support logprobs",
        # otherwise a 401 gets misread as "this model does not support logprobs" and yields a wrong experimental conclusion.
        status = info.get("status")
        if status in (401, 403):
            report["fatal"] = "auth"
            print(f"\n[Assumption 1] ✗ Authentication failed (HTTP {status}): invalid API key, region mismatch, or service not enabled."
                  f"\n  {str(info.get('reason'))[:200]}")
            print("  → This is a preliminary error, not a model-capability issue; no further models will be tried.")
            return report
        if status == 404:
            report["fatal"] = "model-not-found"
            print(f"\n[Assumption 1] ✗ Model not found (HTTP 404): {model!r} may be misspelled or unavailable in this region."
                  f"\n  {str(info.get('reason'))[:200]}")
            return report

        verdict = "pass" if info.get("ok") else "fail"
        print(f"\n[Assumption 1] logprobs available: {verdict}")
        if info.get("ok"):
            print(f"  HTTP {info['status']} | position 0 token={info['generated_token']!r} "
                  f"| top_logprobs returned {info['top_size']} entries")
            print(f"  top tokens: {info['top_tokens']}")
        else:
            print(f"  HTTP {info.get('status')} | {info.get('reason')}")
            print(f"  Model output (first 120 chars): {str(info.get('content'))[:120]!r}")
            print("\n  → Assumption 1 failed; later assumptions need no checking, skipping this model.")
            return report

    # ---- Assumption 2: which parameter disables thinking ----
    thinking = await probe_thinking_mode(client, model, tasks[0])
    report["assumption2_thinking"] = thinking
    print(f"\n[Assumption 2] Ways to disable thinking (does position 0 hit a candidate handle?):")
    for label, info in thinking.items():
        if info.get("ok"):
            mark = "✓" if info.get("position0_is_handle") else "✗"
            print(f"  {mark} {label:<22} token={info['generated_token']!r:<10} "
                  f"top={info['top_size']} finish={info.get('finish_reason')}")
        else:
            print(f"  ✗ {label:<22} {info.get('reason')}")

    # ---- Assumption 3: how many candidates fit into 5 slots (bucket by candidate count for a gradient) ----
    slot_tasks = slot_probe_tasks()
    slots = await probe_slot_capacity(client, model, slot_tasks)
    report["assumption3_slots"] = slots
    print(f"\n[Assumption 3] slot capacity at top_logprobs={client._max_top_logprobs}"
          f" (bucketed by candidate count, {len(slot_tasks)} questions):")
    print(f"  Readable {slots['n_readable']}/{slots['n']} questions; actual top entries returned: {slots['top_size_values']}")
    print(f"  {'Cands':<8}{'N':<6}{'Readable':<8}{'Avg hit':<10}{'Full cov':<8}")
    for count, block in slots["by_candidate_count"].items():
        print(f"  {count:<8}{block['n']:<6}{block['readable']:<8}"
              f"{str(block['avg_hit']):<10}{block['full_coverage']:<8}")

    # ---- Assumption 4: accuracy ----
    accuracy = await probe_accuracy(client, model, tasks)
    report["assumption4_accuracy"] = accuracy
    acc = accuracy["accuracy"]
    print(f"\n[Assumption 4] Strategy A accuracy (this endpoint + this model):")
    print(f"  {accuracy['correct']}/{accuracy['n_valid']} = "
          f"{'—' if acc is None else f'{acc:.1%}'}"
          f"  errors {accuracy['n_errors']}  p50 {accuracy['latency_p50_ms']}ms")
    for error in accuracy["error_samples"]:
        print(f"  ⚠ {error}")
    return report


def api_key_from_config(base_url: str) -> Optional[str]:
    """When --api-key is not passed explicitly, find a route in llm2decision.yaml with a matching base_url and reuse its key.

    This keeps configuration in llm2decision.yaml only, with no extra .env file; the probe can still override with --api-key.
    """
    try:
        from llm2decision.core.config import Config

        config = Config.load()
    except Exception:
        return None
    wanted = base_url.rstrip("/")
    for settings in config.models.values():
        if settings.base_url.rstrip("/") == wanted and settings.api_key:
            return settings.api_key
    return None


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify whether any OpenAI-compatible endpoint can support this project's readout mechanism")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names (real model IDs, not route names in llm2decision.yaml)")
    parser.add_argument("--base-url", default=os.getenv("LLM2DECISION_PROBE_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("LLM2DECISION_PROBE_KEY"),
                        help="When omitted, look up the key of the matching route in llm2decision.yaml by base_url")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of questions for stratified sampling")
    parser.add_argument("--thinking-param", choices=sorted(THINKING_PARAMS), default="enable_thinking",
                        help="Disable-thinking parameter: DashScope uses enable_thinking, Ark uses thinking, none means attach nothing")
    parser.add_argument("--max-top-logprobs", type=int, default=None,
                        help=f"The endpoint's top_logprobs cap; when omitted it is inferred from base_url "
                             f"(use {DASHSCOPE_MAX_TOP_LOGPROBS} when it contains dashscope, otherwise {DEFAULT_MAX_TOP_LOGPROBS})")
    parser.add_argument("--skip-logprobs-probe", action="store_true",
                        help="Skip assumption 1 (use when the model is known to support logprobs)")
    args = parser.parse_args()

    api_key = args.api_key or api_key_from_config(args.base_url)
    if not api_key:
        print("Missing API key. Provide it in one of three ways:\n"
              "  1) Configure a route in llm2decision.yaml with a matching base_url (the probe reuses it automatically)\n"
              "  2) export LLM2DECISION_PROBE_KEY=sk-xxx\n"
              "  3) python3 benchmarks/probe_provider.py --api-key sk-xxx")
        return 2
    args.api_key = api_key

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = choice_tasks(args.limit)
    max_top = args.max_top_logprobs or default_top_logprobs(args.base_url)
    print(f"OpenAI-compatible endpoint probe | base_url={args.base_url} | top_logprobs cap={max_top}")
    print(f"Disable-thinking parameter: {args.thinking_param} -> {THINKING_PARAMS[args.thinking_param] or '(none attached)'}")
    print(f"Models: {', '.join(models)}")
    print(f"Question set: JevBench choice stratified sample, {len(tasks)} questions"
          f" (tiers={sorted({t['_tier'] for t in tasks})})")

    client = OpenAICompatClient(
        args.base_url, args.api_key, thinking=THINKING_PARAMS[args.thinking_param],
        max_top_logprobs=max_top,
    )
    report: Dict[str, object] = {
        "base_url": args.base_url,
        "limit": args.limit,
        "thinking_param": args.thinking_param,
        "models": {},
    }
    try:
        for model in models:
            try:
                block = await run_model(
                    client, model, tasks, logprobs_probe=not args.skip_logprobs_probe
                )
                report["models"][model] = block
            except Exception as exc:  # a single model failing does not affect the others
                print(f"\nModel {model} probe raised: {type(exc).__name__}: {exc}")
                report["models"][model] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            # Auth is a global problem; switching models won't help, so stop
            if block.get("fatal") == "auth":
                print("\nAuthentication failed; stopping remaining models.")
                break
    finally:
        await client.aclose()

    print(f"\n{'=' * 62}\nSummary\n{'=' * 62}")
    for model, block in report["models"].items():
        if "error" in block:
            print(f"  {model:<24} probe raised")
            continue
        if block.get("fatal") == "auth":
            print(f"  {model:<24} ✗ authentication failed (preliminary error, not a model-capability issue)")
            continue
        if block.get("fatal") == "model-not-found":
            print(f"  {model:<24} ✗ model name not found")
            continue
        a1 = block.get("assumption1_logprobs") or {}
        if not a1.get("ok"):
            print(f"  {model:<24} ✗ logprobs unavailable → this model cannot support the mechanism")
            continue
        acc = (block.get("assumption4_accuracy") or {}).get("accuracy")
        slots = block.get("assumption3_slots") or {}
        print(f"  {model:<24} ✓ usable | accuracy {'—' if acc is None else f'{acc:.1%}'}"
              f" | top entries {slots.get('top_size_values')}")

    _save(report)
    return 0


def _save(report: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"probe-provider-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArtifact: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
