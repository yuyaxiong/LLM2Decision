"""Run the Intern-Decision comparison suites against this service's /v1/systemone.

Suites (matching https://github.com/InternLM/Intern-Decision `benchmarks/accuracy-v1/`):
    jevbench         public 231 (72 original + 48 easy + 111 hard), loaded from the upstream
                     repository this project already uses — item ids are identical to the bundle's
    typed_decisions  400 records, 5 questions each (choice / noul / score), 2,000 decisions
    toolace          310 records, tool selection (2-8 candidates)
    agnews           7,600 records, 4-way topic classification
    wildjailbreak    2,210 records, harmful/benign classification
    pilot            the 96-case known-distribution bundle (exact reference distributions shipped),
                     scored with expected Brier / expected ECE

Every request carries a whole record, so a typed_decisions row costs one call and yields five
decisions — the upstream "decisions" counting rule. Distributions are stored for every decision,
which is what makes the uncalibrated Brier / ECE columns computable.

Usage:
    python3 benchmarks/run_intern_suite.py --routes doubao-evolving --limit 20   # smoke run
    python3 benchmarks/run_intern_suite.py --routes doubao-evolving,doubao-2.1-pro,doubao-2.1-lite,deepseek-flash

Results go to benchmarks/results/intern-<timestamp>-<tag>.json, flushed after every suite.
Both upstream bundles are evaluation-only: they must not be used to fit a temperature.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

from llm2decision.llm.client import ChatClient  # noqa: E402
from llm2decision.core.config import Config  # noqa: E402
from llm2decision.core.schema import SystemOneRequest  # noqa: E402
from llm2decision.llm.service import SystemOneService  # noqa: E402

import intern_decision as itd  # noqa: E402
import run_jevbench as jev  # noqa: E402

RESULT_DIR = BENCH_DIR / "results"
SUITES = ("jevbench", "typed_decisions", "toolace", "agnews", "wildjailbreak", "pilot")

# Same retry policy as run_matrix: connection-layer failures only, 1s/3s/9s backoff
RETRY_DELAYS = (1.0, 3.0, 9.0)
CONNECTION_ERROR_MARKERS = ("connection", "all connection attempts failed", "timeout", "timed out")


def _is_connection_error(message: Optional[str]) -> bool:
    return bool(message) and any(marker in message.lower() for marker in CONNECTION_ERROR_MARKERS)


def build_tasks(suite: str, limit: int) -> List[dict]:
    """Build one task per record: one request carrying all of that record's questions."""
    tasks: List[dict] = []

    if suite == "jevbench":
        items = jev.load_tasks()
        if limit:
            items = items[:limit]
        for task in items:
            tasks.append({
                "bench": "jevbench",
                "id": task["id"],
                "tier": task["_tier"],
                "request": jev.to_payload(task),
                "questions": {"q": {"type": task["question"]["type"], "gold": task["expected"]}},
            })
        return tasks

    if suite == "pilot":
        inputs, _ = itd.load_pilot()
        if limit:
            inputs = inputs[:limit]
        for row in inputs:
            tasks.append({
                "bench": "pilot",
                "id": row["id"],
                "tier": "pilot",
                "request": itd.to_request(row),
                "questions": {name: {"type": q["type"], "gold": None} for name, q in row["questions"].items()},
            })
        return tasks

    rows = itd.load_suite(suite)
    if limit:
        rows = rows[:limit]
    for row in rows:
        targets = row.get("targets", {})
        questions = {
            name: {"type": q["type"], "gold": (targets.get(name) or {}).get("label")}
            for name, q in row["questions"].items()
        }
        tasks.append({
            "bench": suite,
            "id": row["id"],
            "tier": suite,
            "request": itd.to_request(row),
            "questions": questions,
        })
    return tasks


async def _attempt(service, task: dict, semaphore: asyncio.Semaphore, route: str) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            payload = {**task["request"], "model": route}
            response = await service.decide(SystemOneRequest(**payload))
            decisions = []
            for name, meta in task["questions"].items():
                decision = response.decisions.get(name)
                if decision is None:
                    raise KeyError(f"response is missing the decision '{name}'")
                if meta["gold"] is None:
                    predicted, correct = decision.label, False  # pilot: correctness comes from references
                else:
                    predicted, correct = itd.predicted_and_correct(meta["type"], meta["gold"], decision)
                decisions.append({
                    "row_id": task["id"],
                    "name": name,
                    "type": meta["type"],
                    "gold": meta["gold"],
                    "predicted": predicted,
                    "correct": correct,
                    "distribution": itd.decision_distribution(decision),
                    "confidence": decision.confidence,
                    "true_probability": decision.true_probability,
                    "coverage": decision.coverage,
                    "reliable": decision.reliable,
                    "generated_token": decision.generated_token,
                    "error": None,
                })
            record = {"bench": task["bench"], "id": task["id"], "tier": task["tier"], "questions": decisions, "error": None}
        except Exception as exc:
            record = {
                "bench": task["bench"], "id": task["id"], "tier": task["tier"],
                "questions": [
                    {"row_id": task["id"], "name": name, "type": meta["type"], "gold": meta["gold"],
                     "predicted": None, "correct": False, "distribution": None, "confidence": None,
                     "true_probability": None, "coverage": None, "reliable": None,
                     "generated_token": None, "error": f"{type(exc).__name__}: {exc}"}
                    for name, meta in task["questions"].items()
                ],
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        for decision in record["questions"]:
            decision["latency_ms"] = record["latency_ms"]
    return record


async def run_with_retry(service, task: dict, semaphore: asyncio.Semaphore, route: str) -> dict:
    record: dict = {}
    total = len(RETRY_DELAYS) + 1
    for index in range(total):
        record = await _attempt(service, task, semaphore, route)
        record["attempts"] = index + 1
        if index >= total - 1 or not _is_connection_error(record.get("error")):
            break
        await asyncio.sleep(RETRY_DELAYS[index])
    return record


def _block(decisions: List[dict]) -> dict:
    latencies = sorted(d["latency_ms"] for d in decisions)
    n = len(decisions)
    correct = sum(1 for d in decisions if d["correct"])
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else None,
        "errors": sum(1 for d in decisions if d["error"]),
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
    }


def _apply_pilot_reference_correctness(records: List[dict], references: Dict[str, dict]) -> None:
    """The pilot has exact reference distributions instead of hard labels; make `correct` mean
    "matched the reference argmax" so the per-suite counts stay interpretable."""
    for record in records:
        reference = references.get(record["id"])
        if not reference:
            continue
        gold = {str(label): float(prob) for label, prob in reference["gold_probs"].items()}
        for decision in record["questions"]:
            if decision["predicted"] and not decision["error"]:
                decision["correct"] = decision["predicted"] == max(gold, key=gold.get)


def summarize_route(route_records: List[dict], pilot_references: Dict[str, dict]) -> dict:
    """Per-suite accuracy (counted per decision) plus the two distribution columns."""
    out: Dict[str, dict] = {}
    suite_accuracy: Dict[str, Optional[float]] = {}

    for bench in SUITES:
        records = [r for r in route_records if r["bench"] == bench]
        decisions = [d for r in records for d in r["questions"]]
        block = _block(decisions)
        entry: Dict = dict(block)
        if bench == "jevbench":
            for tier in ("easy", "original", "hard"):
                entry[tier] = _block([d for r in records if r["tier"] == tier for d in r["questions"]])
                suite_accuracy[tier] = entry[tier]["accuracy"]
            hard = [d for r in records if r["tier"] == "hard" for d in r["questions"]]
            scored = [d for d in hard if d["distribution"] and not d["error"]]
            entry["hard_brier"] = round(sum(itd.brier_multiclass(d["distribution"], d["gold"]) for d in scored) / len(scored), 6) if scored else None
            entry["hard_ece"] = round(itd.ece_from_records(hard), 6) if itd.ece_from_records(hard) is not None else None
            entry["by_type"] = {t: _block([d for d in decisions if d["type"] == t]) for t in ("choice", "noul", "score")}
        elif bench == "pilot":
            entry.update(itd.pilot_metrics(decisions, pilot_references))
        elif bench == "typed_decisions":
            entry["by_type"] = {t: _block([d for d in decisions if d["type"] == t]) for t in ("choice", "noul", "score")}
            suite_accuracy["typed_decisions"] = block["accuracy"]
        else:
            suite_accuracy[bench] = block["accuracy"]
        out[bench] = entry

    out["seven_suite_average"] = itd.seven_suite_average(suite_accuracy)
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="Intern-Decision seven-suite comparison")
    parser.add_argument("--routes", required=True, help="comma-separated route names")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N records per suite (debugging)")
    parser.add_argument("--tag", default="", help="suffix for the result filename")
    args = parser.parse_args()

    config = Config.load()
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    missing = [r for r in routes if r not in config.models]
    if missing:
        raise SystemExit(f"unknown routes: {missing}; available: {config.model_names()}")
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suites: {unknown}; available: {list(SUITES)}")

    datasets: Dict[str, List[dict]] = {}
    for suite in suites:
        datasets[suite] = build_tasks(suite, args.limit)
        print(f"{suite}: {len(datasets[suite])} records", flush=True)
    counts = {suite: len(tasks) for suite, tasks in datasets.items()}
    per_route_calls = sum(counts.values())
    print(f"routes: {', '.join(routes)} | calls per route {per_route_calls} | total {per_route_calls * len(routes)} | concurrency {args.concurrency}", flush=True)

    _, pilot_references = itd.load_pilot()
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "routes": routes,
        "suites": suites,
        "counts": counts,
        "updated_per_suite": True,
        "per_route": {},
    }
    RESULT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULT_DIR / f"intern-{stamp}{('-' + args.tag) if args.tag else ''}.json"
    print(f"result file: {out_path}", flush=True)

    started_all = time.perf_counter()
    for route in routes:
        settings = config.resolve(route)
        print(f"\n=== {route} -> {settings.model}", flush=True)
        client = ChatClient(settings)
        service = SystemOneService({route: client}, config)
        semaphore = asyncio.Semaphore(args.concurrency)
        route_records: List[dict] = []
        payload["per_route"][route] = {"model": settings.model, "records": route_records}
        for suite in suites:
            started = time.perf_counter()
            tasks = datasets[suite]
            records = list(await asyncio.gather(*(run_with_retry(service, t, semaphore, route) for t in tasks)))
            if suite == "pilot":
                _apply_pilot_reference_correctness(records, pilot_references)
            route_records.extend(records)
            elapsed = time.perf_counter() - started
            decisions = [d for r in records for d in r["questions"]]
            block = _block(decisions)
            failures = sum(1 for d in decisions if d["error"])
            print(f"  {suite:<16} {block['correct']:>5}/{block['n']:<5} = {block['accuracy']:.4f}  "
                  f"failures {failures}  {elapsed:.0f}s", flush=True)
            payload["per_route"][route]["summary"] = summarize_route(route_records, pilot_references)
            payload["per_route"][route]["elapsed_s"] = round(time.perf_counter() - started_all, 1)
            out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary ===", flush=True)
    for route in routes:
        summary = payload["per_route"][route]["summary"]
        print(f"{route:<18} seven-suite avg {summary['seven_suite_average']}", flush=True)
        for suite in suites:
            entry = summary[suite]
            print(f"   {suite:<16} acc {entry['accuracy']}  n {entry['n']}  errors {entry['errors']}", flush=True)
    print(f"\ntotal elapsed {time.perf_counter() - started_all:.0f}s | result: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))