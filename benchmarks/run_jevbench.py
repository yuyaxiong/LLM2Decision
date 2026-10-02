"""Run the public 231-question JevBench (72 original + 48 easy + 111 hard) against this project's /v1/systemone.

Data comes from the upstream repository benchmarks/jevbench (MIT); each record is already a
state + typed question structure, so the adapter only does three things: question-type mapping,
field shuffling, and answer comparison.

Usage:
    python3 benchmarks/run_jevbench.py                      # run all 231 questions
    python3 benchmarks/run_jevbench.py --limit 20           # quick smoke run to validate the pipeline
    python3 benchmarks/run_jevbench.py --concurrency 4      # adjust concurrency
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm2decision.llm.client import ChatClient  # noqa: E402
from llm2decision.core.config import Config  # noqa: E402
from llm2decision.core.schema import SystemOneRequest  # noqa: E402
from llm2decision.llm.service import SystemOneService  # noqa: E402

DATA_DIR = pathlib.Path(__file__).resolve().parent / "jevbench" / "datasets" / "public"
RESULT_DIR = pathlib.Path(__file__).resolve().parent / "results"

DATA_HINT = (
    f"Cannot find the JevBench data (expected location: {DATA_DIR}).\n"
    "This dataset is not checked into this repository (it is large, and upstream should be the source of truth); fetch a copy first:\n"
    "    git clone --depth 1 https://github.com/fstandhartinger/jevbench "
    "benchmarks/jevbench\n"
    "or symlink it to an existing copy (saves a download):\n"
    "    ln -s <existing-path>/jevbench benchmarks/jevbench"
)


def _read_tier(name: str) -> str:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(DATA_HINT)
    return path.read_text(encoding="utf-8")


# Following the breakdown published in the Open-Jev-27B-v1.1 model card: 231 = 72 original + 48 easy + 111 hard
TIERS = {"original": ["original"], "easy": ["easy"], "hard": ["hard"]}

PUBLISHED = [
    ("Jev 1.13.0 (hosted, TypeSafe)", 200, 231, 81, 111),
    ("Open-Jev-27B-v1.1", 197, 231, 80, 111),
    ("Open-Jev 9B", 179, 231, 66, 111),
    ("Open-Jev 2B", 150, 231, 46, 111),
]


def load_tasks() -> List[dict]:
    tasks: List[dict] = []
    for name in ("original", "easy", "hard"):
        for line in _read_tier(name).splitlines():
            if line.strip():
                task = json.loads(line)
                task["_tier"] = name
                tasks.append(task)
    return tasks


def to_payload(task: dict) -> dict:
    """Convert a JevBench record into this service's request body."""
    question = task["question"]
    qtype = question["type"]
    instructions = question.get("instructions", "") or ""
    criteria = question.get("criteria")
    labels = list(task["labels"])

    # Upstream allows a structured state (dict); this service only accepts text, so serialize it to a JSON string
    state = task["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)

    if qtype == "choice":
        # criteria is already {label: description}, matching this service's format
        adapted: Dict = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
    elif qtype == "noul":
        # Upstream uses criteria to describe the meaning of true/false, and expected is "yes"/"no"
        # Chinese rubric prefix and separator kept as-is: the v1 prompt locale is Chinese and
        # every published number was measured with this exact string.
        rubric = ""
        if isinstance(criteria, dict):
            parts = [f"{key}：{value}" for key, value in criteria.items()]
            rubric = "\n成立与不成立的标准：" + "；".join(parts)
        adapted = {"type": "noul", "instructions": instructions + rubric}
    else:
        # score: criteria is an array of level descriptions (the index is the level number)
        # Chinese rubric prefix and separator kept as-is: the v1 prompt locale is Chinese and
        # every published number was measured with this exact string.
        levels = ""
        if isinstance(criteria, list) and criteria:
            levels = "\n档位定义：" + "；".join(f"{index}={text}" for index, text in enumerate(criteria))
        adapted = {
            "type": "score",
            "instructions": instructions + levels,
            "scale": labels,
            "values": [float(label) for label in labels],
        }
    return {"state": state, "questions": {"q": adapted}}


def predicted_label(task: dict, decision) -> Optional[str]:
    qtype = task["question"]["type"]
    if qtype == "choice":
        return decision.label
    if qtype == "noul":
        if decision.true_probability is None:
            return None
        return "yes" if decision.true_probability >= 0.5 else "no"
    return decision.label


def is_correct(expected, predicted: Optional[str]) -> bool:
    if predicted is None:
        return False
    try:
        return float(expected) == float(predicted)
    except (TypeError, ValueError):
        return str(expected) == str(predicted)


async def run_task(service, task: dict, semaphore: asyncio.Semaphore, route: str) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            # The service resolves the route from the request body; without "model" here it would
            # fall back to default_model, which is not in this single-route client map (KeyError).
            payload = {**to_payload(task), "model": route}
            response = await service.decide(SystemOneRequest(**payload))
            decision = response.decisions["q"]
            predicted = predicted_label(task, decision)
            record = {
                "id": task["id"],
                "tier": task["_tier"],
                "family": task.get("family"),
                "type": task["question"]["type"],
                "expected": task["expected"],
                "predicted": predicted,
                "correct": is_correct(task["expected"], predicted),
                "coverage": decision.coverage,
                "reliable": decision.reliable,
                "generated_token": decision.generated_token,
                "confidence": decision.confidence,
                "true_probability": decision.true_probability,
                "error": None,
            }
        except Exception as exc:
            record = {
                "id": task["id"],
                "tier": task["_tier"],
                "family": task.get("family"),
                "type": task["question"]["type"],
                "expected": task["expected"],
                "predicted": None,
                "correct": False,
                "coverage": None,
                "reliable": None,
                "generated_token": None,
                "confidence": None,
                "true_probability": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return record


def summarize(records: List[dict]) -> dict:
    def accuracy(subset: List[dict]) -> Optional[float]:
        return round(sum(1 for item in subset if item["correct"]) / len(subset), 4) if subset else None

    def block(subset: List[dict]) -> dict:
        latencies = sorted(item["latency_ms"] for item in subset)
        return {
            "n": len(subset),
            "correct": sum(1 for item in subset if item["correct"]),
            "accuracy": accuracy(subset),
            "errors": sum(1 for item in subset if item["error"]),
            "unreliable": sum(1 for item in subset if item["reliable"] is False),
            "accuracy_on_all": round(sum(1 for item in subset if item["correct"]) / len(subset), 4) if subset else None,
            "latency_p50_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": latencies[int(len(latencies) * 0.95) - 1] if latencies else None,
        }

    return {
        "overall": block(records),
        "by_tier": {tier: block([r for r in records if r["tier"] == tier]) for tier in TIERS},
        "by_type": {qtype: block([r for r in records if r["type"] == qtype])
                    for qtype in ("choice", "noul", "score")},
        "errors": [
            {"id": r["id"], "type": r["type"], "error": r["error"]}
            for r in records if r["error"]
        ][:20],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public JevBench subset against this service")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N questions (for debugging)")
    parser.add_argument("--concurrency", type=int, default=6, help="number of concurrent requests")
    parser.add_argument("--logit-bias", action="store_true", help="enable the logit_bias fallback")
    parser.add_argument("--model", help="model route name; defaults to default_model")
    parser.add_argument("--tag", default="", help="suffix for the result filenames")
    args = parser.parse_args()

    config = Config.load()
    route = args.model or config.default_model
    settings = config.resolve(route)
    if args.logit_bias:
        settings = dataclasses.replace(settings, logit_bias_enabled=True)
    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"route {route} -> model {settings.model} | questions {len(tasks)} | concurrency {args.concurrency} | candidate cap 10")

    client = ChatClient(settings)
    service = SystemOneService({route: client}, config)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        records = []
        for index, record in enumerate(
            await asyncio.gather(*(run_task(service, task, semaphore, route) for task in tasks)), 1
        ):
            records.append(record)
            if index % 25 == 0 or index == len(tasks):
                done = [r for r in records if not r["error"]]
                print(f"  {index}/{len(tasks)} done, {len(done)} scorable, "
                      f"correct so far {sum(1 for r in done if r['correct'])}")
    finally:
        await client.aclose()

    summary = summarize(records)
    summary["model"] = settings.model
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    summary["temperature_scale"] = settings.temperature_scale
    summary["logit_bias_enabled"] = settings.logit_bias_enabled
    summary["published"] = [
        {"model": name, "public_231": f"{hit}/{total}", "hard_111": f"{hard_hit}/{hard_total}"}
        for name, hit, total, hard_hit, hard_total in PUBLISHED
    ]

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    detail_path = RESULT_DIR / f"jevbench-{stamp}{tag}.jsonl"
    summary_path = RESULT_DIR / f"jevbench-{stamp}{tag}.summary.json"
    with detail_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("\n" + "=" * 64)
    print(f"Public 231-question breakdown: {overall['correct']}/{overall['n']} = {overall['accuracy']:.2%}"
          f" ({overall['errors']} failures, {overall['unreliable']} with reliable=false)")
    for tier, block in summary["by_tier"].items():
        if not block["n"]:
            print(f"  {tier:<9} not covered this run")
            continue
        print(f"  {tier:<9} {block['correct']:>3}/{block['n']:<3} = {block['accuracy']:.2%}"
              f" ({block['errors']} failures)")
    for qtype, block in summary["by_type"].items():
        if not block["n"]:
            continue
        print(f"  {qtype:<9} {block['correct']:>3}/{block['n']:<3} = {block['accuracy']:.2%}")
    print(f"  latency p50={overall['latency_p50_ms']}ms p95={overall['latency_p95_ms']}ms"
          f" | total {summary['elapsed_s']}s")
    if summary["errors"]:
        print("\n--- failure samples ---")
        for item in summary["errors"][:5]:
            print(f"  {item['id']} [{item['type']}] {item['error'][:150]}")
    print("\n--- published results (public 231 / hard 111) ---")
    for item in summary["published"]:
        print(f"  {item['model']:<28} {item['public_231']:<10} {item['hard_111']}")
    print(f"\ndetails: {detail_path}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
