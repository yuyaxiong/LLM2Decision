"""Run Kev's 6 subsets (decision-v7 / transfer-v4 / transfer-v9, each with development+test) against this project's /v1/systemone.

Kev's data format is isomorphic to this service's request body: one record per line, with
state + questions fields. Each record can be tagged via _meta.variant as
clean / none_present / none_absent / permuted. This runner only evaluates records with
variant == "clean"; the other variants are skipped and counted (if a subset has no clean
records at all, it is evaluated in full).

Differences from run_jevbench (verified against the real data):
  * label hangs off each question (question["label"]), not off the record's top level;
  * a record may contain several questions (v7 has 1~3), so we expand them one by one and score each;
  * a choice question's criteria values are "descriptions" that may be null or a nested object
    (in a few cases), so we normalize them to strings;
  * questions with more than 10 candidates are skipped outright (service hard cap) rather than aborting with an error.

Usage:
    python3 benchmarks/run_kev.py --route doubao-2.0-mini
    python3 benchmarks/run_kev.py --route doubao-2.0-mini --limit 8    # sample 8 scorable records per subset
    python3 benchmarks/run_kev.py --route deepseek-v4-pro --concurrency 4 --tag smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

from llm2decision.llm.client import ChatClient  # noqa: E402
from llm2decision.core.config import Config  # noqa: E402
from llm2decision.core.schema import SystemOneRequest  # noqa: E402
from llm2decision.llm.service import SystemOneService  # noqa: E402
from run_jevbench import is_correct, summarize  # noqa: E402

KEV_DIR = BENCH_DIR / "kev"
RESULT_DIR = BENCH_DIR / "results"

_RAW = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals"
SUBSETS: List[Tuple[str, str]] = [
    ("decision-v7-dev", f"{_RAW}/v7/decision-v7/development.jsonl"),
    ("decision-v7-test", f"{_RAW}/v7/decision-v7/test.jsonl"),
    ("transfer-v4-dev", f"{_RAW}/v4/transfer-v4/development.jsonl"),
    ("transfer-v4-test", f"{_RAW}/v4/transfer-v4/test.jsonl"),
    ("transfer-v9-dev", f"{_RAW}/v9/transfer-v9/development.jsonl"),
    ("transfer-v9-test", f"{_RAW}/v9/transfer-v9/test.jsonl"),
]

# Service hard cap on the number of candidates (src/llm2decision/core/labels.py: MAX_CANDIDATES)
MAX_CANDIDATES = 10


def ensure_data() -> Dict[str, pathlib.Path]:
    """On first run, download the 6 subsets to benchmarks/kev/<subset-name>.jsonl; skip any that already exist."""
    KEV_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, pathlib.Path] = {}
    for name, url in SUBSETS:
        target = KEV_DIR / f"{name}.jsonl"
        if not target.exists():
            print(f"downloading {name} <- {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        paths[name] = target
    return paths


def load_records(path: pathlib.Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stringify(value) -> str:
    """A criteria description value: None -> empty string, a plain string as-is, a nested object -> the text inside it."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        texts = [str(item) for item in value.values() if isinstance(item, (str, int, float))]
        # Chinese separator kept as-is: it joins criteria descriptions that are sent to the model,
        # and every published number was measured with this exact string.
        return "；".join(texts) if texts else json.dumps(value, ensure_ascii=False)
    return str(value)


def _gold_label(qtype: str, raw) -> str:
    if qtype == "noul":
        return "yes" if raw is True or str(raw).strip().lower() in {"true", "yes", "1"} else "no"
    if qtype == "score":
        return str(int(raw))
    return str(raw)


def adapt(state: str, question: dict) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """Convert one Kev question into (request-body fragment, gold, skip reason). A non-empty skip reason means the question cannot be scored."""
    qtype = question["type"]
    # instructions is a string in the vast majority of cases, but a few in v7 are a {"question": ..., "focus": ...} structure
    instructions = _stringify(question.get("instructions"))
    criteria = question.get("criteria")

    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            return None, None, "missing-criteria"
        if len(criteria) > MAX_CANDIDATES:
            return None, None, f"too-many-candidates({len(criteria)})"
        adapted = {
            "type": "choice",
            "instructions": instructions,
            "criteria": {key: _stringify(value) for key, value in criteria.items()},
        }
    elif qtype == "noul":
        # Chinese rubric prefix and separator kept as-is: the v1 prompt locale is Chinese and
        # every published number was measured with this exact string.
        rubric = ""
        if isinstance(criteria, dict) and criteria:
            parts = [f"{key}：{_stringify(value)}" for key, value in criteria.items()]
            rubric = "\n判定标准：" + "；".join(parts)
        adapted = {"type": "noul", "instructions": instructions + rubric}
    else:  # score: criteria is an ordered list of level descriptions; the index is the level number (gold is also a 0-based level number)
        # Chinese rubric prefix and separator kept as-is: the v1 prompt locale is Chinese and
        # every published number was measured with this exact string.
        levels = criteria if isinstance(criteria, list) else []
        if not levels:
            return None, None, "missing-criteria"
        if len(levels) > MAX_CANDIDATES:
            return None, None, f"too-many-candidates({len(levels)})"
        scale = [str(index) for index in range(len(levels))]
        definition = "\n档位定义：" + "；".join(
            f"{index}={_stringify(text)}" for index, text in enumerate(levels)
        )
        adapted = {
            "type": "score",
            "instructions": instructions + definition,
            "scale": scale,
            "values": [float(index) for index in range(len(levels))],
        }

    return {"state": state, "questions": {"q": adapted}}, _gold_label(qtype, question.get("label")), None


def build_tasks(subset: str, records: List[dict], limit: int) -> dict:
    """Expand the records of a subset into question-level tasks and tally the various skip counts."""
    clean = [record for record in records if (record.get("_meta") or {}).get("variant") == "clean"]
    skipped_variant = len(records) - len(clean)
    note = None
    if clean:
        usable = clean
    else:
        usable = records
        note = "this subset has no variant==clean records; evaluated in full"

    tasks: List[dict] = []
    skipped_too_many = 0
    skipped_other = 0
    used_records = 0
    for record in usable:
        if limit and used_records >= limit:
            break
        meta = record.get("_meta") or {}
        record_id = meta.get("id") or f"{subset}:{meta.get('row')}"
        state = record["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        record_tasks: List[dict] = []
        for question_name, question in record["questions"].items():
            payload, gold, skip = adapt(state, question)
            if skip:
                if skip.startswith("too-many-candidates"):
                    skipped_too_many += 1
                else:
                    skipped_other += 1
                continue
            record_tasks.append({
                "id": f"{record_id}#{question_name}",
                "subset": subset,
                "type": question["type"],
                "src": question.get("src"),
                "gold": gold,
                "payload": payload,
            })
        # --limit is a debugging sample: only count records that actually yield questions, so we don't pick a
        # prefix of records that are all skipped and leave a subset empty
        if record_tasks:
            tasks.extend(record_tasks)
            used_records += 1
    return {
        "tasks": tasks,
        "clean_records": len(clean),
        "used_records": used_records,
        "skipped_variant_records": skipped_variant,
        "skipped_too_many": skipped_too_many,
        "skipped_other": skipped_other,
        "note": note,
    }


def predicted_label(qtype: str, decision) -> Optional[str]:
    if qtype == "noul":
        if decision.true_probability is None:
            return None
        return "yes" if decision.true_probability >= 0.5 else "no"
    return decision.label


async def run_task(service, route: str, task: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await service.decide(SystemOneRequest(model=route, **task["payload"]))
            decision = response.decisions["q"]
            predicted = predicted_label(task["type"], decision)
            record = {
                "id": task["id"],
                "subset": task["subset"],
                "tier": task["subset"],
                "type": task["type"],
                "src": task["src"],
                "expected": task["gold"],
                "predicted": predicted,
                "correct": is_correct(task["gold"], predicted),
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
                "subset": task["subset"],
                "tier": task["subset"],
                "type": task["type"],
                "src": task["src"],
                "expected": task["gold"],
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


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kev's 6 subsets against this service")
    parser.add_argument("--route", required=True, help="model route name (a key under models in llm2decision.yaml)")
    parser.add_argument("--limit", type=int, default=0,
                        help="number of scorable records to sample per subset (for debugging; 0 = all)")
    parser.add_argument("--concurrency", type=int, default=8, help="number of concurrent requests")
    parser.add_argument("--tag", default="", help="suffix for the result filenames")
    args = parser.parse_args()

    config = Config.load()
    settings = config.resolve(args.route)

    data_paths = ensure_data()
    plans: Dict[str, dict] = {}
    for name, _ in SUBSETS:
        plans[name] = build_tasks(name, load_records(data_paths[name]), args.limit)

    total_questions = sum(len(plan["tasks"]) for plan in plans.values())
    print(f"route {args.route} -> model {settings.model} | subsets 6 | questions to evaluate {total_questions} "
          f"| concurrency {args.concurrency} | candidate cap {MAX_CANDIDATES}")

    clients = {args.route: ChatClient(settings)}
    service = SystemOneService(clients, config)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    subset_outputs: List[dict] = []
    all_records: List[dict] = []
    try:
        for name, _ in SUBSETS:
            plan = plans[name]
            records = list(await asyncio.gather(
                *(run_task(service, args.route, task, semaphore) for task in plan["tasks"])
            ))
            all_records.extend(records)
            summary = summarize(records)
            overall = summary["overall"]
            accuracy = overall["accuracy"]
            subset_outputs.append({
                "name": name,
                "clean_records": plan["clean_records"],
                "used_records": plan["used_records"],
                "questions": overall["n"],
                "correct": overall["correct"],
                "accuracy": accuracy,
                "errors": overall["errors"],
                "skipped_variant_records": plan["skipped_variant_records"],
                "skipped_too_many_candidates": plan["skipped_too_many"],
                "skipped_other": plan["skipped_other"],
                "by_type": summary["by_type"],
                "latency_p50_ms": overall["latency_p50_ms"],
                "note": plan["note"],
            })
            flag = f"{accuracy:.2%}" if accuracy is not None else "n/a"
            print(f"  {name:<20} clean acc {overall['correct']:>4}/{overall['n']:<4} = {flag:<8}"
                  f" ({overall['errors']} failures | {plan['skipped_variant_records']} variant-skipped"
                  f" | candidates > {MAX_CANDIDATES}: {plan['skipped_too_many']} questions skipped)")
    finally:
        for client in clients.values():
            await client.aclose()

    accuracies = [item["accuracy"] for item in subset_outputs if item["accuracy"] is not None]
    mean_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else None

    payload_out = {
        "benchmark": "kev",
        "route": args.route,
        "model": settings.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "concurrency": args.concurrency,
        "limit": args.limit,
        "subsets": subset_outputs,
        "clean_accuracy_mean": mean_accuracy,
        "details": all_records,
    }

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    out_path = RESULT_DIR / f"kev-{stamp}{tag}.json"
    out_path.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"Equal-weight mean clean accuracy across the six subsets: {mean_accuracy:.2%}" if mean_accuracy is not None
          else "Equal-weight mean clean accuracy across the six subsets: n/a")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
