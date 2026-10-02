"""Run every benchmark × every model route and print the accuracy matrix.

Four benchmark groups (columns):
    jevbench  public 231 (72 original + 48 easy + 111 hard)
    nimble    the compat282 subset of Nimble's built-in holdout (TokenRhythm protocol)
    vitaminc  the nimble599 subset of VitaminC validation (Nimble manifest protocol)
    kev       Kev's 6 subsets, per-question scoring (only _meta.variant=="clean" is scored;
              questions with >10 candidates are skipped and counted)

Kev is large (~5768 questions), so by default it runs only the routes given by --kev-routes;
if none are specified, Kev is skipped (the other three groups still run).

Usage:
    python3 benchmarks/run_matrix.py --routes doubao-2.0-mini
    python3 benchmarks/run_matrix.py --kev-routes doubao-2.0-pro,doubao-2.1-lite
    python3 benchmarks/run_matrix.py --benches jevbench,nimble --limit 20
    python3 benchmarks/run_matrix.py --resume benchmarks/results/matrix-xxx.json   # resume: skip already-valid cells

Results go to benchmarks/results/matrix-<timestamp>.json, flushed after every route.
--resume rewrites the given json in place (skipping route×benchmark cells whose valid != false, and
re-running only the missing/invalid ones).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

from llm2decision.llm.client import ChatClient  # noqa: E402
from llm2decision.core.config import Config  # noqa: E402
from llm2decision.core.schema import SystemOneRequest  # noqa: E402
from llm2decision.llm.service import SystemOneService  # noqa: E402

import run_jevbench as jev  # noqa: E402
import run_kev as kev  # noqa: E402
import run_nimble as nim  # noqa: E402
import run_vitaminc as vit  # noqa: E402

RESULT_DIR = BENCH_DIR / "results"
BENCHMARKS = ("jevbench", "nimble", "vitaminc", "kev")
NIMBLE_SUBSET = "compat282"
VITAMINC_SUBSET = "nimble599"

# Retry policy for connection errors: up to 3 retries, backing off 1s/3s/9s
# (attempts records the total number of tries, up to 4)
RETRY_DELAYS = (1.0, 3.0, 9.0)
CONNECTION_ERROR_MARKERS = (
    "connection",
    "all connection attempts failed",
    "timeout",
    "timed out",
)
# A cell whose failure rate exceeds this threshold is treated as invalid (result unusable),
# so infrastructure outages are not mistaken for wrong answers
ERROR_RATIO_THRESHOLD = 0.2


def _stratified(tasks: List[dict], limit: int) -> List[dict]:
    """Sample round-robin by tier, so --limit does not cut off a single tier (jevbench's prefix is all original)."""
    if limit >= len(tasks):
        return tasks
    tiers: Dict[str, List[dict]] = {}
    for task in tasks:
        tiers.setdefault(task["_tier"], []).append(task)
    order = [name for name in ("original", "easy", "hard") if name in tiers]
    order += [name for name in tiers if name not in order]
    picked: List[dict] = []
    index = 0
    while len(picked) < limit:
        progressed = False
        for name in order:
            if index < len(tiers[name]):
                picked.append(tiers[name][index])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return picked


def load_items(bench: str, limit: int) -> Tuple[List[dict], str]:
    """Load the items for one benchmark; returns (item list, protocol note)."""
    items: List[dict] = []
    if bench == "jevbench":
        tasks = jev.load_tasks()
        if limit:
            tasks = _stratified(tasks, limit)
        for task in tasks:
            items.append({
                "id": task["id"],
                "tier": task["_tier"],
                "type": task["question"]["type"],
                "gold": task["expected"],
                "payload": jev.to_payload(task),
                "_task": task,
            })
        return items, "public 231 (72 original + 48 easy + 111 hard)"

    if bench == "nimble":
        raw_items, stats = nim.filter_compat(nim.load_items())
        if limit:
            raw_items = raw_items[:limit]
        for raw in raw_items:
            payload, gold, qtype = nim.to_payload(raw)
            items.append({
                "id": raw["id"],
                "tier": raw.get("source_family") or "unknown",
                "type": qtype,
                "gold": gold,
                "payload": payload,
                "_qtype": qtype,
            })
        info = f"{NIMBLE_SUBSET} subset, {len(items)} items (TokenRhythm protocol), tokenizer: {stats['tokenizer']}"
        return items, info

    # vitaminc: switch to the nimble599 subset from the Nimble manifest
    raw_items, missing = vit.load_manifest_items()
    if limit:
        raw_items = raw_items[:limit]
    for raw in raw_items:
        items.append({
            "id": raw["id"],
            "tier": "validation",
            "type": "choice",
            "gold": raw["gold"],
            "payload": vit.to_payload(raw),
        })
    info = f"{VITAMINC_SUBSET} subset, {len(items)} items (manifest protocol)"
    if missing:
        info += f", ⚠ {len(missing)} ids unmatched"
    return items, info


def check(bench: str, item: dict, decision) -> Tuple[Optional[str], bool]:
    if bench == "jevbench":
        predicted = jev.predicted_label(item["_task"], decision)
        return predicted, jev.is_correct(item["gold"], predicted)
    if bench == "nimble":
        predicted = nim.predicted_label(item["_qtype"], decision)
        gold = nim.normalize_gold(item["_qtype"], item["gold"])
        return predicted, predicted is not None and str(predicted).lower() == gold.lower()
    return decision.label, decision.label == item["gold"]


def _is_connection_error(message: Optional[str]) -> bool:
    """Connection-layer failures (httpx connection errors/timeouts, etc.) are retryable; parameter 4xx errors are not."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in CONNECTION_ERROR_MARKERS)


async def _with_retry(attempt_fn) -> dict:
    """Run one call and return a record carrying an error field.

    Connection errors use 1s/3s/9s exponential backoff, up to 3 retries; other errors return immediately.
    attempt_fn acquires the concurrency semaphore itself on each call (the backoff wait happens outside the semaphore).
    """
    record: dict = {}
    total = len(RETRY_DELAYS) + 1
    for index in range(total):
        record = await attempt_fn()
        record["attempts"] = index + 1
        if index >= total - 1 or not _is_connection_error(record.get("error")):
            break
        await asyncio.sleep(RETRY_DELAYS[index])
    return record


async def _one_attempt(service, route: str, bench: str, item: dict) -> dict:
    started = time.perf_counter()
    try:
        payload = {**item["payload"], "model": route}
        response = await service.decide(SystemOneRequest(**payload))
        decision = response.decisions["q"]
        predicted, correct = check(bench, item, decision)
        error = None
    except Exception as exc:
        predicted, correct, error = None, False, f"{type(exc).__name__}: {exc}"
    return {
        "id": item["id"],
        "tier": item["tier"],
        "type": item["type"],
        "gold": item["gold"],
        "predicted": predicted,
        "correct": correct,
        "error": error,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


async def run_one(service, route: str, bench: str, item: dict, semaphore: asyncio.Semaphore) -> dict:
    async def attempt() -> dict:
        async with semaphore:
            return await _one_attempt(service, route, bench, item)

    return await _with_retry(attempt)


def summarize(records: List[dict]) -> dict:
    def block(subset: List[dict]) -> Optional[dict]:
        if not subset:
            return None
        n = len(subset)
        valid = [r for r in subset if not r["error"]]
        n_valid = len(valid)
        n_errors = n - n_valid
        correct_valid = sum(1 for r in valid if r["correct"])
        correct_all = sum(1 for r in subset if r["correct"])
        latencies = sorted(r["latency_ms"] for r in (valid or subset))
        result = {
            "n": n,
            "n_valid": n_valid,
            "n_errors": n_errors,
            "correct": correct_valid,
            # Accuracy is computed only over error-free items, so connection failures are not counted as wrong answers
            "accuracy": round(correct_valid / n_valid, 4) if n_valid else None,
            # The protocol that counts failures as wrong; kept for comparison
            "accuracy_on_all": round(correct_all / n, 4) if n else None,
            "latency_p50_ms": statistics.median(latencies),
        }
        ratio = n_errors / n
        result["valid"] = ratio <= ERROR_RATIO_THRESHOLD
        if not result["valid"]:
            result["validity_reason"] = (
                f"failures {n_errors}/{n} = {ratio:.1%} > {ERROR_RATIO_THRESHOLD:.0%}"
            )
        return result

    return {
        "overall": block(records),
        "by_type": {t: block([r for r in records if r["type"] == t]) for t in ("choice", "noul", "score")},
        "by_tier": {t: block([r for r in records if r["tier"] == t])
                    for t in sorted({r["tier"] for r in records})},
    }


async def run_kev_route(service, route: str, plans: Dict[str, dict],
                        semaphore: asyncio.Semaphore) -> Tuple[List[dict], List[dict]]:
    """Run the Kev 6 subsets for one route with per-question scoring. Returns (all per-question records, per-subset summaries)."""
    all_records: List[dict] = []
    subsets: List[dict] = []
    for name, _ in kev.SUBSETS:
        plan = plans[name]
        records = list(await asyncio.gather(
            *(_with_retry(lambda t=task: kev.run_task(service, route, t, semaphore))
              for task in plan["tasks"])
        ))
        all_records.extend(records)
        overall = summarize(records)["overall"]
        subsets.append({
            "name": name,
            "questions": overall["n"] if overall else 0,
            "n_valid": overall["n_valid"] if overall else 0,
            "correct": overall["correct"] if overall else 0,
            "accuracy": overall["accuracy"] if overall else None,
            "n_errors": overall["n_errors"] if overall else 0,
            "valid": overall["valid"] if overall else True,
            "validity_reason": overall.get("validity_reason") if overall else None,
            "clean_records": plan["clean_records"],
            "used_records": plan["used_records"],
            "skipped_variant_records": plan["skipped_variant_records"],
            "skipped_too_many_candidates": plan["skipped_too_many"],
            "skipped_other": plan["skipped_other"],
            "note": plan["note"],
        })
    return all_records, subsets


def _fmt(accuracy: Optional[float]) -> str:
    return f"{accuracy:7.2%}" if accuracy is not None else "      —"


def print_matrix(routes: List[str], benches: List[str], matrix: dict) -> None:
    header = f"{'route':<20} {'jevbench':>8} {'jev-hard':>9} {'nimble':>8} {'vitaminc':>9} {'kev mean':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for route in routes:
        cells = [f"{route:<20}"]
        for bench in ("jevbench", "nimble", "vitaminc"):
            if bench in benches:
                cells.append(_fmt(matrix.get(route, {}).get(bench, {}).get("overall", {}).get("accuracy")))
            else:
                cells.append("      —")
        hard = matrix.get(route, {}).get("jevbench", {}).get("by_tier", {}).get("hard", {})
        cells.insert(2, _fmt(hard.get("accuracy") if benches and "jevbench" in benches else None))
        kev_block = matrix.get(route, {}).get("kev", {})
        cells.append(_fmt(kev_block.get("mean_accuracy")))
        print(" ".join(cells))
    print("=" * len(header))

    kev_routes = [r for r in routes if matrix.get(r, {}).get("kev")]
    if kev_routes:
        names = [name for name, _ in kev.SUBSETS]
        width = max(len("route"), max((len(r) for r in kev_routes), default=0))
        head = f"{'route':<{width}} " + " ".join(f"{n:>18}" for n in names) + f" {'mean of 6':>9}"
        print("\nKev subset detail (per-question, variant=clean only; questions with >10 candidates skipped):")
        print(head)
        print("-" * len(head))
        for route in kev_routes:
            block = matrix[route]["kev"]
            cells = []
            for name in names:
                sub = next((s for s in block["subsets"] if s["name"] == name), {})
                cells.append(f"{_fmt(sub.get('accuracy')):>18}")
            cells.append(f"{_fmt(block.get('mean_accuracy')):>9}")
            print(f"{route:<{width}} " + " ".join(cells))


def _cell_reusable(cell) -> bool:
    """Decide, under --resume, whether a route×benchmark cell can reuse its old result.

    Cells explicitly marked valid=false must be re-run; for the old format (no valid field) we infer
    from the error rate, treating a rate above the threshold as invalid and in need of a re-run.
    """
    if not isinstance(cell, dict):
        return False
    if cell.get("valid") is False:
        return False
    if "valid" not in cell:
        overall = cell.get("overall") or {}
        n = overall.get("n") or 0
        errors = overall.get("n_errors", overall.get("errors")) or 0
        if n and errors / n > ERROR_RATIO_THRESHOLD:
            return False
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description="full benchmark × model-route matrix")
    parser.add_argument("--routes", help="comma-separated route names; all by default")
    parser.add_argument("--benches", default=",".join(BENCHMARKS))
    parser.add_argument("--kev-routes", default="",
                        help="comma-separated: run Kev only for these routes (it is large); leave empty to skip Kev and still run the other three groups")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="number of items sampled per benchmark (for debugging; 0 = all)")
    parser.add_argument("--resume", default="",
                        help="path to an existing matrix-*.json: skip its valid cells and re-run only the missing/invalid ones (written back in place)")
    args = parser.parse_args()

    config = Config.load()
    routes = [r.strip() for r in args.routes.split(",")] if args.routes else config.model_names()
    missing = [r for r in routes if r not in config.models]
    if missing:
        raise SystemExit(f"unknown routes: {missing}; available: {config.model_names()}")
    kev_routes = [r.strip() for r in args.kev_routes.split(",") if r.strip()]
    unknown_kev = [r for r in kev_routes if r not in config.models]
    if unknown_kev:
        raise SystemExit(f"unknown --kev-routes: {unknown_kev}; available: {config.model_names()}")

    benches = [b.strip() for b in args.benches.split(",")]
    unknown_benches = [b for b in benches if b not in BENCHMARKS]
    if unknown_benches:
        raise SystemExit(f"unknown benchmark: {unknown_benches}; available: {list(BENCHMARKS)}")

    if "kev" in benches and not kev_routes:
        print("--kev-routes not specified: skipping Kev (the other three groups still run)")
        benches = [b for b in benches if b != "kev"]

    # First three groups
    datasets: Dict[str, List[dict]] = {}
    for bench in [b for b in benches if b != "kev"]:
        items, info = load_items(bench, args.limit)
        datasets[bench] = items
        print(f"{bench}: {len(items)} items ({info})")

    # Kev plan (expanded per question)
    kev_plans: Dict[str, dict] = {}
    kev_total = 0
    if "kev" in benches:
        paths = kev.ensure_data()
        for name, _ in kev.SUBSETS:
            plan = kev.build_tasks(name, kev.load_records(paths[name]), args.limit)
            kev_plans[name] = plan
            kev_total += len(plan["tasks"])
        print(f"kev: 6 subsets, {kev_total} questions total (per-question scoring, variant=clean only; "
              f"questions with > {kev.MAX_CANDIDATES} candidates skipped)")

    base_count = sum(len(items) for items in datasets.values())
    planned: Dict[str, int] = {}
    print(f"\nroute plan ({len(routes)} routes; Kev only for: {', '.join(kev_routes) or 'none'}):")
    for route in routes:
        runs_kev = "kev" in benches and route in kev_routes
        count = base_count + (kev_total if runs_kev else 0)
        planned[route] = count
        tail = f" + kev {kev_total}" if runs_kev else " (no kev)"
        print(f"  {route:<20} {base_count}{tail} = {count} questions")
    total_calls = sum(planned.values())
    print(f"total planned calls {total_calls} | concurrency {args.concurrency}")

    RESULT_DIR.mkdir(exist_ok=True)
    reused = 0
    if args.resume:
        out_path = pathlib.Path(args.resume)
        if not out_path.exists():
            raise SystemExit(f"--resume file does not exist: {out_path}")
        payload_out = json.loads(out_path.read_text(encoding="utf-8"))
        payload_out.setdefault("matrix", {})
        payload_out.setdefault("details", {})
        payload_out.update({
            "routes": routes,
            "benches": benches,
            "kev_routes": kev_routes,
            "subsets": {"nimble": NIMBLE_SUBSET, "vitaminc": VITAMINC_SUBSET},
            "counts": {b: len(items) for b, items in datasets.items()},
            "kev_questions": kev_total,
            "planned_calls": planned,
        })
        print(f"resume mode: loading {out_path}, skipping its valid cells and re-running only the missing/invalid ones")
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = RESULT_DIR / f"matrix-{stamp}.json"
        payload_out = {
            "routes": routes,
            "benches": benches,
            "kev_routes": kev_routes,
            "subsets": {"nimble": NIMBLE_SUBSET, "vitaminc": VITAMINC_SUBSET},
            "counts": {b: len(items) for b, items in datasets.items()},
            "kev_questions": kev_total,
            "planned_calls": planned,
            "matrix": {},
            "details": {},
        }

    clients = {name: ChatClient(config.models[name]) for name in routes}
    service = SystemOneService(clients, config)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        for route in routes:
            payload_out["matrix"].setdefault(route, {})
            for bench in benches:
                if bench == "kev" and route not in kev_routes:
                    continue
                old_cell = payload_out["matrix"][route].get(bench)
                if args.resume and _cell_reusable(old_cell):
                    reused += 1
                    print(f"[{route:<19}] {bench:<9} skipped (reusing old result)")
                    continue
                if bench == "kev":
                    records, subsets = await run_kev_route(service, route, kev_plans, semaphore)
                    summary = summarize(records)
                    accuracies = [s["accuracy"] for s in subsets if s["accuracy"] is not None]
                    summary["mean_accuracy"] = (
                        round(sum(accuracies) / len(accuracies), 4) if accuracies else None
                    )
                    summary["subsets"] = subsets
                    overall = summary["overall"]
                    reasons = []
                    if overall and not overall["valid"]:
                        reasons.append(overall.get("validity_reason") or "overall failure rate too high")
                    bad_subsets = [s["name"] for s in subsets if s.get("valid") is False]
                    if bad_subsets:
                        reasons.append("invalid subsets: " + ", ".join(bad_subsets))
                    summary["valid"] = not reasons
                    if reasons:
                        summary["validity_reason"] = "; ".join(reasons)
                else:
                    items = datasets[bench]
                    records = list(await asyncio.gather(
                        *(run_one(service, route, bench, item, semaphore) for item in items)
                    ))
                    summary = summarize(records)
                    overall = summary["overall"]
                    summary["valid"] = overall["valid"] if overall else True
                    if overall and not overall["valid"]:
                        summary["validity_reason"] = overall["validity_reason"]
                payload_out["matrix"][route][bench] = summary
                payload_out["details"][f"{route}|{bench}"] = records
                overall = summary["overall"]
                if overall:
                    flag = "" if summary.get("valid") else "[!] "
                    extra = ""
                    if bench == "kev":
                        extra = f" | mean of 6 {_fmt(summary['mean_accuracy']).strip()}"
                    print(f"{flag}[{route:<19}] {bench:<9} {overall['correct']:>4}/{overall['n_valid']:<4} "
                          f"= {_fmt(overall['accuracy']).strip():>7}  valid {overall['n_valid']}/{overall['n']}  "
                          f"failures {overall['n_errors']}  p50={overall['latency_p50_ms']:.0f}ms{extra}")
                else:
                    print(f"[{route:<19}] {bench:<9} no items")
            out_path.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        for client in clients.values():
            await client.aclose()

    print_matrix(routes, benches, payload_out["matrix"])
    print(f"\ntotal elapsed {time.perf_counter() - started:.0f}s | result: {out_path}")
    if args.resume:
        print(f"resume: reused {reused} old cells")

    invalid_cells = [
        (route, bench, cell.get("validity_reason", ""))
        for route, cells in payload_out["matrix"].items()
        for bench, cell in cells.items()
        if isinstance(cell, dict) and cell.get("valid") is False
    ]
    if invalid_cells:
        print(f"\n[!] {len(invalid_cells)} invalid cells (failure rate > {ERROR_RATIO_THRESHOLD:.0%}); "
              f"results unusable:")
        for route, bench, reason in invalid_cells:
            print(f"[!]   {route} × {bench}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
