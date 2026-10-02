"""Run VitaminC (fact verification, 3-way choice): decide SUPPORTS / REFUTES / NOT ENOUGH INFO from evidence + claim.

Note on the comparison basis: both Nimble's PUBLIC_BENCHMARKS.md and the NeoHorse-Jev-4B model
card from TokenRhythm publish results on **the same 599-record vitaminc-dev subset**
(jev-1.13.0 80.1%, 95% CI 76.8%–83.1%). The default `--subset nimble599` reproduces these 599
records exactly (the id list comes from a local manifest) and can be compared directly against
the published 80.1%; `--subset sample300` keeps the old "draw 300 with a fixed seed" behavior for
historical comparison, and the two are not the same set of questions.

Usage:
    python3 benchmarks/run_vitaminc.py                      # default nimble599 (599 records)
    python3 benchmarks/run_vitaminc.py --subset sample300 --n 300   # old basis
    python3 benchmarks/run_vitaminc.py --logit-bias
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import pathlib
import random
import sys
import time
from collections import Counter

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from llm2decision.llm.client import ChatClient  # noqa: E402
from llm2decision.core.config import Config  # noqa: E402
from llm2decision.core.schema import SystemOneRequest  # noqa: E402
from llm2decision.llm.service import SystemOneService  # noqa: E402
from run_jevbench import summarize  # noqa: E402

RESULT_DIR = pathlib.Path(__file__).resolve().parent / "results"
MANIFEST = (
    pathlib.Path(__file__).resolve().parent
    / "nimble" / "docs" / "assets" / "public-benchmarks" / "subsets" / "vitaminc-dev-manifest.json"
)
ID_PREFIX = "vitaminc-"

CRITERIA = {
    "SUPPORTS": "The evidence confirms the claim.",
    "REFUTES": "The evidence contradicts the claim.",
    "NOT ENOUGH INFO": "The evidence neither confirms nor contradicts the claim.",
}
SEED = 20260918
SUBSETS = ("nimble599", "sample300")


def load_items(n: int, seed: int = SEED):
    """Old basis: draw n records from the validation split with a fixed seed."""
    from datasets import load_dataset

    dataset = load_dataset("tals/vitaminc", split="validation")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    items = []
    for index in indices[:n]:
        row = dataset[index]
        items.append(
            {
                "id": f"vitaminc-{index}",
                "evidence": row["evidence"],
                "claim": row["claim"],
                "gold": row["label"],
            }
        )
    return items


def load_manifest_items():
    """Reproduce Nimble/TokenRhythm's 599-record vitaminc-dev subset exactly.

    Ids in the manifest look like `vitaminc-<unique_id>`; stripping the prefix gives the
    `unique_id` field of the HF dataset tals/vitaminc, which is used to fetch the exact rows
    from the validation split. Returns (items, missing), where missing holds the unique_ids not
    found in the HF split (the upstream zip and the HF version may differ).
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = [
        item_id[len(ID_PREFIX):] if item_id.startswith(ID_PREFIX) else item_id
        for item_id in manifest["ids"]
    ]
    from datasets import load_dataset

    dataset = load_dataset("tals/vitaminc", split="validation")
    index = {uid: position for position, uid in enumerate(dataset["unique_id"])}
    items, missing = [], []
    for uid in wanted:
        position = index.get(uid)
        if position is None:
            missing.append(uid)
            continue
        row = dataset[position]
        items.append(
            {
                "id": f"{ID_PREFIX}{uid}",
                "evidence": row["evidence"],
                "claim": row["claim"],
                "gold": row["label"],
            }
        )
    return items, missing


def to_payload(item: dict) -> dict:
    return {
        "state": f"[Evidence]\n{item['evidence']}\n\n[Claim]\n{item['claim']}",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Is the claim supported by the evidence?",
                "criteria": dict(CRITERIA),
            }
        },
    }


async def run_item(service, item, semaphore, route: str) -> dict:
    async with semaphore:
        started = time.perf_counter()
        try:
            # The service resolves the route from the request body; without "model" here it would
            # fall back to default_model, which is not in this single-route client map (KeyError).
            response = await service.decide(SystemOneRequest(**{**to_payload(item), "model": route}))
            decision = response.decisions["q"]
            record = {
                "predicted": decision.label,
                "correct": decision.label == item["gold"],
                "coverage": decision.coverage,
                "reliable": decision.reliable,
                "distribution": decision.distribution,
                "error": None,
            }
        except Exception as exc:
            record = {
                "predicted": None,
                "correct": False,
                "coverage": None,
                "reliable": None,
                "distribution": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    record.update(
        {
            "id": item["id"],
            "tier": "validation",
            "family": item["gold"],
            "type": "choice",
            "expected": item["gold"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    return record


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a VitaminC validation subset")
    parser.add_argument("--subset", choices=SUBSETS, default="nimble599",
                        help="nimble599 = reproduce the 599-record subset exactly (default); sample300 = old fixed-seed draw")
    parser.add_argument("--n", type=int, default=300, help="only applies to sample300")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N records (for smoke testing)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--logit-bias", action="store_true")
    parser.add_argument("--model", help="model route name; defaults to default_model")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    config = Config.load()
    route = args.model or config.default_model
    settings = config.resolve(route)
    if args.logit_bias:
        settings = dataclasses.replace(settings, logit_bias_enabled=True)

    missing: list = []
    if args.subset == "nimble599":
        manifest_count = len(json.loads(MANIFEST.read_text(encoding="utf-8"))["ids"])
        items, missing = load_manifest_items()
        print(f"subset nimble599: manifest has {manifest_count} ids, HF validation actually matched {len(items)} records")
        if len(items) != manifest_count:
            print(f"  ⚠ {len(missing)} ids unmatched (the HF split and the upstream zip may differ)")
            print(f"  unmatched examples: {missing[:10]}")
    else:
        items = load_items(args.n)
    if args.limit:
        items = items[: args.limit]
    print(f"model {settings.model} | VitaminC {args.subset} {len(items)} records | logit_bias={settings.logit_bias_enabled}")
    print("label distribution:", dict(Counter(item["gold"] for item in items)))

    client = ChatClient(settings)
    service = SystemOneService({route: client}, config)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    try:
        records = list(await asyncio.gather(*(run_item(service, item, semaphore, route) for item in items)))
    finally:
        await client.aclose()

    summary = summarize(records)
    summary["model"] = settings.model
    summary["subset"] = args.subset
    if args.subset == "nimble599":
        summary["dataset"] = (
            f"VitaminC validation nimble599 subset, manifest 599 ids, {len(items)} matched"
        )
        summary["missing_ids"] = missing
    else:
        summary["dataset"] = f"VitaminC validation, seed={SEED}, sample300, n={len(items)}"
    summary["logit_bias_enabled"] = settings.logit_bias_enabled
    summary["published_jev_1_13_0"] = "80.1% (95% CI 76.8%–83.1%), same basis: 599-record vitaminc-dev subset"
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    detail_path = RESULT_DIR / f"vitaminc-{stamp}{tag}.jsonl"
    summary_path = RESULT_DIR / f"vitaminc-{stamp}{tag}.summary.json"
    with detail_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print(f"\nVitaminC: {overall['correct']}/{overall['n']} = {overall['accuracy']:.2%}"
          f" ({overall['errors']} failures)")
    for label in CRITERIA:
        subset = [r for r in records if r["expected"] == label]
        if subset:
            hits = sum(1 for r in subset if r["correct"])
            print(f"  {label:<16} {hits:>3}/{len(subset):<3} = {hits / len(subset):.2%}")
    print(f"  p50={overall['latency_p50_ms']}ms | total {summary['elapsed_s']}s")
    print(f"  comparison (published by Nimble): jev-1.13.0 80.1% on the hand-picked 599 records")
    print(f"\ndetails: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
