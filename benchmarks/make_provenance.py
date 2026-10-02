"""Generate reviewable provenance for the evaluation: dataset hashes / evaluation protocol / run hashes.

Design goal (following the frozen-evaluation practice of zhihz/openjev): any published score can be
re-verified by a third party along the chain "dataset → subset selection rule → evaluation protocol →
run artifacts", instead of being a bare number.

Usage:
    python3 benchmarks/make_provenance.py                     # compute datasets and protocol only
    python3 benchmarks/make_provenance.py --run <matrix.json>  # also hash a specific run and its item sets

Output:
    benchmarks/provenance.json   machine-readable
    a markdown table printed to the terminal, ready to paste into REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

KEV_DIR = BENCH / "kev"
KEV_FILES = [
    "decision-v7-dev.jsonl", "decision-v7-test.jsonl",
    "transfer-v4-dev.jsonl", "transfer-v4-test.jsonl",
    "transfer-v9-dev.jsonl", "transfer-v9-test.jsonl",
]

# Per benchmark: data source, subset selection rule, evaluation protocol (who scores, and what)
PROTOCOLS: Dict[str, dict] = {
    "jevbench": {
        "source": "https://github.com/fstandhartinger/jevbench (MIT)",
        "subset": "public 231 = original 72 + easy 48 + hard 111, all of it, no sampling",
        "protocol": "three types choice/noul/score; per-question hard-label argmax; noul uses P(yes) ≥0.5; score uses the argmax level",
        "data_files": [
            "benchmarks/jevbench/datasets/public/original.jsonl",
            "benchmarks/jevbench/datasets/public/easy.jsonl",
            "benchmarks/jevbench/datasets/public/hard.jsonl",
        ],
        "extra_files": ["benchmarks/jevbench/datasets/manifest.json"],
    },
    "nimble": {
        "source": "https://github.com/bespokelabsai/nimble, data/eval.jsonl (324 synthetic holdout items, no human review)",
        "subset": "compat282: state ≤384 tokens, encoded request body ≤2048 tokens, ≤26 candidates, whole family kept (280 items measured)",
        "protocol": "per-sample decision exact match; choice uses the argmax option, noul uses P(yes) ≥0.5, score uses the argmax level",
        "data_files": ["benchmarks/nimble/data/eval.jsonl"],
        "extra_files": ["benchmarks/nimble/data/manifest.json"],
    },
    "vitaminc": {
        "source": "HuggingFace datasets `tals/vitaminc` (validation split, CC BY-SA 3.0)",
        "subset": "nimble599: the 599 ids recorded in the Nimble manifest, matched row by unique_id (599/599 match)",
        "protocol": "3-way choice (SUPPORTS / REFUTES / NOT ENOUGH INFO); per-question hard-label argmax",
        "data_files": [],
        "extra_files": [
            "benchmarks/nimble/docs/assets/public-benchmarks/subsets/vitaminc-dev-manifest.json"
        ],
    },
    "kev": {
        "source": "https://github.com/jaredpalmer/kev (Apache-2.0) evals/{v7,v4,v9}",
        "subset": "all 6 subsets, only `_meta.variant == \"clean\"`; questions with >10 candidates skipped and counted (not recorded as wrong)",
        "protocol": "per-question hard-label argmax; equally weighted mean of the six clean accuracies",
        "data_files": [f"benchmarks/kev/{name}" for name in KEV_FILES],
        "extra_files": [],
    },
}

# Implementation scope of this run (changes over time and must be recorded alongside the results)
RUNTIME = {
    "engine": "POST /v1/systemone (label-logit readout)",
    "decoding": "temperature=0 (greedy), max_tokens=1, reading only generation position 0",
    "candidate_cap": 10,
    "calibration": "no temperature calibration (temperature_scale=1.0); accuracy judged by argmax only",
    "scoring": "hard-label comparison, no LLM-as-judge, retries never count toward the score (connection errors retried up to 3 times)",
    "counts_as_wrong": "a model that does not answer in the expected format (refusal) is recorded as wrong; connection failures are tracked separately and mark the cell invalid when the failure rate is >20%",
}


def sha256_file(path: pathlib.Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(items: List[str]) -> str:
    payload = "\n".join(sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint() -> dict:
    """This repo has no git, so the implementation version is identified by a content hash of the sources."""
    files = sorted(list((ROOT / "src" / "llm2decision").rglob("*.py")) + list(BENCH.glob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {"files": [str(p.relative_to(ROOT)) for p in files], "sha256": digest.hexdigest()}


def dataset_hashes() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for bench, spec in PROTOCOLS.items():
        files = {}
        for rel in spec["data_files"] + spec["extra_files"]:
            path = ROOT / rel
            files[rel] = sha256_file(path) or "missing"
        out[bench] = {"source": spec["source"], "subset": spec["subset"],
                      "protocol": spec["protocol"], "files": files}
    return out


def run_hashes(run_path: pathlib.Path) -> dict:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    details = data.get("details", {})
    evaluated: Dict[str, dict] = {}
    for bench in data.get("benches", []):
        for route in data.get("routes", []):
            records = details.get(f"{route}|{bench}")
            if not records:
                continue
            ids = [r["id"] for r in records]
            evaluated[bench] = {"n_items": len(ids), "ids_sha256": sha256_lines(ids),
                                "id_source": f"{route}|{bench}"}
            break
    return {
        "run_file": str(run_path.relative_to(ROOT)),
        "run_file_sha256": sha256_file(run_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "routes": data.get("routes"),
        "kev_routes": data.get("kev_routes"),
        "counts": data.get("counts"),
        "planned_calls": data.get("planned_calls"),
        "evaluated_sets": evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="generate reviewable evaluation provenance")
    parser.add_argument("--run", help="path to a matrix-*.json from some run")
    args = parser.parse_args()

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": RUNTIME,
        "implementation": code_fingerprint(),
        "benchmarks": dataset_hashes(),
    }
    run_info = None
    if args.run:
        run_path = pathlib.Path(args.run)
        if not run_path.is_absolute():
            run_path = ROOT / run_path
        run_info = run_hashes(run_path)
        payload["run"] = run_info

    out_path = BENCH / "provenance.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| Item | Value |")
    print("|---|---|")
    print(f"| Implementation fingerprint (src/llm2decision/ + benchmarks/ sources) | `{payload['implementation']['sha256'][:16]}…` ({len(payload['implementation']['files'])} files) |")
    for bench, info in payload["benchmarks"].items():
        digest = info["files"].get("", "")
        keys = list(info["files"].items())
        first = keys[0][1] if keys else ""
        print(f"| {bench} dataset hash | `{first[:16]}…` ({len(keys)} files) |")
    if run_info:
        print(f"| run artifact {run_info['run_file']} | `{run_info['run_file_sha256'][:16]}…` |")
        for bench, info in run_info["evaluated_sets"].items():
            print(f"| {bench} evaluated item set | {info['n_items']} items, id-list hash `{info['ids_sha256'][:16]}…` |")
    print(f"\nmachine-readable: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
