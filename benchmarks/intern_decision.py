"""Adapter for the Intern-Decision evaluation bundles (accuracy-v1 + the known-distribution pilot).

Source: https://github.com/InternLM/Intern-Decision — `benchmarks/accuracy-v1/` and
`benchmarks/known-distribution-pilot-v1/`. Repository code is Apache-2.0; each dataset keeps its
upstream terms (AG News upstream metadata reports an unknown license — consult upstream before any
redistribution). Both bundles are evaluation-only: the upstream protocol forbids fitting a
temperature or selecting a checkpoint on them, and this adapter only scores them.

The records already use the same "state + typed questions" shape this service accepts, so the
adapter mirrors run_jevbench.to_payload: `choice` passes `criteria` through, `noul` folds the
true/false rubric into the statement, `score` maps the level list onto `scale` + `values`.
Accuracy is counted per decision; the seven-suite average is the unweighted mean of the seven
displayed suites, matching the upstream table.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from run_jevbench import is_correct  # noqa: E402,F401  (same answer-comparison rule)

DATA_DIR = pathlib.Path(__file__).resolve().parent / "intern-decision"
ACCURACY_DIR = DATA_DIR / "accuracy-v1"
PILOT_DIR = DATA_DIR / "calibration-pilot-v1"

# The four suites beyond JevBench; JevBench itself is loaded from the upstream repository this
# project already uses (the item ids are identical to the Intern-Decision bundle's copies).
ACCURACY_SUITES = ("agnews", "toolace", "typed_decisions", "wildjailbreak")
SEVEN_SUITES = ("easy", "original", "hard", "typed_decisions", "toolace", "agnews", "wildjailbreak")

DATA_HINT = (
    f"Cannot find the Intern-Decision bundles (expected under: {DATA_DIR}).\n"
    "These datasets are not checked into this repository; fetch the evaluation bundles from upstream\n"
    "(then verify the sha256 values against benchmarks/accuracy-v1/manifest.json):\n"
    "    mkdir -p benchmarks/intern-decision/accuracy-v1/{agnews,toolace,typed_decisions,wildjailbreak}\n"
    "    g='repos/InternLM/Intern-Decision/contents/benchmarks/accuracy-v1'\n"
    "    for s in agnews toolace typed_decisions wildjailbreak; do\n"
    "      gh api -H 'Accept: application/vnd.github.raw' \"$g/$s/test.jsonl\" "
    "> \"benchmarks/intern-decision/accuracy-v1/$s/test.jsonl\"\n"
    "    done\n"
    "    gh api -H 'Accept: application/vnd.github.raw' \"$g/manifest.json\" "
    "> benchmarks/intern-decision/accuracy-v1/manifest.json\n"
    "    # plus the calibration pilot, into benchmarks/intern-decision/calibration-pilot-v1/"
)


def _read_jsonl(path: pathlib.Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(DATA_HINT)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_suite(suite: str) -> List[dict]:
    """Load one accuracy suite as raw records."""
    path = ACCURACY_DIR / suite / "test.jsonl"
    return _read_jsonl(path)


def load_pilot() -> Tuple[List[dict], Dict[str, dict]]:
    """Load the 96 calibration-pilot inputs and their exact reference distributions, keyed by id."""
    inputs = _read_jsonl(PILOT_DIR / "inputs.jsonl")
    references = {row["id"]: row for row in _read_jsonl(PILOT_DIR / "references.jsonl")}
    return inputs, references


def _stringify_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def adapt_question(question: dict) -> dict:
    """Convert one upstream question into this service's question shape.

    The Chinese rubric prefixes are the same conventions run_jevbench uses, so every number this
    project publishes is measured with one consistent prompt rendering.
    """
    qtype = question["type"]
    instructions = question.get("instructions", "") or ""
    criteria = question.get("criteria")

    if qtype == "choice":
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria or {})}

    if qtype == "noul":
        rubric = ""
        if isinstance(criteria, dict):
            rubric = "\n成立与不成立的标准：" + "；".join(f"{key}：{value}" for key, value in criteria.items())
        return {"type": "noul", "instructions": instructions + rubric}

    # score: upstream ships the level descriptions as a list; the level number is the label
    levels = ""
    if isinstance(criteria, list) and criteria:
        levels = "\n档位定义：" + "；".join(f"{index}={text}" for index, text in enumerate(criteria))
    labels = [str(index) for index in range(len(criteria or []))]
    return {
        "type": "score",
        "instructions": instructions + levels,
        "scale": labels,
        "values": [float(label) for label in labels],
    }


def to_request(record: dict) -> dict:
    """Build the service request body for one record (all of its questions in one request)."""
    questions = {name: adapt_question(q) for name, q in record["questions"].items()}
    return {"state": _stringify_state(record["state"]), "questions": questions}


def predicted_and_correct(qtype: str, gold: str, decision) -> Tuple[Optional[str], bool]:
    """Return (predicted label, correct) for one decision, using the upstream label space."""
    if qtype == "noul":
        predicted = None if decision.true_probability is None else ("yes" if decision.true_probability >= 0.5 else "no")
        return predicted, predicted is not None and is_correct(gold, predicted)
    predicted = decision.label
    return predicted, predicted is not None and is_correct(gold, predicted)


def decision_distribution(decision) -> Optional[Dict[str, float]]:
    dist = getattr(decision, "distribution", None)
    if not dist:
        return None
    return {str(label): float(prob) for label, prob in dist.items()}


def brier_multiclass(distribution: Dict[str, float], gold: str) -> float:
    """Upstream definition: sum over labels of (p - 1[label == gold])**2."""
    return sum((prob - (1.0 if label == gold else 0.0)) ** 2 for label, prob in distribution.items())


def expected_brier(distribution: Dict[str, float], reference: Dict[str, float]) -> float:
    """Pilot definition: multiclass sum against the exact reference distribution (missing labels count as 0)."""
    labels = set(distribution) | set(reference)
    return sum((distribution.get(label, 0.0) - reference.get(label, 0.0)) ** 2 for label in labels)


def binned_ece(pairs: List[Tuple[float, float]], bins: int = 10) -> Optional[float]:
    """Equal-width ECE over (confidence, correctness) pairs; correctness may be fractional (pilot)."""
    if not pairs:
        return None
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [p for p in pairs if (low <= p[0] < high) or (index == bins - 1 and p[0] == 1.0)]
        if not bucket:
            continue
        mean_conf = sum(p[0] for p in bucket) / len(bucket)
        mean_correct = sum(p[1] for p in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_correct - mean_conf)
    return error


def ece_from_records(records: List[dict], bins: int = 10) -> Optional[float]:
    """Hard-label ECE: confidence is max(P) over the returned distribution (upstream's hard-table rule)."""
    pairs: List[Tuple[float, float]] = []
    for record in records:
        distribution = record.get("distribution")
        if not distribution or record.get("error"):
            continue
        pairs.append((max(distribution.values()), 1.0 if record["correct"] else 0.0))
    return binned_ece(pairs, bins)


def pilot_metrics(results: List[dict], references: Dict[str, dict]) -> Dict[str, Optional[float]]:
    """Expected Brier / expected ECE against the exact reference distributions.

    `results` are flattened decisions (one per pilot case). ECE uses the returned confidence and
    the reference probability of the predicted outcome as the expected correctness, pooled over
    all cases (upstream's pilot rule).
    """
    briers: List[float] = []
    pairs: List[Tuple[float, float]] = []
    for record in results:
        reference = references.get(record.get("row_id") or record.get("id"))
        distribution = record.get("distribution")
        if reference is None or not distribution or record.get("error"):
            continue
        gold_probs = {str(label): float(prob) for label, prob in reference["gold_probs"].items()}
        briers.append(expected_brier(distribution, gold_probs))
        confidence = max(distribution.values())
        expected_correct = gold_probs.get(str(record["predicted"]), 0.0)
        pairs.append((confidence, expected_correct))
    return {
        "scored": len(briers),
        "expected_brier": round(sum(briers) / len(briers), 6) if briers else None,
        "expected_ece": round(binned_ece(pairs), 6) if pairs else None,
    }


def seven_suite_average(suite_accuracy: Dict[str, Optional[float]]) -> Optional[float]:
    values = [suite_accuracy[name] for name in SEVEN_SUITES if suite_accuracy.get(name) is not None]
    if len(values) != len(SEVEN_SUITES):
        return None
    return round(sum(values) / len(values), 4)