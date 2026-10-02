"""Temperature calibration: fit a scaling temperature for logprobs on labeled samples so the output probabilities approach the true frequencies.

Usage:
    python3 -m llm2decision.calibrate --data data/labels.jsonl --cache data/raw_logprobs.jsonl --write

Data format (JSONL, one sample per line):
    {"state": "The customer says they were charged twice", "question": {"type": "choice", "instructions": "pick the intent",
      "criteria": {"billing": "billing issue", "other": "other"}}, "gold": "billing"}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .core.config import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH, Config, ModelSettings
from .core.labels import build_lookup
from .core.readout import extract_content, read_distribution, softmax
from .core.schema import Question
from .llm.client import ChatClient
from .llm.service import NOUL_ALIASES, question_items
from .prompts import build_messages

_ECE_BINS = 10
NLL_EPSILON = 1e-6


def load_samples(path: Path) -> List[dict]:
    samples: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
            for key in ("state", "question", "gold"):
                if key not in sample:
                    raise ValueError(f"{path} line {line_number} is missing field {key}")
            samples.append(sample)
    if not samples:
        raise ValueError(f"{path} has no samples")
    return samples


async def collect(samples: Sequence[dict], settings: ModelSettings) -> List[dict]:
    client = ChatClient(settings)
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def one(sample: dict) -> dict:
        question = Question(**sample["question"])
        items = question_items(question)
        handles = [handle for handle, _, _ in items]
        aliases = NOUL_ALIASES if question.type == "noul" else None
        messages = build_messages(sample["state"], question, items, version=settings.prompt_version)
        async with semaphore:
            response = await client.chat_completions(messages, max_tokens=settings.max_tokens)
        readout = read_distribution(
            extract_content(response), handles, build_lookup(handles, aliases)
        )
        return {
            "type": question.type,
            "labels": [label for _, label, _ in items],
            "logprobs": readout.logprobs,
            "coverage": readout.coverage,
            "gold": sample["gold"],
        }

    try:
        return list(await asyncio.gather(*(one(sample) for sample in samples)))
    finally:
        await client.aclose()


def save_records(path: Path, records: Sequence[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_rows(records: Sequence[dict], question_type: Optional[str] = None) -> List[dict]:
    rows: List[dict] = []
    for record in records:
        if question_type and record["type"] != question_type:
            continue
        labels = list(record["labels"])
        if record["gold"] not in labels:
            raise ValueError(f"Gold {record['gold']!r} is not among the candidates {labels}; check the data")
        rows.append({"logprobs": list(record["logprobs"]), "gold_index": labels.index(record["gold"])})
    return rows


def mean_nll(rows: Sequence[dict], temperature: float) -> float:
    total = 0.0
    for row in rows:
        probabilities = softmax(row["logprobs"], temperature)
        total -= math.log(max(probabilities[row["gold_index"]], 1e-12))
    return total / len(rows)


def fit_temperature(rows: Sequence[dict], low: float = 0.05, high: float = 200.0, iterations: int = 200) -> float:
    """Golden-section search in log space. NLL is usually unimodal with respect to temperature."""
    inverse_phi = (math.sqrt(5) - 1) / 2
    left, right = math.log(low), math.log(high)
    inner_left = right - inverse_phi * (right - left)
    inner_right = left + inverse_phi * (right - left)
    for _ in range(iterations):
        if mean_nll(rows, math.exp(inner_left)) < mean_nll(rows, math.exp(inner_right)):
            right = inner_right
        else:
            left = inner_left
        inner_left = right - inverse_phi * (right - left)
        inner_right = left + inverse_phi * (right - left)
    return math.exp((left + right) / 2)


def is_degenerate(rows: Sequence[dict], temperature: float, low: float = 0.05, high: float = 200.0) -> bool:
    """Whether the temperature is unidentifiable: the search presses against a bound, or the fitted NLL has hit zero (the model was all-correct and overconfident on the samples)."""
    if temperature <= low * 1.01 or temperature >= high * 0.99:
        return True
    return mean_nll(rows, temperature) <= NLL_EPSILON



def evaluate(rows: Sequence[dict], temperature: float) -> Dict[str, float]:
    bins = [[0.0, 0, 0] for _ in range(_ECE_BINS)]
    correct = 0
    nll = 0.0
    brier = 0.0
    for row in rows:
        probabilities = softmax(row["logprobs"], temperature)
        gold = row["gold_index"]
        best = max(range(len(probabilities)), key=lambda index: probabilities[index])
        correct += int(best == gold)
        nll -= math.log(max(probabilities[gold], 1e-12))
        brier += sum(
            (probability - (1.0 if index == gold else 0.0)) ** 2
            for index, probability in enumerate(probabilities)
        )
        confidence = probabilities[best]
        bucket = min(int(confidence * _ECE_BINS), _ECE_BINS - 1)
        bins[bucket][0] += confidence
        bins[bucket][1] += int(best == gold)
        bins[bucket][2] += 1

    total = len(rows)
    ece = sum(
        abs(confidence_sum / count - correct_sum / count) * count / total
        for confidence_sum, correct_sum, count in bins
        if count
    )
    return {
        "accuracy": correct / total,
        "nll": nll / total,
        "brier": brier / total,
        "ece": ece,
    }


def write_temperature(path: Path, route: str, value: float) -> None:
    """Write the temperature into the models.<route> entry of llm2decision.yaml (preserving comments and other content)."""
    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()

    route_index = None
    route_indent = 0
    inside_models = False
    for index, line in enumerate(lines):
        if re.match(r"^models:\s*(#.*)?$", line):
            inside_models = True
            continue
        if not inside_models:
            continue
        if line.strip() and not line.startswith((" ", "\t")):
            break
        match = re.match(r"^(\s+)([^:#\s][^:]*):\s*(#.*)?$", line)
        if match and len(match.group(1)) == 2 and match.group(2).strip() == route:
            route_index, route_indent = index, len(match.group(1))
            break
    if route_index is None:
        raise ValueError(f"No route {route!r} found in the models section of {target}; cannot write back")

    entry = f"{' ' * (route_indent + 2)}temperature_scale: {value:.6f}"
    for index in range(route_index + 1, len(lines)):
        line = lines[index]
        if line.strip() and len(line) - len(line.lstrip()) <= route_indent:
            lines.insert(index, entry)
            break
        if re.match(r"^\s*temperature_scale:", line):
            lines[index] = entry
            break
    else:
        lines.insert(route_index + 1, entry)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit logprob temperature-calibration parameters")
    parser.add_argument("--data", required=True, help="Labeled samples JSONL")
    parser.add_argument("--cache", help="Raw logprobs cache file; if it already exists, reuse it instead of calling the API")
    parser.add_argument("--model", help="Model route name; defaults to default_model")
    parser.add_argument("--config", help="Config file path; defaults to llm2decision.yaml")
    parser.add_argument("--write", action="store_true", help="Write the fitted temperature back into that route's entry")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    samples = load_samples(args.data)
    config = Config.load(args.config)
    route = args.model or config.default_model
    settings = config.resolve(route)

    if args.cache and Path(args.cache).exists():
        records = load_records(args.cache)
        print(f"Reusing cache: {args.cache} ({len(records)} records)")
    else:
        print(f"Calling model: {settings.model} (route {route}), {len(samples)} samples in total")
        records = asyncio.run(collect(samples, settings))
        if args.cache:
            save_records(args.cache, records)
            print(f"Raw logprobs cached to: {args.cache}")

    rows = to_rows(records)
    if not rows:
        print("No samples available for fitting")
        return 1

    fitted = fit_temperature(rows)
    print(f"\nSamples: {len(rows)} (" + ", ".join(
        f"{kind}={len(to_rows(records, kind))}" for kind in sorted({record['type'] for record in records})
    ) + ")")
    print(f"{'Temp':<12}{'Accuracy':<10}{'NLL':<10}{'Brier':<10}{'ECE':<10}")
    for label, temperature in (("Raw", 1.0), ("Fitted", fitted)):
        metrics = evaluate(rows, temperature)
        print(
            f"{label} T={temperature:<7.4f}{metrics['accuracy']:<10.4f}{metrics['nll']:<10.4f}"
            f"{metrics['brier']:<10.4f}{metrics['ece']:<10.4f}"
        )

    for kind in sorted({record["type"] for record in records}):
        type_rows = to_rows(records, kind)
        if type_rows:
            type_temperature = fit_temperature(type_rows)
            flag = "  <-- unidentifiable: too few samples, or this subset is almost all-correct" if is_degenerate(type_rows, type_temperature) else ""
            print(f"Best temperature per question type: {kind}={type_temperature:.4f}{flag}")

    if is_degenerate(rows, fitted):
        print(
            "\nNote: the fitted temperature is unidentifiable (the search hit a bound, or the fitted NLL has "
            "already hit zero). That means there are too few samples, or the model is almost all-correct on "
            "these samples, so NLL just keeps moving toward a sharper fit and the result is an artifact. Add a "
            "few hundred samples, including hard ones the model gets wrong, before using it."
        )

    if args.write:
        path = Path(args.config or os.getenv(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH)
        write_temperature(path, route, fitted)
        print(f"\nWrote models.{route}: temperature_scale: {fitted:.6f} in {path}")
    else:
        print(f"\nAdd the line below to models.{route} to take effect:  temperature_scale: {fitted:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
