from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .labels import normalize_token

MIN_COVERAGE = 0.5
# Vendor placeholder that shows up in top_logprobs instead of a real probability
# (measured: DeepSeek returns -9999 for every token except the emitted one). Treated as
# "not observed", so it neither counts toward coverage nor enters the distribution.
SENTINEL_LOGPROB = -1000.0


class ReadoutError(RuntimeError):
    pass


@dataclass
class RawCandidate:
    token: str
    logprob: float


@dataclass
class Readout:
    logprobs: List[float]
    probabilities: List[float]
    coverage: float
    generated_token: str
    reliable: bool
    raw_candidates: List[RawCandidate] = field(default_factory=list)


def extract_content(response: dict) -> list:
    choices = response.get("choices") or []
    if not choices:
        raise ReadoutError("The Ark response contains no choices")
    content = (choices[0].get("logprobs") or {}).get("content")
    if not content:
        raise ReadoutError(
            "The model returned no logprobs.content: make sure the model supports logprobs/top_logprobs and is not in thinking mode"
        )
    return content


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    scaled = [value / temperature for value in logits]
    largest = max(scaled)
    exps = [math.exp(value - largest) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def read_distribution(
    content: Sequence[dict],
    handles: Sequence[str],
    lookup: Dict[str, List[int]],
    missing_floor: float = 3.0,
    min_coverage: float = MIN_COVERAGE,
) -> Readout:
    """Read out the probability distribution over candidate labels: only position 0 of the generated sequence.

    The prompt already stops right where "the answer should begin", so the distribution at position 0 is
    the model's raw judgment over the candidates — semantically the same as Jev's single forward pass at
    the answer position. Scanning further gives the conditional distribution "after a prefix has been
    written", which means something else, so we don't scan: if position 0 is not a candidate handle, raise
    instead of forcing a fit. Candidates that fell outside the top-k fall back to
    "lowest observed logprob - missing_floor"; when coverage is low, reliable=false.
    """
    if not content:
        raise ReadoutError("The model generated no token, so the distribution cannot be read")

    entry = content[0] or {}
    generated = entry.get("token", "")
    if normalize_token(generated) not in lookup:
        raise ReadoutError(f"The model did not output a candidate handle as its first token; actual output: {generated!r}")

    top = entry.get("top_logprobs") or []
    if not top:
        raise ReadoutError(f"Position 0 returned no top_logprobs (token={generated!r}), so the distribution cannot be read")

    table: Dict[str, RawCandidate] = {}
    for item in top:
        token = item.get("token", "")
        key = normalize_token(token)
        if not key:
            continue
        logprob = float(item.get("logprob", -1e9))
        if logprob <= SENTINEL_LOGPROB:
            # Some vendors pad top_logprobs with a placeholder instead of a probability
            # (measured: DeepSeek returns -9999 for every token except the emitted one).
            # A placeholder is not an observation, so it must not count toward coverage.
            continue
        candidate = RawCandidate(token=token, logprob=logprob)
        if key not in table or candidate.logprob > table[key].logprob:
            table[key] = candidate

    observed: Dict[int, RawCandidate] = {}
    for key, candidate in table.items():
        for index in lookup.get(key, ()):
            if index not in observed or candidate.logprob > observed[index].logprob:
                observed[index] = candidate
    if not observed:
        raise ReadoutError("No candidate handle appears in the top_logprobs at position 0")

    coverage = len(observed) / len(handles)
    floor = min(candidate.logprob for candidate in observed.values()) - missing_floor
    logprobs = [observed[i].logprob if i in observed else floor for i in range(len(handles))]
    return Readout(
        logprobs=logprobs,
        probabilities=softmax(logprobs),
        coverage=coverage,
        generated_token=generated,
        reliable=coverage >= min_coverage,
        raw_candidates=sorted(table.values(), key=lambda candidate: -candidate.logprob),
    )
