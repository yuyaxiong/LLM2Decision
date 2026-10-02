from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator

from .labels import MAX_CANDIDATES

# Structural cap: the physical limit of the handle scheme (digits for 1-9, letters for 10-20, see
# core/labels.py). This only blocks clearly invalid input; the **per-vendor capability cap** (Ark 10 /
# DashScope 4) is validated at the service layer against the effective route, because the Pydantic layer
# can't see which route a request finally lands on.
MAX_STRUCTURAL_CANDIDATES = MAX_CANDIDATES

_TOO_MANY = (
    f"The candidate count cannot exceed {MAX_STRUCTURAL_CANDIDATES}: that is already the physical limit of the label-handle scheme. "
    "The actual usable cap depends on the route's vendor (Ark 10 / DashScope 4) and is validated at the service layer against the route."
)


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = ""
    criteria: Optional[Dict[str, str]] = None
    scale: Optional[List[str]] = None
    values: Optional[List[float]] = None

    @model_validator(mode="after")
    def _validate(self) -> "Question":
        if self.type == "choice":
            if not self.criteria:
                raise ValueError("choice questions must provide non-empty criteria")
            if len(self.criteria) > MAX_STRUCTURAL_CANDIDATES:
                raise ValueError(f"choice {_TOO_MANY}")
        elif self.type == "noul":
            if not self.instructions.strip():
                raise ValueError("noul questions must provide non-empty instructions (the statement to judge)")
        else:
            if not self.scale or len(self.scale) < 2:
                raise ValueError("score questions must provide a scale with at least 2 levels")
            if len(self.scale) > MAX_STRUCTURAL_CANDIDATES:
                raise ValueError(f"score {_TOO_MANY}")
            if self.values is not None and len(self.values) != len(self.scale):
                raise ValueError("values must have the same length as scale")
        return self


class SystemOneRequest(BaseModel):
    state: str
    questions: Dict[str, Question]
    model: Optional[str] = None
    debug: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "SystemOneRequest":
        if not self.state.strip():
            raise ValueError("state cannot be empty")
        if not self.questions:
            raise ValueError("questions cannot be empty")
        return self


class CandidateLogprob(BaseModel):
    token: str
    logprob: float


class Timing(BaseModel):
    """Timing breakdown for a single question (milliseconds).

    Excludes concurrency queue wait — that never appears here, but shows up in the difference between
    the response-level `latency_ms` (end-to-end wall time) and the sum of this question's three
    segments. To tell "slow model" from "long queue", use that difference.

    `prepare_ms` includes one possible online tokenization request (to resolve the token ids for
    logit_bias); that result is cached by (base_url, model, handle), so later requests in the same
    process measure ≈0 here.
    """

    prepare_ms: float
    call_ms: float
    readout_ms: float


class Decision(BaseModel):
    type: str
    distribution: Optional[Dict[str, float]] = None
    label: Optional[str] = None
    confidence: Optional[float] = None
    true_probability: Optional[float] = None
    decision: Optional[bool] = None
    expected: Optional[float] = None
    coverage: float
    reliable: bool
    generated_token: Optional[str] = None
    raw_candidates: Optional[List[CandidateLogprob]] = None
    timing: Optional[Timing] = None
    # Model calls actually issued for this question (1 with the current strategy; a per-candidate strategy would be the candidate count)
    calls: Optional[int] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class SystemOneResponse(BaseModel):
    decisions: Dict[str, Decision]
    model: str
    route: Optional[str] = None
    latency_ms: float
    # Total model calls issued for this request (summed over all questions). latency_ms is end-to-end
    # wall time, and on its own it can't tell "slow model" from "many calls", so read it together with this number.
    calls: int = 0
    usage: Optional[Usage] = None
