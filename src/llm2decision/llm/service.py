from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .client import ChatClient
from ..core.config import Config, ModelSettings, UnknownModelError
from ..core.labels import build_lookup, make_handles
from ..core.providers import get_profile
from ..core.readout import Readout, extract_content, read_distribution, softmax
from ..core.schema import (
    CandidateLogprob,
    Decision,
    Question,
    SystemOneRequest,
    SystemOneResponse,
    Timing,
    Usage,
)
from ..prompts import build_messages

NOUL_HANDLES = ["1", "2"]
NOUL_ALIASES = {
    "1": ["Y", "YES", "TRUE", "T", "是", "TRUE.", "对"],
    "2": ["N", "NO", "FALSE", "F", "否", "FALSE.", "不"],
}

logger = logging.getLogger(__name__)


class TooManyCandidatesError(ValueError):
    """The candidate count exceeds the capability cap of the route's vendor.

    The cap depends on the vendor's top_logprobs slots: Ark's 20 slots reliably carry 10 candidates;
    DashScope has only 5 slots, and 4 candidates is already the measured cap (6 candidates gave 0/4
    coverage across all 6 models). This check lives in the service layer rather than the Pydantic layer,
    because only after resolving the route can we know which vendor is in play.
    """

    def __init__(self, question: str, count: int, profile) -> None:
        super().__init__(
            f"Question {question!r} has {count} candidates, over the cap {profile.max_candidates} "
            f"of {profile.display_name} (that vendor's top_logprobs goes up to {profile.max_top_logprobs}, "
            "and the slots must also hold non-candidate tokens such as EOS, full-width variants, and punctuation). "
            "Reduce the candidates, split into multiple questions, or switch to a route with a higher candidate cap."
        )
        self.question = question
        self.count = count
        self.max_candidates = profile.max_candidates


def candidate_count(question: Question) -> int:
    """The user-specified candidate count. noul always has two handles, never over the cap, so return 0 to skip the check."""
    if question.type == "choice":
        return len(question.criteria or {})
    if question.type == "score":
        return len(question.scale or [])
    return 0

# (base_url|model, handle) -> token id; handles that are not single-token are cached as None.
# The cache key includes base_url and model: different routes may point at different vendors/tokenizers and must not share entries.
_HANDLE_TOKEN_IDS: Dict[Tuple[str, str], Optional[int]] = {}


async def resolve_handle_token_ids(client, cache_key: str, handles: Sequence[str]) -> Dict[str, int]:
    """handle -> token id, returning only single-token handles; results are cached by (cache_key, handle) to avoid re-tokenizing."""
    resolved: Dict[str, int] = {}
    for handle in handles:
        key = (cache_key, handle)
        if key not in _HANDLE_TOKEN_IDS:
            token_ids = await client.tokenize(handle)
            _HANDLE_TOKEN_IDS[key] = token_ids[0] if len(token_ids) == 1 else None
        token_id = _HANDLE_TOKEN_IDS[key]
        if token_id is not None:
            resolved[handle] = token_id
    return resolved


class SystemOneService:
    """Dispatch a request to the matching model route by route name."""

    def __init__(self, clients: Dict[str, ChatClient], config: Config) -> None:
        self._clients = clients
        self._config = config
        self._semaphores = {
            name: asyncio.Semaphore(settings.max_concurrency)
            for name, settings in config.models.items()
        }

    def route_of(self, requested: Optional[str] = None) -> Tuple[str, ModelSettings]:
        name = requested or self._config.default_model
        if name not in self._config.models:
            raise UnknownModelError(name, self._config.model_names())
        return name, self._config.models[name]

    async def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        route, settings = self.route_of(request.model)
        profile = get_profile(settings.provider)
        # The candidate cap follows the vendor (Ark 10 / DashScope 4) and is enforced before any model call
        for name, question in request.questions.items():
            count = candidate_count(question)
            if count > profile.max_candidates:
                raise TooManyCandidatesError(name, count, profile)
        client = self._clients[route]
        semaphore = self._semaphores[route]
        started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(
                self._decide_one(client, settings, semaphore, name, question, request.state, request.debug)
                for name, question in request.questions.items()
            )
        )
        return SystemOneResponse(
            decisions={name: decision for name, decision, _, _ in outcomes},
            model=settings.model,
            route=route,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            calls=sum(calls for _, _, _, calls in outcomes),
            usage=_merge_usage([usage for _, _, usage, _ in outcomes]),
        )

    async def _logit_bias_for(
        self, client: ChatClient, settings: ModelSettings, handles: Sequence[str]
    ) -> Optional[dict]:
        """Fallback: push all candidate handles up together to raise the rate at which "the first token lands on a candidate".

        The bias uses the same strength for every candidate, so it cancels out automatically when
        renormalizing within the candidate set and does not change the readout's relative probabilities
        (measured). If any single candidate can't get a single-token id, abandon it entirely, to avoid an
        inconsistent distribution where some candidates are biased and others aren't.
        """
        if not settings.logit_bias_enabled:
            return None
        profile = get_profile(settings.provider)
        if not profile.supports_tokenize:
            # DashScope has no online tokenizer, so no token ids and this fallback must be dropped.
            # Measured: 60 questions with no bias also had 0 readout failures, but that safety net is gone.
            logger.warning(
                "%s has no online tokenizer, so this route cannot apply logit_bias (no format fallback): %s",
                profile.display_name,
                list(handles),
            )
            return None
        cache_key = f"{settings.base_url}|{settings.model}"
        token_ids = await resolve_handle_token_ids(client, cache_key, handles)
        if len(token_ids) != len(handles):
            logger.warning("Some candidate handles are not single-token; abandoning logit_bias: %s", list(handles))
            return None
        return {str(token_id): settings.logit_bias_strength for token_id in set(token_ids.values())}

    async def _decide_one(
        self,
        client: ChatClient,
        settings: ModelSettings,
        semaphore: asyncio.Semaphore,
        name: str,
        question: Question,
        state: str,
        debug: bool,
    ) -> Tuple[str, Decision, Optional[dict], int]:
        """Process one question. Returns (question name, decision, usage, model calls issued for this question).

        None of the three timing segments includes concurrency queue wait: each segment covers only its
        own work, and the semaphore wait shows up in the difference between the response-level latency_ms
        and the sum of the three — that's what separates "long queue" from "slow model".
        """
        prepared_started = time.perf_counter()
        items = question_items(question)
        handles = [handle for handle, _, _ in items]
        aliases = NOUL_ALIASES if question.type == "noul" else None
        lookup = build_lookup(handles, aliases)
        messages = build_messages(state, question, items, version=settings.prompt_version)
        # logit_bias may require one online tokenization call to get token ids (the result is cached)
        logit_bias = await self._logit_bias_for(client, settings, handles)
        prepared_at = time.perf_counter()

        async with semaphore:
            call_started = time.perf_counter()
            response = await client.chat_completions(
                messages, max_tokens=settings.max_tokens, logit_bias=logit_bias
            )
            called_at = time.perf_counter()

        content = extract_content(response)
        readout = read_distribution(content, handles, lookup)
        probabilities = softmax(readout.logprobs, settings.temperature_scale)
        timing = Timing(
            prepare_ms=_ms_since(prepared_started, prepared_at),
            call_ms=_ms_since(call_started, called_at),
            readout_ms=_ms_since(called_at, time.perf_counter()),
        )
        decision = _build_decision(
            question, items, probabilities, readout, debug, timing=timing, calls=1
        )
        return name, decision, response.get("usage"), 1


def _ms_since(start: float, end: float) -> float:
    return round((end - start) * 1000, 2)


def question_items(question: Question) -> List[Tuple[str, str, str]]:
    """Returns [(handle, raw label, description)]."""
    if question.type == "choice":
        labels = list(question.criteria.keys())
        handles = make_handles(labels)
        return [(handle, label, question.criteria[label] or "") for handle, label in zip(handles, labels)]
    if question.type == "noul":
        # Labels are display/contract only: true_probability uses the handle index
        # (probabilities[0]), not these strings, so renaming them can't move the math.
        # NOUL_ALIASES is separate and deliberately still accepts 是/否 — those are
        # tokens the *model* may emit under the Chinese v1 prompt.
        return [(handle, label, "") for handle, label in zip(NOUL_HANDLES, ["yes", "no"])]
    labels = list(question.scale)
    handles = make_handles(labels)
    return [(handle, label, "") for handle, label in zip(handles, labels)]


def score_values(question: Question) -> List[float]:
    if question.values is not None:
        return list(question.values)
    values: List[float] = []
    for label in question.scale:
        try:
            values.append(float(label))
        except ValueError:
            values.append(float(len(values)))
    return values


def _build_decision(
    question: Question,
    items: Sequence[Tuple[str, str, str]],
    probabilities: Sequence[float],
    readout: Readout,
    debug: bool,
    timing: Optional[Timing] = None,
    calls: Optional[int] = None,
) -> Decision:
    distribution = {
        label: round(probability, 6) for (_, label, _), probability in zip(items, probabilities)
    }
    best_index = max(range(len(items)), key=lambda index: probabilities[index])
    raw_candidates = (
        [CandidateLogprob(token=item.token, logprob=item.logprob) for item in readout.raw_candidates]
        if debug
        else None
    )
    common = {
        "distribution": distribution,
        "confidence": round(probabilities[best_index], 6),
        "coverage": round(readout.coverage, 6),
        "reliable": readout.reliable,
        "generated_token": readout.generated_token,
        "raw_candidates": raw_candidates,
        "timing": timing,
        "calls": calls,
    }

    if question.type == "choice":
        return Decision(type="choice", label=items[best_index][1], **common)
    if question.type == "noul":
        true_probability = probabilities[0]
        return Decision(
            type="noul",
            label=items[best_index][1],
            true_probability=round(true_probability, 6),
            decision=true_probability >= 0.5,
            **common,
        )
    values = score_values(question)
    expected = sum(value * probability for value, probability in zip(values, probabilities))
    return Decision(type="score", label=items[best_index][1], expected=round(expected, 6), **common)


def _merge_usage(usages: Sequence[Optional[dict]]) -> Optional[Usage]:
    present = [usage for usage in usages if usage]
    if not present:
        return None
    return Usage(
        prompt_tokens=sum(int(usage.get("prompt_tokens") or 0) for usage in present),
        completion_tokens=sum(int(usage.get("completion_tokens") or 0) for usage in present),
        total_tokens=sum(int(usage.get("total_tokens") or 0) for usage in present),
    )
