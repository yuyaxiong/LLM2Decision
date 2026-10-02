"""Provider profiles: every vendor's quirks live in this one file, so the rest of the code never has to care about a specific vendor.

It lives in core/ rather than llm/: config needs to validate per provider (core must not depend back on
llm), and this module is a pure data table with no network calls.

Differences **confirmed by measurement** (see the "cross-vendor same-question-set comparison" section of
benchmarks/REPORT.md):

| Item | Volcengine Ark | Alibaba Cloud DashScope | DeepSeek |
|---|---|---|---|
| Disable-thinking param | top-level `thinking: {"type":"disabled"}` | top-level `enable_thinking: false` (omitted, logprobs come back null) | top-level `thinking: {"type":"disabled"}` (any other spelling leaves logprobs null) |
| `top_logprobs` cap | 20 | 5 | 20 |
| Get token id | online `/tokenization` | none; the vendor ships an id-mapping table file | none |
| Stable candidate cap | 10 | 4 | 10 |

The numbers for the three profiles above are all measured; each one's own comment records how. The
fourth, `openai_compatible`, is a **generic fallback for unverified vendors**: its values follow the
OpenAI spec and are *not* measured — before wiring up a new vendor, run
`benchmarks/probe_provider.py` to measure the real caps, then decide whether it deserves a profile of
its own.

Adding a vendor = add one entry to `PROVIDERS`; select it in `llm2decision.yaml` with
`provider: <name>`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_PROVIDER = "ark"

# Where the candidate caps come from: the top_logprobs slots must also hold the EOS token, full-width
# variants, punctuation, and the words a model reaches for when it wants to start explaining (measured
# noise such as <|im_end|>, The, Let, A/B/C). Measured: Ark's 20 slots reliably carry 10 candidates;
# DashScope's 5 slots averaged 3.2~4.0 hits with 4 candidates, and 6 candidates gave 0/4 coverage across
# all 6 models; DeepSeek's 20 slots carried 10 candidates at full coverage (8/8 questions) but only
# 0.865 average coverage at 12 and 0.588 at 20. These numbers are measured, not inferred from docs.


@dataclass
class ProviderProfile:
    """One vendor's capability and request differences."""

    name: str
    display_name: str
    default_base_url: str
    max_top_logprobs: int
    max_candidates: int
    # Whether an online tokenizer endpoint exists (logit_bias needs token ids; without one this fallback must be dropped)
    supports_tokenize: bool
    # Field name and value placed at the top level of the request body to disable thinking; None means the vendor cannot disable it
    thinking_field: Optional[str]
    thinking_disabled_value: Any

    def thinking_payload(self, disabled: bool) -> Dict[str, Any]:
        """Return the disable-thinking params to merge into the top level of the request body; an empty dict when not needed.

        Uses a deep copy: the value is a nested dict (e.g. {"thinking": {"type": "disabled"}}), and a
        shallow copy would let the object the caller gets share its inner dict with the module-level
        constant, so one mutation would pollute the constant table.
        """
        if not disabled or self.thinking_field is None:
            return {}
        return {self.thinking_field: copy.deepcopy(self.thinking_disabled_value)}

    def describe(self) -> str:
        return (
            f"{self.display_name} (top_logprobs≤{self.max_top_logprobs}"
            f", candidates≤{self.max_candidates}"
            f", {'supports' if self.supports_tokenize else 'does not support'} online tokenization)"
        )


ARK = ProviderProfile(
    name="ark",
    display_name="Volcengine Ark",
    default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    max_top_logprobs=20,
    max_candidates=10,
    supports_tokenize=True,
    thinking_field="thinking",
    thinking_disabled_value={"type": "disabled"},
)

DASHSCOPE = ProviderProfile(
    name="dashscope",
    display_name="Alibaba Cloud DashScope",
    default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_top_logprobs=5,
    max_candidates=4,
    supports_tokenize=False,
    thinking_field="enable_thinking",
    thinking_disabled_value=False,
)

DEEPSEEK = ProviderProfile(
    name="deepseek",
    display_name="DeepSeek",
    default_base_url="https://api.deepseek.com",
    # Measured on api.deepseek.com (probe_provider.py, 60 JevBench choice questions): logprobs are
    # accepted, top_logprobs returns all 20 entries, and thinking MUST be disabled explicitly —
    # `thinking: {"type": "disabled"}` is the only spelling that works (`enable_thinking: false` and
    # `chat_template_kwargs` both leave logprobs.content null, silently, with HTTP 200).
    #
    # This matters more here than elsewhere: DeepSeek's reasoning-first models (deepseek-flash,
    # deepseek-v4-pro) put their single generated token into the reasoning chain, so with max_tokens=1
    # the answer position is empty and position 0 carries no candidate at all. Without the disable
    # param this vendor cannot drive the mechanism.
    #
    # Candidate cap measured by sweeping the candidate count at top_logprobs=20 over 8 ambiguous
    # 10-way questions: 10 candidates → coverage 1.000 (8/8 full), 12 → 0.865 (0/8 full),
    # 16 → 0.695, 20 → 0.588. So 10, the same as Ark with the same slot count.
    max_top_logprobs=20,
    max_candidates=10,
    # No online tokenizer endpoint is documented for this vendor, so the logit_bias fallback is dropped.
    supports_tokenize=False,
    thinking_field="thinking",
    thinking_disabled_value={"type": "disabled"},
)

OPENAI_COMPATIBLE = ProviderProfile(
    name="openai_compatible",
    display_name="OpenAI-compatible endpoint (generic, unverified)",
    default_base_url="https://api.openai.com/v1",
    # The values below follow the OpenAI spec and are **not measured**. Verify with probe_provider.py before adding a vendor:
    #   slot cap — 20 is the OpenAI protocol cap; endpoints that don't support logprobs silently return null
    #   candidate cap — conservatively 8 (measured: Ark's 20 slots reliably carry 10, but noise tokens
    #             vary with the tokenizer, so an unverified vendor shouldn't just use 10; tune once you
    #             measure the real value)
    #   disable thinking — the OpenAI protocol has no such switch; if a vendor enables thinking by
    #             default (e.g. DashScope's qwen3.8), this profile can't send a disable param, position 0
    #             lands on the reasoning chain, and you should switch to that vendor's dedicated profile
    max_top_logprobs=20,
    max_candidates=8,
    supports_tokenize=False,
    thinking_field=None,
    thinking_disabled_value=None,
)

PROVIDERS: Dict[str, ProviderProfile] = {
    profile.name: profile for profile in (ARK, DASHSCOPE, DEEPSEEK, OPENAI_COMPATIBLE)
}


class UnknownProviderError(ValueError):
    def __init__(self, name: str, available: List[str]) -> None:
        super().__init__(f"Unknown provider {name!r}, available: {', '.join(available)}")
        self.name = name
        self.available = available


def get_profile(name: Optional[str]) -> ProviderProfile:
    key = name or DEFAULT_PROVIDER
    profile = PROVIDERS.get(key)
    if profile is None:
        raise UnknownProviderError(key, sorted(PROVIDERS))
    return profile
