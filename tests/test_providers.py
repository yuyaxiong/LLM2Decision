from __future__ import annotations

import pytest

from llm2decision.core.providers import (
    ARK,
    DASHSCOPE,
    DEEPSEEK,
    OPENAI_COMPATIBLE,
    PROVIDERS,
    UnknownProviderError,
    get_profile,
)


def test_vendors_are_registered() -> None:
    assert set(PROVIDERS) == {"ark", "dashscope", "deepseek", "openai_compatible"}


def test_ark_profile_matches_measured_limits() -> None:
    """The numbers come from measurement, not documentation: Ark's top_logprobs cap is 20,
    the stable candidate cap is 10, and online tokenization is available."""
    assert ARK.max_top_logprobs == 20
    assert ARK.max_candidates == 10
    assert ARK.supports_tokenize is True


def test_dashscope_profile_matches_measured_limits() -> None:
    """Measured: DashScope has only 5 slots (4 candidates is already the cap) and no online
    tokenization endpoint."""
    assert DASHSCOPE.max_top_logprobs == 5
    assert DASHSCOPE.max_candidates == 4
    assert DASHSCOPE.supports_tokenize is False


def test_deepseek_profile_matches_measured_limits() -> None:
    """Measured on api.deepseek.com by sweeping the candidate count over 8 ambiguous 10-way
    questions at top_logprobs=20: 10 candidates held full coverage (8/8), 12 dropped to 0.865 and
    16/20 fell to 0.695/0.588. Same slot count and same candidate cap as Ark."""
    assert DEEPSEEK.max_top_logprobs == 20
    assert DEEPSEEK.max_candidates == 10
    # No documented online tokenizer, so the logit_bias fallback must be dropped here
    assert DEEPSEEK.supports_tokenize is False


def test_deepseek_needs_the_disable_thinking_param() -> None:
    """DeepSeek is the vendor where this is load-bearing rather than an optimisation: its
    reasoning-first models (deepseek-flash, deepseek-v4-pro) put the single generated token into
    the reasoning chain, so with max_tokens=1 position 0 carries no candidate and logprobs.content
    comes back null. Measured: thinking={"type":"disabled"} is the only spelling that works —
    enable_thinking=false and chat_template_kwargs both fail silently with HTTP 200."""
    assert DEEPSEEK.thinking_payload(True) == {"thinking": {"type": "disabled"}}
    assert DEEPSEEK.thinking_payload(False) == {}
    # Same spelling as Ark, different from DashScope
    assert DEEPSEEK.thinking_field == ARK.thinking_field
    assert DEEPSEEK.thinking_field != DASHSCOPE.thinking_field


def test_thinking_payload_differs_by_vendor() -> None:
    """The thinking-disable parameter name is the hardest difference between the two vendors:
    Ark uses thinking, DashScope uses enable_thinking."""
    assert ARK.thinking_payload(True) == {"thinking": {"type": "disabled"}}
    assert DASHSCOPE.thinking_payload(True) == {"enable_thinking": False}
    # No parameter is appended when not disabling
    assert ARK.thinking_payload(False) == {}
    assert DASHSCOPE.thinking_payload(False) == {}


def test_thinking_payload_is_a_copy() -> None:
    """The return value must not be a reference to internal state, otherwise a caller's mutation
    would pollute the constants table."""
    payload = ARK.thinking_payload(True)
    payload["thinking"]["type"] = "enabled"
    assert ARK.thinking_payload(True) == {"thinking": {"type": "disabled"}}


def test_get_profile_defaults_and_rejects_unknown() -> None:
    assert get_profile(None).name == "ark"
    assert get_profile("dashscope") is DASHSCOPE
    with pytest.raises(UnknownProviderError) as error:
        get_profile("nope")
    assert "dashscope" in str(error.value)


def test_describe_mentions_limits() -> None:
    assert "Alibaba Cloud DashScope" in DASHSCOPE.describe()
    assert "5" in DASHSCOPE.describe()


def test_openai_compatible_is_a_conservative_fallback() -> None:
    """The generic profile is the fallback for unverified vendors: values follow the OpenAI spec
    and must be more conservative than the verified Ark.

    Its reason for existing is "let users reach any OpenAI-compatible endpoint", at the cost of
    capping candidates at only 8 (Ark's 20 slots measured a stable 10, but noise tokens vary with
    the tokenizer, so an unverified vendor should not be assumed to match).
    """
    assert OPENAI_COMPATIBLE.max_top_logprobs == 20
    assert OPENAI_COMPATIBLE.max_candidates < ARK.max_candidates
    # The OpenAI protocol has no thinking-disable switch and offers no online tokenization
    assert OPENAI_COMPATIBLE.thinking_field is None
    assert OPENAI_COMPATIBLE.thinking_payload(True) == {}
    assert OPENAI_COMPATIBLE.supports_tokenize is False
    assert "unverified" in OPENAI_COMPATIBLE.describe()
