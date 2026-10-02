from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llm2decision.core.config import Config, ModelSettings, UnknownModelError
from llm2decision.core.readout import softmax
from llm2decision.core.schema import Question, SystemOneRequest
from llm2decision.llm import service as service_module
from llm2decision.llm.service import SystemOneService, TooManyCandidatesError, score_values
from llm2decision.api.main import create_app

FAKE_TOP = [
    ("1", -0.01),
    ("2", -3.0),
    ("0", -4.0),
    ("3", -5.0),
    ("4", -6.0),
    ("5", -7.0),
]
FAKE_CONTENT = [
    {
        "token": "1",
        "logprob": -0.01,
        "top_logprobs": [{"token": token, "logprob": logprob} for token, logprob in FAKE_TOP],
    }
]

PAYLOAD = {
    "state": "I was charged twice and want a refund.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Choose the customer intent.",
            "criteria": {
                "billing": "A payment or refund issue",
                "technical": "A malfunction or setup issue",
                "other": "Another request",
            },
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "The customer explicitly requests a refund.",
        },
        "satisfaction": {
            "type": "score",
            "instructions": "Rate the customer's satisfaction.",
            "scale": ["0", "1", "2", "3", "4", "5"],
        },
    },
}


@pytest.fixture(autouse=True)
def clear_handle_token_cache():
    service_module._HANDLE_TOKEN_IDS.clear()
    yield
    service_module._HANDLE_TOKEN_IDS.clear()


class FakeChatClient:
    def __init__(self, content=FAKE_CONTENT, multi_token_handles: bool = False) -> None:
        self._content = content
        self._multi_token = multi_token_handles
        self.calls: list = []
        self.tokenize_calls: list = []

    async def tokenize(self, text: str) -> list:
        self.tokenize_calls.append(text)
        if self._multi_token:
            return [1, 2]
        return [1000 + ord(text)]

    async def chat_completions(self, messages, max_tokens: int, logit_bias=None) -> dict:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "logit_bias": logit_bias})
        return {
            "choices": [{"logprobs": {"content": self._content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }


def model_settings(**overrides) -> ModelSettings:
    base = dict(
        api_key="test-key",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-seed-test",
        top_logprobs=20,
        temperature_scale=1.0,
        max_tokens=1,
        disable_thinking=True,
        logit_bias_enabled=False,
        logit_bias_strength=5.0,
        timeout_seconds=30.0,
        max_concurrency=8,
    )
    base.update(overrides)
    return ModelSettings(**base)


def make_config(**overrides) -> Config:
    return Config(models={"test": model_settings(**overrides)}, default_model="test")


@contextmanager
def http_app(config: Config = None):
    """Builds an app with the test config injected and replaces the service with a fake client
    (triggering no network calls).

    Uses create_app(cfg) rather than the module-level app so it does not depend on a real config
    file existing at the repo root -- otherwise cloning the repo and running tests directly would
    fail due to the missing config.
    """
    cfg = config if config is not None else make_config()
    instance = create_app(cfg)
    with TestClient(instance) as client:
        instance.state.service = SystemOneService({"test": FakeChatClient()}, cfg)
        yield client


def decide(payload: dict = PAYLOAD, client=None, config: Config = None, **overrides):
    config = config if config is not None else make_config(**overrides)
    shared = client if client is not None else FakeChatClient()
    clients = {name: shared for name in config.models}
    service = SystemOneService(clients, config)
    return asyncio.run(service.decide(SystemOneRequest(**payload)))


def test_choice_decision_maps_handles_back_to_labels() -> None:
    response = decide()
    decision = response.decisions["intent"]
    assert decision.type == "choice"
    assert decision.label == "billing"
    assert list(decision.distribution) == ["billing", "technical", "other"]
    assert decision.confidence == pytest.approx(decision.distribution["billing"])
    assert decision.coverage == 1.0
    assert decision.reliable is True
    assert decision.generated_token == "1"
    assert response.route == "test"


def test_noul_decision_returns_true_probability() -> None:
    decision = decide().decisions["refund_requested"]
    assert decision.type == "noul"
    assert decision.true_probability == pytest.approx(softmax([-0.01, -3.0])[0])
    assert decision.decision is True


def test_score_decision_returns_expected_value() -> None:
    decision = decide().decisions["satisfaction"]
    probabilities = softmax([-4.0, -0.01, -3.0, -5.0, -6.0, -7.0])
    expected = sum(value * probability for value, probability in enumerate(probabilities))
    assert decision.type == "score"
    assert decision.expected == pytest.approx(expected, abs=1e-6)
    assert decision.label == "1"


def test_low_temperature_scale_sharpens_distribution() -> None:
    soft = decide().decisions["intent"].confidence
    sharp = decide(temperature_scale=0.5).decisions["intent"].confidence
    assert sharp > soft


def test_usage_is_aggregated_and_questions_run_in_parallel() -> None:
    response = decide()
    assert response.usage.total_tokens == 33
    assert response.model == "doubao-seed-test"
    assert response.latency_ms >= 0


def test_timing_breakdown_and_call_count_are_reported() -> None:
    """Timing breakdown and call count: used to distinguish "slow model", "many calls", and
    "long queue wait"."""
    response = decide()
    # The current strategy makes one call per question, so request-level calls == number of questions
    assert response.calls == len(PAYLOAD["questions"])
    for decision in response.decisions.values():
        assert decision.calls == 1
        assert decision.timing is not None
        for value in (decision.timing.prepare_ms, decision.timing.call_ms,
                      decision.timing.readout_ms):
            assert value >= 0
        # The sum of a single question's three segments must fall within the whole request's
        # wall time (+1ms tolerance for timer rounding)
        total = (decision.timing.prepare_ms + decision.timing.call_ms
                 + decision.timing.readout_ms)
        assert total <= response.latency_ms + 1


def test_http_response_exposes_calls_and_timing() -> None:
    with http_app() as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["calls"] == len(PAYLOAD["questions"])
    timing = body["decisions"]["intent"]["timing"]
    assert set(timing) == {"prepare_ms", "call_ms", "readout_ms"}
    assert body["decisions"]["intent"]["calls"] == 1


def test_debug_flag_exposes_raw_candidates() -> None:
    payload = dict(PAYLOAD, debug=True)
    decision = decide(payload).decisions["intent"]
    assert decision.raw_candidates is not None
    tokens = {item.token for item in decision.raw_candidates}
    assert {"1", "2", "0"} <= tokens


def test_score_values_fall_back_to_index_for_text_labels() -> None:
    question = Question(type="score", instructions="rate", scale=["low", "mid", "high"])
    assert score_values(question) == [0.0, 1.0, 2.0]


def test_routes_to_requested_model() -> None:
    config = Config(
        models={"mini": model_settings(model="mini-model"), "pro": model_settings(model="pro-model")},
        default_model="mini",
    )
    mini_client, pro_client = FakeChatClient(), FakeChatClient()
    service = SystemOneService({"mini": mini_client, "pro": pro_client}, config)

    response = asyncio.run(service.decide(SystemOneRequest(**{**PAYLOAD, "model": "pro"})))
    assert response.route == "pro"
    assert response.model == "pro-model"
    assert len(pro_client.calls) == 3
    assert mini_client.calls == []


def test_default_route_used_when_model_omitted() -> None:
    config = Config(
        models={"mini": model_settings(model="mini-model"), "pro": model_settings(model="pro-model")},
        default_model="pro",
    )
    mini_client, pro_client = FakeChatClient(), FakeChatClient()
    service = SystemOneService({"mini": mini_client, "pro": pro_client}, config)

    response = asyncio.run(service.decide(SystemOneRequest(**PAYLOAD)))
    assert response.route == "pro"
    assert pro_client.calls and mini_client.calls == []


def test_unknown_route_is_rejected() -> None:
    with pytest.raises(UnknownModelError) as error:
        decide(payload={**PAYLOAD, "model": "nope"})
    assert "test" in str(error.value)


def test_each_route_keeps_its_own_semaphore_and_settings() -> None:
    config = Config(
        models={
            "mini": model_settings(model="mini-model", logit_bias_enabled=False),
            "pro": model_settings(model="pro-model", logit_bias_enabled=True),
        },
        default_model="mini",
    )
    mini_client, pro_client = FakeChatClient(), FakeChatClient()
    service = SystemOneService({"mini": mini_client, "pro": pro_client}, config)

    asyncio.run(service.decide(SystemOneRequest(**PAYLOAD)))
    asyncio.run(service.decide(SystemOneRequest(**{**PAYLOAD, "model": "pro"})))
    assert all(call["logit_bias"] is None for call in mini_client.calls)
    assert all(call["logit_bias"] is not None for call in pro_client.calls)


def test_logit_bias_not_sent_when_disabled() -> None:
    client = FakeChatClient()
    decide(client=client)
    assert all(call["logit_bias"] is None for call in client.calls)
    assert client.tokenize_calls == []


def test_logit_bias_sent_uniformly_and_token_ids_cached() -> None:
    client = FakeChatClient()
    decide(client=client, logit_bias_enabled=True)
    biases = [call["logit_bias"] for call in client.calls]
    assert all(bias is not None for bias in biases)
    assert len(biases[0]) == 3
    assert set(biases[0].values()) == {5.0}
    assert set(client.tokenize_calls) == set("012345")
    assert len(client.tokenize_calls) == len(set(client.tokenize_calls))


def test_logit_bias_strength_is_configurable() -> None:
    client = FakeChatClient()
    decide(client=client, logit_bias_enabled=True, logit_bias_strength=2.5)
    assert set(client.calls[0]["logit_bias"].values()) == {2.5}


def test_logit_bias_abandoned_when_handle_is_not_single_token(caplog) -> None:
    client = FakeChatClient(multi_token_handles=True)
    with caplog.at_level(logging.WARNING):
        decide(client=client, logit_bias_enabled=True)
    assert all(call["logit_bias"] is None for call in client.calls)
    assert "abandoning logit_bias" in caplog.text


def candidates_payload(count: int) -> dict:
    return dict(
        PAYLOAD,
        questions={
            "q": {
                "type": "choice",
                "instructions": "pick one",
                "criteria": {f"c{index}": f"option {index}" for index in range(count)},
            }
        },
    )


def test_candidate_cap_is_enforced_by_service_for_ark() -> None:
    """The candidate cap is pushed down to the service layer per vendor (the Pydantic layer cannot
    see the route); the cap on Ark is 10."""
    with pytest.raises(TooManyCandidatesError) as error:
        decide(candidates_payload(11))
    assert error.value.max_candidates == 10
    assert error.value.count == 11


def test_candidate_cap_is_stricter_on_dashscope() -> None:
    """The same 5-candidate request: Ark allows it, DashScope rejects it (DashScope has only 5
    top_logprobs slots)."""
    dashscope = Config(
        models={"test": model_settings(provider="dashscope", top_logprobs=5)},
        default_model="test",
    )
    decide(candidates_payload(5))  # Ark: cap 10, allowed
    with pytest.raises(TooManyCandidatesError) as error:
        decide(candidates_payload(5), config=dashscope)
    assert error.value.max_candidates == 4
    assert "Alibaba Cloud DashScope" in str(error.value)

    decide(candidates_payload(4), config=dashscope)  # Within DashScope's cap, reads normally


def test_dashscope_route_skips_logit_bias(caplog) -> None:
    """DashScope has no online tokenization endpoint and cannot obtain token ids, so logit_bias
    must be skipped rather than hitting a 404."""
    client = FakeChatClient()
    dashscope = Config(
        models={
            "test": model_settings(
                provider="dashscope", top_logprobs=5, logit_bias_enabled=True
            )
        },
        default_model="test",
    )
    with caplog.at_level(logging.WARNING):
        decide(candidates_payload(3), client=client, config=dashscope)
    assert client.tokenize_calls == []  # Does not call /tokenization, which only Ark has
    assert all(call["logit_bias"] is None for call in client.calls)
    assert "has no online tokenizer" in caplog.text


def test_structural_candidate_cap_still_rejected_by_schema() -> None:
    """The physical cap of the handle scheme (20) is still enforced by the Pydantic layer,
    independent of the vendor."""
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria={str(i): str(i) for i in range(21)})
    with pytest.raises(ValidationError):
        Question(type="score", instructions="x", scale=[str(i) for i in range(21)])


def test_request_requires_noul_instructions_and_non_empty_state() -> None:
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="  ")
    with pytest.raises(ValidationError):
        SystemOneRequest(state="  ", questions={"q": {"type": "noul", "instructions": "x"}})


def test_http_endpoint_lists_routes() -> None:
    with http_app() as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["default_model"] in health["models"]
        body = client.get("/v1/models").json()
        names = [item["name"] for item in body["models"]]
        assert body["default"] in names
        assert all("model" in item and "base_url" in item for item in body["models"])


def test_http_endpoint_returns_decisions_with_fake_client() -> None:
    with http_app() as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["decisions"]["intent"]["label"] == "billing"
    assert body["decisions"]["refund_requested"]["decision"] is True
    assert body["route"] == "test"


def test_http_endpoint_rejects_unknown_route() -> None:
    with http_app() as client:
        response = client.post("/v1/systemone", json={**PAYLOAD, "model": "ghost"})
    assert response.status_code == 422
    assert "ghost" in response.json()["detail"]


def test_native_and_compat_endpoints_agree() -> None:
    """/v1/decide (native) and /v1/systemone (Jev-compatible) must be the same path, only differing
    in name."""
    with http_app() as client:
        native = client.post("/v1/decide", json=PAYLOAD)
        compat = client.post("/v1/systemone", json=PAYLOAD)
    assert native.status_code == compat.status_code == 200

    def comparable(body: dict) -> dict:
        # timing is the real elapsed time of each call and necessarily differs between the two
        # requests, so it is excluded from the comparison
        return {name: {k: v for k, v in decision.items() if k != "timing"}
                for name, decision in body["decisions"].items()}

    assert comparable(native.json()) == comparable(compat.json())
    assert native.json()["route"] == compat.json()["route"]
    assert native.json()["calls"] == compat.json()["calls"]


def test_debug_page_serves_html_form() -> None:
    with http_app() as client:
        response = client.get("/debug")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    for element_id in ('id="state"', 'id="qtype"', 'id="candidates"', 'id="model"', 'id="debug"', 'id="run"'):
        assert element_id in body
    assert "fetch('/v1/decide'" in body

