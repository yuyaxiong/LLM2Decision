from __future__ import annotations

import math

import pytest

from llm2decision.core.labels import build_lookup
from llm2decision.core.readout import ReadoutError, read_distribution, softmax

LOOKUP_12 = build_lookup(["1", "2"])


def entry(token: str, logprob: float, top: list) -> dict:
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [{"token": item[0], "logprob": item[1]} for item in top],
    }


def test_reads_distribution_at_first_position() -> None:
    content = [entry("2", -0.1, [(" 2", -0.1), (" 1", -1.2), (" 3", -3.0)])]
    readout = read_distribution(content, ["1", "2"], LOOKUP_12)
    assert readout.coverage == 1.0
    assert readout.generated_token == "2"
    assert readout.reliable is True
    assert readout.probabilities == pytest.approx(softmax([-1.2, -0.1]))
    assert readout.logprobs == pytest.approx([-1.2, -0.1])


def test_ignores_later_positions() -> None:
    """Only position 0 is read: if the model first writes a prefix and only emits the label later,
    this raises rather than scanning forward to make it fit."""
    content = [
        entry("由于", -0.66, [("由于", -0.66), ("1", -16.0)]),
        entry("1", -0.05, [("1", -0.05), ("2", -3.4)]),
    ]
    with pytest.raises(ReadoutError) as error:
        read_distribution(content, ["1", "2"], LOOKUP_12)
    assert "由于" in str(error.value)


def test_rejects_prose_at_first_position() -> None:
    """Regression: a candidate letter that incidentally appears in the top-k of explanatory text
    must not be taken as the answer."""
    content = [entry("你的", -0.4, [("你的", -0.4), ("A", -16.0), ("B", -17.0)])]
    with pytest.raises(ReadoutError):
        read_distribution(content, ["A", "B"], build_lookup(["A", "B"]))


def test_missing_candidate_gets_floor_and_reduces_coverage() -> None:
    content = [entry("1", -0.1, [("1", -0.1), ("9", -5.0)])]
    readout = read_distribution(content, ["1", "2"], LOOKUP_12, missing_floor=3.0)
    assert readout.coverage == 0.5
    assert readout.logprobs[1] == pytest.approx(-3.1)
    assert readout.probabilities[1] < readout.probabilities[0]


def test_reliable_flag_false_when_coverage_low() -> None:
    content = [entry("1", -0.1, [("1", -0.1)])]
    readout = read_distribution(content, ["1", "2", "3", "4"], build_lookup(["1", "2", "3", "4"]))
    assert readout.coverage == 0.25
    assert readout.reliable is False


def test_full_width_generated_token_is_accepted() -> None:
    content = [entry("１", -0.02, [("１", -0.02), ("２", -2.0)])]
    readout = read_distribution(content, ["1", "2"], LOOKUP_12)
    assert readout.coverage == 1.0


def test_raises_when_top_logprobs_missing() -> None:
    content = [{"token": "1", "logprob": -0.1}]
    with pytest.raises(ReadoutError):
        read_distribution(content, ["1", "2"], LOOKUP_12)


def test_raises_on_empty_content() -> None:
    with pytest.raises(ReadoutError):
        read_distribution([], ["1", "2"], LOOKUP_12)


def test_vendor_sentinel_logprobs_are_not_observations() -> None:
    """Measured on DeepSeek: every token except the emitted one comes back as -9999.
    A placeholder is not an observation: it must not count toward coverage, and the
    distribution falls back to the documented floor for the unobserved candidates."""
    content = [entry("3", 0.0, [("3", 0.0), ("1", -9999.0), ("2", -9999.0), ("4", -9999.0)])]
    readout = read_distribution(content, ["1", "2", "3", "4"], build_lookup(["1", "2", "3", "4"]), missing_floor=3.0)
    assert readout.coverage == 0.25
    assert readout.reliable is False
    assert readout.logprobs[0] == pytest.approx(-3.0)
    assert readout.probabilities[2] > 0.8
    assert [candidate.token for candidate in readout.raw_candidates] == ["3"]


def test_raw_candidates_sorted_by_logprob() -> None:
    content = [entry("1", -0.1, [("1", -0.1), ("2", -2.0), ("3", -1.0)])]
    readout = read_distribution(content, ["1", "2"], LOOKUP_12)
    assert [item.token for item in readout.raw_candidates] == ["1", "3", "2"]


def test_softmax_is_temperature_scaled() -> None:
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])
    sharpened = softmax([2.0, 0.0], temperature=2.0)
    assert sharpened == pytest.approx(softmax([1.0, 0.0]))
    assert sharpened[0] == pytest.approx(1 / (1 + math.exp(-1)))
