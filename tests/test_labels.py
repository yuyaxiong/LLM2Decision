from __future__ import annotations

import pytest

from llm2decision.core.labels import build_lookup, make_handles, normalize_token, numbered_handles


def test_numbered_handles_uses_digits_then_letters() -> None:
    assert numbered_handles(3) == ["1", "2", "3"]
    assert numbered_handles(9) == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    assert numbered_handles(12) == list("ABCDEFGHIJKL")


def test_numbered_handles_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        numbered_handles(21)


def test_make_handles_reuses_single_char_labels() -> None:
    assert make_handles(["0", "1", "2", "3"]) == ["0", "1", "2", "3"]
    assert make_handles(["billing", "technical", "other"]) == ["1", "2", "3"]


def test_make_handles_rejects_duplicate_single_chars() -> None:
    assert make_handles(["a", "a"]) == ["1", "2"]


def test_normalize_token_strips_and_uppercases() -> None:
    assert normalize_token(" 1") == "1"
    assert normalize_token("b.") == "B"
    assert normalize_token("Yes\n") == "YES"


def test_normalize_token_folds_full_width_chars() -> None:
    assert normalize_token("１") == "1"
    assert normalize_token("０") == "0"
    assert normalize_token("Ａ") == "A"


def test_build_lookup_maps_aliases_to_index() -> None:
    lookup = build_lookup(["1", "2"], {"1": ["Y", "YES"], "2": ["N"]})
    assert lookup["1"] == [0]
    assert lookup["YES"] == [0]
    assert lookup["N"] == [1]
    assert "3" not in lookup
