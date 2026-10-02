from __future__ import annotations

import json
import math

import pytest

from llm2decision.calibrate import (
    evaluate,
    fit_temperature,
    is_degenerate,
    load_records,
    load_samples,
    main,
    save_records,
    to_rows,
    write_temperature,
)
from llm2decision.core.config import Config

CONFIG_TEXT = "defaults:\n  api_key: k\n  base_url: https://example.com\nmodels:\n  mini:\n    model: m\n"

GAP = 4.0
SMALL_GAP = 2.0


def two_way_rows(correct: int, wrong: int) -> list:
    rows = [{"logprobs": [0.0, -GAP], "gold_index": 0} for _ in range(correct)]
    rows += [{"logprobs": [0.0, -GAP], "gold_index": 1} for _ in range(wrong)]
    return rows


def test_fit_temperature_matches_closed_form() -> None:
    """With two classes, a fixed logit gap d, and positive rate p, the optimal temperature is d / logit(p)."""
    rows = two_way_rows(correct=10, wrong=5)
    expected = GAP / math.log((10 / 15) / (1 - 10 / 15))
    assert fit_temperature(rows) == pytest.approx(expected, rel=1e-3)
    assert fit_temperature(rows) > 1.0


def test_fit_temperature_lowers_nll() -> None:
    rows = two_way_rows(correct=6, wrong=6)
    fitted = fit_temperature(rows)
    assert fit_temperature(rows, iterations=200) == pytest.approx(fitted)
    assert evaluate(rows, fitted)["nll"] < evaluate(rows, 1.0)["nll"]


def test_evaluate_reports_accuracy_and_ece() -> None:
    rows = [{"logprobs": [0.0, -20.0], "gold_index": index % 2} for index in range(4)]
    metrics = evaluate(rows, 1.0)
    assert metrics["accuracy"] == 0.5
    assert metrics["ece"] == pytest.approx(0.5, abs=0.01)


def test_temperature_scaling_improves_ece_for_overconfident_models() -> None:
    rows = two_way_rows(correct=4, wrong=4)
    fitted = fit_temperature(rows)
    assert evaluate(rows, fitted)["ece"] < evaluate(rows, 1.0)["ece"]


def test_to_rows_rejects_unknown_gold() -> None:
    records = [{"type": "choice", "labels": ["a", "b"], "logprobs": [0.0, -1.0], "coverage": 1.0, "gold": "c"}]
    with pytest.raises(ValueError):
        to_rows(records)


def test_to_rows_filters_by_question_type() -> None:
    records = [
        {"type": "choice", "labels": ["a", "b"], "logprobs": [0.0, -1.0], "coverage": 1.0, "gold": "a"},
        {"type": "noul", "labels": ["是", "否"], "logprobs": [0.0, -2.0], "coverage": 1.0, "gold": "否"},
    ]
    assert to_rows(records, "noul") == [{"logprobs": [0.0, -2.0], "gold_index": 1}]


def test_write_temperature_keeps_comments_and_other_keys(tmp_path) -> None:
    path = tmp_path / "llm2decision.yaml"
    path.write_text(
        "# 注释保留\nmodels:\n  # 默认档\n  mini:\n    model: x\n    temperature_scale: 1.0\n    max_tokens: 8\n",
        encoding="utf-8",
    )
    write_temperature(path, "mini", 2.5343690298)
    text = path.read_text(encoding="utf-8")
    assert "# 注释保留" in text
    assert "temperature_scale: 2.534369" in text
    assert "max_tokens: 8" in text
    assert "1.0" not in text


def test_write_temperature_inserts_when_route_has_no_value(tmp_path) -> None:
    path = tmp_path / "llm2decision.yaml"
    path.write_text("models:\n  mini:\n    model: x\n", encoding="utf-8")
    write_temperature(path, "mini", 3.0)
    assert Config.load(path).models["mini"].temperature_scale == pytest.approx(3.0)


def test_write_temperature_requires_existing_route(tmp_path) -> None:
    path = tmp_path / "llm2decision.yaml"
    path.write_text("models:\n  mini:\n    model: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        write_temperature(path, "ghost", 2.0)


def test_load_samples_validates_required_fields(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"state": "s", "question": {"type": "noul", "instructions": "x"}, "gold": "是"}) + "\n",
        encoding="utf-8",
    )
    assert len(load_samples(path)) == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"state": "s"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_samples(bad)


def test_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    records = [{"type": "choice", "labels": ["a", "b"], "logprobs": [0.0, -1.0], "coverage": 1.0, "gold": "a"}]
    save_records(path, records)
    assert load_records(path) == records


def test_all_correct_subset_is_unidentifiable() -> None:
    """When every prediction is correct, NLL collapses to zero and the temperature is unidentifiable;
    this must be detected rather than treated as a valid calibration."""
    rows = [{"logprobs": [0.0, -SMALL_GAP], "gold_index": 0} for _ in range(5)]
    assert is_degenerate(rows, fit_temperature(rows)) is True


def test_interior_optimum_is_not_flagged() -> None:
    rows = two_way_rows(correct=10, wrong=5)
    assert is_degenerate(rows, fit_temperature(rows)) is False


def test_cli_reuses_cache_and_warns_on_bound(tmp_path, capsys) -> None:
    data = tmp_path / "labels.jsonl"
    data.write_text(
        json.dumps({"state": "s", "question": {"type": "noul", "instructions": "x"}, "gold": "是"}) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "raw.jsonl"
    save_records(
        cache,
        [{"type": "noul", "labels": ["是", "否"], "logprobs": [0.0, -SMALL_GAP], "coverage": 1.0, "gold": "是"}],
    )
    config_file = tmp_path / "llm2decision.yaml"
    config_file.write_text(CONFIG_TEXT, encoding="utf-8")
    assert main(["--data", str(data), "--cache", str(cache), "--config", str(config_file)]) == 0
    output = capsys.readouterr().out
    assert "Reusing cache" in output
    assert "unidentifiable" in output


def test_cli_writes_fitted_temperature_to_config(tmp_path, capsys) -> None:
    data = tmp_path / "labels.jsonl"
    data.write_text(
        json.dumps({"state": "s", "question": {"type": "noul", "instructions": "x"}, "gold": "是"}) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "raw.jsonl"
    row = {"type": "noul", "labels": ["是", "否"], "logprobs": [0.0, -4.0], "coverage": 1.0}
    save_records(
        cache,
        [dict(row, gold="是"), dict(row, gold="是"), dict(row, gold="否")],
    )
    config_file = tmp_path / "llm2decision.yaml"
    config_file.write_text(CONFIG_TEXT, encoding="utf-8")
    assert main(["--data", str(data), "--cache", str(cache), "--config", str(config_file), "--write"]) == 0
    assert Config.load(config_file).models["mini"].temperature_scale == pytest.approx(5.7708, rel=1e-3)
    assert "Wrote" in capsys.readouterr().out
