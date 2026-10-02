from __future__ import annotations

import pytest

from llm2decision.core.config import Config, UnknownModelError
from llm2decision.core.providers import ARK, DASHSCOPE, UnknownProviderError


def write_config(tmp_path, content: str):
    path = tmp_path / "llm2decision.yaml"
    path.write_text(content, encoding="utf-8")
    return path


BASIC = """
default_model: mini
defaults:
  base_url: https://example.com/v1/
  api_key: shared-key
  top_logprobs: 15
  max_tokens: 1
models:
  mini:
    model: mini-model
  pro:
    model: pro-model
    base_url: https://pro.example.com/v1
    temperature_scale: 2.5
    max_tokens: 2
"""


def test_model_section_overrides_defaults(tmp_path) -> None:
    config = Config.load(write_config(tmp_path, BASIC))
    assert config.default_model == "mini"
    assert config.model_names() == ["mini", "pro"]

    mini, pro = config.models["mini"], config.models["pro"]
    assert mini.model == "mini-model"
    assert mini.base_url == "https://example.com/v1"
    assert mini.temperature_scale == 1.0
    assert mini.max_tokens == 1
    # Per-route config overrides defaults
    assert pro.model == "pro-model"
    assert pro.base_url == "https://pro.example.com/v1"
    assert pro.temperature_scale == 2.5
    assert pro.max_tokens == 2
    # Items supplied by neither the route nor defaults fall back to built-in defaults
    assert mini.disable_thinking is True
    assert mini.logit_bias_enabled is False


def test_resolve_falls_back_to_default_and_rejects_unknown(tmp_path) -> None:
    config = Config.load(write_config(tmp_path, BASIC))
    assert config.resolve().model == "mini-model"
    assert config.resolve("").model == "mini-model"
    assert config.resolve("pro").model == "pro-model"
    with pytest.raises(UnknownModelError) as error:
        config.resolve("nope")
    assert "mini" in str(error.value) and "pro" in str(error.value)


def test_default_model_must_exist(tmp_path) -> None:
    path = write_config(tmp_path, "default_model: ghost\nmodels:\n  mini:\n    model: m\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_default_model_defaults_to_first_entry(tmp_path) -> None:
    path = write_config(tmp_path, "models:\n  only:\n    model: only-model\n")
    assert Config.load(path).default_model == "only"


def test_env_used_when_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM2DECISION_API_KEY", "env-key")
    monkeypatch.setenv("LLM2DECISION_DISABLE_THINKING", "false")
    monkeypatch.setenv("LLM2DECISION_TOP_LOGPROBS", "5")
    path = write_config(tmp_path, "models:\n  mini:\n    model: m\n")
    settings = Config.load(path).resolve()
    assert settings.api_key == "env-key"
    assert settings.disable_thinking is False
    assert settings.top_logprobs == 5
    assert settings.max_tokens == 1


def test_yaml_wins_over_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM2DECISION_API_KEY", "env-key")
    path = write_config(tmp_path, "defaults:\n  api_key: file-key\nmodels:\n  mini:\n    model: m\n")
    assert Config.load(path).resolve().api_key == "file-key"


def test_missing_model_id_rejected(tmp_path) -> None:
    path = write_config(tmp_path, "models:\n  mini:\n    base_url: https://example.com\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_empty_models_rejected(tmp_path) -> None:
    path = write_config(tmp_path, "defaults:\n  api_key: k\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_top_logprobs_above_ark_limit_rejected(tmp_path) -> None:
    path = write_config(
        tmp_path, f"models:\n  mini:\n    model: m\n    top_logprobs: {ARK.max_top_logprobs + 1}\n"
    )
    with pytest.raises(ValueError):
        Config.load(path)


def test_provider_defaults_to_ark(tmp_path) -> None:
    settings = Config.load(write_config(tmp_path, "models:\n  mini:\n    model: m\n")).resolve()
    assert settings.provider == ARK.name


def test_top_logprobs_limit_follows_provider(tmp_path) -> None:
    """Ark's limit of 20 does not hold for DashScope: DashScope has only 5 slots, so writing 10
    must fail at startup."""
    dashscope_route = (
        "models:\n  qwen:\n    provider: dashscope\n    model: qwen3.7-plus\n"
        f"    base_url: {DASHSCOPE.default_base_url}\n    top_logprobs: {DASHSCOPE.max_top_logprobs}\n"
    )
    config = Config.load(write_config(tmp_path, dashscope_route))
    assert config.resolve("qwen").top_logprobs == DASHSCOPE.max_top_logprobs

    too_many = dashscope_route.replace(
        f"top_logprobs: {DASHSCOPE.max_top_logprobs}",
        f"top_logprobs: {DASHSCOPE.max_top_logprobs + 1}",
    )
    with pytest.raises(ValueError, match="Alibaba Cloud DashScope"):
        Config.load(write_config(tmp_path, too_many))


def test_unknown_provider_rejected(tmp_path) -> None:
    path = write_config(tmp_path, "models:\n  mini:\n    model: m\n    provider: nope\n")
    with pytest.raises(UnknownProviderError) as error:
        Config.load(path)
    assert ARK.name in str(error.value)


def test_cross_vendor_base_url_mismatch_rejected(tmp_path) -> None:
    """The most common cross-vendor misconfiguration: provider is set to dashscope, but base_url
    inherits Ark's address from the defaults section."""
    content = (
        f"defaults:\n  base_url: {ARK.default_base_url}\n"
        "models:\n  qwen:\n    provider: dashscope\n    model: qwen3.7-plus\n"
    )
    with pytest.raises(ValueError, match="cross-vendor route must set base_url"):
        Config.load(write_config(tmp_path, content))


def test_missing_file_uses_env_and_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM2DECISION_MODEL", "env-model")
    config = Config.load(tmp_path / "nope.yaml")
    assert config.models["default"].model == "env-model"
    assert config.default_model == "default"


def test_default_path_is_app_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM2DECISION_CONFIG_PATH", raising=False)
    write_config(tmp_path, "default_model: routed\nmodels:\n  routed:\n    model: from-project-root\n")
    monkeypatch.chdir(tmp_path)
    assert Config.load().resolve().model == "from-project-root"


def test_prompt_version_defaults_and_can_be_overridden(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM2DECISION_PROMPT_VERSION", raising=False)
    # Falls back to the built-in default v1 when unconfigured
    assert Config.load(write_config(tmp_path, BASIC)).models["mini"].prompt_version == "v1"
    # The defaults section can set it globally
    override = write_config(tmp_path, "defaults:\n  prompt_version: v2\nmodels:\n  mini:\n    model: m\n")
    assert Config.load(override).models["mini"].prompt_version == "v2"
    # Per-route config takes precedence over defaults
    per_route = write_config(
        tmp_path,
        "defaults:\n  prompt_version: v2\nmodels:\n  mini:\n    model: m\n    prompt_version: v3\n",
    )
    assert Config.load(per_route).models["mini"].prompt_version == "v3"
    # Environment variable as fallback
    monkeypatch.setenv("LLM2DECISION_PROMPT_VERSION", "v4")
    env_only = write_config(tmp_path, "models:\n  mini:\n    model: m\n")
    assert Config.load(env_only).models["mini"].prompt_version == "v4"
