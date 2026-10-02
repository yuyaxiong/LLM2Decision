from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .providers import DEFAULT_PROVIDER, PROVIDERS, get_profile

DEFAULT_CONFIG_PATH = "llm2decision.yaml"
CONFIG_PATH_ENV = "LLM2DECISION_CONFIG"

# Built-in defaults: used when the config's defaults section doesn't provide a value and no env var is set
_DEFAULTS: Dict[str, Any] = {
    "provider": DEFAULT_PROVIDER,
    "api_key": "",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "",
    "top_logprobs": 20,
    "temperature_scale": 1.0,
    "max_tokens": 1,
    "disable_thinking": True,
    "logit_bias_enabled": False,
    "logit_bias_strength": 5.0,
    "timeout_seconds": 60.0,
    "max_concurrency": 8,
    "prompt_version": "v1",
}

_FIELD_ENV: Dict[str, str] = {
    "provider": "LLM2DECISION_PROVIDER",
    "api_key": "LLM2DECISION_API_KEY",
    "base_url": "LLM2DECISION_BASE_URL",
    "model": "LLM2DECISION_MODEL",
    "top_logprobs": "LLM2DECISION_TOP_LOGPROBS",
    "temperature_scale": "LLM2DECISION_TEMPERATURE_SCALE",
    "max_tokens": "LLM2DECISION_MAX_TOKENS",
    "disable_thinking": "LLM2DECISION_DISABLE_THINKING",
    "logit_bias_enabled": "LLM2DECISION_LOGIT_BIAS_ENABLED",
    "logit_bias_strength": "LLM2DECISION_LOGIT_BIAS_STRENGTH",
    "timeout_seconds": "LLM2DECISION_TIMEOUT_SECONDS",
    "max_concurrency": "LLM2DECISION_MAX_CONCURRENCY",
    "prompt_version": "LLM2DECISION_PROMPT_VERSION",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


_CONVERTERS: Dict[str, Callable[[Any], Any]] = {
    "provider": str,
    "api_key": str,
    "base_url": str,
    "model": str,
    "top_logprobs": int,
    "temperature_scale": float,
    "max_tokens": int,
    "disable_thinking": _to_bool,
    "logit_bias_enabled": _to_bool,
    "logit_bias_strength": float,
    "timeout_seconds": float,
    "max_concurrency": int,
    "prompt_version": str,
}


class UnknownModelError(ValueError):
    def __init__(self, name: str, available: List[str]) -> None:
        super().__init__(f"Unknown model route {name!r}, available routes: {', '.join(available)}")
        self.name = name
        self.available = available


@dataclass(frozen=True)
class ModelSettings:
    """The complete config for one model route."""

    api_key: str
    base_url: str
    model: str
    top_logprobs: int
    temperature_scale: float
    max_tokens: int
    disable_thinking: bool
    logit_bias_enabled: bool
    logit_bias_strength: float
    timeout_seconds: float
    max_concurrency: int
    prompt_version: str = "v1"
    # Provider profile name (see core/providers.py). Has a default so tests can construct it directly.
    provider: str = DEFAULT_PROVIDER


def _read_config(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    target = Path(path or os.getenv(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"The top level of config file {target} must be a mapping (key: value)")
    return data


def _resolve_field(field_name: str, section: Dict[str, Any], defaults: Dict[str, Any]) -> Any:
    """Resolve as route section > defaults section > env var; return None if none of the three has it."""
    converter = _CONVERTERS[field_name]
    for source in (section, defaults):
        raw = (source or {}).get(field_name)
        if raw not in (None, ""):
            return converter(raw)
    env_raw = os.getenv(_FIELD_ENV[field_name]) or None
    return converter(env_raw) if env_raw is not None else None


def _build_model(name: str, section: Dict[str, Any], defaults: Dict[str, Any]) -> ModelSettings:
    """Resolution precedence for one model route: its own section > defaults section > env var > built-in default.

    provider is resolved first on its own because some built-in defaults depend on it — most notably
    top_logprobs: the built-in default of 20 is Ark's slot count and DashScope only has 5, so carrying it
    over would be rejected by the startup validation with a confusing error.
    """
    provider_name = _resolve_field("provider", section, defaults) or DEFAULT_PROVIDER
    profile = get_profile(provider_name)  # raises UnknownProviderError here when provider is invalid

    values: Dict[str, Any] = {"provider": provider_name}
    for field in fields(ModelSettings):
        field_name = field.name
        if field_name == "provider":
            continue
        resolved = _resolve_field(field_name, section, defaults)
        if resolved is None:
            resolved = (
                profile.max_top_logprobs if field_name == "top_logprobs"
                else _DEFAULTS[field_name]
            )
        values[field_name] = resolved

    values["base_url"] = str(values["base_url"]).rstrip("/")
    if not values["model"]:
        raise ValueError(f"Model route {name!r} is missing model (model ID or inference endpoint ep-xxx)")

    if values["top_logprobs"] > profile.max_top_logprobs:
        raise ValueError(
            f"Model route {name!r} has top_logprobs above the cap {profile.max_top_logprobs}"
            f" (a hard limit of {profile.display_name}; exceeding it is rejected),"
            f" currently {values['top_logprobs']}"
        )

    # The most common cross-vendor misconfiguration: a new-vendor route omits base_url and inherits
    # another vendor's address from the defaults section. That sends requests to the wrong endpoint
    # with a confusing error, so reject it outright at startup.
    for other in PROVIDERS.values():
        if other.name != profile.name and values["base_url"] == other.default_base_url:
            raise ValueError(
                f"Model route {name!r} has provider={profile.name!r}, "
                f"but its base_url points at {other.display_name} ({other.default_base_url}). "
                f"A cross-vendor route must set base_url and api_key explicitly; it cannot inherit them from the defaults section."
            )
    return ModelSettings(**values)


@dataclass(frozen=True)
class Config:
    """Multi-model routing config: models maps route name -> config, default_model is the route used when none is specified."""

    models: Dict[str, ModelSettings]
    default_model: str

    @property
    def default(self) -> ModelSettings:
        return self.models[self.default_model]

    def model_names(self) -> List[str]:
        return sorted(self.models)

    def resolve(self, name: Optional[str] = None) -> ModelSettings:
        if not name:
            return self.default
        if name not in self.models:
            raise UnknownModelError(name, self.model_names())
        return self.models[name]

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        data = _read_config(path)
        defaults = data.get("defaults") or {}
        sections = data.get("models") or {}
        if not isinstance(defaults, dict):
            raise ValueError("The defaults section of the config file must be a mapping")
        if not isinstance(sections, dict):
            raise ValueError("The models section of the config file must be a mapping")
        if not sections:
            # With no config file, allow running from env vars alone (container setups): synthesize a route named default
            if not os.getenv("LLM2DECISION_MODEL"):
                raise ValueError(
                    "The config has no models section and the LLM2DECISION_MODEL env var is unset, so the model to call is unknown"
                )
            sections = {"default": {}}

        models = {
            str(name): _build_model(str(name), section or {}, defaults)
            for name, section in sections.items()
        }
        default_model = str(data.get("default_model") or "") or next(iter(models))
        if default_model not in models:
            raise ValueError(
                f"default_model={default_model!r} is not in models, available: {', '.join(models)}"
            )
        return cls(models=models, default_model=default_model)
