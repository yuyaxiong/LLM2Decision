"""Versioned loading and rendering of prompt templates.

Template bodies live under ``templates/``, named by version (e.g. ``choice_v1.txt``). Each version has
four templates: ``system`` / ``choice`` / ``noul`` / ``score``. Templates use ``{placeholder}`` syntax
(``str.format``) to mark runtime interpolation points: ``{state}``, ``{instructions}``,
``{options}`` / ``{levels}``, ``{example}``.

A version change affects evaluation results, so benchmark results must be tied to the prompt version
that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..core.schema import Question

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_KINDS = ("system", "choice", "noul", "score")
# In-process cache: each version is read from disk only once.
_CACHE: Dict[str, "PromptSet"] = {}


class UnknownPromptVersionError(ValueError):
    def __init__(self, version: str, available: List[str]) -> None:
        listed = ", ".join(available) if available else "none"
        super().__init__(f"Unknown prompt version {version!r}, available versions: {listed}")
        self.version = version
        self.available = available


@dataclass(frozen=True)
class PromptSet:
    """The four template bodies contained in one prompt version."""

    version: str
    system: str
    choice: str
    noul: str
    score: str


def available_versions() -> List[str]:
    """List the versions present under templates/ (keyed on system_<version>.txt)."""
    versions = {
        path.stem[len("system_"):]
        for path in _TEMPLATES_DIR.glob("system_*.txt")
        if path.stem.startswith("system_") and path.stem != "system_"
    }
    return sorted(versions)


def _read_template(path: Path) -> str:
    # A trailing newline is just editor habit, not template content: drop one so the rendered result
    # stays byte-for-byte identical to the old implementation.
    text = path.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def load_prompt_set(version: str = "v1") -> PromptSet:
    """Load a version's template set; skip disk on an in-process cache hit, and on a missing version raise with the available versions listed."""
    cached = _CACHE.get(version)
    if cached is not None:
        return cached
    templates: Dict[str, str] = {}
    for kind in _KINDS:
        path = _TEMPLATES_DIR / f"{kind}_{version}.txt"
        if not path.exists():
            raise UnknownPromptVersionError(version, available_versions())
        templates[kind] = _read_template(path)
    prompt_set = PromptSet(version=version, **templates)
    _CACHE[version] = prompt_set
    return prompt_set


def build_messages(
    state: str,
    question: Question,
    items: Sequence[Tuple[str, str, str]],
    version: str = "v1",
) -> List[dict]:
    """items is [(handle, raw label, description)], and the order is the candidate order."""
    prompt_set = load_prompt_set(version)
    locale = _locale_for(version)
    if question.type == "choice":
        content = _choice_content(prompt_set, locale, state, question, items)
    elif question.type == "noul":
        content = _noul_content(prompt_set, state, question)
    else:
        content = _score_content(prompt_set, locale, state, question, items)
    return [
        {"role": "system", "content": prompt_set.system},
        {"role": "user", "content": content},
    ]


@dataclass(frozen=True)
class _Locale:
    """Language-dependent bits that are assembled in code rather than interpolated from a file.

    Two things aren't in the template bodies: the separator inside each candidate line, and
    the fallback instructions used when a caller leaves `instructions` empty. Both are
    rendered here, so they need a per-version table.

    **v1 is frozen.** Its rendering is byte-locked by a test because every published benchmark
    number was measured on it — so the v1 locale must never change, only be added to.
    """

    option_separator: str
    default_choice_instructions: str
    default_score_instructions: str


_LOCALES: Dict[str, _Locale] = {
    "v1": _Locale(
        option_separator="：",
        default_choice_instructions="从下列候选中选出最合适的一个。",
        default_score_instructions="请把输入放到下列有序档位中的某一档。",
    ),
    "v2": _Locale(
        option_separator=": ",
        default_choice_instructions="Pick the most fitting option below.",
        default_score_instructions="Place the input in one of the ordered levels below.",
    ),
}

# Unreachable in practice: load_prompt_set rejects an unknown version before rendering starts.
# Kept so _locale_for can't raise on a version that somehow slipped through.
_FALLBACK_LOCALE = _LOCALES["v1"]


def _locale_for(version: str) -> _Locale:
    return _LOCALES.get(version, _FALLBACK_LOCALE)


def _render_options(items: Sequence[Tuple[str, str, str]], locale: _Locale) -> str:
    lines = []
    for handle, label, description in items:
        if description:
            lines.append(f"{handle}. {label}{locale.option_separator}{description}")
        else:
            lines.append(f"{handle}. {label}")
    return "\n".join(lines)


def _choice_content(
    prompt_set: PromptSet,
    locale: _Locale,
    state: str,
    question: Question,
    items: Sequence[Tuple[str, str, str]],
) -> str:
    handles = [handle for handle, _, _ in items]
    example = handles[0] if handles else "1"
    instructions = question.instructions.strip() or locale.default_choice_instructions
    return prompt_set.choice.format(
        state=state,
        instructions=instructions,
        options=_render_options(items, locale),
        example=example,
    )


def _noul_content(prompt_set: PromptSet, state: str, question: Question) -> str:
    return prompt_set.noul.format(
        state=state,
        instructions=question.instructions.strip(),
    )


def _score_content(
    prompt_set: PromptSet,
    locale: _Locale,
    state: str,
    question: Question,
    items: Sequence[Tuple[str, str, str]],
) -> str:
    handles = [handle for handle, _, _ in items]
    example = handles[0] if handles else "1"
    instructions = question.instructions.strip() or locale.default_score_instructions
    levels = "\n".join(f"{handle}. {label}" for handle, label, _ in items)
    return prompt_set.score.format(
        state=state,
        instructions=instructions,
        levels=levels,
        example=example,
    )
