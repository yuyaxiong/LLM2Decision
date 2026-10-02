from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence

MAX_CANDIDATES = 20

_SAFE_HANDLE_CHARS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TRIM_CHARS = ".:、,，)）]】\"'`*#；;"


def numbered_handles(count: int) -> List[str]:
    """Digits for 1..9, letters for 10..20. Both digits and letters are single tokens in mainstream tokenizers."""
    if count < 1 or count > MAX_CANDIDATES:
        raise ValueError(f"the candidate count must be between 1 and {MAX_CANDIDATES}, currently {count}")
    if count <= 9:
        return [str(i + 1) for i in range(count)]
    return list(_LETTERS[:count])


def make_handles(labels: Sequence[str]) -> List[str]:
    """When the labels themselves are all unique single characters (e.g. a 0-5 scale), reuse them directly; otherwise fall back to numbered handles."""
    if labels and len(labels) <= MAX_CANDIDATES:
        upper = [label.strip().upper() for label in labels]
        is_single = all(len(item) == 1 and item in _SAFE_HANDLE_CHARS for item in upper)
        if is_single and len(set(upper)) == len(upper):
            return upper
    return numbered_handles(len(labels))


def normalize_token(token: str) -> str:
    # NFKC folds full-width digits/letters (e.g. １ Ａ) to half-width, so a label can't miss a match because of its encoding form
    return unicodedata.normalize("NFKC", token).strip().strip(_TRIM_CHARS).strip().upper()


def build_lookup(handles: Sequence[str], aliases: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, List[int]]:
    """Normalized token text -> list of candidate indices it may map to."""
    lookup: Dict[str, List[int]] = {}
    for index, handle in enumerate(handles):
        lookup.setdefault(normalize_token(handle), []).append(index)
    for handle, extra_tokens in (aliases or {}).items():
        if handle not in handles:
            continue
        index = handles.index(handle)
        for alias in extra_tokens:
            key = normalize_token(alias)
            if key and index not in lookup.setdefault(key, []):
                lookup[key].append(index)
    return lookup
