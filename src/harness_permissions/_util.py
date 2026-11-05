"""Small shared helpers: canonical hashing, text normalisation for fuzzy (deny-side) matching."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

# Zero-width and bidi-control characters used to smuggle look-alike text past string matchers.
_INVISIBLE = re.compile("[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮⁠-⁤⁦-⁯﻿]")


def canonical_json(value: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) used for hashing and audit records."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_hex(value: Any) -> str:
    """SHA-256 of ``value`` (strings hashed as UTF-8, everything else via :func:`canonical_json`)."""
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_invisible(text: str) -> bool:
    """True when ``text`` contains zero-width or bidi-override characters."""
    return bool(_INVISIBLE.search(text))


def fold(text: str) -> str:
    """Aggressive normalisation for *deny* matching: NFKC, strip invisibles, casefold.

    Deny rules match conservatively (a false positive denies a harmless look-alike); allow rules
    never use this, so a look-alike can never be *allowed* by a rule written for the real thing.
    """
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).casefold()


def non_ascii_lookalikes(text: str) -> list[str]:
    """Characters whose NFKC form differs from themselves (full-width letters, ligatures...)."""
    return sorted({ch for ch in text if ord(ch) > 127 and unicodedata.normalize("NFKC", ch) != ch})


def truncate(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
