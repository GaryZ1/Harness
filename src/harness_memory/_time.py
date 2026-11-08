"""Time helpers. All timestamps are timezone-aware UTC and serialised as ISO-8601 with a ``Z`` suffix."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

Clock = Callable[[], datetime]


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime truncated to whole seconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_iso(dt: datetime | None) -> str | None:
    """Serialise ``dt`` as ``YYYY-MM-DDTHH:MM:SSZ`` (UTC). ``None`` passes through."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | datetime | None) -> datetime | None:
    """Parse an ISO-8601 string (``Z`` or offset) into an aware UTC datetime.

    YAML loaders may already have produced a ``datetime``; naive values are assumed UTC.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compact_stamp(dt: datetime) -> str:
    """Return a filename-safe timestamp, e.g. ``20261103T100000Z``."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
