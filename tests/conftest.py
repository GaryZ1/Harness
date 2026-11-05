from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_memory import FileMemoryStore, Scope  # noqa: E402


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw) -> None:
        self.now = self.now + timedelta(**kw)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "mem", scope=Scope.PROJECT, clock=clock)
