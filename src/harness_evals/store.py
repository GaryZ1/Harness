"""Append-only JSONL result store with atomic headers, transcripts and reports.

Layout::

    <root>/runs/<run_id>/
      header.json             # run conditions (atomic write; immutable once trials exist)
      trials.jsonl            # one TrialRecord per line (append under a cross-process lock + fsync)
      transcripts/<trial>.json
      reports/<name>          # markdown / JSON reports (atomic)

Invariants: run and trial ids are filename-safe; every JSONL append is a single ``write`` of one
complete line under :func:`harness_memory.locking.file_lock` and fsynced, so concurrent threads and
processes never interleave; readers skip a torn trailing line left by a crash (and report it)
rather than failing the whole run. Writes reuse :func:`harness_memory.locking.atomic_write_text`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

from harness_memory.locking import atomic_write_text, file_lock

from .errors import ResultStoreError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


def _check_id(value: str, what: str) -> str:
    if not _ID.match(value or ""):
        raise ResultStoreError(f"invalid {what}: {value!r}")
    return value


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        (self.root / "runs").mkdir(parents=True, exist_ok=True)
        self.torn_lines: dict[str, int] = {}

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / _check_id(run_id, "run id")

    # ------------------------------------------------------------------ header
    def write_header(self, run_id: str, header: Mapping[str, Any], *, overwrite: bool = False) -> None:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "header.json"
        with file_lock(d / ".lock"):
            if path.exists() and not overwrite:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("conditions") != header.get("conditions") or existing.get("suite", {}).get("hash") != header.get("suite", {}).get("hash"):
                    raise ResultStoreError(f"run {run_id} exists with different conditions or suite; use a new run id")
                return
            atomic_write_text(path, json.dumps(dict(header), indent=2, sort_keys=True, default=str))

    def read_header(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "header.json"
        if not path.is_file():
            raise ResultStoreError(f"no such run: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ trials
    def append(self, run_id: str, record: Mapping[str, Any]) -> None:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        if "\n" in line:
            raise ResultStoreError("record serialises to more than one line")
        with file_lock(d / ".lock"):
            with open(d / "trials.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def iter_records(self, run_id: str) -> Iterator[dict[str, Any]]:
        path = self.run_dir(run_id) / "trials.jsonl"
        if not path.is_file():
            return
        torn = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    torn += 1
        self.torn_lines[run_id] = torn

    def load_records(self, run_id: str) -> list[dict[str, Any]]:
        return list(self.iter_records(run_id))

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in (self.root / "runs").iterdir() if p.is_dir() and (p / "header.json").is_file())

    # ------------------------------------------------------------------ transcripts & reports
    def write_transcript(self, run_id: str, trial_id: str, transcript: Mapping[str, Any]) -> str:
        path = self.run_dir(run_id) / "transcripts" / f"{_check_id(trial_id, 'trial id')}.json"
        atomic_write_text(path, json.dumps(dict(transcript), ensure_ascii=False, default=str))
        return str(path.relative_to(self.root))

    def read_transcript(self, run_id: str, trial_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "transcripts" / f"{_check_id(trial_id, 'trial id')}.json"
        if not path.is_file():
            raise ResultStoreError(f"no transcript for {trial_id} in {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_report(self, run_id: str, name: str, text: str) -> Path:
        path = self.run_dir(run_id) / "reports" / _check_id(name, "report name")
        atomic_write_text(path, text)
        return path


__all__ = ["ResultStore"]
