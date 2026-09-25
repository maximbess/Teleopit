"""Append-only task event log.

`events.jsonl` is the execution history. A crash may leave a partial final
line; readers ignore that line and leave every complete line untouched.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_event(path: Path, event: str, **fields: Any) -> dict[str, Any]:
    """Append one event and fsync it. Existing lines are never rewritten."""
    record: dict[str, Any] = {"time": utc_now(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_RDWR, 0o644)
    try:
        if os.lseek(fd, 0, os.SEEK_END) > 0:
            os.lseek(fd, -1, os.SEEK_END)
            if os.read(fd, 1) != b"\n":
                os.write(fd, b"\n")
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def read_events(path: Path) -> list[dict[str, Any]]:
    """Return complete events. An incomplete final line is ignored."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    lines = text.splitlines()
    if not text.endswith("\n"):
        lines = lines[:-1]
    events: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A crash can leave a partial line. The next append seals it with a
            # newline so later events stay readable; the fragment is not an event.
            continue
    return events
