"""Cursor CLI adapter.

The controller talks to Cursor only through this module so a later SDK
backend can replace the subprocess without changing task transitions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from autoresearch.events import utc_now


INSTALL_HINT = (
    "Cursor CLI 'cursor-agent' was not found. Install it with "
    "`curl -sS https://cursor.com/install | bash`, then run `cursor-agent login`."
)


@dataclass
class AgentResult:
    session_id: str | None
    model: str | None
    exit_code: int
    output_path: Path
    result_text: str | None
    is_error: bool
    started_at: str
    ended_at: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.is_error and self.result_text is not None


OnStart = Callable[[int], None]
OnSession = Callable[[str, str | None, int], None]


class CursorAgent:
    def __init__(self, bin_name: str, model: str, sandbox: str = "disabled", force: bool = True):
        self.bin_name = bin_name
        self.model = model
        self.sandbox = sandbox
        self.force = force

    def start(
        self,
        prompt: str,
        cwd: Path,
        output_path: Path,
        on_start: OnStart,
        on_session: OnSession,
    ) -> AgentResult:
        return self._run(prompt, cwd, output_path, on_start, on_session, resume=None)

    def resume(
        self,
        session_id: str,
        prompt: str,
        cwd: Path,
        output_path: Path,
        on_start: OnStart,
        on_session: OnSession,
    ) -> AgentResult:
        return self._run(prompt, cwd, output_path, on_start, on_session, resume=session_id)

    def _run(
        self,
        prompt: str,
        cwd: Path,
        output_path: Path,
        on_start: OnStart,
        on_session: OnSession,
        resume: str | None,
    ) -> AgentResult:
        executable = _resolve_bin(self.bin_name)
        command = [
            executable,
            "--print",
            "--trust",
            "--output-format",
            "stream-json",
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--workspace",
            str(cwd),
        ]
        if self.force:
            command.append("--force")
        if resume:
            command.extend(["--resume", resume])
        command.append(prompt)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = utc_now()
        session_id: str | None = None
        model: str | None = None
        result_text: str | None = None
        is_error = True
        with output_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            on_start(process.pid)
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                parsed = _parse_json_line(line)
                if parsed is None:
                    continue
                if parsed.get("type") == "system" and parsed.get("subtype") == "init":
                    session_id = str(parsed.get("session_id") or "") or None
                    model = str(parsed.get("model") or self.model)
                    if session_id:
                        on_session(session_id, model, process.pid)
                elif parsed.get("type") == "result":
                    result_text = parsed.get("result")
                    if result_text is not None:
                        result_text = str(result_text)
                    session_id = str(parsed.get("session_id") or session_id or "") or session_id
                    is_error = bool(parsed.get("is_error")) or parsed.get("subtype") not in (None, "success")
            exit_code = process.wait()
        ended_at = utc_now()
        exit_path = output_path.with_suffix(".exit")
        exit_path.write_text(f"{exit_code}\n", encoding="utf-8")
        if exit_code != 0:
            is_error = True
        if result_text is None:
            is_error = True
        return AgentResult(
            session_id=session_id,
            model=model or self.model,
            exit_code=exit_code,
            output_path=output_path,
            result_text=result_text,
            is_error=is_error,
            started_at=started_at,
            ended_at=ended_at,
        )


def _resolve_bin(bin_name: str) -> str:
    candidate = Path(bin_name)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(bin_name)
    if found:
        return found
    raise FileNotFoundError(INSTALL_HINT)


def _parse_json_line(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None
