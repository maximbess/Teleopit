"""Task lifecycle for the autoresearch controller.

State is reconstructed from ``events.jsonl`` plus the worktree and git history.
``run`` is the restart entry point and repeats the same transitions after a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from autoresearch.agent import INSTALL_HINT, CursorAgent
from autoresearch.events import append_event, read_events

CREATED = "CREATED"
AGENT_RUNNING = "AGENT_RUNNING"
AGENT_FINISHED = "AGENT_FINISHED"
AGENT_FAILED = "AGENT_FAILED"
LOCAL_VALIDATION = "LOCAL_VALIDATION"
VALIDATION_FAILED = "VALIDATION_FAILED"
LOCAL_COMPLETE = "LOCAL_COMPLETE"
REVIEW = "REVIEW"
REVIEW_DONE = "REVIEW_DONE"
REVIEW_FAILED = "REVIEW_FAILED"
COMMITTED = "COMMITTED"
PUSHED = "PUSHED"
AWAITING_CLUSTER = "AWAITING_CLUSTER"
COMPLETE = "COMPLETE"
NEEDS_HUMAN = "NEEDS_HUMAN"
FAILED = "FAILED"
ABORTED = "ABORTED"

TERMINAL = {COMPLETE, AWAITING_CLUSTER, NEEDS_HUMAN, FAILED, ABORTED}
REPAIR_EVENTS = {
    "local_validation_failed",
    "protected_paths_modified",
    "agent_failed",
    "review_failed",
    "artifacts_missing",
    "agent_interrupted",
}
VALIDATION_FAILURE_MARKER = "Independent validation failed"
PROTECTED_MARKER = "Restore these protected files"
REPORT_MARKER = "Write the final report"
INTERRUPT_MARKER = "The previous run was interrupted"


class ControllerError(Exception):
    """A task cannot continue without a change in the checkout or the task files."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoresearch.controller")
    sub = parser.add_subparsers(dest="command", required=True)

    new_cmd = sub.add_parser("new", help="create a task directory from an idea")
    new_cmd.add_argument("--from-file", type=Path, help="read the idea from this file")
    new_cmd.add_argument("--run", action="store_true", help="run the task after creating it")

    run_cmd = sub.add_parser("run", help="advance a task until it stops")
    run_cmd.add_argument("task_id")

    status_cmd = sub.add_parser("status", help="show one task or every task")
    status_cmd.add_argument("task_id", nargs="?")

    continue_cmd = sub.add_parser("continue", help="resume a task with a follow-up")
    continue_cmd.add_argument("task_id")
    continue_cmd.add_argument("message", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    try:
        repo = find_repo(Path.cwd())
        if args.command == "new":
            idea = _read_idea(args.from_file)
            task_id = create_task(repo, idea)
            print(f"Created autoresearch/runs/{task_id}/idea.md")
            if args.run or _confirm_run():
                state = run_task(repo, task_id)
                print(f"{task_id}: {state}")
            return 0
        if args.command == "run":
            state = run_task(repo, args.task_id)
            print(f"{args.task_id}: {state}")
            return 0 if state in TERMINAL else 1
        if args.command == "status":
            print(format_status(repo, args.task_id))
            return 0
        if args.command == "continue":
            message = " ".join(args.message).strip()
            if not message:
                raise ControllerError("continue requires a follow-up message")
            state = continue_task(repo, args.task_id, message)
            print(f"{args.task_id}: {state}")
            return 0 if state in TERMINAL else 1
    except ControllerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


def find_repo(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "autoresearch" / "config.yaml").is_file() and (candidate / ".git").exists():
            return candidate
    raise ControllerError("could not find an autoresearch checkout (config.yaml next to .git)")


def create_task(repo: Path, idea: str) -> str:
    text = idea.strip()
    if not text:
        raise ControllerError("the task idea is empty")
    runs = repo / "autoresearch" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in runs.iterdir() if path.is_dir() and path.name.isdigit()]
    task_id = f"{(max(existing) if existing else 0) + 1:04d}"
    task = runs / task_id
    task.mkdir()
    (task / "idea.md").write_text(text + "\n", encoding="utf-8")
    _log(repo, task_id, "created")
    return task_id


def run_task(repo: Path, task_id: str, follow_up: str | None = None) -> str:
    cfg = load_config(repo)
    _require_task(repo, task_id)
    _prepare(repo, task_id, cfg, allow_dirty=follow_up is not None)
    if follow_up:
        _begin_follow_up(repo, task_id, cfg, follow_up)
    for _ in range(40):
        _raise_if_agent_alive(repo, task_id)
        state = derive_state(read_events(_events_path(repo, task_id)))
        if _step(repo, task_id, cfg, state) == "stop":
            return derive_state(read_events(_events_path(repo, task_id)))
    raise ControllerError(f"{task_id} did not reach a terminal state within 40 steps")


def continue_task(repo: Path, task_id: str, message: str) -> str:
    if not message.strip():
        raise ControllerError("continue requires a follow-up message")
    return run_task(repo, task_id, follow_up=message.strip())


def format_status(repo: Path, task_id: str | None) -> str:
    runs = repo / "autoresearch" / "runs"
    if task_id is None:
        if not runs.is_dir():
            return "no tasks"
        rows = []
        for path in sorted(child for child in runs.iterdir() if child.is_dir() and child.name.isdigit()):
            rows.append(_status_line(repo, path.name))
        return "\n".join(rows) if rows else "no tasks"
    _require_task(repo, task_id)
    events = read_events(_events_path(repo, task_id))
    lines = [_status_line(repo, task_id), ""]
    session = _read_session(repo, task_id)
    if session.get("session_id"):
        lines.append(f"session: {session['session_id']}")
    commit = _latest(events, "committed", "commit")
    if commit:
        lines.append(f"commit: {commit}")
    reason = _latest(events, "needs_human", "reason")
    if reason:
        lines.append(f"needs human: {reason}")
    lines.append("")
    tail = events[-10:]
    if not tail:
        lines.append("no events")
    else:
        lines.extend(f"{event['time']}  {event['event']}" for event in tail)
    return "\n".join(lines)


def load_config(repo: Path):
    path = repo / "autoresearch" / "config.yaml"
    if not path.is_file():
        raise ControllerError(f"missing {path}")
    return OmegaConf.load(path)


def derive_state(events: list[dict[str, Any]]) -> str:
    state = CREATED
    for event in events:
        name = event.get("event")
        if name == "created":
            state = CREATED
        elif name == "agent_started":
            state = AGENT_RUNNING
        elif name == "agent_finished":
            state = AGENT_FINISHED
        elif name == "agent_failed":
            state = AGENT_FAILED
        elif name == "local_validation_started":
            state = LOCAL_VALIDATION
        elif name == "local_validation_failed":
            state = VALIDATION_FAILED
        elif name == "local_validation_passed":
            state = LOCAL_COMPLETE
        elif name == "protected_paths_modified":
            state = VALIDATION_FAILED
        elif name in {
            "artifacts_missing",
            "agent_interrupted",
            "resume_failed",
            "reconciled",
            "base_recorded",
            "human_follow_up",
            "repair_budget_reset",
        }:
            continue
        elif name == "review_started":
            state = REVIEW
        elif name == "review_finished":
            state = REVIEW_DONE
        elif name == "review_failed":
            state = REVIEW_FAILED
        elif name == "committed":
            state = COMMITTED
        elif name == "pushed":
            state = PUSHED
        elif name == "awaiting_cluster":
            state = AWAITING_CLUSTER
        elif name == "completed":
            state = COMPLETE
        elif name == "needs_human":
            state = NEEDS_HUMAN
        elif name == "failed":
            state = FAILED
        elif name == "aborted":
            state = ABORTED
    return state


def repair_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        name = event.get("event")
        if name == "repair_budget_reset":
            count = 0
        elif name in REPAIR_EVENTS:
            count += 1
    return count


def changed_paths(repo: Path) -> list[str]:
    tracked = _git(repo, "diff", "--name-only", "HEAD").stdout.splitlines()
    staged = _git(repo, "diff", "--name-only", "--cached").stdout.splitlines()
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    return sorted({path for path in (*tracked, *staged, *untracked) if path})


def protected_hits(paths: list[str], protected: list[str], task_id: str) -> list[str]:
    allowed = f"autoresearch/runs/{task_id}/"
    hits = []
    for path in paths:
        if path.startswith(allowed):
            continue
        for guarded in protected:
            guarded = guarded.rstrip("/")
            if path == guarded or path.startswith(guarded + "/"):
                hits.append(path)
                break
    return hits


def find_task_commit(repo: Path, task_id: str) -> str | None:
    prefix = f"autoresearch: task {task_id} - "
    result = _git(repo, "log", "--format=%H%x1f%s", "-n", "50", check=False)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        sha, separator, subject = line.partition("\x1f")
        if separator and subject.startswith(prefix):
            return sha
    return None


def _step(repo: Path, task_id: str, cfg, state: str) -> str:
    if state in TERMINAL:
        return "stop"
    if state == AGENT_RUNNING:
        _resume_interrupted(repo, task_id, cfg)
        return "continue"
    if state == AGENT_FINISHED:
        if _stop_for_human_file(repo, task_id):
            return "stop"
        if not _has_implementation_notes(repo, task_id):
            _request_notes(repo, task_id, cfg, kind="agent")
            return "continue"
        _validate(repo, task_id, cfg)
        return "continue"
    if state in {AGENT_FAILED, VALIDATION_FAILED}:
        _resume_repair(repo, task_id, cfg, kind="agent")
        return "continue"
    if state == LOCAL_VALIDATION:
        _validate(repo, task_id, cfg)
        return "continue"
    if state == LOCAL_COMPLETE:
        if _reject_protected(repo, task_id, cfg):
            return "continue"
        if (_task_dir(repo, task_id) / "cluster.md").is_file():
            _commit(repo, task_id)
            return "continue"
        if not (_task_dir(repo, task_id) / "report.md").is_file():
            _invoke(repo, task_id, cfg, _prompt(repo, "review.md", task_id), kind="review")
            return "continue"
        _commit(repo, task_id)
        return "continue"
    if state == REVIEW:
        _invoke(repo, task_id, cfg, _prompt(repo, "review.md", task_id), kind="review")
        return "continue"
    if state == REVIEW_FAILED:
        _resume_repair(repo, task_id, cfg, kind="review")
        return "continue"
    if state == REVIEW_DONE:
        if _stop_for_human_file(repo, task_id):
            return "stop"
        if _reject_protected(repo, task_id, cfg):
            return "continue"
        if not (_task_dir(repo, task_id) / "report.md").is_file():
            _request_notes(repo, task_id, cfg, kind="review")
            return "continue"
        _validate(repo, task_id, cfg)
        return "continue"
    if state == COMMITTED:
        _push(repo, task_id, cfg)
        return "continue"
    if state == PUSHED:
        if (_task_dir(repo, task_id) / "cluster.md").is_file():
            _log(repo, task_id, "awaiting_cluster")
        else:
            _log(repo, task_id, "completed")
        return "stop"
    if state == CREATED:
        _invoke(repo, task_id, cfg, _prompt(repo, "start_task.md", task_id), kind="agent")
        return "continue"
    raise ControllerError(f"unhandled state {state} for task {task_id}")


def _prepare(repo: Path, task_id: str, cfg, allow_dirty: bool) -> None:
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != str(cfg.branch):
        raise ControllerError(
            f"autoresearch tasks must run on branch {cfg.branch}; current branch is {branch}"
        )
    _reconcile(repo, task_id)
    events = read_events(_events_path(repo, task_id))
    if not any(event.get("event") == "base_recorded" for event in events):
        task_sha = find_task_commit(repo, task_id)
        if task_sha:
            parent = _git(repo, "rev-parse", f"{task_sha}^").stdout.strip()
            _log(repo, task_id, "base_recorded", commit=parent, reconciled=True)
        else:
            _log(repo, task_id, "base_recorded", commit=_head(repo))
        events = read_events(_events_path(repo, task_id))
    _assert_history(repo, task_id, events, allow_dirty=allow_dirty)


def _assert_history(repo: Path, task_id: str, events: list[dict[str, Any]], allow_dirty: bool) -> None:
    head = _head(repo)
    base = _latest(events, "base_recorded", "commit")
    task_sha = _latest(events, "committed", "commit") or find_task_commit(repo, task_id)
    if task_sha and head != task_sha:
        raise ControllerError(f"HEAD {head} does not match task commit {task_sha}")
    if not task_sha and base and head != base:
        raise ControllerError(f"HEAD {head} does not match recorded base {base}")
    started = any(event.get("event") == "agent_started" for event in events)
    task_prefix = f"autoresearch/runs/{task_id}/"
    if task_sha and not allow_dirty:
        extra = [path for path in changed_paths(repo) if not path.startswith(task_prefix)]
        if extra:
            raise ControllerError(
                "unexpected uncommitted changes after the task commit: " + ", ".join(extra)
            )
    if not started and not allow_dirty:
        extra = [path for path in changed_paths(repo) if not path.startswith(f"autoresearch/runs/{task_id}/")]
        if extra:
            raise ControllerError("unexpected uncommitted changes: " + ", ".join(extra))


def _reconcile(repo: Path, task_id: str) -> None:
    events = read_events(_events_path(repo, task_id))
    if _open_span(events, "agent_started", "agent_finished") and not _session_pid_alive(repo, task_id):
        if _output_succeeded(_latest_output(repo, task_id)):
            _log(repo, task_id, "agent_finished", exit_code=0, reconciled=True)
            _log(repo, task_id, "reconciled", what="agent_finished")
    events = read_events(_events_path(repo, task_id))
    if _open_span(events, "review_started", "review_finished") and not _session_pid_alive(repo, task_id):
        if _output_succeeded(_latest_output(repo, task_id)):
            _log(repo, task_id, "review_finished", exit_code=0, reconciled=True)
            _log(repo, task_id, "reconciled", what="review_finished")
    events = read_events(_events_path(repo, task_id))
    if not any(event.get("event") == "committed" for event in events):
        sha = find_task_commit(repo, task_id)
        if sha:
            _log(repo, task_id, "committed", commit=sha, reconciled=True)
            _log(repo, task_id, "reconciled", what="committed")


def _begin_follow_up(repo: Path, task_id: str, cfg, message: str) -> None:
    pending = _task_dir(repo, task_id) / "needs_human.md"
    if pending.is_file():
        pending.rename(_task_dir(repo, task_id) / "needs_human.acknowledged.md")
    _log(repo, task_id, "repair_budget_reset")
    _log(repo, task_id, "human_follow_up")
    prompt = (
        f"{_system(repo)}\n\nTask {task_id}\n\nThe researcher replied:\n\n{message}\n\n"
        "Continue the task. Update plan.md and implementation.md in the task directory."
    )
    _invoke(repo, task_id, cfg, prompt, kind="agent")


def _resume_interrupted(repo: Path, task_id: str, cfg) -> None:
    if not _budget_left(repo, task_id, cfg, "agent stopped responding before it finished"):
        return
    _log(repo, task_id, "agent_interrupted")
    prompt = (
        f"{_system(repo)}\n\nTask {task_id}\n\n{INTERRUPT_MARKER} before it finished. "
        "Continue from the current files. Do not repeat work that is already present."
    )
    _invoke(repo, task_id, cfg, prompt, kind="agent")


def _resume_repair(repo: Path, task_id: str, cfg, kind: str) -> None:
    events = read_events(_events_path(repo, task_id))
    cause = _latest_event_named(
        events,
        {"local_validation_failed", "protected_paths_modified", "agent_failed", "review_failed"},
    )
    cause_name = cause.get("event") if cause else None
    if cause_name == "local_validation_failed" and cause is not None:
        reason = "validation failed and the retry limit was reached"
        prompt = _validation_prompt(repo, task_id, cause)
    elif cause_name == "protected_paths_modified":
        reason = "protected files stayed modified and the retry limit was reached"
        prompt = _protected_prompt(repo, task_id, cause)
    else:
        reason = "the agent failed and the retry limit was reached"
        prompt = (
            f"{_system(repo)}\n\nTask {task_id}\n\nThe previous agent invocation failed. "
            "Inspect the task directory and continue the task."
        )
    if not _budget_left(repo, task_id, cfg, reason):
        return
    _invoke(repo, task_id, cfg, prompt, kind=kind)


def _request_notes(repo: Path, task_id: str, cfg, kind: str) -> None:
    missing = "report.md" if kind == "review" else "plan.md and implementation.md"
    reason = f"agent finished without {missing}"
    if not _budget_left(repo, task_id, cfg, reason):
        return
    _log(repo, task_id, "artifacts_missing", missing=missing)
    if kind == "review":
        prompt = _prompt(repo, "review.md", task_id)
    else:
        prompt = (
            f"{_system(repo)}\n\nTask {task_id}\n\n"
            f"{missing} is missing. Write both files in the task directory, then stop."
        )
    _invoke(repo, task_id, cfg, prompt, kind=kind)


def _reject_protected(repo: Path, task_id: str, cfg) -> bool:
    hits = protected_hits(changed_paths(repo), [str(path) for path in cfg.protected_paths], task_id)
    if not hits:
        return False
    if not _budget_left(repo, task_id, cfg, "protected files were modified and the retry limit was reached"):
        return True
    event = _log(repo, task_id, "protected_paths_modified", paths=hits)
    _invoke(repo, task_id, cfg, _protected_prompt(repo, task_id, event), kind="agent")
    return True


def _budget_left(repo: Path, task_id: str, cfg, reason: str) -> bool:
    events = read_events(_events_path(repo, task_id))
    if repair_count(events) >= int(cfg.max_validation_attempts):
        _log(repo, task_id, "needs_human", reason=reason)
        return False
    return True


def _validate(repo: Path, task_id: str, cfg) -> None:
    commands = [str(command) for command in cfg.validation.commands]
    attempt = 1 + sum(
        1
        for event in read_events(_events_path(repo, task_id))
        if event.get("event") == "local_validation_started"
    )
    log_path = _task_dir(repo, task_id) / "validation" / f"{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(repo, task_id, "local_validation_started", attempt=attempt, commands=commands)
    chunks: list[str] = []
    failed: tuple[str, int] | None = None
    for command in commands:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=repo,
            text=True,
            capture_output=True,
        )
        chunks.append(
            f"$ {command}\nexit {completed.returncode}\n{completed.stdout}{completed.stderr}\n"
        )
        if completed.returncode != 0:
            failed = (command, completed.returncode)
            break
    log_path.write_text("".join(chunks), encoding="utf-8")
    relative = log_path.relative_to(repo).as_posix()
    if failed is None:
        _log(repo, task_id, "local_validation_passed", log=relative)
        return
    command, exit_code = failed
    _log(
        repo,
        task_id,
        "local_validation_failed",
        command=command,
        exit_code=exit_code,
        log=relative,
    )


def _commit(repo: Path, task_id: str) -> None:
    hits = protected_hits(
        changed_paths(repo),
        [str(path) for path in load_config(repo).protected_paths],
        task_id,
    )
    if hits:
        raise ControllerError("refusing to commit protected paths: " + ", ".join(hits))
    existing = find_task_commit(repo, task_id)
    if existing:
        if not any(
            event.get("event") == "committed" and event.get("commit") == existing
            for event in read_events(_events_path(repo, task_id))
        ):
            _log(repo, task_id, "committed", commit=existing, reconciled=True)
        return
    message = f"autoresearch: task {task_id} - {_task_title(repo, task_id)}"
    _git(repo, "add", "-A")
    if _git(repo, "status", "--porcelain").stdout.strip():
        _git(repo, "commit", "-m", message)
    else:
        _git(repo, "commit", "--allow-empty", "-m", message)
    _log(repo, task_id, "committed", commit=_head(repo))


def _push(repo: Path, task_id: str, cfg) -> None:
    if not bool(cfg.push):
        _log(repo, task_id, "pushed", remote=None, skipped=True)
        return
    remote = str(cfg.remote)
    branch = str(cfg.branch)
    if remote != "origin":
        raise ControllerError(f"refusing to push to remote {remote}; only origin is allowed")
    completed = _git(repo, "push", "-u", remote, f"HEAD:{branch}", check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ControllerError(f"git push failed: {detail}")
    _log(repo, task_id, "pushed", remote=remote, branch=branch)


def _invoke(repo: Path, task_id: str, cfg, prompt: str, kind: str) -> None:
    _raise_if_agent_alive(repo, task_id)
    agent = CursorAgent(
        bin_name=str(cfg.cursor.bin),
        model=str(cfg.model),
        sandbox=str(cfg.cursor.sandbox),
        force=bool(cfg.cursor.force),
    )
    start_event = "agent_started" if kind == "agent" else "review_started"
    finish_event = "agent_finished" if kind == "agent" else "review_finished"
    fail_event = "agent_failed" if kind == "agent" else "review_failed"
    session_id = _latest_session_id(read_events(_events_path(repo, task_id)))
    output = _next_output(repo, task_id)
    try:
        result = _spawn(repo, task_id, agent, prompt, output, start_event, session_id)
        if session_id and result.exit_code != 0 and not result.result_text:
            _log(repo, task_id, "resume_failed", session_id=session_id, exit_code=result.exit_code)
            fresh = (
                f"{_system(repo)}\n\nConversation resume failed. "
                "Continue from the task files already on disk.\n\n"
                + prompt
            )
            output = _next_output(repo, task_id)
            result = _spawn(repo, task_id, agent, fresh, output, start_event, None)
    except FileNotFoundError as exc:
        raise ControllerError(str(exc) or INSTALL_HINT) from exc
    finally:
        _write_session(repo, task_id, pid=None)
    if result.ok:
        _log(
            repo,
            task_id,
            finish_event,
            exit_code=result.exit_code,
            session_id=result.session_id,
            output=output.relative_to(repo).as_posix(),
        )
        return
    _log(
        repo,
        task_id,
        fail_event,
        exit_code=result.exit_code,
        session_id=result.session_id,
        output=output.relative_to(repo).as_posix(),
    )


def _spawn(repo, task_id, agent: CursorAgent, prompt: str, output: Path, start_event: str, session_id: str | None):
    logged = {"done": False}

    def on_start(pid: int) -> None:
        session = _read_session(repo, task_id)
        session["pid"] = pid
        _write_session(repo, task_id, **session)

    def on_session(new_session: str, model: str | None, pid: int) -> None:
        if logged["done"]:
            return
        logged["done"] = True
        _log(
            repo,
            task_id,
            start_event,
            session_id=new_session,
            model=model or agent.model,
            pid=pid,
        )
        _write_session(
            repo,
            task_id,
            session_id=new_session,
            model=model or agent.model,
            pid=pid,
        )

    if session_id:
        return agent.resume(session_id, prompt, repo, output, on_start, on_session)
    return agent.start(prompt, repo, output, on_start, on_session)


def _stop_for_human_file(repo: Path, task_id: str) -> bool:
    path = _task_dir(repo, task_id) / "needs_human.md"
    if not path.is_file():
        return False
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reason = lines[0] if lines else "agent requested human input"
    _log(repo, task_id, "needs_human", reason=reason)
    return True


def _has_implementation_notes(repo: Path, task_id: str) -> bool:
    task = _task_dir(repo, task_id)
    return (task / "plan.md").is_file() and (task / "implementation.md").is_file()


def _validation_prompt(repo: Path, task_id: str, failure: dict[str, Any]) -> str:
    log_path = repo / str(failure.get("log", ""))
    body = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    excerpt = "\n".join(body.splitlines()[-200:])
    if len(excerpt) > 8000:
        excerpt = excerpt[-8000:]
    return (
        f"{_system(repo)}\n\nTask {task_id}\n\n{VALIDATION_FAILURE_MARKER}.\n"
        f"Command: {failure.get('command')}\n"
        f"Exit code: {failure.get('exit_code')}\n"
        f"Output tail:\n{excerpt}\n\n"
        "Diagnose and fix the failure. Do not claim success unless the fix is applied."
    )


def _protected_prompt(repo: Path, task_id: str, event: dict[str, Any] | None) -> str:
    paths = ", ".join(event.get("paths", [])) if event else ""
    return (
        f"{_system(repo)}\n\nTask {task_id}\n\n{PROTECTED_MARKER}:\n{paths}\n\n"
        "Return those files to the current HEAD contents. "
        f"Put task notes only under autoresearch/runs/{task_id}/."
    )


def _prompt(repo: Path, name: str, task_id: str) -> str:
    body = (repo / "autoresearch" / "prompts" / name).read_text(encoding="utf-8")
    return f"{_system(repo)}\n\n{body.replace('{task_id}', task_id)}"


def _system(repo: Path) -> str:
    return (repo / "autoresearch" / "prompts" / "system.md").read_text(encoding="utf-8").strip()


def _task_title(repo: Path, task_id: str) -> str:
    idea = (_task_dir(repo, task_id) / "idea.md").read_text(encoding="utf-8")
    line = next((item.strip() for item in idea.splitlines() if item.strip()), "task")
    line = " ".join(line.split())
    if len(line) > 72:
        return line[:69] + "..."
    return line


def _log(repo: Path, task_id: str, event: str, **fields: Any) -> dict[str, Any]:
    record = append_event(_events_path(repo, task_id), event, task_id=task_id, **fields)
    print(f"{task_id}: {event}")
    return record


def _events_path(repo: Path, task_id: str) -> Path:
    return _task_dir(repo, task_id) / "events.jsonl"


def _task_dir(repo: Path, task_id: str) -> Path:
    return repo / "autoresearch" / "runs" / task_id


def _require_task(repo: Path, task_id: str) -> None:
    if not task_id.isdigit() or len(task_id) < 4:
        raise ControllerError(f"task id must be a zero-padded number, got {task_id}")
    if not _task_dir(repo, task_id).is_dir():
        raise ControllerError(f"task {task_id} does not exist")


def _status_line(repo: Path, task_id: str) -> str:
    events = read_events(_events_path(repo, task_id))
    state = derive_state(events) if events else CREATED
    commit = _latest(events, "committed", "commit")
    suffix = f"  {commit[:12]}" if commit else ""
    return f"{task_id}  {state}{suffix}"


def _latest(events: list[dict[str, Any]], name: str, field: str) -> Any:
    event = _latest_event(events, name)
    if event is None:
        return None
    return event.get(field)


def _latest_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return _latest_event_named(events, {name})


def _latest_event_named(events: list[dict[str, Any]], names: set[str]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") in names:
            return event
    return None


def _latest_session_id(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("event") in {"agent_started", "review_started"} and event.get("session_id"):
            return str(event["session_id"])
    return None


def _open_span(events: list[dict[str, Any]], start: str, end: str) -> bool:
    last_start = -1
    last_end = -1
    for index, event in enumerate(events):
        if event.get("event") == start:
            last_start = index
        elif event.get("event") == end:
            last_end = index
    return last_start >= 0 and last_end < last_start


def _output_succeeded(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    exit_path = path.with_suffix(".exit")
    if exit_path.is_file() and exit_path.read_text(encoding="utf-8").strip() not in {"", "0"}:
        return False
    result = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") == "result":
            result = parsed
    if not result or result.get("is_error") or result.get("subtype") not in (None, "success"):
        return False
    return result.get("result") is not None


def _latest_output(repo: Path, task_id: str) -> Path | None:
    output = _task_dir(repo, task_id) / "agent" / "output"
    files = sorted(output.glob("*.jsonl")) if output.is_dir() else []
    return files[-1] if files else None


def _next_output(repo: Path, task_id: str) -> Path:
    output = _task_dir(repo, task_id) / "agent" / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{len(list(output.glob('*.jsonl'))) + 1:03d}.jsonl"


def _session_path(repo: Path, task_id: str) -> Path:
    return _task_dir(repo, task_id) / "agent" / "session.json"


def _read_session(repo: Path, task_id: str) -> dict[str, Any]:
    path = _session_path(repo, task_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_session(repo: Path, task_id: str, **fields: Any) -> None:
    path = _session_path(repo, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_session(repo, task_id)
    current.update(fields)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _session_pid_alive(repo: Path, task_id: str) -> bool:
    pid = _read_session(repo, task_id).get("pid")
    return _pid_alive(pid)


def _raise_if_agent_alive(repo: Path, task_id: str) -> None:
    events = read_events(_events_path(repo, task_id))
    open_agent = _open_span(events, "agent_started", "agent_finished") or _open_span(
        events, "review_started", "review_finished"
    )
    if open_agent and _session_pid_alive(repo, task_id):
        pid = _read_session(repo, task_id).get("pid")
        raise ControllerError(f"Cursor agent is already running (pid {pid})")


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ControllerError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _read_idea(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    if sys.stdin.isatty():
        print("Describe task/research idea:\n")
        lines: list[str] = []
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip() and lines:
                break
            lines.append(line)
        return "\n".join(lines)
    return sys.stdin.read()


def _confirm_run() -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("Run now? [Y/n] ").strip().lower()
    except EOFError:
        return False
    return answer in {"", "y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
