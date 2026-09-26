"""Controller tests using a fake cursor-agent binary and temporary git repos."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from autoresearch.controller import (
    COMPLETE,
    AWAITING_CLUSTER,
    NEEDS_HUMAN,
    ControllerError,
    continue_task,
    create_task,
    find_task_commit,
    format_status,
    main,
    protected_hits,
    run_task,
)
from autoresearch.events import append_event, read_events

FAKE_AGENT = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
prompt = args[-1]
workspace = Path(args[args.index("--workspace") + 1])
scenario = os.environ["FAKE_SCENARIO"]
session = os.environ.get("FAKE_SESSION", "session-1")
task = workspace / "autoresearch" / "runs" / "0001"

def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}), flush=True)

def record():
    task.mkdir(parents=True, exist_ok=True)
    (task / "agent").mkdir(parents=True, exist_ok=True)
    mode = "resume" if "--resume" in args else "fresh"
    flags = []
    if prompt.startswith("system"):
        flags.append("system")
    if "Independent validation failed" in prompt:
        flags.append("validation")
    if "Write the final report" in prompt:
        flags.append("report")
    if "idea.md:" in prompt:
        flags.append("brief")
    with (task / "agent" / "invocations.log").open("a", encoding="utf-8") as handle:
        handle.write(mode + "\\t" + " ".join(flags) + "\\n")

record()

if scenario == "resume_fail" and "--resume" in args:
    emit("system", subtype="init", session_id=session, model="grok-4.7")
    sys.exit(2)

emit("system", subtype="init", session_id=session, model="grok-4.7")
task.mkdir(parents=True, exist_ok=True)

if "Write the final report" in prompt:
    (task / "report.md").write_text("report\\n", encoding="utf-8")
    emit("result", subtype="success", is_error=False, result="report", session_id=session)
    sys.exit(0)

def notes():
    (task / "plan.md").write_text("plan\\n", encoding="utf-8")
    (task / "implementation.md").write_text("impl\\n", encoding="utf-8")

if scenario == "needs_human":
    notes()
    (task / "needs_human.md").write_text("which module should be removed\\n", encoding="utf-8")
elif scenario == "cluster":
    notes()
    (workspace / "marker.txt").write_text("ok\\n", encoding="utf-8")
    (task / "cluster.md").write_text("train the ladder policy\\n", encoding="utf-8")
elif scenario == "protect":
    notes()
    (workspace / "marker.txt").write_text("ok\\n", encoding="utf-8")
    target = workspace / "autoresearch" / "controller.py"
    if "Restore these protected files" in prompt:
        target.write_text(os.environ["FAKE_ORIGINAL"], encoding="utf-8")
    else:
        target.write_text(target.read_text(encoding="utf-8") + "# agent touched\\n", encoding="utf-8")
elif scenario == "fail_once":
    notes()
    target = workspace / "broken.py"
    if "Independent validation failed" in prompt:
        target.write_text("def f():\\n    return 1\\n", encoding="utf-8")
    else:
        target.write_text("def f(:\\n", encoding="utf-8")
elif scenario == "always_broken":
    notes()
    (workspace / "broken.py").write_text("def f(:\\n", encoding="utf-8")
else:
    notes()
    (workspace / "marker.txt").write_text("ok\\n", encoding="utf-8")

emit("result", subtype="success", is_error=False, result="done", session_id=session)
"""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "Autoresearch Test")
    env.setdefault("GIT_AUTHOR_EMAIL", "autoresearch@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "Autoresearch Test")
    env.setdefault("GIT_COMMITTER_EMAIL", "autoresearch@example.com")
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        text=True,
        capture_output=True,
        env=env,
    )


def _init_repo(
    path: Path,
    *,
    branch: str = "autoresearch",
    commands: list[str] | None = None,
    max_attempts: int = 4,
    push: bool = False,
) -> None:
    path.mkdir()
    _git(path, "init", "-b", branch)
    package = path / "autoresearch"
    (package / "prompts").mkdir(parents=True)
    (package / "prompts" / "system.md").write_text("system\n", encoding="utf-8")
    (package / "prompts" / "start_task.md").write_text("Task {task_id}\n", encoding="utf-8")
    (package / "prompts" / "review.md").write_text(
        "Write the final report to autoresearch/runs/{task_id}/report.md\n",
        encoding="utf-8",
    )
    (package / "controller.py").write_text("original\n", encoding="utf-8")
    config = {
        "branch": "autoresearch",
        "model": "grok-4.7",
        "max_validation_attempts": max_attempts,
        "push": push,
        "remote": "origin",
        "cursor": {"bin": "cursor-agent", "sandbox": "disabled", "force": True},
        "validation": {
            "commands": commands
            or ["python -c \"import pathlib; assert pathlib.Path('marker.txt').read_text() == 'ok\\n'\""]
        },
        "protected_paths": ["autoresearch/controller.py"],
    }
    OmegaConf.save(OmegaConf.create(config), package / "config.yaml")
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "base")


def _install_fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "cursor-agent"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SCENARIO", scenario)
    monkeypatch.setenv("FAKE_SESSION", "session-1")
    monkeypatch.setenv("FAKE_ORIGINAL", "original\n")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Autoresearch Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "autoresearch@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Autoresearch Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "autoresearch@example.com")


def _events(repo: Path, task_id: str = "0001") -> list[dict]:
    return read_events(repo / "autoresearch" / "runs" / task_id / "events.jsonl")


def _names(repo: Path) -> list[str]:
    return [event["event"] for event in _events(repo)]


def _invocations(repo: Path) -> list[tuple[str, str]]:
    path = repo / "autoresearch" / "runs" / "0001" / "agent" / "invocations.log"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        mode, _, flags = line.partition("\t")
        rows.append((mode, flags))
    return rows


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "repo"
    _init_repo(checkout)
    _install_fake(monkeypatch, tmp_path, "implement")
    return checkout


def test_read_events_ignores_truncated_tail_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"created"}\n{"event":"brok', encoding="utf-8")
    assert read_events(path) == [{"event": "created"}]
    append_event(path, "agent_finished", exit_code=0)
    text = path.read_text(encoding="utf-8")
    assert text.startswith('{"event":"created"}\n{"event":"brok')
    assert text.rstrip().endswith("}")
    assert [event["event"] for event in read_events(path)] == ["created", "agent_finished"]


def test_read_events_skips_unparseable_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"event":"created"}\nnot-json\n{"event":"agent_finished"}\n', encoding="utf-8")
    assert [event["event"] for event in read_events(path)] == ["created", "agent_finished"]


def test_protected_hits_allow_task_directory() -> None:
    hits = protected_hits(
        [
            "autoresearch/controller.py",
            "autoresearch/runs/0001/plan.md",
            "autoresearch/prompts/system.md",
        ],
        ["autoresearch/controller.py", "autoresearch/prompts"],
        "0001",
    )
    assert hits == ["autoresearch/controller.py", "autoresearch/prompts/system.md"]


def test_new_command_writes_idea_without_running(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idea = tmp_path / "idea.txt"
    idea.write_text("Add a marker\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    assert main(["new", "--from-file", str(idea)]) == 0
    assert (repo / "autoresearch" / "runs" / "0001" / "idea.md").read_text(encoding="utf-8") == "Add a marker\n"
    assert _names(repo) == ["created"]


def test_run_validates_commits_and_is_idempotent(repo: Path) -> None:
    create_task(repo, "Add a marker")
    assert run_task(repo, "0001") == COMPLETE
    subject = _git(repo, "log", "-1", "--format=%s").stdout.strip()
    assert subject == "autoresearch: task 0001 - Add a marker"
    assert "local_validation_passed" in _names(repo)
    assert "completed" in _names(repo)
    assert (repo / "autoresearch" / "runs" / "0001" / "report.md").is_file()
    count = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    assert run_task(repo, "0001") == COMPLETE
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == count


def test_validation_failure_starts_a_fresh_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    _init_repo(checkout, commands=["python -m py_compile broken.py"])
    _install_fake(monkeypatch, tmp_path, "fail_once")
    create_task(checkout, "Fix the compile error")
    assert run_task(checkout, "0001") == COMPLETE
    assert _names(checkout).count("local_validation_failed") == 1
    assert (checkout / "broken.py").read_text(encoding="utf-8").startswith("def f()")
    start, repair, review = _invocations(checkout)
    assert start == ("fresh", "system")
    assert repair[0] == "fresh"
    assert "system" in repair[1] and "validation" in repair[1] and "brief" in repair[1]
    assert review[0] == "fresh"
    assert "system" in review[1] and "report" in review[1] and "brief" in review[1]


def test_retry_limit_stops_for_a_human(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    _init_repo(checkout, commands=["python -m py_compile broken.py"], max_attempts=2)
    _install_fake(monkeypatch, tmp_path, "always_broken")
    create_task(checkout, "This will not compile")
    assert run_task(checkout, "0001") == NEEDS_HUMAN
    assert _names(checkout).count("local_validation_failed") == 2
    assert find_task_commit(checkout, "0001") is None


def test_protected_edit_is_reverted_before_commit(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "protect")
    create_task(repo, "Add a marker")
    assert run_task(repo, "0001") == COMPLETE
    assert "protected_paths_modified" in _names(repo)
    committed = _git(repo, "show", "HEAD:autoresearch/controller.py").stdout
    assert committed == "original\n"


def test_needs_human_file_skips_commit(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "needs_human")
    create_task(repo, "Add a marker")
    assert run_task(repo, "0001") == NEEDS_HUMAN
    assert find_task_commit(repo, "0001") is None
    assert "local_validation_started" not in _names(repo)
    assert "which module should be removed" in format_status(repo, "0001")


def test_continue_after_needs_human(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "needs_human")
    create_task(repo, "Add a marker")
    assert run_task(repo, "0001") == NEEDS_HUMAN
    monkeypatch.setenv("FAKE_SCENARIO", "implement")
    assert continue_task(repo, "0001", "Keep the public controller and add the marker.") == COMPLETE
    assert not (repo / "autoresearch" / "runs" / "0001" / "needs_human.md").exists()


def test_cluster_handoff_commits_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    _init_repo(checkout, commands=["python -c \"print(1)\""])
    _install_fake(monkeypatch, tmp_path, "cluster")
    create_task(checkout, "Train the ladder policy")
    assert run_task(checkout, "0001") == AWAITING_CLUSTER
    assert find_task_commit(checkout, "0001") is not None
    assert "completed" not in _names(checkout)
    assert "review_started" not in _names(checkout)
    assert (checkout / "autoresearch" / "runs" / "0001" / "cluster.md").is_file()


def test_resume_failure_starts_a_fresh_session(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SCENARIO", "resume_fail")
    create_task(repo, "Add a marker")
    task = repo / "autoresearch" / "runs" / "0001"
    output = task / "agent" / "output"
    output.mkdir(parents=True)
    (output / "001.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": "session-1", "model": "grok-4.7"}) + "\n",
        encoding="utf-8",
    )
    append_event(task / "events.jsonl", "agent_started", task_id="0001", session_id="session-1")
    assert run_task(repo, "0001") == COMPLETE
    assert "resume_failed" in _names(repo)
    assert (task / "report.md").is_file()
    resumed, recovered = _invocations(repo)[:2]
    assert resumed == ("resume", "")
    assert recovered[0] == "fresh" and "system" in recovered[1]


def test_reconciles_successful_output_without_finish_event(repo: Path) -> None:
    create_task(repo, "Add a marker")
    task = repo / "autoresearch" / "runs" / "0001"
    (task / "plan.md").write_text("plan\n", encoding="utf-8")
    (task / "implementation.md").write_text("impl\n", encoding="utf-8")
    (repo / "marker.txt").write_text("ok\n", encoding="utf-8")
    output = task / "agent" / "output"
    output.mkdir(parents=True)
    (output / "001.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init", "session_id": "session-1", "model": "grok-4.7"}),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "done",
                        "session_id": "session-1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "001.exit").write_text("0\n", encoding="utf-8")
    append_event(task / "events.jsonl", "agent_started", task_id="0001", session_id="session-1", pid=None)
    assert run_task(repo, "0001") == COMPLETE
    assert "reconciled" in _names(repo)


def test_interrupted_agent_is_resumed(repo: Path) -> None:
    create_task(repo, "Add a marker")
    task = repo / "autoresearch" / "runs" / "0001"
    output = task / "agent" / "output"
    output.mkdir(parents=True)
    (output / "001.jsonl").write_text(
        json.dumps({"type": "system", "subtype": "init", "session_id": "session-1", "model": "grok-4.7"}) + "\n",
        encoding="utf-8",
    )
    append_event(task / "events.jsonl", "agent_started", task_id="0001", session_id="session-1")
    assert run_task(repo, "0001") == COMPLETE
    assert "agent_interrupted" in _names(repo)
    assert _invocations(repo)[0] == ("resume", "")


def test_existing_task_commit_is_not_duplicated(repo: Path) -> None:
    create_task(repo, "Add a marker")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "autoresearch: task 0001 - Add a marker")
    before = _git(repo, "rev-list", "--count", "HEAD").stdout.strip()
    assert run_task(repo, "0001") == COMPLETE
    assert _git(repo, "rev-list", "--count", "HEAD").stdout.strip() == before
    committed = [event for event in _events(repo) if event["event"] == "committed"]
    assert committed[0]["reconciled"] is True


def test_live_pid_blocks_a_second_agent(repo: Path) -> None:
    create_task(repo, "Add a marker")
    process = subprocess.Popen(["sleep", "60"])
    try:
        task = repo / "autoresearch" / "runs" / "0001"
        append_event(task / "events.jsonl", "agent_started", task_id="0001", session_id="session-1", pid=process.pid)
        (task / "agent").mkdir()
        (task / "agent" / "session.json").write_text(json.dumps({"pid": process.pid}) + "\n", encoding="utf-8")
        with pytest.raises(ControllerError, match="already running"):
            run_task(repo, "0001")
    finally:
        process.kill()
        process.wait(timeout=5)


def test_wrong_branch_and_unexpected_changes_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    _init_repo(checkout, branch="master")
    _install_fake(monkeypatch, tmp_path, "implement")
    create_task(checkout, "Add a marker")
    with pytest.raises(ControllerError, match="must run on branch"):
        run_task(checkout, "0001")

    clean = tmp_path / "clean"
    _init_repo(clean)
    create_task(clean, "Add a marker")
    (clean / "stray.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(ControllerError, match="unexpected uncommitted changes"):
        run_task(clean, "0001")


def test_push_to_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    _init_repo(checkout, push=True)
    _install_fake(monkeypatch, tmp_path, "implement")
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "--bare", "-b", "autoresearch", str(bare)], check=True, capture_output=True)
    _git(checkout, "remote", "add", "origin", str(bare))
    create_task(checkout, "Add a marker")
    assert run_task(checkout, "0001") == COMPLETE
    remote_head = subprocess.run(
        ["git", "--git-dir", str(bare), "rev-parse", "autoresearch"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert remote_head == _git(checkout, "rev-parse", "HEAD").stdout.strip()
