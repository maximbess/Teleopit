# Autoresearch

Autoresearch runs one repository task at a time on the `autoresearch` branch. A Python controller decides when to call Cursor, when validation has passed, and when to commit. Cursor CLI (`cursor-agent`) does the editing and debugging. The controller does not treat the agent's own report as proof that the change works.

Task files live under `autoresearch/runs/<id>/`. `events.jsonl` is the execution history. Agent stdout and stderr are stored under `agent/output/`. Restart a task with `run`; the controller reads the log, the worktree, and `git log`, then continues from the last finished step.

## Prerequisite

Install and log in to Cursor CLI before `run`:

```bash
curl -sS https://cursor.com/install | bash
cursor-agent login
cursor-agent --list-models
```

Set `model` in `config.yaml` to a slug from that list. The default is `grok-4.7`.

Check out the `autoresearch` branch and start from a clean worktree. `run` refuses other branches and refuses uncommitted files outside the task directory.

## Commands

From the repository root:

```bash
python -m autoresearch.controller new --from-file idea.txt
python -m autoresearch.controller run 0001
python -m autoresearch.controller status
python -m autoresearch.controller status 0001
python -m autoresearch.controller continue 0001 "what to do next"
```

`new` without `--from-file` reads an idea from the terminal when stdin is a TTY. `continue` resumes the stored Cursor session and gives the agent another validation budget. Use it after `NEEDS_HUMAN`. A resumed session already has the standing system prompt, so the follow-up does not send it again.

## What the agent must write

- `plan.md` and `implementation.md` for every task
- `needs_human.md` when a person has to decide before any commit
- `cluster.md` when the next step is expensive training; the controller then commits and stops in `AWAITING_CLUSTER`
- `report.md` when the controller asks for the closing report

The controller commits as `autoresearch: task 0001 - <first line of idea.md>` and pushes `autoresearch` to `origin`. Protected paths in `config.yaml` cannot be part of that commit. The task directory is writable.

## Validation

`config.yaml` lists the commands. The defaults compile the Python packages and run `pytest tests/ -q`. A failure starts a new Cursor session with the idea, implementation notes, diff stat, and log tail. The review that writes `report.md` is also a new session. After `max_validation_attempts` (default 4), the task stops in `NEEDS_HUMAN`. An agent that stops mid-run is still resumed.

## Crash recovery

`run` again after the process, Cursor, or the machine stops. In particular it will:

- resume an agent that exited before finishing, or adopt a finished output log that was not yet recorded
- rerun a validation command that was interrupted
- record a task commit that already exists instead of creating a second one
- push when the commit was recorded but the push was not

A live Cursor pid recorded in `agent/session.json` blocks a second agent.
