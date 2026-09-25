# Autoresearch v1 — Cursor + Grok + Git Pipeline

## Goal

Build a small, robust autoresearch controller for this repository.

The intended interaction is:

```text
Researcher provides an idea/task
        ↓
Python controller
        ↓
Cursor CLI + Grok 4.7
        ↓
agent understands repository/task
agent plans
agent implements
agent tests/debugs
agent performs cheap local validation
        ↓
controller independently validates
        ↓
commit + push to dedicated autoresearch branch
        ↓
if cluster training required:
    wait for researcher to run experiment manually
        ↓
researcher copies one result archive back
        ↓
resume same Cursor conversation
        ↓
agent analyzes numerical + visual results
        ↓
report
```

The first end-to-end task will be **repository cleanup**, not an ML experiment. This should be treated as a calibration task for how reliably the system can understand the repository, make nontrivial changes, test them, and produce an auditable report.

Do not build a large agent framework. Prefer straightforward Python, files, subprocesses, Git, and Cursor CLI.

---

# 1. Design principles

### Files are the state database

Do not introduce SQLite or another database in v1.

Every task/experiment gets its own permanent directory. State transitions are recorded in an append-only `events.jsonl`.

The filesystem should contain enough information to understand and recover the pipeline after the controller, Cursor, SSH session, or machine crashes.

### Git is the code provenance mechanism

Use one dedicated long-lived branch:

```text
autoresearch
```

Do not use Git worktrees in v1.

The pipeline operates sequentially on this branch.

Each completed task/experiment should result in an identifiable Git commit.

Never rewrite published experiment history.

### The controller owns orchestration

The Python controller decides:

- what state the task is in;
- when an agent should be invoked;
- whether validation succeeded;
- when a commit should be created;
- when human intervention is required;
- whether results have arrived.

Grok is allowed substantial freedom *inside* a task, but should not be the durable source of workflow state.

### Cursor/Grok owns the difficult coding loop

Do not implement custom file-search/edit/debug agents.

Use Cursor CLI with Grok 4.7 to perform:

```text
understand → plan → edit → test → inspect failure → debug → retest
```

The controller should invoke Cursor rather than recreate its coding-agent functionality.

### Expensive cluster execution is human-gated

The autoresearch machine has no Slurm credentials and does not submit cluster jobs.

For experiments requiring training, the pipeline ends the pre-training phase by producing an exact Git commit and an easy cluster launch instruction.

---

# 2. Proposed repository structure

Add something approximately like:

```text
autoresearch/
├── README.md
├── controller.py
├── config.yaml
│
├── prompts/
│   ├── system.md
│   ├── start_task.md
│   ├── continue_with_results.md
│   └── review.md
│
├── scripts/
│   ├── validate.py
│   ├── pack_results.sh
│   └── import_results.py
│
└── runs/
    ├── 0001/
    │   ├── idea.md
    │   ├── events.jsonl
    │   ├── agent/
    │   │   ├── session.json
    │   │   └── output/
    │   ├── plan.md
    │   ├── implementation.md
    │   ├── cluster.md
    │   ├── results/
    │   └── report.md
    │
    └── 0002/
        └── ...
```

Names can be adjusted to fit the existing repository.

Do not generate empty files merely to satisfy this structure. Files should appear when their corresponding stage occurs.

---

# 3. Task lifecycle

The controller should reconstruct state from `events.jsonl`.

Conceptual lifecycle:

```text
CREATED
   ↓
AGENT_RUNNING
   ↓
LOCAL_VALIDATION
   ↓
   ├── validation failure → AGENT_RUNNING
   │
   ↓
LOCAL_COMPLETE
   ↓
   ├── no cluster required → REVIEW → COMPLETE
   │
   └── cluster required
             ↓
       AWAITING_CLUSTER
             ↓
       AWAITING_RESULTS
             ↓
       ANALYZING_RESULTS
             ↓
           REVIEW
             ↓
          COMPLETE
```

Also support:

```text
NEEDS_HUMAN
FAILED
ABORTED
```

These do not need to be represented by a complicated class hierarchy. A few Python functions and explicit transition logic are preferable.

---

# 4. Append-only event log

`events.jsonl` is authoritative.

Example:

```json
{"time":"...", "event":"created", "task_id":"0001"}
{"time":"...", "event":"agent_started", "session_id":"..."}
{"time":"...", "event":"agent_finished"}
{"time":"...", "event":"local_validation_started"}
{"time":"...", "event":"local_validation_failed", "command":"pytest", "exit_code":1}
{"time":"...", "event":"agent_resumed", "session_id":"..."}
{"time":"...", "event":"local_validation_passed"}
{"time":"...", "event":"committed", "commit":"abc123..."}
{"time":"...", "event":"awaiting_cluster"}
```

Requirements:

- append events only after the corresponding operation has definitely happened;
- flush/fsync important events where appropriate;
- tolerate an incomplete final JSONL line after a crash;
- never silently delete/rewrite historical events;
- include enough information to reconcile ambiguous operations after restart;
- store full agent stdout/stderr separately rather than stuffing everything into JSONL.

Human-readable Markdown files are research artifacts. `events.jsonl` is machine-readable execution history.

---

# 5. Controller interface

Start with a simple CLI.

Desired usage should be roughly:

```bash
python -m autoresearch.controller new
python -m autoresearch.controller run 0001
python -m autoresearch.controller status
python -m autoresearch.controller status 0001
python -m autoresearch.controller continue 0001
python -m autoresearch.controller import-results 0001 result.tar.zst
```

`new` should either accept an idea interactively or from a file.

For example:

```text
$ python -m autoresearch.controller new

Describe task/research idea:

> Clean up the repository. Remove obsolete/dead code and
> unnecessary duplication, improve organization where justified,
> and make the repository easier for future agents to understand.
> Preserve behavior.

Created autoresearch/runs/0001/idea.md

Run now? [Y/n]
```

Do not spend significant effort building a UI in v1.

---

# 6. Cursor interface

Create one small adapter around Cursor CLI.

Conceptually:

```python
class CursorAgent:
    def start(self, prompt, cwd) -> AgentResult:
        ...

    def resume(self, session_id, prompt, cwd) -> AgentResult:
        ...
```

Store:

```text
session_id
model
start/end time
exit status
raw output
```

Use Grok 4.7 initially.

Keep the model and reasoning configuration configurable rather than scattered through code.

Do not introduce the Cursor SDK in v1 unless CLI limitations actually block implementation.

The architecture must make it easy to replace the CLI adapter with the SDK later without changing experiment logic.

---

# 7. Main agent instructions

The primary agent should receive persistent instructions along these lines:

```text
You are the implementation and experimental-design agent for this
robotics repository.

The researcher gives you a task or research idea. Your responsibility
is to take it as far as possible without unnecessary human involvement.

For each task:

1. Understand the requested idea.
2. Inspect the repository and relevant previous experiment reports.
3. Determine what the task actually requires.
4. Think through risks, assumptions, and expected effects.
5. Form a concrete plan.
6. Implement the plan.
7. Run appropriate cheap tests and checks.
8. Diagnose and fix problems you introduce.
9. Perform cheap local behavioral/simulation checks when relevant.
10. Prepare the task for its next stage.
11. Document what you did, what you observed, and remaining uncertainty.

For research experiments additionally:

- state the mechanism/hypothesis being tested;
- identify expected observations before seeing the results;
- identify important confounders;
- preserve a meaningful baseline;
- do not change fixed evaluation definitions simply to improve results;
- prepare an exact reproducible training configuration;
- do not launch expensive cluster jobs.

Exercise engineering judgment rather than mechanically following this
checklist when a step is irrelevant.

Stop and request human input when:
- an important requirement is genuinely ambiguous;
- proceeding risks destructive/unrecoverable changes;
- credentials or external access are required;
- the next action is expensive cluster execution;
- evidence is insufficient to make a responsible decision.

Do not claim tests succeeded unless they were actually executed.
Do not hide failed approaches or unexpected behavior.
```

The actual task prompt should then be intentionally small:

```text
Task 0001

The researcher's idea/request is in:

autoresearch/runs/0001/idea.md

Take this task from the supplied idea to the next point requiring
human intervention.

Record a concise plan and final summary in the task directory.
```

The researcher should not normally have to manually specify baselines, files to modify, tests, predictions, etc. Discovering those is part of the agent's job.

---

# 8. Independent validation

Do not trust the agent's textual claim that its implementation works.

After Cursor finishes, the controller should inspect the repository and independently run configured validation commands.

Configuration could contain:

```yaml
validation:
  commands:
    - pytest
    - python -m compileall ...
```

Adapt these to the actual repository after inspecting it.

For robotics experiments, later add a cheap simulation/smoke-test command.

If validation fails:

```text
controller records failure
        ↓
resume same Cursor session
        ↓
provide exact command + stdout/stderr
        ↓
agent diagnoses/fixes
        ↓
controller validates again
```

Put a configurable retry limit on this loop, e.g. 3–5 attempts, after which transition to `NEEDS_HUMAN`.

---

# 9. Git behavior

The autoresearch server operates on the dedicated:

```text
autoresearch
```

branch.

Before starting a task:

- ensure branch is correct;
- ensure unexpected uncommitted changes do not exist;
- record the starting commit.

After successful validation:

- inspect the diff;
- ensure protected infrastructure/evaluation files were not unexpectedly modified;
- create a commit containing the task ID;
- push the branch;
- record the exact commit SHA in `events.jsonl`.

Example:

```text
autoresearch: task 0007 - observation history experiment
```

Cluster execution must always use an exact commit SHA, not simply whatever happens to be the current tip of `autoresearch`.

---

# 10. Cluster handoff

For a research task requiring expensive training, create:

```text
autoresearch/runs/0042/cluster.md
```

containing:

```text
Experiment: 0042
Commit: <full SHA>

Purpose:
<short description>

Configuration:
<exact config>

Expected runs:
<seeds / resources if known>

Cluster command:
<the simplest command the researcher needs to execute>

Expected result bundle:
0042-results.tar.zst
```

Then stop in `AWAITING_CLUSTER`.

Do not require the controller process to remain running.

---

# 11. Result bundle

Cluster-side execution should gather everything needed for later analysis into ONE directory.

Target format:

```text
0042-results/
├── manifest.json
├── config.yaml
├── metrics.json
├── training.log
├── evaluation/
│   ├── summary.json
│   ├── episodes.csv
│   └── learning_curve.csv
├── videos/
│   ├── representative.mp4
│   ├── random.mp4
│   └── failure.mp4
└── images/
    └── representative_contact_sheet.png
```

Large checkpoints should be optional and should not automatically be included if they make transfer inconvenient.

Then provide:

```bash
./autoresearch/scripts/pack_results.sh 0042
```

which produces exactly:

```text
0042-results.tar.zst
```

The researcher should only need to move this one file from the cluster.

On the autoresearch machine:

```bash
python -m autoresearch.controller import-results \
    0042 \
    0042-results.tar.zst
```

should validate and unpack it into the appropriate task directory.

---

# 12. Behavioral/visual evaluation

Robotics experiments must not be evaluated only through scalar reward.

The evaluation pipeline should produce at least one reproducible rendered rollout and preferably several:

- representative/fixed-seed rollout;
- randomly selected rollout;
- interesting failure rollout where available.

Generate a contact sheet from the representative rollout so a vision-capable model can inspect behavior even if direct video ingestion through the CLI is inconvenient.

The post-training agent should be explicitly asked to assess:

```text
Is the robot actually doing the intended task?

Does it climb, or is it exploiting reward by jumping/falling?

Does the movement look physically plausible?

Are there oscillations, repeated impacts, saturation, pathological
contacts, or termination/reward exploits?

Is the visual behavior consistent with the numerical metrics?

Is it consistent with the mechanism proposed before training?
```

Numerical analysis remains necessary. Visual inspection complements rather than replaces it.

---

# 13. Post-training continuation

Preserve the Cursor session ID from the pre-training phase.

After results are imported, attempt to resume the same conversation with something approximately like:

```text
The cluster experiment has completed.

Results are available under:

autoresearch/runs/0042/results/

Continue the research task.

Inspect numerical results, logs, plots, and available visual evidence.

Compare the outcome against the predictions you made before training.

Determine:
- what actually changed;
- whether performance improved;
- whether the original hypothesis gained or lost support;
- anomalies/confounders;
- likely failure modes;
- what experiment should logically follow.

Write the final report to:
autoresearch/runs/0042/report.md
```

If conversation resumption fails, the system should be capable of starting a new agent using the persisted task artifacts. Correctness must not depend on Cursor retaining conversation state forever.

---

# 14. Reports

Every completed task gets `report.md`.

For research experiments, reports should clearly separate:

```text
Hypothesis

Change made

Pre-experiment prediction

Experiment configuration

Results

Visual behavior

Interpretation

Evidence for/against hypothesis

Confounders / uncertainty

Decision

Possible follow-up
```

Do not force every task into this format. Engineering/refactoring tasks can use an appropriate engineering report instead.

---

# 15. First calibration task: repository cleanup

After implementing the minimum pipeline, use it on the repository itself.

Create task `0001` with approximately this idea:

```text
Clean up this repository in preparation for autonomous research work.

Inspect the repository before deciding what "cleanup" means.

Look for obsolete code, duplicated implementations, dead scripts,
stale configurations, confusing structure, unused utilities,
temporary/debug artifacts, inconsistent naming, and documentation
that no longer reflects reality.

Improve organization where there is a clear benefit.

Preserve existing behavior.

Be conservative about deleting things whose purpose you cannot
establish.

Run the repository's relevant tests/checks after making changes.

Document:
- what you found;
- what you changed and why;
- things that look suspicious but you deliberately left alone;
- tests/checks performed;
- recommendations for additional cleanup that would require
  researcher judgment.
```

This task should deliberately exercise the real agent pipeline rather than being manually implemented as a special case.

It is a useful calibration because the agent must:

```text
explore unfamiliar repo
        ↓
form its own plan
        ↓
distinguish dead code from important code
        ↓
make changes across files
        ↓
test
        ↓
debug
        ↓
exercise restraint when uncertain
        ↓
produce an auditable explanation
```

There is no cluster phase for task 0001.

After task 0001, the researcher will manually inspect the diff and report. Treat feedback from this task as input for improving the standing agent instructions before allowing autonomous ML experiments.

---

# 16. Crash/restart requirements

Design explicitly for the process dying at any point.

At minimum test interruption:

- before invoking Cursor;
- while Cursor is running;
- after Cursor succeeds but before event logging;
- during validation;
- after Git commit but before recording its SHA;
- while waiting for cluster results;
- during result import.

On restart, the controller should inspect both the event log and real external state where necessary.

For example, if there is no `committed` event but HEAD already contains a commit for task 0042, reconcile it rather than creating a duplicate commit.

Operations should be idempotent where practical.

---

# 17. Scope restrictions for v1

Do NOT add unless required by an observed problem:

```text
database
LangGraph
Google ADK
web server
frontend
Kubernetes
distributed workers
Git worktrees
automatic Slurm access
multi-agent framework
vector database
custom coding-agent implementation
Cursor SDK
```

Prefer understandable code over extensible abstractions.

It should be possible for one person to read the controller and understand the entire execution model.

---

# 18. Implementation order

Implement in this order:

### Phase A — minimal task runner

Implement:

```text
task directories
idea.md
events.jsonl
Cursor CLI start/resume wrapper
raw output logging
status reconstruction
```

Demonstrate with a harmless trivial task.

### Phase B — robust coding loop

Add:

```text
validation commands
validation → agent repair loop
retry limits
Git commit/push
crash reconciliation
protected-file checks
```

Then run **task 0001: repository cleanup** through the actual pipeline.

Stop here for researcher review.

### Phase C — research experiment support

Only after feedback from task 0001, add:

```text
research-specific prompt
experiment manifest
AWAITING_CLUSTER state
cluster.md
result bundle format
pack/import scripts
```

### Phase D — post-training analysis

Add:

```text
result validation
resume Cursor conversation
metric analysis
render/contact-sheet support
visual policy inspection
report generation
```

### Phase E — autonomous iteration

Do not implement until the user-driven loop has proven reliable.

Eventually support:

```text
user idea
    OR
agent proposes next idea from previous reports
```

The downstream pipeline should be identical regardless of where the idea originated.

---

# 19. Definition of success for v1

Do not consider v1 complete merely because the controller can call Cursor.

It is successful when the following works:

```text
researcher writes one idea
        ↓
one controller command
        ↓
Grok understands repository
        ↓
plans + implements + debugs
        ↓
controller independently validates
        ↓
commit is created and pushed
        ↓
all reasoning/actions remain inspectable in task directory
        ↓
controller can restart without losing task state
```

Task 0001 (repository cleanup) should demonstrate this end-to-end.

After that succeeds, implement the cluster handoff and result-analysis path and demonstrate:

```text
idea
→ implementation
→ exact Git commit
→ human cluster submission
→ one portable results archive
→ import
→ numerical + visual analysis
→ report
```

Optimize for **robustness, inspectability, and simplicity**, not maximum autonomy in the first version.