Task {task_id}

The researcher's idea/request is in:

autoresearch/runs/{task_id}/idea.md

Take this task from the supplied idea to the next point requiring human intervention.

Record a concise plan in:

autoresearch/runs/{task_id}/plan.md

Record what you changed, what you observed, and remaining uncertainty in:

autoresearch/runs/{task_id}/implementation.md

If you must stop for a human, write:

autoresearch/runs/{task_id}/needs_human.md

If the task requires expensive cluster training, write:

autoresearch/runs/{task_id}/cluster.md

Then stop. Do not launch cluster jobs, and do not claim the experiment has already run.
