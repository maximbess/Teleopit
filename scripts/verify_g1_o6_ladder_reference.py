"""Compare fresh pre-refactor reference traces to the reviewed campaign."""

import json
from itertools import zip_longest
from pathlib import Path

from teleopit.sim.g1_o6_support.ladder_campaign import file_hash

ROOT = Path("outputs/g1_o6/ladder_material")


def compare_traces(first, second):
    count, differences = 0, []
    keys = (
        "time",
        "phase",
        "base_pose",
        "joint_positions",
        "base_drift_m",
        "contacts",
        "supports",
        "wrist_world_on_hand",
        "failure_reasons",
    )
    with first.open() as a, second.open() as b:
        for old, new in zip_longest(a, b):
            count += 1
            if old is None or new is None:
                differences.append({"row": count, "reason": "row count differs"})
                break
            before, after = json.loads(old), json.loads(new)
            changed = [k for k in keys if before.get(k) != after.get(k)]
            if changed and len(differences) < 10:
                differences.append({"row": count, "fields": changed})
    return {
        "rows": count,
        "identical": not differences,
        "first_differences": differences,
    }


def main():
    results = {}
    for seconds, stage in ((30, "screen"), (120, "120s")):
        old = Path("outputs/g1_o6/pose_search/runs/reference_" + stage)
        new = ROOT / "reference" / f"{seconds}s"
        comparison = compare_traces(old / "steps.jsonl", new / "steps.jsonl")
        summary = json.loads((new / "summary.json").read_text())
        comparison.update(
            summary=summary,
            previous_path=old.as_posix(),
            new_path=new.as_posix(),
            previous_trace_sha256=file_hash(old / "steps.jsonl"),
            new_trace_sha256=file_hash(new / "steps.jsonl"),
        )
        results[str(seconds)] = comparison
    destination = ROOT / "reference_verification.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {n: v[n] for n in ("rows", "identical", "first_differences")}
                for k, v in results.items()
            }
        )
    )


if __name__ == "__main__":
    main()
