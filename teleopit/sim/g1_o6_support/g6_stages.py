"""Evidence-based milestone reporting and the finite downstream request schedule."""

from .g6_gates import initial_pose_gate, loaded_hold_gate
from .qualification import Evidence, read_run


def qualified_pose(campaign):
    for name in campaign.state["candidates"]:
        records = pose_records(campaign.state["runs"], name)
        accepts = getattr(campaign, "passes", None)
        gates_passed = accepts is None or all(accepts(r["result"], r["stage"]) for r in records)
        if len(records) >= 5 and gates_passed:
            runs = [read_run(campaign.root / r["output"]) for r in records]
            if initial_pose_gate(runs):
                return name, runs
    return None, []


def pose_records(records, name):
    return [
        r
        for r in records
        if r["candidate"] == name
        and r["stage"] in ("screen", "60s", "120s", "repeat", "fine")
        and r["status"] == "COMPLETE"
        and not r.get("imported_control")
    ]


def transfer_schedule():
    requests = []
    for share in (0.05, 0.1, 0.2, 0.3):
        requests.append(dict(mode="load", target_share=share))
    for duration, dt in ((30, 0.001), (60, 0.001), (60, 0.001), (60, 0.0005)):
        requests.append(
            dict(mode="load", target_share=0.3, plateau_seconds=duration, physics_dt=dt)
        )
    requests.extend(
        dict(mode="unload", side=s, target_share=0.3) for s in ("left", "right")
    )
    for duration in (1, 3, 10):
        requests.extend(
            dict(mode="release", side=s, three_seconds=duration, target_share=0.3)
            for s in ("left", "right")
        )
    for cycles, alternating in ((1, False), (3, False), (3, True)):
        requests.extend(
            dict(
                mode="reacquire",
                side=s,
                cycles=cycles,
                alternating=alternating,
                target_share=0.3,
            )
            for s in ("left", "right")
        )
    return requests + adjacent_schedule()


def adjacent_schedule():
    requests = []
    for side in ("left", "right"):
        requests.append(dict(mode="adjacent-first", side=side, target_share=0.3))
        requests.extend(
            dict(mode="adjacent", side=side, target_share=0.3) for _ in range(3)
        )
        requests.append(
            dict(mode="adjacent-first", side=side, target_share=0.3, physics_dt=0.0005)
        )
        requests.append(
            dict(mode="adjacent", side=side, target_share=0.3, physics_dt=0.0005)
        )
    return requests


def milestone_report(campaign):
    name, _ = qualified_pose(campaign)
    initial = "PASS" if name else "FAIL" if campaign.state["runs"] else "NOT RUN"
    report = {"initial_pose": {"status": initial, "candidate": name}}
    book = campaign.root / "evidence.json"
    if name and book.exists():
        return downstream_milestones(report, Evidence(book))
    report.update(
        {
            stage: {"status": "BLOCKED", "reason": "initial G6 pose not qualified"}
            for stage in (
                "load_bearing_grasp",
                "controlled_release",
                "same_tread_reacquisition",
                "adjacent_tread_transfer",
            )
        }
    )
    return report


def downstream_milestones(report, proof):
    qualified = proof.qualified_transfers()
    experiments = [r["metadata"]["experiment"] for r in qualified]
    loaded = loaded_hold_gate(qualified)
    released = all(
        count_experiments(experiments, mode="release", side=s, three_seconds=d)
        for s in ("left", "right")
        for d in (1, 3, 10)
    )
    same = all(
        count_experiments(
            experiments, mode="reacquire", side=s, cycles=3, alternating=True
        )
        for s in ("left", "right")
    )
    adjacent = adjacent_confirmed(experiments)
    for stage, prerequisite, passed in (
        ("load_bearing_grasp", True, loaded),
        ("controlled_release", loaded, released),
        ("same_tread_reacquisition", released, same),
        ("adjacent_tread_transfer", same, adjacent),
    ):
        report[stage] = {
            "status": "PASS" if passed else "NOT RUN" if prerequisite else "BLOCKED"
        }
    return report


def count_experiments(experiments, **criteria):
    return sum(all(e.get(k) == v for k, v in criteria.items()) for e in experiments)


def adjacent_confirmed(experiments):
    return all(
        count_experiments(experiments, mode="adjacent", side=s, physics_dt=dt) >= count
        for s in ("left", "right")
        for dt, count in ((0.001, 3), (0.0005, 1))
    )
