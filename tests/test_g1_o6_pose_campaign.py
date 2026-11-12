import numpy as np
import pytest


def test_catalogue_is_one_factor_and_mandatory_rung10_is_exact():
    from teleopit.sim.g1_o6_support.pose_candidates import (
        controller_candidates,
        initial_geometry,
        offset_and_branch_candidates,
    )

    candidates = initial_geometry()
    assert len(candidates) == 16
    reference, lower = candidates[:2]
    assert lower.changes == {"hand_rungs": (10, 10)}
    assert lower.pose.support_targets()["right_foot"].endswith("04")
    assert all(len(c.changes) <= 1 for c in candidates)
    assert len(offset_and_branch_candidates([reference, lower])) == 12
    names = {
        s: [s + "_o6_thumb_cmc_pitch", s + "_o6_thumb_cmc_yaw", "a", "b", "c", "d"]
        for s in ("left", "right")
    }
    controllers = controller_candidates([reference, lower], names)
    assert len(controllers) == 12
    assert len({c.digest() for c in controllers}) == 12


def test_budget_counts_failed_attempts_and_does_not_promote_them():
    from teleopit.sim.g1_o6_support.pose_campaign import validate_trial

    budgets = {"promoted_60s": 1, "first_120s": 1, "finalists": 1, "robustness": 1}
    failed = {
        "candidate": "a",
        "stage": "screen",
        "status": "COMPLETE",
        "result": {"strict_status": "FAIL"},
    }
    with pytest.raises(ValueError, match="already attempted"):
        validate_trial([failed], "a", "screen", budgets)
    with pytest.raises(ValueError, match="preceding strict"):
        validate_trial([failed], "a", "60s", budgets)
    passed = {**failed, "result": {"strict_status": "PASS"}}
    used = {"candidate": "b", "stage": "60s", "status": "ERROR"}
    with pytest.raises(ValueError, match="budget"):
        validate_trial([passed, used], "a", "60s", budgets)
    with pytest.raises(ValueError, match="qualified"):
        validate_trial([passed], "a", "repeat", budgets)


@pytest.mark.parametrize("rungs", [(10, 10), (10, 11), (11, 10)])
def test_rung_targets_and_reset_are_deterministic(rungs):
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate
    from teleopit.sim.g1_o6_support.pose_trial import SearchEnvironment, reset_report

    env = SearchEnvironment(
        Candidate("test", pose=GraspPose(hand_rungs=rungs)), 0.02, 0.001
    )
    qpos = env.data.qpos.copy()
    assert [env.support_targets[s + "_hand"] for s in env.hands] == [
        f"left_ladder_rung_{r:02d}" for r in rungs
    ]
    inspection = reset_report(env)
    assert inspection["rung_11_minus_10_world_m"][2] > 0
    env.reset()
    np.testing.assert_array_equal(env.data.qpos, qpos)
    assert env.hold_controller.state()["latch_time_s"] is None


def test_infeasible_candidate_is_rejected_without_stepping(tmp_path, monkeypatch):
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate
    from teleopit.sim.g1_o6_support.pose_trial import SearchEnvironment, run_candidate

    def forbidden(*args):
        raise AssertionError("infeasible reset stepped")

    monkeypatch.setattr(SearchEnvironment, "step", forbidden)
    result = run_candidate(
        Candidate("bad_reach", pose=GraspPose(hand_spacing_m=5)), tmp_path / "bad"
    )
    assert result["reset_status"] == "INFEASIBLE"
    assert result["physics_steps"] == 0 and not result["progression_justified"]


def test_parallel_workers_have_one_ledger_writer_and_resume_completed_jobs(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    from teleopit.sim.g1_o6_support import pose_campaign

    calls = []

    def simulated_result(candidate, output, **kwargs):
        calls.append(candidate.id)
        return {"strict_status": "PASS", "progression_justified": False}

    monkeypatch.setattr(pose_campaign, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(pose_campaign, "run_candidate", simulated_result)
    monkeypatch.setattr(pose_campaign, "prior_resets", lambda *args: [])
    monkeypatch.setattr(pose_campaign, "preflight_geometry", lambda *args: None)
    campaign = pose_campaign.Campaign(tmp_path)
    jobs = [("reference", "screen"), ("rung10", "screen")]
    campaign.run_many(jobs, workers=2)
    campaign.run_many(jobs, workers=2)
    assert sorted(calls) == ["reference", "rung10"]
    assert len(campaign.state["runs"]) == 2
    assert all(r["status"] == "COMPLETE" for r in campaign.state["runs"])


def test_failed_registration_leaves_frozen_candidates_unchanged(tmp_path):
    import copy
    from dataclasses import replace

    from teleopit.sim.g1_o6_support.pose_campaign import Campaign

    campaign = Campaign(tmp_path)
    before = copy.deepcopy(campaign.state)
    with pytest.raises(ValueError, match="budget"):
        campaign.register(
            [
                replace(campaign.candidate("reference"), id=f"overflow_{i}")
                for i in range(33)
            ]
        )
    assert campaign.state == before


def test_frozen_choices_resume_without_reselecting_and_batches_are_atomic(tmp_path):
    import copy

    from teleopit.sim.g1_o6_support.pose_campaign import Campaign

    campaign = Campaign(tmp_path)
    chosen = campaign.decision("parents", lambda: ["reference"])
    assert chosen == ["reference"]
    again = Campaign(tmp_path)
    assert again.decision("parents", lambda: ["rung10"]) == chosen
    before = copy.deepcopy(again.state)
    with pytest.raises(ValueError, match="preceding strict"):
        again.reserve_many([("reference", "screen"), ("rung10", "60s")])
    assert again.state == before
