"""Measured outcomes select bounded, frozen follow-up requests, never passes."""

from copy import deepcopy


def outcome(campaign, name, stage="screen", cost=2.0, passed=True):
    return {
        "candidate": name,
        "stage": stage,
        "configuration": campaign.candidate(name).record(),
        "strict": "PASS" if passed else "FAIL",
        "requested_s": 30,
        "achieved_s": 30 if passed else 2,
        "late_motion_cost": cost,
        "hands": {},
        "materially_reduced_hand_load": {"left": False, "right": False},
        "result": {
            "progression_justified": False,
            "contact_quality": {"status": "PASS" if passed else "FAIL"},
        },
    }


def test_adaptive_choices_are_frozen_and_arm_feedback_replaces_one_slot(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_selection as selection
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign

    campaign = Campaign(tmp_path)
    rows = [
        outcome(campaign, "reference"),
        outcome(campaign, "rung10", passed=False),
        outcome(campaign, "reference_hand_spacing_m_0p16", cost=1),
    ]
    monkeypatch.setattr(selection, "campaign_results", lambda _: rows)
    requests = []
    monkeypatch.setattr(campaign, "run_many", lambda jobs: requests.extend(jobs))
    selection.adaptive_geometry(campaign)
    assert len(requests) == 12
    first = deepcopy(campaign.state["decisions"])
    rows[0]["late_motion_cost"] = 0.1
    selection.adaptive_geometry(campaign)
    assert campaign.state["decisions"] == first
    monkeypatch.setattr(
        selection,
        "read",
        lambda _: {
            "command_joint_mapping": {
                "left": ["left_pitch", "left_thumb_cmc_yaw"],
                "right": ["right_pitch", "right_thumb_cmc_yaw"],
            }
        },
    )
    evidence = {
        "source": "saved reference",
        "hands": {"left": {"mount_excursion_m": 0.02}},
    }
    monkeypatch.setattr(selection, "arm_tracking_evidence", lambda _: evidence)
    selection.select_controllers(campaign)
    controllers = [
        c for c in campaign.state["candidates"].values() if c["kind"] == "controller"
    ]
    assert len(controllers) == 12
    assert sum(c["controller"] == "arm_feedback" for c in controllers) == 1
    assert "reference_preload_0p14" not in campaign.state["candidates"]
    assert "reference_hand_spacing_m_0p16_preload_0p14" in campaign.state["candidates"]


def test_promotions_count_existing_attempts_and_stop_after_failed_repeat(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_selection as selection
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign

    campaign = Campaign(tmp_path)
    names = list(campaign.state["candidates"])
    rows = [outcome(campaign, n, "60s", cost=i) for i, n in enumerate(names[:8])]
    monkeypatch.setattr(selection, "campaign_results", lambda _: rows)
    campaign.state["runs"] = [
        {"candidate": "reference", "stage": "120s", "status": "COMPLETE"}
    ]
    assert selection.long_requests(campaign) == names[1:6]
    rows = [outcome(campaign, n, "120s", cost=i) for i, n in enumerate(names[:2])]
    for r in rows:
        r["result"]["progression_justified"] = True
    requested = []

    def record(jobs):
        requested.extend(jobs)
        for n, stage in jobs:
            campaign.state["runs"].append(
                {
                    "candidate": n,
                    "stage": stage,
                    "status": "COMPLETE",
                    "result": {"progression_justified": n == names[0]},
                }
            )

    monkeypatch.setattr(campaign, "run_many", record)
    selection.qualify(campaign)
    assert requested == [(names[0], "repeat"), (names[1], "repeat"), (names[0], "fine")]
    assert campaign.decision("frozen_finalists", list) == [
        {"id": n, "configuration_sha256": campaign.candidate(n).digest()}
        for n in names[:2]
    ]


def test_duplicate_reset_is_saved_without_physics_and_keeps_original_reference(
    tmp_path,
):
    from teleopit.sim.g1_o6_support.pose_candidates import Candidate
    from teleopit.sim.g1_o6_support.pose_trial import preflight_geometry

    references = []
    assert (
        preflight_geometry(Candidate("first"), tmp_path / "first", references) is None
    )
    assert len(references) == 1
    result = preflight_geometry(Candidate("second"), tmp_path / "second", references)
    assert result["reset_status"] == "DUPLICATE"
    assert result["physics_steps"] == 0 and result["strict_status"] == "NOT RUN"
    assert result["reasons"] == ["near_identical_reset_to:first"]
    assert len(references) == 1 and (tmp_path / "second" / "manifest.json").exists()


def test_sixty_second_quota_counts_the_existing_reference_and_failed_screens(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_selection as selection
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign
    from teleopit.sim.g1_o6_support.pose_candidates import controller_candidates

    campaign = Campaign(tmp_path)
    controls = controller_candidates(
        [campaign.candidate("reference")], {"left": ["a_thumb_cmc_yaw"]}
    )
    campaign.register(controls)
    names = list(campaign.state["candidates"])
    rows = [outcome(campaign, n, cost=i) for i, n in enumerate(names)]
    rows[1]["strict"] = "FAIL"
    campaign.state["runs"] = [
        {"candidate": "reference", "stage": "60s", "status": "COMPLETE"}
    ]
    monkeypatch.setattr(selection, "campaign_results", lambda _: rows)
    requests = selection.sixty_requests(campaign)
    assert len(requests) == 11
    assert ("reference", "60s") not in requests and (names[1], "60s") not in requests
    assert sum(campaign.candidate(n).kind == "controller" for n, _ in requests) == 6


def test_robustness_uses_four_untuned_offsets_after_complete_qualification(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_results
    from teleopit.sim.g1_o6_support.pose_campaign import Campaign
    from teleopit.sim.g1_o6_support.pose_validation import robustness

    campaign = Campaign(tmp_path)
    parent = campaign.candidate("reference")
    campaign.state["runs"] = [
        {
            "candidate": parent.id,
            "configuration_sha256": parent.digest(),
            "stage": stage,
            "status": "COMPLETE",
            "result": {"progression_justified": True},
        }
        for stage in ("120s", "repeat", "fine")
    ]
    monkeypatch.setattr(
        pose_results, "campaign_results", lambda _: [outcome(campaign, parent.id)]
    )
    requests = []
    monkeypatch.setattr(campaign, "run_many", lambda jobs: requests.extend(jobs))
    robustness(campaign)
    assert len(requests) == 4 and all(s == "robustness" for _, s in requests)
    for name, _ in requests:
        child = campaign.candidate(name)
        assert child.pose == parent.pose and child.controller == parent.controller
        assert next(iter(child.reset_offsets.values())) in (-0.002, 0.002)
    assert (
        len({tuple(campaign.candidate(n).reset_offsets.items()) for n, _ in requests})
        == 4
    )
