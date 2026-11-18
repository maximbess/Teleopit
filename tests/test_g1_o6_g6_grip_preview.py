import numpy as np
import pytest


def test_grip_preview_favors_an_executed_interval_with_all_four_support_conditions():
    from teleopit.sim.g1_o6_support.g6_grip_preview import GraspForecast, choose_grip, alternatives

    unsafe = GraspForecast(12., 1., 0., .7)
    safe = GraspForecast(20., 1., .30, .42)
    healthy = GraspForecast(12., 1., .58, .54)
    assert choose_grip(healthy, [("unused", 1)], lambda _: pytest.fail("extra simulation")) == ("nominal", None, healthy, 0)
    candidates = [("thumb", 1), ("raise", 2), ("lower", 3)]
    forecasts = {1: GraspForecast(25., 1., 1., 1.), 2: safe, 3: GraspForecast(13., 0., .8, .8)}
    assert choose_grip(unsafe, candidates, forecasts.get) == ("raise", 2, safe, 3)
    assert choose_grip(unsafe, candidates, lambda _: healthy) == ("thumb", 1, healthy, 1)
    assert choose_grip(safe, candidates, lambda _: safe) == ("nominal", None, safe, 3)
    assert not forecasts[1].healthy() and forecasts[1].grip_rank()[0] == 1
    assert GraspForecast(24.99, .35, .5, .5).healthy()
    anchor = np.full((2,6), .4)
    choices = dict(alternatives(anchor, anchor, anchor, anchor+1., .02))
    assert len(choices)==3
    assert choices["thumb"][0,0] == pytest.approx(.4008)
    assert choices["raise"][1,3] == pytest.approx(.4008)
    assert choices["lower"][0,4] == pytest.approx(.3992)
    assert all(np.array_equal(x[:,1],anchor[:,1]) for x in choices.values())
    near = anchor + .12
    assert np.max(dict(alternatives(near, near, anchor, near+1., .02))["raise"]-anchor) <= .12+1e-12


def test_forecast_needs_an_opposed_loaded_thumb_on_its_own_tread():
    from teleopit.sim.g1_o6_support.g6_grip_preview import PreviewGrip, opposed_load

    with pytest.raises(ValueError):
        PreviewGrip(thumb_preview="endpoint")
    thumb = [(np.array([1., 0., 0.]), .4), (np.array([1., 0., 0.]), .2)]
    assert opposed_load(thumb, []) == 0.
    assert opposed_load(thumb, [np.array([-.5, .8660254, 0.])]) == 0.
    assert opposed_load(thumb, [np.array([-1., 0., 0.])]) == pytest.approx(.6)


def test_native_grip_forecast_matches_each_solver_step_and_final_endpoint(tmp_path):
    import gzip
    import json
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_candidates import G6Candidate
    from teleopit.sim.g1_o6_support.g6_grip_preview_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from scripts.verify_g1_o6_g6_grip_feedback import grasp_sample
    from scripts.verify_g1_o6_g6_grip_preview import verify_intervals

    candidate = candidates()[-1]
    assert G6Candidate.from_dict(candidate.record()) == candidate
    env = make_environment(candidate, 6, .001)
    ref = env.hold_controller
    while ref.grip.capture_s is None:
        env.step()
        assert env.phase not in ("ABORT","HOLD_COMPLETE")
    controls, contacts = [], []
    def sample(e):
        row = snapshot(e)
        row["support_targets"] = e.support_targets
        readings = [grasp_sample(row, s) for s in ("left", "right")]
        actual.append([np.linalg.norm(row["static_balance_residual_N"]), min(readings[0][0][1:]),
                       *[forces[0] if opposed else 0. for forces,opposed in readings]])
        return {"phase": e.phase, "support_targets": e.support_targets, "contacts": row["contacts"]}
    for _ in range(35):
        actual=[]
        env.step(lambda e: contacts.append(sample(e)))
        controls.append({**sample(env), "pose_search": {"controller_state": ref.state()}})
        force, finger, left, right = np.asarray(actual).T
        assert [ref.last.peak_N, ref.last.finger_N, ref.last.left_thumb_N, ref.last.right_thumb_N] == pytest.approx(
            [max(force), min(finger), min(left), min(right)],abs=1e-8)
        assert 0 <= ref.extra <= 3 and ref.count <= 21
        assert len(ref.state()["grip_preview"]["tested_forecasts"]) == ref.extra
        if ref.extra:
            break
    assert ref.extra > 0
    meta = {"config": {"control_dt": .02, "physics_dt": .001}, "support_targets": env.support_targets}
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in controls))
    with gzip.open(tmp_path / "physics_contacts.jsonl.gz", "wt") as out:
        out.writelines(json.dumps(row) + "\n" for row in contacts)
    receipt = verify_intervals(tmp_path, meta)
    assert receipt["max_thumb_forecast_error_N"] < 1e-8
    assert sum(receipt["choices"].values()) == len(controls)


def test_reserved_preview_requires_the_three_archived_failures(tmp_path):
    import json
    from teleopit.sim.g1_o6_support.g6_grip_feedback_experiment import LAWS, SUPPORT, passes, specification
    from teleopit.sim.g1_o6_support.g6_grip_preview_experiment import PreviewCampaign
    from teleopit.sim.g1_o6_support.g6_knee_retention_experiment import prior_failures
    from teleopit.sim.g1_o6_support.ladder_campaign import digest, file_hash
    from test_g1_o6_g6_ground_reaction import passing_hold

    campaign = PreviewCampaign(tmp_path / "g6_grip_preview")
    assert campaign.state["specification"]["budgets"]["screen"] == 1
    with pytest.raises(ValueError, match="exhausted"):
        campaign.prepare()
    prior = tmp_path / "g6_grip_feedback"
    prior.mkdir()
    milestone = prior / "milestones.json"
    milestone.write_text(json.dumps({"initial_pose": "PASS", "candidate": "palm_height"}))
    with pytest.raises(ValueError, match="exhausted"):
        campaign.prepare()
    milestone.write_text(json.dumps({"initial_pose": "FAIL", "candidate": None}))
    ledger = prior / "campaign_state.json"
    ledger.write_text(json.dumps({"spec_sha256": "other", "runs": []}))
    with pytest.raises(ValueError, match="unexpected preceding"):
        campaign.prepare()
    result = passing_hold()
    result["strict"]["failure_reasons"] = []
    audit = result["physics_audit"]
    audit["scored_steps"] = 120000
    audit["settled_grip"] = {"capture_s": 5.}
    audit["grip_support"] = {k: {"samples": n, "loss_steps": dict.fromkeys(SUPPORT, 0)}
                             for k, n in (("native", 120000), ("control", 6000))}
    audit["knee_rung3"] = {k: {"samples": n, "minimum_m": .04, "contact_steps": 0}
                           for k, n in (("native", 120000), ("control", 6000))}
    assert passes(result, "120s")
    def reconcile():
        return prior_failures(prior, specification=specification, names=LAWS, gate=passes)
    state = {"spec_sha256": digest(specification()), "runs": []}
    for name in LAWS:
        ledger.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="terminal physical failure"):
            reconcile()
        folder = prior / name
        folder.mkdir()
        summary = folder / "summary.json"
        # The original anatomy passes. A single native thumb loss disqualifies this run.
        audit["grip_support"]["native"]["loss_steps"]["left_thumb"] = 1
        summary.write_text(json.dumps(result))
        state["runs"].append({"candidate": name, "id": name + "_120s", "identity": name, "stage": "120s",
                              "status": "COMPLETE", "output": name, "result": result,
                              "files_sha256": {"summary.json": file_hash(summary)}})
    ledger.write_text(json.dumps(state))
    assert set(reconcile()) == set(LAWS)
    summary.write_text("{}")
    with pytest.raises(ValueError, match="raw evidence"):
        reconcile()
