import json
from types import SimpleNamespace


def test_capacity_reports_an_unsupported_live_law_without_generating_reserve_evidence(tmp_path):
    from teleopit.sim.g1_o6_support.g6_capacity import run_g6_capacity

    (tmp_path / "resolved.json").write_text(json.dumps({"controller": {
        "schema": "g6_grasp_acquisition_v1", "isolated_runner_support": False}}))
    result = run_g6_capacity(tmp_path, tmp_path / "bench")
    assert result["capacity_status"] == "BLOCKED" and result["states"] == []
    assert result["controller_schema"] == "g6_grasp_acquisition_v1"
    assert result == json.loads((tmp_path / "bench/summary.json").read_text())


def test_transfer_retains_sourced_live_controller_telemetry():
    from teleopit.sim.g1_o6_support.control import TransferController

    controller = TransferController.__new__(TransferController)
    controller.original = SimpleNamespace(state=lambda: {"live_wrist_updates": 72})
    assert controller.state() == {"live_wrist_updates": 72}


def test_loaded_transfer_scores_each_required_hand_and_always_scores_knee_and_residual():
    from teleopit.sim.g1_o6_support.g6_transfer_retention import retention_losses

    quality = {"thumb_opposition": True, "thumb_normal_N": .05, "minimum_finger_normal_N": .05}
    hands = {s: {"quality":dict(quality)} for s in ("left","right")}
    assert retention_losses(hands, set(hands), 24.999, .00001, False) == []
    hands["left"]["quality"]["minimum_finger_normal_N"] = .049
    hands["right"]["quality"]["thumb_opposition"] = False
    assert set(retention_losses(hands, set(hands), 25., 0., False)) == {
        "left_fingers", "right_thumb", "force_balance", "right_knee_rung3"}
    assert retention_losses(hands, {"left"}, 10., .01, False) == ["left_fingers"]
    assert retention_losses(hands, {"right"}, 10., .01, True) == ["right_thumb", "right_knee_rung3"]


def test_relocated_saved_model_preserves_exact_identity_and_rejects_changed_mass(tmp_path):
    import mujoco
    import numpy as np
    import pytest
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6_support.g6_thumb_retention_experiment import candidates
    from teleopit.sim.g1_o6_support.g6_trial import make_environment
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import restore_environment

    env = make_environment(candidates()[0], 5, .001)
    env.model.mesh_pathadr[1] = env.model.mesh_pathadr[0]
    scene = tmp_path / "scene.mjb"
    meta = experiment_metadata(env, snapshot(env), {"mode":"hold"})
    mujoco.mj_saveModel(env.model, str(scene), None)
    restored = restore_environment(meta, .0005, saved_scene=scene)
    np.testing.assert_array_equal(restored.model.mesh_pathadr, env.model.mesh_pathadr)
    assert restored.model.opt.timestep == .0005
    env.model.body_mass[env.model.body("pelvis").id] += .1
    mujoco.mj_saveModel(env.model, str(scene), None)
    with pytest.raises(ValueError, match="undeclared physical"):
        restore_environment(meta, .001, saved_scene=scene)
