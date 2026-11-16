"""G6 geometry and physical evidence must survive every experiment boundary."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest


def test_resolved_g6_rebuild_preserves_model_pose_and_native_contacts(tmp_path):
    from teleopit.sim.g1_o6.config import Config
    from teleopit.sim.g1_o6.diagnostics import snapshot
    from teleopit.sim.g1_o6.env import Environment
    from teleopit.sim.g1_o6.grasp_pose import GraspPose
    from teleopit.sim.g1_o6_support.long_hold import experiment_metadata
    from teleopit.sim.g1_o6_support.scene_contract import restore_environment
    from teleopit.sim.ladder_config import geometry_matrix

    env = Environment(
        Config(root_offset_x=-0.02),
        pose=GraspPose(hand_rungs=(7, 7), foot_rungs=(2, 2)),
        ladder=geometry_matrix()["G6"],
    )
    meta = experiment_metadata(env, snapshot(env), {"mode": "fixture"})
    restored = restore_environment(meta, 0.0005)
    assert restored.ladder == env.ladder and restored.material is None
    assert restored.config.friction_scale == 1
    assert restored.config.control_dt == 0.02 and restored.model.opt.timestep == 0.0005
    np.testing.assert_array_equal(restored.data.qpos, env.data.qpos)
    for field in (
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_size",
        "geom_pos",
        "actuator_forcerange",
        "body_mass",
    ):
        np.testing.assert_array_equal(
            getattr(restored.model, field), getattr(env.model, field)
        )
    mujoco.MjModel.from_xml_string("<mujoco/>")
    restored.save_scene(tmp_path / "scene.xml")
    exported = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert exported.nq == restored.model.nq
    assert exported.geom("right_ladder_rung_08").contype[0] == 1
    broken = deepcopy(meta)
    del broken["environment_spec"]
    with pytest.raises(ValueError, match="environment"):
        restore_environment(broken, 0.001)


def test_30_percent_and_sixty_second_requests_have_ordered_prerequisites():
    from teleopit.sim.g1_o6_support.qualification import transition_prerequisites
    from teleopit.sim.g1_o6_support.transfer import validate_request

    request = dict(
        mode="load",
        side="left",
        target_share=0.3,
        cycles=1,
        alternating=False,
        three_seconds=1,
        plateau_seconds=60,
        physics_dt=0.001,
    )
    validate_request(request)
    before = [
        {
            "metadata": {
                "experiment": dict(request, target_share=0.2, plateau_seconds=10)
            }
        }
    ]
    assert "previous_load_level" in transition_prerequisites(request, [])
    assert "thirty_second_load_plateau" in transition_prerequisites(request, before)
    before.append({"metadata": {"experiment": dict(request, plateau_seconds=30)}})
    assert not transition_prerequisites(request, before)


def test_capacity_matching_rejects_geometry_material_and_source_state_changes():
    from teleopit.sim.g1_o6_support.qualification import capacity_matches

    source = dict(
        grasp_pose={},
        controller={},
        support_targets={},
        assets={},
        config={},
        ladder={"geometry_sha256": "g6"},
        material=None,
        environment_spec={"ladder": "G6"},
        initial={"base_pose": [0], "joint_positions": [1]},
    )
    meta = dict(
        source_grasp_pose={},
        controller={},
        source_support_targets={},
        assets={},
        source_config={},
        source_ladder=source["ladder"],
        source_material=None,
        source_environment_spec=source["environment_spec"],
        source_initial=source["initial"],
    )
    assert capacity_matches(meta, source)
    for key, changed in (
        ("source_ladder", {"geometry_sha256": "g0"}),
        ("source_material", {"mu": 0.4}),
        ("source_initial", {"base_pose": [1], "joint_positions": [1]}),
    ):
        bad = dict(meta, **{key: changed})
        assert not capacity_matches(bad, source), key
    bad = dict(meta)
    del bad["source_ladder"]
    assert not capacity_matches(bad, source)


def test_30_percent_balance_tolerance_requires_explicit_capacity_bounds():
    from teleopit.sim.g1_o6_support.guards import share_settled

    def row(left, right):
        return {
            "supports": {
                s + "_hand": {"force_world": [0, 0, f]}
                for s, f in zip(("left", "right"), (left, right))
            }
        }

    assert not share_settled(row(15.5, 14.5), 0.3, 100)
    assert share_settled(
        row(15.5, 14.5), 0.3, 100, capacity_shares={"left": 0.16, "right": 0.16}
    )
    assert not share_settled(
        row(15.5, 14.5), 0.3, 100, capacity_shares={"left": 0.15, "right": 0.16}
    )
    assert not share_settled(
        row(17, 13), 0.3, 100, capacity_shares={"left": 0.25, "right": 0.25}
    )


def test_capacity_search_reports_uncovered_demands_and_uses_declared_bounds():
    from teleopit.sim.g1_o6_support.capacity import CapacityBounds
    from teleopit.sim.g1_o6_support.capacity_matrix import load_cases

    references = [{"time_s": 10, "wrench_world_at_mount": [0, 0, 51, 0, 0, 1]}]
    limited = load_cases(references)
    reserve = next(
        c for c in limited if c["family"] == "combined" and c["scale"] == 1.2
    )
    assert not reserve["within_budget"]
    declared = CapacityBounds(force_N=70, moment_Nm=4)
    expanded = load_cases(references, bounds=declared)
    reserve = next(
        c for c in expanded if c["family"] == "combined" and c["scale"] == 1.2
    )
    assert reserve["within_budget"]
    np.testing.assert_allclose(reserve["wrench"], [0, 0, 61.2, 0, 0, 1.2])
    with pytest.raises(ValueError):
        CapacityBounds(force_N=-1)
