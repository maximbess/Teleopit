import numpy as np
import pytest

from teleopit.sim.g1_o6_support.regrasp_actions import DigitCommand, fixed_model


def test_independent_signed_commands_open_from_encoders_and_hold_target():
    measured = np.array([.26, .48, 1.1, .8, .75, .7])
    closed = np.array([.35, .5, 1.35, 1.35, 1.35, 1.35])
    hands = [DigitCommand(closed, np.full(6, 4.), .12) for _ in range(2)]
    for hand in hands:
        hand.follow(measured + .02, 1.)
    def tick(hand, request, angles=measured, dt=.02):
        return hand.advance(request, angles, dt).copy()
    start = tick(hands[1], 0)
    left = tick(hands[0], -3)
    np.testing.assert_allclose(left, measured * .98)
    np.testing.assert_array_equal(tick(hands[0], 0, measured / 2), left)
    np.testing.assert_array_equal(tick(hands[1], 0), start)
    for _ in range(49):
        tick(hands[0], -1)
    np.testing.assert_allclose(hands[0].target, 0, atol=1e-12)
    np.testing.assert_allclose(tick(hands[0], 1, measured * 0, .5),
                               np.minimum(.5 * closed, .12 / 4))
    with pytest.raises(ValueError):
        tick(hands[0], np.nan)


def test_pelvis_fixture_holds_six_axes_and_leaves_fingers_free():
    import mujoco
    xml = """<mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom size=".1"/><body pos="0 0 .2"><joint name="digit"/>
      <geom size=".03"/></body></body></worldbody></mujoco>"""
    pose = np.array([.1, .2, .8, 1, 0, 0, 0])
    model, _ = fixed_model(xml, pose)
    data = mujoco.MjData(model)
    data.qpos[:7] = pose
    data.qfrc_applied[[0, 4, 6]] = [.3, .2, .001]
    for _ in range(200):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[:7], pose, atol=2e-5)
    assert abs(data.qpos[7]) > .001
    assert model.neq == 1


def test_contact_classes_require_the_correct_loaded_opposing_surfaces():
    from teleopit.sim.g1_o6_support.regrasp_measure import hand_state

    def pad(digit, z, normal, target=True, load=1.):
        return dict(group=digit, target=target, normal_force_N=load,
                    normal_on_hand_world=[0, 0, normal], position_world_m=[0, 0, z],
                    rung_position_m=[0, 0, z])
    contacts = [pad("thumb", -.018, -1),
                *[pad(d, .018, 1) for d in ("index", "middle", "ring", "pinky")]]
    assert hand_state([], 0.) == "air"
    assert hand_state([dict(c, target=False) for c in contacts], 0.) == "wrong_surface"
    assert hand_state(contacts, 0.) == "grasp"
    for change in ({"normal_force_N": .01}, {"normal_on_hand_world": [0, 0, 1]},
                   {"position_world_m": [0, 0, .02]}):
        assert hand_state([dict(contacts[0], **change), *contacts[1:]], 0.) == "touch"
