"""CPU-only reference authoring/IK checks; also runnable with unittest."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from teleopit.runtime.cartesian_reference import load_reference, solve_reference
from teleopit.runtime.ladder_scene import build_ladder_preview_model


class CartesianReferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
          <body name="first"><joint name="x" type="slide" axis="1 0 0" range="-.5 .5"/>
            <joint name="yaw" type="hinge" axis="0 0 1" range="-2 2"/>
            <geom type="sphere" size=".02"/><site name="hand"/></body>
          <body name="second" pos="0 2 0"><joint name="z" type="slide" axis="0 0 1" range="-.5 .5"/>
            <geom type="sphere" size=".02"/></body>
        </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.spec = {"version": 1, "duration_s": 1., "fps": 10, "tracks": [
            {"name": "hand", "kind": "site", "target": "hand", "keyframes": [
                {"time_s": 0, "offset_m": [0, 0, 0]},
                {"time_s": 1, "offset_m": [.2, 0, 0],
                 "quaternion_wxyz": [np.cos(.3), 0, 0, np.sin(.3)]}]},
            {"name": "body", "kind": "body", "target": "second", "keyframes": [
                {"time_s": 0, "offset_m": [0, 0, 0]},
                {"time_s": 1, "offset_m": [0, 0, .1]}]},
        ]}

    def load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.json"
            path.write_text(json.dumps(self.spec), encoding="utf-8")
            return load_reference(path, self.model, self.data)

    def test_interpolation_and_orientation_have_exact_endpoints(self):
        _, tracks = self.load()
        np.testing.assert_allclose(tracks[0].sample(0)[0], [0, 0, 0])
        np.testing.assert_allclose(tracks[0].sample(.5)[0], [.1, 0, 0])
        np.testing.assert_allclose(tracks[0].sample(1)[0], [.2, 0, 0])
        np.testing.assert_allclose(tracks[0].sample(.5)[1], [np.cos(.15), 0, 0, np.sin(.15)])
        self.assertLess(np.linalg.norm(tracks[0].sample(.0001)[0])/ .0001, 1e-6)

    def test_multiple_body_and_site_tracks_solve_together(self):
        spec, tracks = self.load()
        clip, report = solve_reference(self.model, self.data, spec, tracks)
        self.assertTrue(report["tracking_ok"])
        self.assertLess(clip["position_error_m"].max(), 1e-4)
        np.testing.assert_allclose(clip["qpos"][-1], [.2, .6, .1], atol=1e-4)
        np.testing.assert_allclose(self.data.qpos, 0)  # Solver restores the initial state.

    def test_unreachable_goal_is_reported_and_joint_limits_are_respected(self):
        self.spec["tracks"][0]["keyframes"][-1]["offset_m"] = [2, 0, 0]
        spec, tracks = self.load()
        clip, report = solve_reference(self.model, self.data, spec, tracks)
        self.assertFalse(report["tracking_ok"])
        self.assertLessEqual(clip["qpos"][:, 0].max(), .5)
        self.assertGreater(clip["position_error_m"][-1, 0], 1)

    def test_ambiguous_or_out_of_order_keys_are_rejected(self):
        self.spec["tracks"][0]["keyframes"][1]["time_s"] = 0
        with self.assertRaisesRegex(ValueError, "key times"):
            self.load()
        self.spec["tracks"][0]["keyframes"][1]["time_s"] = 1
        self.spec["tracks"][0]["keyframes"][0]["position_m"] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.load()

    def test_supplied_ladder_reference_preserves_support_and_has_no_reported_penetration(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "assets/robots/unitree_g1/omniretarget_collision/manifest.json").exists():
            self.skipTest("Downloaded collision assets are unavailable")
        model, data = build_ladder_preview_model()
        np.testing.assert_allclose(data.qpos[:3], [-.913384, 0, 1.364509])
        np.testing.assert_allclose(data.site("left_grip_site").xpos,
                                   [-.5713792109299028, .1673431160355614, 1.4771781962009825])
        spec, tracks = load_reference(root / "assets/motions/ladder_first_hand.json", model, data)
        clip, report = solve_reference(model, data, spec, tracks)
        self.assertTrue(report["tracking_ok"], report["tracks"])
        self.assertEqual(report["collision_frames"], [])
        for name in ("right_hand", "left_foot", "right_foot"):
            self.assertLess(report["tracks"][name]["max_position_error_m"], 1e-4)
        self.assertLess(np.abs(np.diff(clip["qpos"][:, 7:], axis=0)).max(), .25)


if __name__ == "__main__":
    unittest.main()
