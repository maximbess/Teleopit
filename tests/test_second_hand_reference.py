"""Geometry and contact-order checks independent of a downloaded policy or pose bank."""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from teleopit.runtime.second_hand_reference import build_second_hand_spec


class SecondHandReferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <site name="left_hand" pos=".1 .16 1.8"/>
          <site name="right_hand" pos="0 -.16 1.5"/>
          <site name="left_foot" pos="-.3 .1 .6"/>
          <site name="right_foot" pos="-.3 -.1 .6"/>
          <site name="left_ladder_rung_01_grip" pos=".03 0 1.42" zaxis="0 1 0"/>
          <site name="left_ladder_rung_02_grip" pos=".15 0 1.75" zaxis="0 1 0"/>
          <body name="pelvis" pos="-.3 0 1.1"/>
          <body name="torso" pos="-.3 0 1.2"/>
        </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.meta = {"hand_site_ids": [0,1], "foot_site_ids": [2,3],
                     "pelvis_body_id": 1, "torso_body_id": 2,
                     "track_names": ["left_hand","right_hand","left_foot","right_foot","pelvis","torso"]}
        self.state = {"attached": np.array([[True,True]]), "foot_contact": np.array([[True,True]]),
                      "held_rung": np.array([[1,0]])}
        self.design = json.loads((Path(__file__).resolve().parents[1] / "assets/motions/ladder_second_hand.json").read_text())

    def test_rung_retarget_keeps_grip_offset_and_supports(self):
        spec = build_second_hand_spec(self.model,self.data,self.state,self.meta,self.design)
        right = spec["tracks"][1]
        np.testing.assert_allclose(right["keyframes"][-1]["position_m"], [.12,-.16,1.83])
        self.assertEqual(spec["geometry"]["target_rung_index"],1)
        for i in (0,2,3):
            track = spec["tracks"][i]
            np.testing.assert_array_equal(track["keyframes"][0]["position_m"],track["keyframes"][-1]["position_m"])
        start = np.array(right["keyframes"][0]["position_m"])
        retreat = np.array(right["keyframes"][2]["position_m"])
        self.assertLess(retreat[0], start[0])
        self.assertLess(retreat[1],start[1])
        self.assertAlmostEqual(retreat[2],start[2])
        self.assertEqual(right["keyframes"][1]["time_s"],self.design["times_s"]["release"])

    def test_invalid_start_support_and_phase_order_fail(self):
        self.state["attached"][0,0] = False
        with self.assertRaisesRegex(ValueError,"attachments"):
            build_second_hand_spec(self.model,self.data,self.state,self.meta,self.design)
        self.state["attached"][0,0] = True
        self.state["held_rung"][0] = [0,0]
        with self.assertRaisesRegex(ValueError,"one rung above"):
            build_second_hand_spec(self.model,self.data,self.state,self.meta,self.design)
        self.state["held_rung"][0] = [1,0]
        self.design["times_s"]["release"] = self.design["times_s"]["prepare"]
        with self.assertRaisesRegex(ValueError,"increasing"):
            build_second_hand_spec(self.model,self.data,self.state,self.meta,self.design)


if __name__ == "__main__":
    unittest.main()
