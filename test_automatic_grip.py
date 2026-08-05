from __future__ import annotations

import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np


SITE = 0
BODY = 1
EQUALITY = 2
CAPSULE = 3
SPHERE = 4
WELD = 5


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    raise AssertionError("Test helper only expects positive-trace rotations")


fake_mujoco = types.ModuleType("mujoco")
fake_mujoco.MjModel = object
fake_mujoco.MjData = object
fake_mujoco.mjtObj = types.SimpleNamespace(
    mjOBJ_SITE=SITE,
    mjOBJ_BODY=BODY,
    mjOBJ_EQUALITY=EQUALITY,
)
fake_mujoco.mjtGeom = types.SimpleNamespace(
    mjGEOM_CAPSULE=CAPSULE,
    mjGEOM_SPHERE=SPHERE,
)
fake_mujoco.mjtEq = types.SimpleNamespace(mjEQ_WELD=WELD)


def _name2id(model: object, obj_type: int, name: str) -> int:
    names = model.names[obj_type]
    try:
        return names.index(name)
    except ValueError:
        return -1


def _id2name(model: object, obj_type: int, obj_id: int) -> str | None:
    names = model.names[obj_type]
    return names[obj_id] if 0 <= obj_id < len(names) else None


def _object_velocity(
    model: object,
    data: object,
    obj_type: int,
    obj_id: int,
    result: np.ndarray,
    local: int,
) -> None:
    del model, obj_type, local
    result[:] = data.site_velocities[obj_id]


def _quat2mat(result: np.ndarray, quaternion: np.ndarray) -> None:
    result[:] = _quat_to_matrix(quaternion).reshape(9)


def _mat2quat(result: np.ndarray, matrix: np.ndarray) -> None:
    result[:] = _matrix_to_quat(matrix.reshape(3, 3))


def _forward(model: object, data: object) -> None:
    del model
    data.forward_calls += 1


fake_mujoco.mj_name2id = _name2id
fake_mujoco.mj_id2name = _id2name
fake_mujoco.mj_objectVelocity = _object_velocity
fake_mujoco.mju_quat2Mat = _quat2mat
fake_mujoco.mju_mat2Quat = _mat2quat
fake_mujoco.mj_forward = _forward
sys.modules["mujoco"] = fake_mujoco

PATCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PATCH_ROOT))

from teleopit.robots.automatic_grip import AutomaticGrip  # noqa: E402


class FakeModel:
    def __init__(self) -> None:
        self.names = {
            SITE: [
                "left_grip_site",
                "right_grip_site",
                "left_grip_anchor",
                "right_grip_anchor",
                "left_ladder_rung_01_grip",
            ],
            BODY: ["world", "left_grip_anchor_body", "right_grip_anchor_body"],
            EQUALITY: ["left_grip_weld", "right_grip_weld"],
        }
        self.nsite = len(self.names[SITE])
        self.site_type = np.array(
            [SPHERE, SPHERE, SPHERE, SPHERE, CAPSULE], dtype=np.int32
        )
        self.site_size = np.zeros((self.nsite, 3), dtype=np.float64)
        self.site_size[4] = [0.075, 0.35, 0.0]
        self.site_bodyid = np.array([0, 0, 1, 2, 0], dtype=np.int32)
        self.body_mocapid = np.array([-1, 0, 1], dtype=np.int32)
        self.eq_type = np.array([WELD, WELD], dtype=np.int32)
        self.site_pos = np.zeros((self.nsite, 3), dtype=np.float64)
        self.site_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (self.nsite, 1),
        )


class FakeData:
    def __init__(self) -> None:
        self.site_xpos = np.zeros((5, 3), dtype=np.float64)
        # Near the positive end of the 0.70 m rung, not its center.
        self.site_xpos[0] = [0.07, 0.34, 0.0]
        self.site_xpos[1] = [1.0, 0.0, 0.0]
        self.site_xmat = np.tile(np.eye(3).reshape(1, 9), (5, 1))
        # The capsule's local +Z axis points along world +Y.
        self.site_xmat[4] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        ).reshape(9)
        self.site_velocities = np.zeros((5, 6), dtype=np.float64)
        self.mocap_pos = np.zeros((2, 3), dtype=np.float64)
        self.mocap_quat = np.zeros((2, 4), dtype=np.float64)
        self.eq_active = np.zeros(2, dtype=np.uint8)
        self.forward_calls = 0


def _make_grip(*, auto_attach: bool = True) -> tuple[AutomaticGrip, FakeData]:
    model = FakeModel()
    data = FakeData()
    grip = AutomaticGrip(
        model=model,
        data=data,
        attach_distance=0.0,
        max_hand_speed=0.25,
        auto_attach=auto_attach,
        rung_site_pattern=r".*_ladder_rung_[0-9]+_grip",
        left_hand_site="left_grip_site",
        right_hand_site="right_grip_site",
        left_anchor_body="left_grip_anchor_body",
        right_anchor_body="right_grip_anchor_body",
        left_anchor_site="left_grip_anchor",
        right_anchor_site="right_grip_anchor",
        left_weld="left_grip_weld",
        right_weld="right_grip_weld",
    )
    return grip, data


class AutomaticGripTest(unittest.TestCase):
    def test_attaches_near_full_length_and_without_position_snap(self) -> None:
        grip, data = _make_grip()

        grip.update()

        self.assertTrue(grip.is_attached("left"))
        self.assertEqual(
            grip.attached_rung("left"), "left_ladder_rung_01_grip"
        )
        self.assertEqual(int(data.eq_active[0]), 1)
        np.testing.assert_allclose(data.mocap_pos[0], data.site_xpos[0])
        self.assertFalse(grip.status("left").requested)
        self.assertFalse(grip.is_attached("right"))

    def test_release_disables_weld_and_requires_new_request(self) -> None:
        grip, data = _make_grip()
        grip.update()

        grip.release("left")
        grip.update()

        self.assertEqual(int(data.eq_active[0]), 0)
        self.assertFalse(grip.is_attached("left"))
        np.testing.assert_allclose(data.mocap_pos[0], [0.0, 0.0, -1.0])

        grip.request_attach("left")
        grip.update()
        self.assertTrue(grip.is_attached("left"))

    def test_hand_speed_gate(self) -> None:
        grip, data = _make_grip()
        data.site_velocities[0, 3:6] = [0.3, 0.0, 0.0]

        grip.update()
        self.assertFalse(grip.is_attached("left"))

        data.site_velocities[0, 3:6] = 0.0
        grip.update()
        self.assertTrue(grip.is_attached("left"))

    def test_manual_mode_and_reset(self) -> None:
        grip, data = _make_grip(auto_attach=False)
        grip.update()
        self.assertFalse(grip.is_attached("left"))

        grip.request_attach("left")
        grip.update()
        self.assertTrue(grip.is_attached("left"))

        grip.reset()
        self.assertEqual(int(data.eq_active[0]), 0)
        self.assertFalse(grip.status("left").requested)
        self.assertFalse(grip.is_attached("left"))

    def test_capsule_endcap_is_used(self) -> None:
        grip, data = _make_grip()
        # Beyond the endpoint in Y, but still inside the spherical cap.
        data.site_xpos[0] = [0.02, 0.40, 0.0]
        grip.update()
        self.assertTrue(grip.is_attached("left"))

    def test_invalid_hand_fails_fast(self) -> None:
        grip, _ = _make_grip()
        with self.assertRaisesRegex(ValueError, "left.*right"):
            grip.release("middle")


if __name__ == "__main__":
    unittest.main()
