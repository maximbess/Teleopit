import numpy as np
import pytest

from scripts.g1_o6_g6_thumb_replay import edge_margin, first_event, rotation_in_tread
from scipy.spatial.transform import Rotation


def test_signed_margin_uses_nearest_usable_edge_even_outside():
    points = np.array([[-.013, -.084, -.019], [.039, .2698, -.010], [-.043, 0, 0]])
    assert edge_margin(points, [.04, .27, .019]).tolist() == pytest.approx([.027, .0002, -.003])


def test_spatial_rotation_uses_the_tread_axes_and_the_actual_reference_pose():
    tread = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    rest = Rotation.from_euler("x", .3).as_matrix()
    new = Rotation.from_euler("y", -.01).as_matrix() @ rest
    np.testing.assert_allclose(rotation_in_tread(rest, new, tread), [-.01, 0, 0], atol=1e-12)


def test_first_event_preserves_a_single_native_step_and_absence():
    t = np.array([52.028, 52.029, 52.030])
    assert first_event(t, [False, True, False]) == 52.029
    assert first_event(t, [False, False, False]) is None
