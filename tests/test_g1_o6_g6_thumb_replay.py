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


def test_missing_contact_exports_json_null_and_first_failure_uses_the_earliest_clock():
    import json
    from scripts.g1_o6_g6_thumb_report import first_failure
    from scripts.g1_o6_g6_thumb_replay import json_values

    assert json.dumps(json_values({"position":[1., float("nan")]}), allow_nan=False) == '{"position": [1.0, null]}'
    result = first_failure({"right_thumb":52.029, "left_thumb":52.184, "knee":None},
                           {"drift":55., "left_fingers":52.029})
    assert result == {"hold_s":52.029, "conditions":["left_fingers","right_thumb"]}
    assert first_failure({"t":None}, {}) is None


def test_reference_can_be_repeated_from_the_tracked_evidence_without_raw_archives():
    from pathlib import Path
    from scripts.g1_o6_g6_reference_replay import reference_inputs

    candidate, expected = reference_inputs(Path("docs/g1_o6/g6_wrist_grip/results.json"))
    assert candidate.controller.wrist_mode == "right"
    assert candidate.pose.hand_rungs == (7,7) and candidate.pose.foot_rungs == (2,2)
    assert expected["60s"] == "135a4b2bc2b71e7509fbb233f3ba94a921687b0b7b1d173d97ed815ec7b0eb98"


def test_unreachable_reset_has_model_time_zero_and_no_hold_timestamp():
    from scripts.g1_o6_g6_thumb_report import prehold_failure

    result = {"reset_status":"INFEASIBLE", "reasons":["initial_joint_range"]}
    assert prehold_failure(result) == {"hold_s":None, "model_s":0., "phase":"RESET", "conditions":["initial_joint_range"]}


def test_native_reference_hash_tracks_sample_bytes_independently_of_gzip_time(tmp_path):
    import gzip
    from scripts.g1_o6_g6_reference_replay import decoded_hash

    a, b = tmp_path/"a.gz", tmp_path/"b.gz"
    a.write_bytes(gzip.compress(b"contact at 52.029", mtime=1))
    b.write_bytes(gzip.compress(b"contact at 52.029", mtime=2))
    assert a.read_bytes() != b.read_bytes() and decoded_hash(a) == decoded_hash(b)
    b.write_bytes(gzip.compress(b"contact at 52.030", mtime=1))
    assert decoded_hash(a) != decoded_hash(b)
