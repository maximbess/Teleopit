import numpy as np


def test_matrix_keeps_measured_pairs_reserve_and_budget():
    from teleopit.sim.g1_o6_support.capacity_matrix import load_cases

    reference = {
        "time_s": 30,
        "selection": ["settled"],
        "wrench_world_at_mount": [2, -1, 3, 0.1, 0.2, 0.3],
    }
    cases = load_cases([reference])
    combined = next(c for c in cases if c["family"] == "combined" and c["scale"] == 1.2)
    np.testing.assert_allclose(
        combined["wrench"], np.array(reference["wrench_world_at_mount"]) * 1.2
    )
    assert combined["reference"]["time_s"] == 30
    assert len(load_cases([reference] * 3)) <= 39
    assert all(np.linalg.norm(c["wrench"][:3]) <= 50 for c in cases)


def test_failed_zero_blocks_loading_instead_of_claiming_retention(
    tmp_path, monkeypatch
):
    import teleopit.sim.g1_o6_support.capacity_matrix as matrix

    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return {
            "strict_pass": False,
            "physical_retention": False,
            "plateau_complete": False,
        }

    monkeypatch.setattr(matrix, "run_capacity", run)
    result = matrix.evaluate_side(object(), "left", tmp_path, [], 0.001, 48)
    assert len(calls) == 1
    assert result["physical_status"] == "FAIL"
    assert result["loading_status"] == "BLOCKED"
    assert not result["reserve_qualified"]
