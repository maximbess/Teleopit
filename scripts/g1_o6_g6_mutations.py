"""Focused G6 behavioral mutations using the exact-byte restoration runner."""

from pathlib import Path
import argparse

import g1_o6_ladder_mutations as runner

BASE = "teleopit/sim/g1_o6_support/"
runner.MUTATIONS = (
    (
        "skip_twenty_percent",
        BASE + "qualification.py",
        "0.3: 0.2",
        "0.3: 0.1",
        "tests/test_g1_o6_g6_plumbing.py",
        "30_percent",
    ),
    (
        "sample_state_ignored",
        BASE + "capacity.py",
        'value = {"qpos": [*row["base_pose"], *row["joint_positions"]], "targets": targets}',
        'value = {"qpos": [], "targets": targets}',
        "tests/test_g1_o6_g6_state_evidence.py",
        "sampled_state",
    ),
    (
        "released_hand_still_required",
        BASE + "g6_gates.py",
        'return {"left", "right"} - ({selected} if phase in MOVING_PHASES else set())',
        'return {"left", "right"}',
        "tests/test_g1_o6_g6_search.py",
        "phase_contract",
    ),
    (
        "short_loaded_hold_accepted",
        BASE + "g6_gates.py",
        ">= 59.98",
        ">= 0",
        "tests/test_g1_o6_g6_transfer.py",
        "loaded_gate",
    ),
    (
        "same_tread_called_upward",
        BASE + "g6_reach.py",
        'int(env.support_targets[side + "_hand"].rsplit("_", 1)[1]) + 1',
        'int(env.support_targets[side + "_hand"].rsplit("_", 1)[1]) + 0',
        "tests/test_g1_o6_g6_search.py",
        "adjacent_destination",
    ),
    (
        "reach_rate_tenfold",
        BASE + "g6_transfer.py",
        "limit = 0.05 * env.config.control_dt",
        "limit = 0.5 * env.config.control_dt",
        "tests/test_g1_o6_g6_motion.py",
        "actuator_targets",
    ),
    (
        "changed_source_resumed",
        BASE + "g6_campaign.py",
        "if code_provenance() != frozen:",
        "if False:",
        "tests/test_g1_o6_g6_orchestration.py",
        "negative_runs",
    ),
    (
        "capacity_upper_bound_lost",
        BASE + "guards.py",
        "min(capacity_shares[side], target / 2 + cfg.per_hand_balance_tolerance)",
        "target / 2 + cfg.per_hand_balance_tolerance",
        "tests/test_g1_o6_g6_plumbing.py",
        "balance_tolerance",
    ),
    *[
        m
        for m in runner.MUTATIONS
        if m[0] in ("global_xml_export", "external_assistance_accepted")
    ],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_stability/mutations"))
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.only:
        runner.MUTATIONS = tuple(m for m in runner.MUTATIONS if m[0] == args.only)
        if not runner.MUTATIONS:
            parser.error("unknown mutation")
    runner.main(args.output)
