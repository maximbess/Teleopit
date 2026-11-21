"""Behavioral mutations for thumb-edge reconstruction, wrist direction and qualification."""

import argparse
from pathlib import Path

import g1_o6_ladder_mutations as runner

BASE = "teleopit/sim/g1_o6_support/"
RETENTION = "tests/test_g1_o6_g6_thumb_retention.py"
HANDOFF = "tests/test_g1_o6_g6_retention_handoff.py"
runner.MUTATIONS = (
    ("edge_sign", "scripts/g1_o6_g6_thumb_replay.py", "- np.abs(np.asarray(points)", "+ np.abs(np.asarray(points)",
     "tests/test_g1_o6_g6_thumb_replay.py", "signed_margin"),
    ("opposite_preorientation", BASE + "g6_thumb_retention.py", "frame @ (PREORIENT_RAD", "frame @ (-PREORIENT_RAD",
     RETENTION, "single_corrections"),
    ("wrist_away_from_center", BASE + "g6_thumb_retention.py", "recenter = -RECENTER_GAIN", "recenter = RECENTER_GAIN",
     RETENTION, "releases_wrist"),
    ("four_mm_without_margin", BASE + "g6_thumb_retention_experiment.py", '"eligible": margin > BASE_MARGIN[when]',
     '"eligible": True', RETENTION, "four_mm"),
    ("unexplained_promotion", BASE + "g6_thumb_retention_experiment.py", 'or not self.explained(name):', ':',
     RETENTION, "explanation_are_both_required"),
    ("inclusive_force_gate", BASE + "g6_transfer_retention.py", "not residual < BALANCE_LIMIT_N",
     "not residual <= BALANCE_LIMIT_N", HANDOFF, "scores_each_required"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    runner.main(parser.parse_args().output)
