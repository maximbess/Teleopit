"""Focused assertion-killed mutations for the bounded acquisition wrapper."""

from pathlib import Path
import argparse

import g1_o6_ladder_mutations as runner

BASE = "teleopit/sim/g1_o6_support/"
TEST = "tests/test_g1_o6_g6_arm_feedback.py"
CAMPAIGN = "tests/test_g1_o6_g6_feedback_campaign.py"
runner.MUTATIONS = (
    ("activation_delayed_to_hold", BASE + "g6_arm.py",
     'if env.phase in ("REMOVE_ASSISTANCE", "HOLD"):', 'if env.phase == "HOLD":',
     TEST, "starts_at_removal"),
    ("feedback_rate_tenfold", BASE + "g6_arm.py",
     'rate, self.settings.rate_rad_s)', 'rate, 10 * self.settings.rate_rad_s)',
     TEST, "starts_at_removal"),
    ("command_projection_bypassed", BASE + "g6_arm.py",
     'if delta[0] <= CARTESIAN_LIMIT_M and delta[1] <= ANGULAR_LIMIT_RAD:\n            return offset, *delta',
     'if True:\n            return offset, *delta', TEST, "cartesian_bound_checks"),
    ("reference_relatched", BASE + "g6_arm.py",
     'if self.reference:\n            return', 'if False:\n            return',
     TEST, "starts_at_removal"),
    ("loaded_thumb_retargeted", BASE + "g6_arm.py",
     'self.quality[side]["thumb_normal_N"] < LOADED_N', 'not self.quality[side]["valid"]',
     TEST, "loaded_thumb"),
    ("physical_command_clipping_ignored", BASE + "g6_arm.py",
     'np.clip(command, limits[:, 0], limits[:, 1])', 'command',
     TEST, "physically_clipped"),
    ("premise_audit_skipped", BASE + "g6_feedback_experiment.py",
     'if name.startswith("seek_") and not any', 'if False and not any',
     CAMPAIGN, "seek_requires"),
    ("feedback_identity_ignored", BASE + "g6_candidates.py",
     '"controller": self.record()["controller"],', '"controller": "reference",',
     CAMPAIGN, "identity_and_resume"),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_arm_feedback/mutations"))
    args = parser.parse_args()
    runner.main(args.output)
