"""Behavioral mutation probes for named acquisition, evidence and stage gates."""

import argparse
from pathlib import Path

import g1_o6_ladder_mutations as runner

BASE = "teleopit/sim/g1_o6_support/"
LAW = "tests/test_g1_o6_g6_acquisition.py"
CAMPAIGN = "tests/test_g1_o6_g6_acquisition_campaign.py"
EVIDENCE = "tests/test_g1_o6_g6_acquisition_evidence.py"
RECOVERY = "tests/test_g1_o6_g6_evidence_recovery.py"
runner.MUTATIONS = (
    ("thumb_axes_swapped", BASE + "g6_acquisition.py",
     '{"rh_thumb_cmc_pitch": pitch, "rh_thumb_cmc_yaw": yaw}',
     '{"rh_thumb_cmc_pitch": yaw, "rh_thumb_cmc_yaw": pitch}', LAW, "named_thumb"),
    ("finger_delay_removed", BASE + "g6_acquisition.py",
     'FINGER_DELAY_FRACTION = .25', 'FINGER_DELAY_FRACTION = 0.', LAW, "thumb_first_schedule"),
    ("preload_increased", BASE + "g6_acquisition.py",
     'self.base.kp[indices], self.base.preload)', 'self.base.kp[indices], 100 * self.base.preload)', LAW, "thumb_first_schedule"),
    ("single_hand_accepted", BASE + "g6_acquisition_audit.py",
     'all(hand["quality"]["valid"] for hand in hands.values())',
     'any(hand["quality"]["valid"] for hand in hands.values())', LAW, "half_a_second"),
    ("instantaneous_contact_accepted", BASE + "g6_acquisition_audit.py",
     '>= self.required_seconds - 1e-9', '>= 0', LAW, "half_a_second"),
    ("structural_violation_ignored", BASE + "g6_acquisition_audit.py",
     'and not reasons', 'and True', LAW, "half_a_second"),
    ("invalid_reference_latched_early", BASE + "g6_acquisition.py",
     'return env.phase == "HOLD"', 'return env.phase in ("REMOVE_ASSISTANCE", "HOLD")', LAW, "identity_scene"),
    ("absent_grasp_stabilized", BASE + "g6_acquisition_experiment.py",
     'and grasp_established(result)', 'and True', CAMPAIGN, "derived_only_once"),
    ("premise_audit_bypassed", BASE + "g6_acquisition_experiment.py",
     'if preceding and not any', 'if False and not any', CAMPAIGN, "two_failures"),
    ("anatomical_order_ignored", "scripts/g1_o6_g6_acquisition_report.py",
     '"valid": ordered and opposed', '"valid": opposed', EVIDENCE, "requires_real"),
    ("unloaded_thumb_accepted", "scripts/g1_o6_g6_acquisition_report.py",
     'contact["normal_force"] < .05', 'contact["normal_force"] < 0', EVIDENCE, "requires_real"),
    ("physical_failure_retried", "scripts/verify_g1_o6_g6.py",
     'assert previous["status"] in ("ERROR", "RUNNING")', 'assert True', RECOVERY, "physical_retry"),
    ("experiment_identity_rewritten", "scripts/verify_g1_o6_g6.py",
     'assert record["identity"] == previous["identity"] == digest(record["payload"])',
     'assert record["identity"] == digest(record["payload"])', RECOVERY, "changed_experiment"),
    ("unfinished_attempt_ignored", "scripts/verify_g1_o6_g6.py",
     'assert all(r["status"] == "COMPLETE" for r in previous.values()), "unfinished attempt lacks completed recovery"',
     'assert True', RECOVERY, "unfinished"),
    ("original_attempt_tampered", "scripts/verify_g1_o6_g6.py",
     'assert previous["identity"] == digest(previous["payload"])', 'assert True', RECOVERY, "changed_prior_payload"),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/g1_o6/g6_grasp_acquisition/mutations"))
    runner.main(parser.parse_args().output)
