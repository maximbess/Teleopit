"""Targeted behavioral mutations; restore exact source bytes after each probe."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("outputs/g1_o6/ladder_material/mutations")
MUTATIONS = (
    ("mixing_floor", "teleopit/sim/g1_o6/material.py",
     'parameters["friction"][:2] = [material.mu, material.mu]',
     'parameters["friction"][:2] = [max(.8, material.mu), max(.8, material.mu)]',
     "tests/test_g1_o6_ladder_material.py", "actual_contact_coefficients"),
    ("hand_scope_changes_feet", "teleopit/sim/g1_o6/material.py", 'if scope == "H":', 'if False:',
     "tests/test_g1_o6_ladder_material.py", "actual_contact_coefficients"),
    ("vertical_pitch_misread", "teleopit/sim/ladder_config.py", 'pitch_convention="vertical",', 'pitch_convention="along_rail",',
     "tests/test_g1_o6_ladder_geometry.py", "ladderman_and_commercial"),
    ("rear_collision_disabled", "teleopit/sim/ladder_config.py", 'rear_collision=True,', 'rear_collision=False,',
     "tests/test_g1_o6_ladder_geometry.py", "ladderman_and_commercial"),
    ("lost_opposition_promoted", "teleopit/sim/g1_o6_support/ladder_campaign.py",
     'and result.get("contact_quality", {}).get("status") == "PASS"', 'and True',
     "tests/test_g1_o6_ladder_campaign.py", "promotions_reject"),
    ("external_assistance_accepted", "teleopit/sim/g1_o6_support/ladder_campaign.py",
     'and audit.get("no_external_support") is True', 'and True',
     "tests/test_g1_o6_ladder_campaign.py", "promotions_reject"),
    ("housing_support_accepted", "teleopit/sim/g1_o6_support/ladder_campaign.py",
     'and audit.get("housing_contact_steps") == 0', 'and True',
     "tests/test_g1_o6_ladder_campaign.py", "promotions_reject"),
    ("identity_ignored", "teleopit/sim/g1_o6_support/ladder_campaign.py",
     'if record.get("identity") != identity:', 'if False:',
     "tests/test_g1_o6_ladder_ledger.py", "completed_runs_resume"),
)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, file, old, new, test, selector in MUTATIONS:
        path = Path(file)
        original = path.read_bytes()
        source = original.decode()
        if source.count(old) != 1:
            raise ValueError("mutation target is not unique: " + name)
        try:
            path.write_bytes(source.replace(old, new).encode())
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=str((ROOT / (name + "_cache")).resolve()))
            with (ROOT / (name + ".txt")).open("w") as output:
                completed = subprocess.run([sys.executable, "-m", "pytest", test, "-k", selector,
                    "--basetemp", str(ROOT / (name + "_tmp")), "-q", "-x"],
                    stdout=output, stderr=subprocess.STDOUT, env=environment, check=False)
            text = (ROOT / (name + ".txt")).read_text()
            killed = completed.returncode == 1 and ("AssertionError" in text or "Failed: DID NOT RAISE" in text)
            results.append({"mutation": name, "killed_by_behavior_assertion": killed,
                            "returncode": completed.returncode})
            print(json.dumps(results[-1]), flush=True)
        finally:
            path.write_bytes(original)
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["killed_by_behavior_assertion"] for r in results):
        raise SystemExit("mutation survived or failed for an unintended reason")


if __name__ == "__main__":
    main()
