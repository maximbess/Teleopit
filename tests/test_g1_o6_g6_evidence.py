"""Saved physics-step evidence must reject coefficient changes and omissions."""

import gzip
import json

import pytest

from scripts.verify_g1_o6_g6 import contacts


def test_native_contact_verification_rejects_tampering_and_missing_steps(tmp_path):
    row = {"phase": "HOLD", "contacts": [{
        "geoms": ["left_o6_thumb", "left_ladder_rung_07"],
        "friction": [1.8, 1.8, .02, .002, .002]}]}
    path = tmp_path / "physics_contacts.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    result = contacts(tmp_path, 1)
    assert result["scored_rows"] == result["physics_rows"] == 1
    assert result["observed_hand_vectors"] == [[1.8, 1.8, .02, .002, .002]]
    with pytest.raises(AssertionError):
        contacts(tmp_path, 2)
    row["contacts"][0]["friction"][2] = .2
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        contacts(tmp_path, 1)
