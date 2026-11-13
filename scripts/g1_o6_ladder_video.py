"""Replay recorded states through a verified scene without changing run evidence."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from g1_o6_state_video import video

from teleopit.sim.g1_o6_support.ladder_campaign import file_hash


def replay(root, identity, tail):
    state = json.loads((root / "campaign_state.json").read_text())
    record = next(r for r in state["runs"] if r["id"] == identity)
    if record["status"] != "COMPLETE":
        raise ValueError("replay requires a completed immutable run")
    run = root / record["output"]
    scene = root / "reconstructions" / identity
    proof = json.loads((scene / "proof.json").read_text())
    assert file_hash(scene / "scene.xml") == proof["files_sha256"]["scene.xml"]
    output = root / "videos" / identity
    scratch = (root / "replay_inputs").resolve()
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=identity + "_", dir=scratch) as temporary:
        view = Path(temporary).resolve()
        assert view.is_relative_to(
            scratch
        )  # Verify cleanup stays inside workspace scratch.
        for name in ("steps.jsonl", "summary.json", "resolved.json"):
            assert file_hash(run / name) == record["files_sha256"][name]
            os.link(run / name, view / name)  # Offline renderer only reads these links.
        os.link(scene / "scene.xml", view / "scene.xml")
        video(view, output, tail)
    receipt_path = output / "replay.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(
        source_run=record["output"],
        run_identity=record["identity"],
        verified_scene=(scene / "scene.xml").as_posix(),
        scene_sha256=proof["files_sha256"]["scene.xml"],
        reconstruction_proof_sha256=file_hash(scene / "proof.json"),
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identities", nargs="+")
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/g1_o6/ladder_material")
    )
    parser.add_argument("--tail-seconds", type=float, default=10)
    args = parser.parse_args()
    for identity in args.identities:
        replay(args.root, identity, args.tail_seconds)
