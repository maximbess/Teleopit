"""Exercise the actual downstream controller wrappers without issuing a load or transfer request."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.g1_o6_g6_thumb_replay import loads
from teleopit.sim.g1_o6_support.scene_contract import restore_environment
from teleopit.sim.g1_o6_support.transfer import TransferTrial
from teleopit.sim.g1_o6_support.long_hold import enrich
from teleopit.sim.o6_diagnostics.harness import Source
from teleopit.sim.o6_diagnostics.nominal import coupled_telemetry


def compare_commands(run, meta, mode, output):
    env = restore_environment(meta, meta["config"]["physics_dt"], saved_scene=run / "scene.mjb")
    source_controller = env.hold_controller
    request = dict(mode=mode, side="right", cycles=1, alternating=False, target_share=.3,
                   three_seconds=1, plateau_seconds=10)
    output.mkdir(parents=True, exist_ok=True)
    trial = TransferTrial(env, request, output)
    telemetry = coupled_telemetry(env)
    needs_proximal = meta["controller"].get("right_retention", {}).get("intervention") == "recenter"
    exercised = False
    try:
        with (run / "steps.jsonl").open("rb") as stream:
            next(stream)
            ticks = 0
            for row in map(loads, stream):
                # Keep transfer overlays idle. Compare the wrapped live command to the source
                # at every tick, including the captured wrist and bilateral grip decision.
                measured = env.step(trial.observe if row["phase"] == "HOLD" else None)
                # Exercise acquisition and the initial transfer baseline. Stop advancing
                # the transfer protocol before any change to the requested load.
                if (row["hold_time_s"] or 0) < .12:
                    enrich(env, measured, telemetry, trial.hold_start, trial.audit)
                    trial.update(measured)
                np.testing.assert_array_equal(env.data.ctrl, row["applied_position_targets_rad"])
                np.testing.assert_array_equal(measured["joint_positions"], row["joint_positions"])
                ticks += 1
                state = trial.controller.state()
                wrist = state["wrist_grip"]
                exercised = wrist["updates"]["right"] > 0 and (not needs_proximal or state["right_proximal_updates"] >= 2)
                if exercised:
                    break
        if not exercised or trial.controller.original is not source_controller:
            raise ValueError("sourced wrist and grip never ran through the transfer wrapper")
        return {"wrapper": type(trial.controller).__name__, "live_grip": type(source_controller).__name__,
                "live_arm": type(source_controller.parent).__name__, "matched_control_ticks": ticks,
                "native_hold_samples": trial.audit.samples["physics"], "hold_s": row["hold_time_s"],
                "right_wrist_updates": wrist["updates"]["right"], "proximal_updates": state.get("right_proximal_updates", 0),
                "support_targets": env.support_targets}
    finally:
        trial.audit.close()


def audit(run, output):
    meta = json.loads((run / "resolved.json").read_bytes())
    try:
        Source(run)
    except ValueError as error:
        capacity = {"status": "BLOCKED", "reason": str(error)}
    else:
        capacity = {"status": "SOURCE_ACCEPTED"}
    return {"source": run.name, "capacity": capacity, "coupled_setup": {
        mode: compare_commands(run, meta, mode, output / mode) for mode in ("load", "adjacent")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    (args.output / "handoff.json").write_text(json.dumps(result, indent=2) + "\n")
