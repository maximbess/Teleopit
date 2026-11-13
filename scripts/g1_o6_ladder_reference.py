"""Reproduce the reviewed reference before geometry parameterization."""

import argparse
import json
from pathlib import Path

import numpy as np

from teleopit.sim.g1_o6_support.pose_candidates import Candidate
from teleopit.sim.g1_o6_support.pose_trial import SearchEnvironment, run_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, choices=(30, 120), required=True)
    args = parser.parse_args()
    candidate = Candidate("reference")
    if args.seconds == 30:
        env = SearchEnvironment(candidate, 30, 0.001)
        arrays = {
            name: value for name in dir(env.model)
            if isinstance(value := getattr(env.model, name), np.ndarray)
        }
        arrays["reset_qpos"] = env.data.qpos.copy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.parent / "reviewed_model.npz", **arrays)
    result = run_candidate(candidate, args.output, seconds=args.seconds)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
