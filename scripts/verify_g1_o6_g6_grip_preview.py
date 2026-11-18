"""Validate every bilateral shadow forecast against the interval actually executed."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.verify_g1_o6_g6_grip_feedback import SIDES, FeedbackAudit, grasp_sample, verify_run as verify_feedback
from scripts.verify_g1_o6_g6_whole_body import read, verify

NAMES = ("thumb", "raise", "lower")


def check_alternate(cap, name, nominal, previous, anchor, native, dt):
    expected = nominal
    if name != "nominal":
        delta = np.zeros_like(previous)
        delta[:, 0] = .04 * dt
        delta[:, 2:] = {"thumb": 0., "raise": .04, "lower": -.04}[name] * dt
        delta[(previous > native) & (delta > 0.)] = 0.
        expected = np.clip(previous + delta, anchor - .12, anchor + .12)
        if name == "thumb":
            expected[:, 2:] = nominal[:, 2:]
    if not np.allclose(cap, expected, rtol=0., atol=1e-10):
        raise ValueError("executed alternative differs from the bounded hand command")


class PreviewAudit(FeedbackAudit):
    def nominal_cap(self, control):
        return control["grip_preview"]["nominal_cap_rad"]

    def executed_cap(self, control, nominal, previous, anchor, native):
        cap = np.asarray(control["grip_feedback_cap_rad"]).reshape(2, 6)
        check_alternate(cap, control["grip_preview"]["choice"], nominal, previous, anchor, native, self.dt)
        return cap


def score(value):
    loads = [value[k] for k in ("finger_N", "left_thumb_N", "right_thumb_N")]
    hard = int(value["peak_N"] >= 25.) + sum(v < .05 for v in loads)
    deficits = [max(0., 1. - v/t) for v, t in zip(loads, (.35, .50, .50))]
    return hard, max(max(deficits), value["peak_N"]/25. - 1.), sum(deficits)


def choose_receipt(primary, forecasts):
    winner, best = "nominal", primary
    for name, forecast in zip(NAMES, forecasts):
        if score(forecast) < score(best):
            winner, best = name, forecast
    return winner, best


def requested_count(primary, trials, enabled):
    if not enabled or score(primary) == (0, 0., 0.):
        return 0
    return next((i for i, item in enumerate(trials, 1) if score(item) == (0, 0., 0.)), 3)


def check_selection(state, executed, enabled):
    trials, primary = state["tested_forecasts"], state["nominal_forecast"]
    count = requested_count(primary, trials, enabled)
    if len(trials) != count or state["extra_evaluations"] != count or count > 3:
        raise ValueError("unexpected number of native grip previews")
    name, winner = choose_receipt(primary, trials)
    if state["choice"] != name or executed != winner:
        raise ValueError("selected forecast differs from the bounded search")


def check_thumb_forecast(forecast, measured):
    predicted = [forecast["left_thumb_N"], forecast["right_thumb_N"]]
    error = float(np.max(np.abs(np.subtract(predicted, measured))))
    if not error <= 1e-8:
        raise ValueError("selected thumb forecast differs from the executed contact interval")
    return error


def thumb_values(row):
    return np.array([loads[0] if opposed else 0. for loads, opposed in (grasp_sample(row, s) for s in SIDES)])


def verify_intervals(path, meta):
    stride = round(meta["config"]["control_dt"] / meta["config"]["physics_dt"])
    choices, extras = Counter(), Counter()
    maximum = 0.
    with (path/"steps.jsonl").open() as stream, gzip.open(path/"physics_contacts.jsonl.gz", "rt") as raw:
        native = (r for r in map(json.loads, raw) if r["phase"] == "HOLD")
        rows = (r for r in map(json.loads, stream) if r["phase"] == "HOLD")
        for row in rows:
            row["support_targets"] = meta["support_targets"]
            measured = thumb_values(row)
            for _ in range(stride):
                measured = np.minimum(measured, thumb_values(next(native)))
            control = row["pose_search"]["controller_state"]
            saved = control["grip_preview"]
            forecast = control["ground_reaction_preview"]["forecast"]
            enabled = control["settled_grip"]["capture_s"] is not None and not control["inhibition"]
            check_selection(saved, forecast, enabled)
            maximum = max(maximum, check_thumb_forecast(forecast, measured))
            choices[saved["choice"]] += 1
            extras[saved["extra_evaluations"]] += 1
        assert next(native, None) is None
    return {"choices": dict(choices), "extra_evaluations_per_interval": dict(extras), "max_thumb_forecast_error_N": maximum}


def verify_run(path, report):
    meta = read(path/"resolved.json")
    if "grip_preview" not in meta["controller"]:
        return verify_feedback(path, report)
    return {**verify_feedback(path, report, adaptive_type=PreviewAudit), "grip_preview": verify_intervals(path, meta)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/g1_o6/g6_grip_preview"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    verify(args.root, args.run_id, verify_function=verify_run)
