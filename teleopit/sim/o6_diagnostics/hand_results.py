"""Pure result accounting: strict thresholds, contact retention and completion."""

import numpy as np

SCORED_PHASES = ("SETTLE", "LOAD_RAMP", "PLATEAU")


def maximum(rows, key):
    return max(r[key] for r in rows)


def drift_peak(rows, key):
    values = [r["from_hold"][key] for r in rows if r["from_hold"][key] is not None]
    return max(values, default=None)


def contact_bodies(rows, bodies, minimum):
    contacts = (c for r in rows for c in r["contacts"])
    return sorted(
        {
            tuple(c["bodies"])
            for c in contacts
            if set(c["body_ids"]).issubset(bodies) and c["normal_N"] >= minimum
        }
    )


def metrics(rows, h, cfg):
    return {
        "max_penetration_m": maximum(rows, "penetration_m"),
        "max_joint_error_rad": maximum(rows, "max_joint_error_rad"),
        "max_mimic_error_rad": maximum(rows, "max_mimic_error_rad"),
        "max_translation_m": drift_peak(rows, "translation_m"),
        "max_rotation_rad": drift_peak(rows, "angle_rad"),
        "achieved_force_norm_N": max(
            np.linalg.norm(r["external_wrench_world_at_mount"][:3]) for r in rows
        ),
        "achieved_moment_norm_Nm": max(
            np.linalg.norm(r["external_wrench_world_at_mount"][3:]) for r in rows
        ),
        "self_contact_bodies": contact_bodies(rows, h.bodies, cfg.min_normal_N),
    }


def terminal_failure(failure, terminal):
    numerical = any(terminal["warnings"]) or terminal["reason"] == "nonfinite_state"
    if numerical:
        failure.setdefault("numerical_warning", terminal["time_s"])
    if terminal["reason"] == "excessive_penetration":
        failure.setdefault("penetration", terminal["time_s"])
    if terminal["reason"] in ("fall_drop", "escape"):
        failure.setdefault("grasp_loss", terminal["time_s"])
    return numerical


def outcome(complete, failure, numerical):
    if numerical:
        return "numerical_termination"
    if "grasp_loss" in failure:
        return "grasp_loss"
    return "pass" if complete and not failure else "threshold_failure"


def support_result(rows, h, cfg, failure, complete):
    scored = [r for r in rows if r["phase"] in SCORED_PHASES]
    retained = contact_retained(scored, h.side)
    physical = set(failure) - {"effort"}
    plateau = sum(r["phase"] == "PLATEAU" for r in rows) * h.control_dt
    settle = sum(r["phase"] == "SETTLE" for r in rows) * h.control_dt
    return {
        "physical_retention": retained and complete,
        "zero_load_qualified": bool(scored) and not physical and complete,
        "contact_retained_until_termination": retained,
        "plateau_achieved_s": plateau,
        "plateau_complete": plateau >= cfg.plateau_s - 1e-8,
        "unsupported_elapsed_s": len(scored) * h.control_dt,
        "settle_completed": settle >= cfg.zero_fixture_settle_s - 1e-8,
    }


def contact_retained(rows, side):
    return bool(rows) and all(r["supports"][side + "_hand"]["loaded"] for r in rows)


def endpoint_errors(row):
    return {n: abs(a["target_error_rad"]) for n, a in row["actuators"].items()}


def peak_velocity(rows):
    return max(abs(j["velocity_rad_s"]) for r in rows for j in r["joints"].values())


def closing_endpoints(rows, h, cfg):
    closed = [r for r in rows if r["phase"] == "CLOSED"]
    end = closed[-1] if closed else rows[-1]
    errors = endpoint_errors(end)
    settled = max(endpoint_errors(rows[-1]).values()) < cfg.settling_position_rad
    settled &= peak_velocity(rows[-1:]) < cfg.settling_velocity_rad_s
    blocked = any(c["normal_N"] >= cfg.min_normal_N for c in end["contacts"])
    return {
        "closed_endpoint_errors_rad": errors,
        "endpoint_reached_without_self_contact": max(errors.values())
        < cfg.settling_position_rad
        and not blocked,
        "reopened_settled": bool(settled),
        "peak_joint_velocity_rad_s": peak_velocity(rows),
        "physical_retention": None,
        "zero_load_qualified": None,
        "contact_retained_until_termination": None,
    }


def hand_summary(rows, failure, acquisition, saturation, terminal, h, cfg):
    complete = terminal["reason"] == "schedule_complete"
    numerical = terminal_failure(failure, terminal)
    result = {
        "schema": "o6_hand_result_v1",
        "strict_pass": complete and not failure,
        "outcome": outcome(complete, failure, numerical),
        "first_failure_s": failure,
        "acquisition_failures": sorted(acquisition),
        "diagnostic_end": terminal,
        "saturation": saturation.summary(),
        **metrics(rows, h, cfg),
        **support_result(rows, h, cfg, failure, complete),
    }
    if h.fixed:
        result.update(closing_endpoints(rows, h, cfg))
    return result
