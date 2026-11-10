"""Optional scientific figures; no plotting import on a simulation-only path."""

import numpy as np


def figure(rows, n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n, 1, figsize=(11, 2 * n), sharex=True, constrained_layout=True
    )
    for ax in np.atleast_1d(axes):
        ax.grid(alpha=0.2)
    return plt, fig, np.atleast_1d(axes), [r.get("time", r.get("time_s")) for r in rows]


def save(fig, path, name):
    fig.savefig(path / (name + ".png"), dpi=130)
    fig.savefig(path / (name + ".pdf"))


def curve(ax, t, rows, value, label, **options):
    ax.plot(t, [value(r) for r in rows], label=label, **options)


def norm_or_missing(value):
    return np.nan if value is None else np.linalg.norm(value)


def label_axes(axes, labels):
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.legend(fontsize=7, loc="upper left", ncol=2)


def support_curves(force, slip, t, diagnostic):
    for name in diagnostic[0]["supports"]:
        curve(
            force,
            t,
            diagnostic,
            lambda r, name=name: r["supports"][name]["wrench_world_about_rung"][2],
            name,
        )
        curve(
            slip,
            t,
            diagnostic,
            lambda r, name=name: r["supports"][name]["slip_m_s"],
            name,
        )
        missing = [
            x for x, r in zip(t, diagnostic) if r["supports"][name]["contact_loss"]
        ]
        slip.plot(missing, [0.025] * len(missing), "rx", markersize=2)
    curve(
        force,
        t,
        diagnostic,
        lambda r: r["assistance_wrench_world_at_base"][2],
        "pelvis assistance",
    )


def wrist_curves(force, moment, t, rows):
    for side in ("left", "right"):
        curve(
            force,
            t,
            rows,
            lambda r, side=side: np.linalg.norm(r["wrist_world_on_hand"][side][:3]),
            side,
        )
        curve(
            moment,
            t,
            rows,
            lambda r, side=side: np.linalg.norm(r["wrist_world_on_hand"][side][3:]),
            side,
        )


def drift_curves(ax, t, rows):
    rotation = ax.twinx()
    for name, color in [("from_reset", "orange"), ("from_hold", "green")]:
        curve(ax, t, rows, lambda r, name=name: r[name]["translation_m"], name)
        curve(
            rotation,
            t,
            rows,
            lambda r, name=name: r[name]["angle_rad"],
            name + " rotation",
            color=color,
            linestyle="--",
        )
    ax.axhline(0.03, color="red", linestyle=":")
    rotation.set_ylabel("orientation change (rad)")
    rotation.legend(fontsize=6, loc="upper right")


def base_speed(ax, t, rows):
    curve(
        ax,
        t,
        rows,
        lambda r: np.linalg.norm(r["base_velocity_world_m_s"]),
        "linear m/s",
    )
    angular = ax.twinx()
    curve(
        angular,
        t,
        rows,
        lambda r: np.linalg.norm(r["base_angular_velocity_body_rad_s"]),
        "angular rad/s",
        color="orange",
    )
    angular.set_ylabel("rad/s")


def finger_curves(ax, t, rows):
    for name in rows[0]["actuators"]:
        if "_o6_" not in name:
            continue
        if max(r["actuators"][name]["effort_fraction"] for r in rows) < 0.98:
            continue
        curve(
            ax,
            t,
            rows,
            lambda r, name=name: r["actuators"][name]["effort_fraction"],
            name.replace("_servo", ""),
        )
    ax.axhline(0.98, color="red", linestyle=":")


def limit_curves(ax, t, rows):
    for name in rows[0]["joints"]:
        if "thumb" not in name:
            continue
        curve(
            ax,
            t,
            rows,
            lambda r, name=name: max(
                r["joints"][name]["lower_violation_rad"],
                r["joints"][name]["upper_violation_rad"],
            ),
            name,
        )
    ax.axhline(0.02, color="red", linestyle=":")


def residual_part(row, part):
    value = row["momentum_rate"]["balance_residual_N_Nm"]
    return np.nan if value is None else np.linalg.norm(value[part])


def residual_curves(ax, t, rows):
    curve(
        ax,
        t,
        rows,
        lambda r: norm_or_missing(r["momentum_rate"]["linear_N"]),
        "linear momentum rate N",
        alpha=0.5,
    )
    curve(
        ax,
        t,
        rows,
        lambda r: residual_part(r, slice(0, 3)),
        "linear balance residual N",
    )
    moment = ax.twinx()
    curve(
        moment,
        t,
        rows,
        lambda r: residual_part(r, slice(3, 6)),
        "moment residual Nm",
        color="orange",
    )
    moment.set_ylabel("moment residual Nm")


def phase_markers(axes, rows):
    boundaries = [r for r in rows if r.get("next_phase") not in (None, r["phase"])]
    for row in boundaries:
        for ax in axes:
            ax.axvline(row["time"], color="black", linestyle=":", linewidth=0.6)
        axes[0].annotate(
            row["next_phase"],
            (row["time"], 1),
            xycoords=("data", "axes fraction"),
            rotation=45,
            fontsize=7,
        )


def nominal_plot(rows, path):
    plt, fig, axes, t = figure(rows, 9)
    diagnostic = [r["diagnostic_v1"] for r in rows]
    support_curves(axes[0], axes[7], t, diagnostic)
    wrist_curves(axes[1], axes[2], t, rows)
    drift_curves(axes[3], t, diagnostic)
    base_speed(axes[4], t, diagnostic)
    finger_curves(axes[5], t, diagnostic)
    limit_curves(axes[6], t, diagnostic)
    residual_curves(axes[8], t, diagnostic)
    label_axes(
        axes,
        [
            "Support Fz (N)",
            "Wrist force (N)",
            "Mount moment (Nm)",
            "Base displacement (m)",
            "Base speed (m/s)",
            "Finger effort / limit",
            "Thumb limit error (rad)",
            "Slip (m/s); red x = lost",
            "Momentum / residual (N)",
        ],
    )
    phase_markers(axes, rows)
    axes[-1].set_xlabel("Simulated time (s); 20 ms curves, physics-step event table")
    fig.suptitle(
        "Nominal failure diagnostics; continuation remains failed and unscored"
    )
    save(fig, path, "nominal_failure_timeline")
    axes[-1].set_xlim(2.0, max(2.1, min(t[-1], 6.0)))
    save(fig, path, "nominal_failure_zoom")
    plt.close(fig)


def hand_tracking(axes, t, rows):
    for name in rows[0]["actuators"]:
        label = name.replace("_o6_", " ").replace("_servo", "")
        curve(
            axes[0],
            t,
            rows,
            lambda r, name=name: r["actuators"][name]["target_error_rad"],
            label,
        )
        curve(
            axes[1],
            t,
            rows,
            lambda r, name=name: r["actuators"][name]["effort_fraction"],
            label,
        )
    curve(axes[2], t, rows, lambda r: r["max_joint_error_rad"], "limit violation")
    curve(axes[2], t, rows, lambda r: r["max_mimic_error_rad"], "mimic residual")
    curve(axes[3], t, rows, lambda r: r["penetration_m"], "penetration m")
    curve(
        axes[4],
        t,
        rows,
        lambda r: max(abs(j["velocity_rad_s"]) for j in r["joints"].values()),
        "peak absolute joint speed",
    )


def hand_loads(axes, t, rows):
    curve(
        axes[5],
        t,
        rows,
        lambda r: np.linalg.norm(r["external_wrench_world_at_mount"][:3]),
        "external force N",
    )
    moments = axes[5].twinx()
    curve(
        moments,
        t,
        rows,
        lambda r: np.linalg.norm(r["external_wrench_world_at_mount"][3:]),
        "external moment Nm",
        color="orange",
    )
    moments.set_ylabel("moment Nm")
    curve(axes[6], t, rows, lambda r: r["fixture_gain"], "acquisition fixture gain")
    curve(
        axes[6],
        t,
        rows,
        lambda r: r["from_hold"]["translation_m"],
        "unsupported drift m",
    )


def hand_plot(rows, path, title):
    fixed = rows[0]["fixed_base"]
    plt, fig, axes, t = figure(rows, 5 if fixed else 7)
    hand_tracking(axes, t, rows)
    if not fixed:
        hand_loads(axes, t, rows)
    label_axes(
        axes,
        [
            "Target error (rad)",
            "Effort fraction",
            "Joint error (rad)",
            "Penetration (m)",
            "Joint speed (rad/s)",
            "Applied force (N)",
            "Fixture gain / drift (m)",
        ],
    )
    axes[-1].set_xlabel("Simulated time (s)")
    fig.suptitle(title)
    save(fig, path, "trajectory")
    plt.close(fig)


def load_group(axes, cases, side, outcome):
    selected = [
        c
        for c in cases
        if c.get("executed") and (c["side"], c["outcome"]) == (side, outcome)
    ]
    durations = [c["plateau_achieved_s"] for c in selected]
    for ax, part in zip(axes, [slice(0, 3), slice(3, 6)]):
        ax.scatter(
            [
                np.linalg.norm(c["requested_wrench_world_at_mount"][part])
                for c in selected
            ],
            durations,
            label=side + ": " + outcome,
            alpha=0.65,
        )


def load_plot(cases, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    groups = sorted({(c["side"], c["outcome"]) for c in cases if c.get("executed")})
    for side, outcome in groups:
        load_group(axes, cases, side, outcome)
    for ax, label in zip(
        axes, ["Requested force norm (N)", "Requested moment norm (Nm)"]
    ):
        ax.set_xlabel(label)
        ax.set_ylabel("Achieved plateau (s)")
        ax.set_ylim(-0.2, 10.5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Measured load response; fresh resets; unexecuted scales remain unknown"
    )
    save(fig, path, "load_response")
    plt.close(fig)
