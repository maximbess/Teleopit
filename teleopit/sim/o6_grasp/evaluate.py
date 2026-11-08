"""Reproducible deterministic scenarios, JSONL trajectories and JSON summaries."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import mujoco
import numpy as np

from .env import O6GraspEnv


def scenarios(config):
    """One-variable comparisons; finite rung length is evaluated separately."""
    return {
        'free_motion': replace(config, rung_enabled=False, evaluation_mode='acquisition'),
        'acquisition': replace(config, evaluation_mode='acquisition', load_force=(0, 0, 0)),
        'fixture_removal': replace(config, load_force=(0, 0, 0)),
        'pullout_small': config,
        'pullout_large': replace(config, load_force=(0, 30, 0)),
        'twist_along': replace(config, load_force=(0, 0, 0), load_torque=(0, 0.15, 0)),
        'twist_across': replace(config, load_force=(0, 0, 0), load_torque=(0.15, 0, 0)),
        'combined': replace(config, load_torque=(0, 0.1, 0)),
        'impulse': replace(config, load_force=(0, 0, -10), load_profile='impulse'),
        'release': replace(config, load_profile='constant'),
        'friction_low': replace(config, friction=0.25),
        'alignment_offset': replace(config, palm_pos=tuple(np.asarray(config.palm_pos) + [0.004, 0, 0])),
        'rung_small': replace(config, radius=0.010),
        'rung_large': replace(config, radius=0.014),
        'rectangular': replace(config, cross_section='rectangular'),
        'actuator_delay': replace(config, actuator_delay_steps=3),
        'finite_edge': replace(config, length=0.06, rung_pos=tuple(np.asarray(config.rung_pos) + [0, 0.04, 0])),
        'fine_timestep': replace(config, physics_dt=config.physics_dt / 2,
                                 solver_iterations=config.solver_iterations * 2, solver_tolerance=config.solver_tolerance / 10),
    }


def summarize(name, config, initial, rows):
    unsupported = [row for row in rows if row['unsupported']]
    assisted = [row for row in rows if row['fixture_gain'] > 0 and row['mode'] != 'RELEASE']
    verified = [row for row in unsupported if row['verification'] == 'VERIFIED']
    failures = [row for row in unsupported if row['verification'] in ('FAILED', 'SLIPPING')]
    warnings = any(any(row['numerical_warnings']) for row in rows)
    penetration = max(row['max_penetration'] for row in rows)
    duration = len(verified) * config.control_dt
    return dict(scenario=name, evaluation_mode=config.evaluation_mode,
                initial_penetration=initial['max_penetration'],
                unsupported_seconds=len(unsupported) * config.control_dt,
                verified_seconds=duration, unsupported_success=bool(duration >= 1.0 and not failures and not warnings
                    and initial['max_penetration'] < 1e-6 and penetration < config.max_penetration),
                assisted_contact_seconds=sum(bool(row['contacts']) for row in assisted) * config.control_dt,
                failure_observed=bool(failures), release_clear=rows[-1]['release_clear'],
                max_translation=max((row['translation_drift'] for row in unsupported), default=0.),
                max_rotation=max((row['rotation_drift'] for row in unsupported), default=0.),
                max_contact_slip=max(row['contact_slip_speed'] for row in rows),
                max_penetration=penetration, max_coupling_error=max(row['coupling_error'] for row in rows),
                max_effort_fraction=max(row['effort_fraction'] for row in rows),
                numerical_warnings=warnings,
                peak_external_force=max(np.linalg.norm(row['external_wrench'][:3]) for row in rows),
                final_verification=rows[-1]['verification'])


def run_trial(name, config, output, *, seed=0, callback=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = O6GraspEnv(config)
    _, initial = env.reset(seed=seed)
    (output / f'{name}.config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    (output / f'{name}.xml').write_text(env.scene_xml)
    rows = []
    with (output / f'{name}.jsonl').open('w') as stream:
        stream.write(json.dumps(dict(event='reset', **initial)) + '\n')
        while True:
            obs, _, _, done, info = env.step(np.zeros(6))
            info['actor_observation'] = obs.tolist()
            stream.write(json.dumps(info) + '\n')
            rows.append(info)
            if callback is not None:
                callback(env, info)
            if done:
                break
    summary = summarize(name, config, initial, rows)
    summary.update(seed=seed, mujoco_version=mujoco.__version__, config=asdict(config))
    (output / f'{name}.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def run_suite(config, output, *, seed=0):
    results = [run_trial(name, cfg, output, seed=seed) for name, cfg in scenarios(config).items()]
    path = Path(output)
    (path / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# Deterministic O6 evaluation', '',
             '| Scenario | Unsupported verified (s) | Success | Failure/sliding | Release clear | Peak penetration (mm) |',
             '|---|---:|---|---|---|---:|']
    for row in results:
        lines.append(f"| {row['scenario']} | {row['verified_seconds']:.2f} | {row['unsupported_success']} | "
                     f"{row['failure_observed']} | {row['release_clear']} | {1000 * row['max_penetration']:.2f} |")
    (path / 'summary.md').write_text('\n'.join(lines) + '\n')
    return results
