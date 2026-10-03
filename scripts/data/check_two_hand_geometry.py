"""Compare the cluster's compiled scene with the measured pose bank on CPU."""
import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from mjlab.scene import Scene
from teleopit.runtime.assets import PROJECT_ROOT
from train_mimic.data.two_hand_bank import MODEL_FIELDS, validate_model
from train_mimic.tasks.tracking.config.two_hand import make_two_hand_env_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path,
                        default=PROJECT_ROOT / 'assets/motions/ladder_two_hand_bank.npz')
    parser.add_argument('--output', type=Path, default=Path('outputs/two_hand_geometry.json'))
    args = parser.parse_args()
    cfg = make_two_hand_env_cfg()
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device='cpu')
    model = scene.compile()
    names = [model.geom(i).name for i in range(model.ngeom)]
    baseline = json.loads((PROJECT_ROOT / 'assets/motions/ladder_two_hand_geometry.json').read_text())
    report = {'versions': {}, 'bank_geom_count': len(baseline['geom_names']),
              'scene_geom_count': model.ngeom,
              'added_geoms': [n for n in names if n not in baseline['geom_names']],
              'removed_geoms': [n for n in baseline['geom_names'] if n not in names],
              'geoms': [], 'differences': {}}
    for package in ('mujoco', 'mjlab', 'mujoco-warp', 'warp-lang'):
        try:
            report['versions'][package] = version(package)
        except PackageNotFoundError:
            report['versions'][package] = 'unknown'
    for i, name in enumerate(names):
        report['geoms'].append(dict(name=name, body=model.body(int(model.geom_bodyid[i])).name,
                                   type=int(model.geom_type[i]), pos=model.geom_pos[i].tolist(),
                                   size=model.geom_size[i].tolist(),
                                   contype=int(model.geom_contype[i]),
                                   conaffinity=int(model.geom_conaffinity[i])))
    with np.load(args.bank, allow_pickle=False) as bank:
        meta = json.loads(str(bank['metadata']))
        for field in MODEL_FIELDS:
            expected = bank['model_' + field]
            actual = np.asarray(getattr(model, field))
            if expected.shape != actual.shape:
                report['differences'][field] = {'bank_shape': list(expected.shape),
                                                'scene_shape': list(actual.shape)}
            elif not np.allclose(actual, expected, rtol=0., atol=1e-7):
                report['differences'][field] = {'max_delta': float(np.max(np.abs(actual - expected)))}
        try:
            validate_model(model, bank, meta)
            report['compatible'] = True
        except ValueError as error:
            report['compatible'] = False
            report['error'] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'geoms'}, indent=2))
    print(f'Full geometry inventory: {args.output}')


if __name__ == '__main__':
    main()
