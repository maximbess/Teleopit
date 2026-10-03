"""Package measured starts and validated right-hand clips for portable training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import mujoco
from train_mimic.data.two_hand_bank import model_signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, default=Path('outputs/first_hand_pose_bank'))
    p.add_argument('--references', type=Path, default=Path('outputs/second_hand_reference'))
    p.add_argument('--output', type=Path, default=Path('assets/motions/ladder_two_hand_bank.npz'))
    a = p.parse_args()
    meta = json.loads((a.bank/'manifest.json').read_text())
    result = {k: [] for k in ('qpos','qvel','position','quaternion','joint_pos','joint_vel')}
    specs = []
    for index in meta['representatives']:
        folder = a.references/f'pose_{index:04d}'
        report = json.loads((folder/'validation.json').read_text())
        if not report['kinematic_checks_ok']:
            raise ValueError(f'Pose {index} failed validation')
        with np.load(a.bank/f'states/state_{index:04d}.npz') as state, np.load(folder/'reference.npz') as clip:
            spec = json.loads(str(clip['reference_json']))
            if hashlib.sha256((a.bank/f'states/state_{index:04d}.npz').read_bytes()).hexdigest() != spec['provenance']['state_sha256']:
                raise ValueError('Reference and starting state do not match')
            np.testing.assert_allclose(clip['qpos'][0],state['qpos'][-1],atol=1e-6)
            result['qpos'].append(state['qpos'][-1])
            result['qvel'].append(state['qvel'][-1])
            result['joint_pos'].append(clip['qpos'][:,clip['joint_qposadr'][1:]])
            result['joint_vel'].append(clip['qvel'][:,clip['joint_dofadr'][1:]])
            result['position'].append(clip['target_position'])
            result['quaternion'].append(clip['target_quaternion_wxyz'])
            specs.append(spec)
    model = mujoco.MjModel.from_binary_path(str(a.bank/'scene.mjb'))
    info = dict(version=1, model_signature=model_signature(model), pose_ids=meta['representatives'], joint_names=meta['joint_names'],
                physics_joint_names=meta['physics_joint_names'], fps=specs[0]['fps'],
                contact_schedule=specs[0]['contact_schedule'], provenance=[s['provenance'] for s in specs],
                source_checkpoint_sha256=meta['checkpoint_sha256'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output, **{k:np.asarray(v,dtype=np.float32) for k,v in result.items()}, metadata=json.dumps(info))
    print(a.output, a.output.stat().st_size)


if __name__ == '__main__':
    main()
