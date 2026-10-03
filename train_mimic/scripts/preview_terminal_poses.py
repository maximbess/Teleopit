"""View or render measured terminal poses and their short holding histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def restore_pose(model, data, state, frame=-1):
    """Restore actual generalized coordinates and per-world weld parameters."""
    import mujoco
    mujoco.mj_resetData(model,data)
    for k in ('eq_data','eq_solref','eq_solimp'):
        getattr(model,k)[:] = state['model_'+k]
    model.opt.gravity[:] = state['gravity'][frame]
    data.qpos[:] = state['qpos'][frame]
    data.qvel[:] = state['qvel'][frame]
    data.ctrl[:] = state['ctrl'][frame]
    data.eq_active[:] = state['eq_active']
    mujoco.mj_forward(model,data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True)
    p.add_argument('--index',type=int,help='Record id; defaults to the selected nominal pose')
    p.add_argument('--render_dir',help='Render all representatives to PNG and exit')
    p.add_argument('--video',help='Render the selected holding history to MP4 and exit')
    p.add_argument('--history',action='store_true',help='Loop the recorded hold in the native viewer')
    args=p.parse_args(argv)
    import mujoco
    bank=Path(args.bank)
    meta=json.loads((bank/'manifest.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(bank/'scene.mjb'))
    data=mujoco.MjData(model)
    index=meta['nominal_id'] if args.index is None else args.index
    cam=mujoco.MjvCamera()
    cam.distance=3.2; cam.azimuth=30.; cam.elevation=-8.
    def load(i):
        with np.load(bank/meta['records'][i]['file']) as f:
            return {k:f[k] for k in f.files}
    state=load(index)
    cam.lookat[:]=state['track_pos'][-1,4]+np.array([0.,0.,.25])
    if args.render_dir or args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(model,height=720,width=960)
        try:
            if args.render_dir:
                out=Path(args.render_dir);out.mkdir(parents=True,exist_ok=True)
                for i in meta['representatives']:
                    s=load(i);restore_pose(model,data,s)
                    renderer.update_scene(data,camera=cam)
                    imageio.imwrite(out/f'pose_{i:04d}.png',renderer.render())
            if args.video:
                Path(args.video).parent.mkdir(parents=True,exist_ok=True)
                with imageio.get_writer(args.video,fps=round(1/meta['step_dt'])) as writer:
                    for t in range(len(state['qpos'])):
                        restore_pose(model,data,state,t)
                        renderer.update_scene(data,camera=cam)
                        writer.append_data(renderer.render())
        finally:
            renderer.close()
        return
    import mujoco.viewer
    restore_pose(model,data,state)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=cam.lookat
        viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
        frame=0
        while viewer.is_running():
            if args.history:
                restore_pose(model,data,state,frame % len(state['qpos']))
                frame+=1
            viewer.sync()
            time.sleep(meta['step_dt'])


if __name__=='__main__':
    main()
