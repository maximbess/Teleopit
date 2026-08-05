import time

import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("./assets/ladder_g1/scene_ladder.xml")
data = mujoco.MjData(model)

root_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "floating_base_joint",
)
qadr = model.jnt_qposadr[root_id]

# Перед левой стороной лестницы, лицом в направлении +X.
data.qpos[qadr:qadr + 3] = [-1.60, 0.0, 0.793]
data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    with viewer.lock():
        viewer.opt.sitegroup[3] = 1

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)