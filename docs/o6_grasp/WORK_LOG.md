# O6 standalone experiment work log

- Read the requested specification and ub-mode; confirmed clean `dex-hand-exp`, no applicable AGENTS.md, and MuJoCo 3.8.1 in conda `teleopit`.
- Inspected existing robot, hardware driver, configuration and test conventions. Kept this experiment in `teleopit.sim` to avoid changing humanoid runtime behavior.
- Inspected manufacturer URDF and linker-sim sources. Pinned URDF revision `49f26021b0a91ff5c33150df22970c07bb548672`; it provides five explicit mimic ratios, full inertias, per-link collision meshes and six independent joints. The URDF's uniform 100 Nm effort fields are not calibrated hardware specifications.
- Added behavioral tests before implementation for action/observation/reset contracts, frames, limits, coupling, unsupported support removal and release.
- The requested crapper, dryer and mutator tools are not exposed in this session or present in the repository. Use available lint, coverage, complexity inspection and targeted fault checks; report this limitation rather than claim those tools ran.
- Confirmed the initial test collection failed because the new module did not exist. Installed pytest, pytest-cov, Ruff and Radon in the requested conda environment for validation.
- Bundled only the pinned, Apache-licensed right-hand URDF and per-link collision meshes (about 280 KB). Kept visual geoms separate and documented the deliberately low-resolution rendering.
- Implemented scene generation, physical fixture/load application, deterministic control, observations/history/sensing imperfections, privileged diagnostics, rewards, viewer and evaluation CLI.
- Initial free-motion tests exposed large thumb coupling error and chatter. Increasing equality impedance alone did not solve it. The shared premise was that constraint softness caused the error; active damping acted against very small moving inertias. Measured alternatives with less damping and explicit provisional reflected inertia; selected 0.0002 kg m² armature, 0.02 Nm s/rad actuator damping and 0.002 Nm s/rad joint damping. Kept the actual URDF inertia tensors unchanged.
- Measured a six-pose alignment comparison. Selected a nonpenetrating rung center at (0.032, 0, 0.608) m and less aggressive closed targets to avoid persistent saturation, without relaxing grasp verification thresholds.
- Checked the manufacturer's SDK joint order and made all six-vector interfaces follow hardware order (thumb pitch, thumb yaw, index, middle, ring, pinky), independently of URDF tree traversal.
- Verified reset, action bounds, rate/effort bounds, coupling, unsupported free fall, observation dimensions/frames, sensing noise/delay/dropout, wrench offsets/nonaccumulation and release authorization.
