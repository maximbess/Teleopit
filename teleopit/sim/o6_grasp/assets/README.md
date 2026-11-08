# Pinned manufacturer assets

Copyright 2025 LinkerHand,Ltd. Apache-2.0; see LICENSE.

Unmodified URDF and collision OBJ files from
https://github.com/linker-bot/linkerhand-urdf/tree/49f26021b0a91ff5c33150df22970c07bb548672/O6/right
are bundled for offline, standalone execution. `manifest.json` records SHA-256 digests.
The upstream GLB visual files are not included. Rendering uses the low-resolution
collision meshes in separate, noncolliding visual geoms. MuJoCo uses each link's
convex hull for collision; no hull spans multiple articulated links.

The runtime translator in `scene.py` retains the original joint frames, axes,
limits, masses and full inertia tensors. Five explicit joint equalities implement
the URDF's linear mimic approximations, not measured nonlinear linkage behavior.
The source effort fields (100 for every revolute joint) are retained in the URDF
for provenance but are deliberately not used as hardware torque ratings.

Command ordering follows the manufacturer's Python SDK correspondence table:
https://github.com/linker-bot/linkerhand-python-sdk#position-and-finger-joint-correspondence-table
Thumb flexion, thumb opposition, index, middle, ring, pinky. This differs from
the source URDF tree traversal order, which places thumb yaw before pitch.
