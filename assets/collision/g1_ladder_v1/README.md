# Fixed G1 ladder collision geometry

This directory contains the 180 prebuilt convex collision meshes used by the
measured two-hand ladder pose bank. The manifest records the upstream source
revision, canonical source checksums, build settings and SHA256 of every mesh.
The complete training scene contains 229 geoms.

Source: amazon-far/holosoma, revision
`bccd4d7451640a2800ddc77e469d911a84f91994`, G1 OmniRetarget model;
the torso uses the canonical G1 rev. 1.0 surface recorded in the manifest.
These files were generated with `scripts/setup/download_g1_collision.py`.

Training and preview use this tracked snapshot by default. Locally generated
files in `assets/robots/unitree_g1/omniretarget_collision` remain available for
development via the explicit `asset_dir` argument. Do not replace the snapshot
without rebuilding and validating the associated references and pose bank.
