# Manufacturer model audit

Pinned source: `linker-bot/linkerhand-urdf@49f26021b0a91ff5c33150df22970c07bb548672`, `O6/right/linkerhand_o6_right.urdf`. Values below are copied from that source, not measured here.

## Joint frames

Axes are in the local joint frame; origin xyz/rpy is relative to the parent link. Units: m, rad. Source effort/velocity are uncalibrated metadata, not imposed torque capabilities.

| Joint | Parent | Origin xyz | Origin rpy | Axis | Limits | Coupling |
|---|---|---|---|---|---|---|
| rh_thumb_cmc_yaw | rh_hand_base_link | 0.010457 0.022975 0.030794 | 0 0 0 | 0 0 -1 | 0 .. 1.36 | active |
| rh_thumb_cmc_pitch | rh_thumb_metacarpals_base2 | 0.0061649 0.010678 -0.004891 | -1.1529 0 -0.5236 | 1 0 0 | 0 .. 0.58 | active |
| rh_thumb_ip | rh_thumb_metacarpals | 0 -0.0037776 0.045368 | 0 0 0 | 1 0 0 | 0 .. 1.08 | 1.86 * rh_thumb_cmc_pitch + 0 |
| rh_index_mcp_pitch | rh_hand_base_link | 0.0024758 0.02419 0.098779 | -0.05236 0 0 | 0 1 0 | 0 .. 1.60 | active |
| rh_index_dip | rh_index_proximal | -0.0052516 0 0.036625 | 0 0 0 | 0 1 0 | 0 .. 1.43 | 0.89 * rh_index_mcp_pitch + 0 |
| rh_middle_mcp_pitch | rh_hand_base_link | 0.00052576 0.00634 0.1027 | 0 0 0 | 0 1 0 | 0 .. 1.60 | active |
| rh_middle_dip | rh_middle_proximal | -0.0052516 0 0.036625 | 0 0 0 | 0 1 0 | 0 .. 1.43 | 0.89 * rh_middle_mcp_pitch + 0 |
| rh_ring_mcp_pitch | rh_hand_base_link | 0.0010258 -0.011135 0.098767 | 0.05236 0 0 | 0 1 0 | 0 .. 1.60 | active |
| rh_ring_dip | rh_ring_proximal | -0.0052516 0 0.036625 | 0 0 0 | 0 1 0 | 0 .. 1.43 | 0.89 * rh_ring_mcp_pitch + 0 |
| rh_pinky_mcp_pitch | rh_hand_base_link | 0.0024758 -0.028372 0.092741 | 0.087266 0 0 | 0 1 0 | 0 .. 1.60 | active |
| rh_pinky_dip | rh_pinky_proximal | -0.0052516 0 0.036625 | 0 0 0 | 0 1 0 | 0 .. 1.43 | 0.89 * rh_pinky_mcp_pitch + 0 |

All 11 source joints specify effort 100 and velocity 1. The six-command API order is pitch, yaw, index, middle, ring, pinky; it intentionally differs from tree order.

## Link inertias

URDF inertia axes have zero rpy on every link. MuJoCo diagonalizes the full symmetric tensor into principal axes internally. The source tensors and centers below are retained. No inertia is inferred from collision meshes.

| Link | Mass (kg) | Center (m) | Ixx Iyy Izz (kg m²) | Ixy Ixz Iyz (kg m²) |
|---|---:|---|---|---|
| rh_hand_base_link | 0.2374 | -0.00157788568070457 0.00034876725935859 0.0538452219801211 | 5.63750E-05 3.59830E-05 2.46490E-05 | -3.31172E-07 -4.21137E-07 8.37956E-07 |
| rh_thumb_metacarpals_base2 | 0.00258 | 0.00162795738393474 0.0027667197833771 -0.00184227725602705 | 9.7974575996e-08 8.9844836854e-08 5.3717259400e-08 | -7.3138244367e-09 4.5379328624e-09 6.9226596212e-09 |
| rh_thumb_metacarpals | 0.04724 | -1.84622133177631E-05 -0.000164684333964604 0.022696504519568 | 7.2358317019e-06 4.5390480908e-06 3.7252650599e-06 | 2.5325516490e-09 -1.7874737118e-09 -4.8023043280e-07 |
| rh_thumb_distal | 0.0142 | -4.4744E-07 0.0041369 0.02301 | 2.2781763396e-06 2.4286781082e-06 5.1587318224e-07 | 4.6616481069e-12 3.0102214070e-11 -1.3376426045e-07 |
| rh_index_proximal | 0.01082 | -0.00208656556336452 -2.53838713251686E-06 0.0172734941980264 | 3.53128E-07 3.62322E-07 1.13645E-07 | 2.21942E-11 -5.72262E-08 -2.49015E-10 |
| rh_index_distal | 0.00632 | 0.00658986616416549 -5.13855642059191E-06 0.0194393600039178 | 1.68896E-07 1.91417E-07 6.51218E-08 | -7.23157E-12 -5.99162E-08 1.10820E-10 |
| rh_middle_proximal | 0.01082 | -0.00208660654138235 -2.56918180119303E-06 0.0172735011723351 | 3.53128E-07 3.62323E-07 1.13646E-07 | 2.25449E-11 -5.72260E-08 -2.48906E-10 |
| rh_middle_distal | 0.00632 | 0.00658986516319194 -5.10862924271716E-06 0.0194393315699782 | 1.68897E-07 1.91417E-07 6.51221E-08 | -7.18465E-12 -5.99163E-08 1.10269E-10 |
| rh_ring_proximal | 0.01082 | -0.00208667931345812 -2.63000419462034E-06 0.0172734926448377 | 3.53129E-07 3.62324E-07 1.13647E-07 | 2.32215E-11 -5.72256E-08 -2.48472E-10 |
| rh_ring_distal | 0.00632 | 0.00658986648895491 -5.05883528165163E-06 0.019439397215532 | 1.68895E-07 1.91416E-07 6.51217E-08 | -7.11364E-12 -5.99163E-08 1.09335E-10 |
| rh_pinky_proximal | 0.01082 | -0.00208673599799501 -2.68062251165824E-06 0.0172734848131236 | 3.53130E-07 3.62324E-07 1.13648E-07 | 2.37682E-11 -5.72252E-08 -2.48138E-10 |
| rh_pinky_distal | 0.00632 | 0.00658987588903027 -5.03087534178825E-06 0.0194394491065091 | 1.68894E-07 1.91415E-07 6.51216E-08 | -7.11370E-12 -5.99164E-08 1.08691E-10 |

Total source mass: **0.36998 kg**. Provisional joint armature is additional reflected motor inertia, not a change to these link inertias.

## Collision and frames

Each of the 12 links has one upstream OBJ collision mesh. MuJoCo uses each separately as a convex collision piece, preserving articulation and the palm/finger cavity. The source links and mesh origins coincide. Visual copies have contype/conaffinity zero; collision copies are in viewer group 3. Only parent-child link pairs are excluded. No hand/rung contact exclusions or attachments exist.

`wrist_mount` is at the base-link origin. `palm_frame` is translated +0.06 m in its Z direction. Both use base-link axes. Distal tip sites are approximate local [0.006, 0, 0.036] m points for diagnostics. `rung_frame` is the fixed rung center with longitudinal local X.
