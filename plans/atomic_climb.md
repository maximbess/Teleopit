# Atomic climb

The policy learns one skill: from a stable stance, move both hands and both feet up one rung and hold. A script outside the policy runs that skill again from the pose where the hold ended. The script is how a rollout reaches the top and how later attempts see higher, less scripted starting poses. It is not a second skill and not a curriculum.

## Task

Rung indices are 0-based: index 0 is the lowest rung. At the start of an attempt, record the rung under the feet and the rung three above it under the hands, the same gap as the climbing keyframe (feet on 1, hands on 4). The attempt succeeds only when both feet and both hands are exactly one rung higher and the hold stays true for a fixed number of frames:

- both hands attached
- both feet in contact with their rungs
- torso speed, joint-speed RMS, torso and pelvis spin, waist speed, and support offset inside the current stabilize limits

A broken gate clears the hold counter. The first attempt in a rollout starts from the climbing keyframe. The environment does not choose which limb moves, or when.

The action is the 29 joint targets plus a grip request for each hand. A request attaches only when that hand is close to a rung. Dropping the request releases the weld. The soft-release ramp and the one-limb-at-a-time rule go away.

## Repeat

When the hold completes, give the reward and keep the physics state. The rungs just reached become the baseline, and the same policy is asked for the next rung. After `N` successes the rollout ends. A fall or a deadline ends it with no reward.

`N` only continues the rollout. It does not unlock phases or change the policy.

## Reward

One term: the completed hold. No failure penalty, distance shaping, height term, foot-placement term, pose cost, or per-limb bonus.

## Code

Remove the five-phase machine from `LadderClimbCommand`: the phase enum, hand and foot advance, `_start_phase`, and the curriculum bank. Keep the baseline rungs, the one-rung-higher test, the hold counter, and grip requests.

The ladder mesh, rung contacts, and weld anchors stay.

The command is whether each hand is attached. There is no stage index and no target point on a rung. `LadderTemporalCNNModel` no longer copies a categorical phase prefix. `LadderOnPolicyRunner` stops storing curriculum state. `--ladder_phase` goes away; the outer limit is `N`.

Actor and critic see the same six rung tokens, in order: one below the feet, the foot rung, the next foot rung, the rung between, the hand rung, and one above the hands. Those 0-based indices are fixed until the hold completes, then they shift up by one. With four successes on nine rungs the windows are 0–5, 1–6, 2–7, and 3–8.

Delete the reward classes and the per-phase copies in `env.py`. Replace the ladder tests that assert phases, the unlock window, and those rewards. New tests cover one attempt, a hold that resets when a gate breaks, success only after the hold, continuation from the held pose, and a checkpoint with no curriculum state.

Old ladder checkpoints do not match this observation or reward. Train from scratch.
