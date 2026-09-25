# Plan

The current branch contains maximbess's climbing work (`ab84379`, `0896b0f`). His two newest commits (`d8cfb2f`, `9772e19`) are only on `origin/ladder_dev` and are not in this worktree. This task cleans the artifacts those climbing commits left here, shrinks the unused ladder reward surface, and fixes two defects that still apply to this tree. It does not merge the later reward redesign.

## Remove unreferenced artifacts

Delete and ignore:

- Jupyter notebooks and `.ipynb_checkpoints/`
- `assets/ladder_g1/` (copied G1 meshes and experimental XML, about 63 MB) and `assets/ladders/`
- `show_scene.py` and `generate_double_ladder.py`, which only exist to view or emit that unused scene
- `PROJECT_CONTEXT.md` (scratch notes; the later branch already deleted it)
- crawl-filter scripts, `seed_crawl.yaml`, and `seed_crawl_metadata.csv` (paths do not match, and nothing else references them)

Keep `teleopit/robots/automatic_grip.py`, `teleopit/configs/robot/g1_ladder.yaml`, and `test_automatic_grip.py`. `MuJoCoRobot` still loads that grip config.

## Shrink the ladder reward surface

The training config only uses upward progress, foot placement, phase progress, phase completion, stabilization orientation, failure, and success. Remove the older reward helpers that nothing in the environment calls, and drop the tests that exist only to exercise them. Leave the FSM flags those live rewards still read.

## Fix two current defects

1. Box rung and trunk-blocker contacts use `margin=0.002`. MuJoCo Warp multi-CCD rejects a nonzero margin on box/mesh contacts, so ladder training cannot start on the pinned Warp stack. Set those margins to 0 in both the generated spec and the collision editor. Leave the capsule-rail margin at 2 mm.
2. The ladder actor and critic normalize the leading five phase one-hot values. A phase switch then stops being a clean categorical input. Use a ladder-only normalizer that copies those five values through, including history. Leave the tracking policy unchanged.

## Do not port

Do not bring over `ladder_dev`'s posture and unwanted-contact rewards, the critic `remaining_time` feature, the timeout-bootstrap change, or the shared self-collision pattern edit. Those change the documented reward and observation contract, and they sit on top of intermediate commits this branch does not contain.

## Checks

Compile the packages. Run the grip unittest. Run the ladder and model tests that this environment can import. Do not start cluster training.
