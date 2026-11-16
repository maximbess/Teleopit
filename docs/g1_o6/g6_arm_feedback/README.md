# Lowered native G6 arm-feedback experiment

The starting supports are **hands 7 / feet 2**, one below the original 8/3
control. Root X remains −40 mm, with the original thumb-below-balanced grasp,
native material and unchanged physical limits. This is a new experiment;
previous failed trials and the project default remain intact.

The lowered control is reproduced before editing the controller. Its entire
trace matches the previous `e_00_probe` SHA-256
`e5e03184430b635b94e5cf76328f5c36489a6b66b3b79eb462fa1f4f7f053a67`.
It fails both hand gates after 0.50 s HOLD. Low drift and knee clearance are not
substitutes for anatomical opposition.

| Candidate | Activation and target | Correction rate |
|---|---|---:|
| control | Original controller | None |
| retain_slow | Latch measured wrists at assistance-removal entry | 0.005 rad/s |
| retain_fast | Same measured reference | 0.020 rad/s |
| seek_slow | At 2 s of acquisition, shift toward measured intended-tread thumb gap | 0.005 rad/s |
| seek_fast | Same bounded contact-seeking rule | 0.020 rad/s |

All five configurations are frozen together before feedback trials. Seek
variants require an immutable premise audit identifying two failed retained-
reference trials, the observed actors and the assignment being removed. They
do not simply retry a failed physical run. Complete the audit before testing
the alternate acquisition premise.

The wrapper reuses the existing damped six-dimensional arm Jacobian and
position servos. It changes only arm targets. Joint offsets stay within
±0.03 rad. Scratch forward kinematics limits the difference between original
and corrected arm-command poses to **5 mm translation and 0.03 rad rotation**.
Projection interpolates the bounded command increment so it also preserves the
rate bound. It does not write live qpos, applied forces, physical gains or
collision settings. Finger commands, feet and global RESET references remain
under their original controller and evaluator.

The seeking target is computed once from an actual native thumb-to-tread
distance witness. It permits at most 0.5 mm beyond that witness to establish
contact, with total target shift capped at 5 mm. An already-loaded thumb is
not retargeted to repair a missing finger. Unreliable/zero distance witnesses
do not invent a target. Global structural violations or hand slip back off
the correction; saturated actuators freeze their increment. Every original
physical, anatomical, native-friction and zero-assistance gate still applies.

Scoring uses 1 ms physics and 20 ms control. A qualifying candidate must pass
fresh **5 s probe → 30 → 60 → 120 s** trials, then a fresh nominal 120 s repeat
and a 0.5 ms confirmation. Final 120 s trials include the existing final-30-s
convergence checks. There are at most four feedback configurations; no result
is accepted on duration or appearance alone. This task qualifies a stationary
pose; capacity/loading/hand transfers remain separate physical milestones.

Use the repository root and activated `teleopit` Conda environment. The ledger
rejects changed source/assets, changed configurations, failed promotion,
unrecorded premise changes and silent reruns. All commands take the same root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment prepare
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment import-control outputs/g1_o6/g6_arm_feedback/reference
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment pair retain
# Inspect the measured failures and write premise_audit.json before the next command.
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment audit outputs/g1_o6/g6_arm_feedback/premise_audit.json
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment pair seek
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment promote
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_feedback_experiment status
```

Reproduction requires the source revision and exact byte snapshot recorded in
the ledger, physical assets and a fresh output directory. Completed evidence
is immutable. `retry <candidate> <stage> --reason "..."` is reserved for
technical ERROR/interruption with identical source identity, never a physical
FAIL. See [WORK_LOG.md](WORK_LOG.md) for validation and campaign decisions.
