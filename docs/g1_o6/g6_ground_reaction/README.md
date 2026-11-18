# Ground-reaction refinement on G6

Each fresh MuJoCo trial uses the acquired G1/O6 grasp, hands on tread 7, feet on tread 2, and the 1.6-gain ground-reaction controller. A fresh five-second parent probe must reproduce the previous high-gain full trace. The grip, friction, actuator gains, commanded wrench limits, joint and rate bounds, assistance removal and original scoring are shared with the parent.

During unsupported HOLD the controller previews the next 20 ms on a private copy of the native model state. It evaluates the signed robot–environment contact force plus gravity and the minimum intended-tread normal load of the four left fingers at each native solver step and at the ending control sample. It first tries the parent's leg and waist update. Its preview targets are a residual below 16 or 21 N, depending on the configuration, and at least 0.35 N on each finger. If the nominal update misses either target, it tries a half update, holding the previous target, then bounded vertical trims of ±0.5, ±2 and ±8 N. If every alternative misses a target, the controller selects the best measured prediction, preferring commands inside the physical limits. Only the original joint position servos actuate the live model. Parent structural and slip backoff takes precedence.

The first pair distributes the trim as a common vertical force. If both configurations fail qualification, the campaign records the support forces at the first and worst balance crossings before trying the second pair. That pair can direct the vertical trim at either foot, including the associated moment, or distribute it across both feet. The resultant virtual wrench stays within 30 N and 5 Nm. The preview evaluates at most 9 or 21 commands per update, respectively.

**Qualification** requires the actual force-balance residual to be strictly below 25 N at every unsupported solver step and every control endpoint. All four left fingers must each carry at least 0.05 N on their intended tread at every unsupported solver step. All original physical, anatomical and convergence gates also apply. Each candidate progresses through 5, 30, 60 and 120 seconds, a fresh 120-second nominal repeat and a 120-second half-step confirmation. The search ends after a confirmed stable pose or four refinements. The parent has one probe.

Run from the repository root with the native G1 assets installed. The recorded campaign used simulator revision `fd2b332`. For a separate fresh campaign, give `--output` a new directory.

```powershell
conda run --no-capture-output -n teleopit python -m teleopit.sim.g1_o6_support.g6_ground_reaction_experiment --output outputs/g1_o6/g6_ground_reaction run
```

The following command reconstructs the measured peaks, continuous finger loading, servo bounds and selected previews. The `--run-id` option checks one completed trial. Use `python -m scripts.verify_g1_o6_g6 --root outputs/g1_o6/g6_ground_reaction` for the archived source, scene, material and external-force checks.

```powershell
conda run --no-capture-output -n teleopit python -m scripts.verify_g1_o6_g6_ground_reaction --root outputs/g1_o6/g6_ground_reaction
```

The full-length attempt reached 107.54 s. The per-step force gate passed, and the original unintended-support gate ended the hold after the right knee loaded rung 3. [Measured outcomes](OUTCOMES.md) · [Per-trial results](results.json) · [Work log](WORK_LOG.md)
