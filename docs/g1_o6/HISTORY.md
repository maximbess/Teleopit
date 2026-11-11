# G1 / O6 experiment history

Use the [current quickstart](README.md) for the supported default. Historical commands and configurations require the specified Git revision in a separate checkout. Old results describe their original controller, not input to the current isolated-hand runner.

| Experiment | Exact reproduction reference | Evidence and outcome |
|---|---|---|
| Original overhand | Measured `4fe09d98cacd802b23dc634f67ef4c56a47adfe2`; report `f5c7067fb9e640a17501764df4eaf818679a146d` | [Exact commands/report](https://github.com/maximbess/Teleopit/blob/f5c7067fb9e640a17501764df4eaf818679a146d/docs/g1_o6/README.md), [preserved report](OVERHAND_BASELINE.md), [asset hashes](audit.json), [results](results.json). HOLD failed after 0.5 s. |
| Intermediate thumb-below and diagnostics | Implementation `9160225267d66ae63a4a986277e1974d9ae2749d`; published `919bbc6b26b8a99e56232b0dcfb33c644ded73ec` | [Exact commands](https://github.com/maximbess/Teleopit/blob/919bbc6b26b8a99e56232b0dcfb33c644ded73ec/docs/g1_o6/diagnostics/README.md), [report](diagnostics/README.md), [archive manifest](diagnostics/artifact.json). Unintended wrist-housing support and failed hold. |
| Balanced qualification before consolidation | Implementation `fa0a1d8f333779188b26e10a99ed89498b91be6e`; report `b34ba73683986ef4252d953def71569b77a22eac` | [Exact commands](https://github.com/maximbess/Teleopit/blob/b34ba73683986ef4252d953def71569b77a22eac/docs/g1_o6/posture/README.md), [report](posture/README.md), [archive manifest](posture/artifact.json). Qualified 10 s and 30 s four-contact holds. |

The collision source is pinned to `amazon-far/holosoma` revision `bccd4d7451640a2800ddc77e469d911a84f91994`; authoritative O6 models to `linker-bot/linkerhand-urdf` revision `49f26021b0a91ff5c33150df22970c07bb548672`. The [asset audit](audit.json) records exact installed canonical G1 hashes; no upstream G1 revision was invented. Use `scripts/setup/install_climb_assets.py` for missing canonical assets. Archived runs also contain resolved assets, model audits and compiled scenes.

These archives are local provenance, **not public download links**. Both were accessible on 2026-11-11 and their complete SHA256 hashes were verified again during consolidation. Existing output directories, reports, JSON results, work logs, figures and manifests were preserved.

| Local archive | Verified SHA256 |
|---|---|
| `C:\Users\maxim\Downloads\G1_O6_Diagnostics_2026-11-11_9160225.zip` | `d340625f1b09351b10e60894c8246dded9415b661fad804b822a21202b37b4bd` |
| `C:\Users\maxim\Downloads\G1_O6_Balanced_Grasp_2026-11-11_fa0a1d8.zip` | `5249e00b940fc7d92b06b011f7dc6ed741448ce27ce559d9da2f2fa0a8f25eb7` |

The archives do not replace canonical mesh installation. The original main README is preserved as [OVERHAND_BASELINE.md](OVERHAND_BASELINE.md); diagnostic and balanced reports only received historical notices. Current validation and its implementation revisions are recorded [separately](consolidation/README.md).
