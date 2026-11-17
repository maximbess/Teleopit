# Portable native G6 acquisition evidence

Archive: `outputs/g1_o6/g6_grasp_acquisition_evidence.zip`.
Its SHA-256 sidecar and `artifact.json` are generated after a complete archive
hash verification. The archive is local; it is not uploaded. Git contains
implementation, tests and documentation; the large output directory remains
ignored.

The archive includes all 15 completed physical runs, the full campaign ledger
with two startup errors and their identical technical recoveries, frozen
candidate/physics/source definitions, original scenes and run manifests,
raw contact/applied-force/control traces, native physical assets, verification
reports, plots, three recorded-state replays, tests/fault logs, measured
coverage, exact runtime source bytes and a `dex-hand-exp` Git bundle. Scratch
pytest fixtures and mutation bytecode caches are excluded. The initial
axis-order diagnostic is explicitly marked discarded and retained only as
an audit trail; the corrected geometric scan is reproducible from the
packaged fresh control.

Original `scene.xml` and `scene.mjb` bytes are preserved. Additional
`scene_portable.xml` files rewrite only asset paths and are checked against
the originals across all public numeric model fields. They reference the
included assets after relocation. Use the MJB or portable XML for inspection;
the original XML retains the original machine's paths.

The physical runtime is revision
`8134ee06f595cd064dfcecc08cbdbd71cef3ab77`, with exact working-tree bytes in
`outputs/g1_o6/g6_grasp_acquisition/source/runtime_source.zip` and their
hashes in the neighboring JSON. Git checkout line endings can differ, so
exact-byte reproduction needs that snapshot. The archive manifest also
identifies the later delivery revision containing report/recovery fixes;
source trees are under `provenance/<revision>/`. The source bundle retains
the named branch history. Restore code beside the included native assets
and use the recorded Python/MuJoCo/runtime versions from the frozen source
provenance. Use a fresh campaign output directory for new physical runs.

From the restored repository root, in the recorded runtime environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/verify_g1_o6_g6.py --root outputs/g1_o6/g6_grasp_acquisition
python scripts/g1_o6_g6_acquisition_report.py --root outputs/g1_o6/g6_grasp_acquisition
```

Use the delivery report tools when verifying the preserved technical-error
history. The physical campaign was closed with its source identity unchanged
before those report-only fixes. Audit PASS confirms source, scene, contact,
force, command and history integrity; [OUTCOMES.md](OUTCOMES.md) retains the
failed stable-pose result and the exact promotion limits.
