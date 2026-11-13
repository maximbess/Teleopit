"""Package immutable originals plus separate portable scenes and frozen sources."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from package_g1_o6_support import (
    ASSETS,
    Archive,
    add_sources,
    collect_files,
    compare_models,
    implementation_revisions,
    portable_scene,
    sha_file,
    verify,
)

ROOT = Path("outputs/g1_o6/ladder_material")
DOCS = Path("docs/g1_o6/ladder_material")


def package(output, staging):
    if output.exists() or staging.exists():
        raise FileExistsError("choose unused archive and staging paths")
    for name in ASSETS:
        shutil.copytree(name, staging / name)
    files = collect_files(
        ROOT,
        DOCS,
        (
            "reference",
            "runs",
            "reconstructions",
            "acquisition_windows",
            "geometry_views",
            "videos",
            "report",
            "source",
            "validation",
        ),
    )
    archive = Archive(output)
    scenes = []
    try:
        for path in files:
            archive.file(path)
            if path.name == "scene.xml":
                portable = path.with_name("scene_portable.xml")
                payload = portable_scene(path)
                target = staging / portable
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                fields = compare_models(path, target)
                archive.data(portable.as_posix(), payload)
                scenes.append(
                    {
                        "original": path.as_posix(),
                        "portable": portable.as_posix(),
                        "identical_public_numeric_fields": fields,
                    }
                )
        revisions = implementation_revisions(files)
        add_sources(archive, revisions)
        archive.data(
            "evidence_manifest.json",
            {
                "schema": "g1_o6_ladder_archive_v1",
                "implementation_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "physical_assets_included": True,
                "source_revisions_included": revisions,
                "original_bytes_preserved": True,
                "scene_policy": "Use reconstructions/<run_id>/scene.mjb or its scene_portable.xml. Original scene.xml is immutable evidence and some exports contain only the auxiliary ladder; consult proof.json. Portable scenes are additional files, never replacements.",
                "portable_scenes": scenes,
                "files_sha256": dict(archive.hashes),
            },
        )
    finally:
        archive.zip.close()
    manifest = verify(output)
    receipt = {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha_file(output),
        "implementation_revision": manifest["implementation_revision"],
        "verified_files": len(manifest["files_sha256"]),
        "portable_scenes": len(scenes),
        "original_bytes_preserved": True,
        "physical_assets_included": True,
        "publication_status": "local archive; not uploaded",
    }
    output.with_suffix(".zip.sha256").write_text(
        receipt["sha256"] + "  " + output.name + "\n"
    )
    (DOCS / "artifact.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    package(args.output, args.staging)
