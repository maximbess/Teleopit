"""Download pinned OmniRetarget surfaces and build convex G1 collision assets.

Called by download_assets.py --only g1_collision. Build dependencies:
    pip install -e ".[collision-build]"
No training stack or GPU is needed. Generated meshes are external ignored assets.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Bound internal OpenMP threads when several links build concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from teleopit.runtime.g1_collision_assets import (
    ASSET_DIR, BODY_MAP, MODEL_PATH, REPOSITORY, REVISION, SCHEMA_VERSION,
    CANONICAL_MESH_OVERRIDES,
)


def _download(path: str, output: Path, blob_sha: str) -> None:
    def git_hash(data: bytes) -> str:
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if output.is_file() and git_hash(output.read_bytes()) == blob_sha:
        return
    request = urllib.request.Request(
        f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{path}",
        headers={"User-Agent": "Teleopit-collision-assets"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if git_hash(data) != blob_sha:
        raise RuntimeError(f"Source checksum mismatch: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def _build_parts(task):
    import coacd
    import trimesh
    coacd.set_log_level("warn")
    source_body, target_body, index, mesh, settings = task
    # Motor housings and ankle hinges use a solid convex envelope. Decomposing
    # their CAD screw bores/internal rotor cavities adds inaccessible contacts
    # and very large build costs without improving external support geometry.
    solid_housing = any(source_body.endswith(f"_{part}_link") for part in
                        ("hip_pitch", "ankle_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw"))
    settings = dict(settings)
    if solid_housing:
        mesh = mesh.convex_hull
        settings.update(preprocess_mode="off", max_convex_hull=1)
    torso_sections = 4 if source_body == "torso_link" else 0
    # The donor meshes often contain open internal CAD surfaces. CoACD
    # preprocesses them before approximate convex decomposition.
    cache = ASSET_DIR / "parts" / f"{source_body}_{index}.json"
    fingerprint = hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes() + (json.dumps((settings, torso_sections), sort_keys=True) if torso_sections or solid_housing else json.dumps(settings, sort_keys=True)).encode()).hexdigest()
    try:
        cached = json.loads(cache.read_text()) if cache.is_file() else None
    except json.JSONDecodeError:
        cached = None
    if cached and cached.get("fingerprint") == fingerprint and cached["parts"] and all(
        (ASSET_DIR / p["file"]).is_file()
        and hashlib.sha256((ASSET_DIR / p["file"]).read_bytes()).hexdigest() == p["sha256"]
        for p in cached["parts"]
    ):
        print(f"Cached {source_body}", flush=True)
        return cached["parts"]
    print(f"Building {source_body}: {len(mesh.faces)} triangles", flush=True)
    if torso_sections:
        import numpy as np
        levels = np.linspace(mesh.bounds[0, 2], mesh.bounds[1, 2], torso_sections + 1)
        hulls = []
        # Slice triangles at shared planes before taking envelopes, so adjacent
        # torso sections meet without gaps. Internal CAD cavities are ignored.
        for low, high in zip(levels[:-1], levels[1:]):
            section = mesh.slice_plane([0, 0, low], [0, 0, 1])
            section = section.slice_plane([0, 0, high], [0, 0, -1]).convex_hull
            section_settings = dict(settings, preprocess_mode="off", max_convex_hull=1)
            hulls.extend(coacd.run_coacd(coacd.Mesh(section.vertices, section.faces), **section_settings))
    else:
        hulls = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
    built = []
    for hull_index, (vertices, faces) in enumerate(hulls):
        name = f"{source_body}_{index}_{hull_index}"
        path = ASSET_DIR / "parts" / f"{name}.obj"
        path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices, faces, process=False).export(path)
        built.append(dict(body=target_body, source_body=source_body,
                          file=path.relative_to(ASSET_DIR).as_posix(),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(fingerprint=fingerprint, parts=built), indent=2))
    temporary.replace(cache)
    print(f"Ready {source_body}: {len(hulls)} convex parts", flush=True)
    return built


def build_assets() -> None:
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    from importlib.metadata import version
    if version("coacd") != "1.0.14":
        raise RuntimeError("Collision assets require coacd==1.0.14 for reproducible builds")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1",
        headers={"User-Agent": "Teleopit-collision-assets"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        tree = json.load(response)
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    sources = {}
    canonical_sources = {}
    def fetch(relative: str) -> Path:
        path = f"{MODEL_PATH}/{relative}"
        output = ASSET_DIR / "source" / relative
        _download(path, output, blobs[path])
        sources[relative] = blobs[path]
        return output
    urdf = ET.parse(fetch("g1_29dof.urdf")).getroot()
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_LICENSES"):
        if name in blobs:
            _download(name, ASSET_DIR / "source" / name, blobs[name])
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    parents = {joint.find("child").attrib["link"]: joint for joint in urdf.findall("joint")}
    def origin(node):
        transform = np.eye(4)
        if node is not None:
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            transform[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        return transform
    parts = []
    settings = dict(threshold=0.035, max_convex_hull=12, preprocess_mode="auto",
                    preprocess_resolution=30, resolution=500, mcts_nodes=4,
                    mcts_iterations=5, mcts_max_depth=2, merge=True,
                    decimate=True, max_ch_vertex=64, seed=0)
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=4)
    futures = []
    for source_body, target_body in sorted(BODY_MAP.items()):
        transform = np.eye(4)
        cursor = source_body
        while cursor != target_body:
            joint = parents[cursor]
            if joint.get("type") != "fixed":
                raise RuntimeError(f"Cannot fold moving donor link {cursor} into {target_body}")
            transform = origin(joint.find("origin")) @ transform
            cursor = joint.find("parent").attrib["link"]
        collisions = links[source_body].findall("collision")
        for index, collision in enumerate(collisions):
            mesh_node = collision.find("geometry/mesh")
            if mesh_node is None:
                raise RuntimeError(f"Expected donor mesh for {source_body}")
            relative = mesh_node.attrib["filename"]
            source = fetch(relative)
            if source_body in CANONICAL_MESH_OVERRIDES:
                source = ASSET_DIR.parent / CANONICAL_MESH_OVERRIDES[source_body]
                if not source.is_file():
                    raise FileNotFoundError("Download canonical G1 first: download_assets.py --only robots")
                canonical_sources[source_body] = dict(
                    file=CANONICAL_MESH_OVERRIDES[source_body],
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                )
            mesh = trimesh.load(source, force="mesh", process=True)
            # CAD exports duplicate positions at UV/normal seams. Weld those
            # seams to 1 micrometre before topology validation/decomposition.
            mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
            mesh.apply_scale(np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(transform @ origin(collision.find("origin")))
            futures.append(pool.submit(_build_parts, (source_body, target_body, index, mesh, settings)))
    for future in futures:
        parts.extend(future.result())
    pool.shutdown()
    manifest = dict(schema_version=SCHEMA_VERSION, repository=REPOSITORY, revision=REVISION,
                    sources=sources, canonical_sources=canonical_sources,
                    settings=settings, torso_sections=4,
                    solid_housing_parts=["hip_pitch", "ankle_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw"],
                    parts=parts)
    target = ASSET_DIR / "manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(target)
    print(f"Ready: {len(parts)} convex parts in {ASSET_DIR}")


if __name__ == "__main__":
    build_assets()
