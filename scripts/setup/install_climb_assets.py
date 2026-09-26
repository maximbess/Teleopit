#!/usr/bin/env python3
"""Install the Unitree G1 assets used by ladder climbing.

This is the asset-only counterpart of ``pip install -e '.[sim2real]'``:
one command, and nothing from the training stack, checkpoints, retargeting
assets, sample BVH, or motion datasets.

    python scripts/setup/install_climb_assets.py
    python scripts/setup/install_climb_assets.py --source huggingface

The canonical robot lands in ``assets/robots/unitree_g1/``. Convex collision
parts are built beside it. Pinned CoACD, trimesh, and shapely packages are
installed only when the active environment does not already have them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep these pins aligned with the collision-build extra in pyproject.toml.
COLLISION_PINS = ("trimesh==5.1.0", "coacd==1.0.14", "shapely==2.1.2")


def _installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def missing_collision_pins() -> list[str]:
    missing = []
    for spec in COLLISION_PINS:
        name, pinned = spec.split("==")
        if _installed_version(name) != pinned:
            missing.append(spec)
    return missing


def ensure_collision_build(run=subprocess.check_call) -> None:
    missing = missing_collision_pins()
    if not missing:
        return
    print("Installing collision build packages:", ", ".join(missing))
    run([sys.executable, "-m", "pip", "install", *missing])


def install_climb_assets(
    *,
    source: str = "modelscope",
    cache_dir: Path | None = None,
    skip_deps: bool = False,
) -> None:
    """Download the canonical G1 model, then build its collision parts."""
    if source not in {"modelscope", "huggingface"}:
        raise ValueError(f"Unsupported asset source: {source}")
    if not skip_deps:
        ensure_collision_build()

    from scripts.setup import download_assets
    from scripts.setup.download_g1_collision import build_assets

    if source == "huggingface":
        cache = cache_dir or PROJECT_ROOT / "data" / "huggingface_cache"
        download_assets.download_all_hf(["robots"], cache)
    else:
        cache = cache_dir or PROJECT_ROOT / "data" / "modelscope_cache"
        download_assets.download_all(["robots"], cache)
    build_assets()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Install Unitree G1 assets for ladder climbing training."
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="Hosted source for the canonical G1 model (default: modelscope)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Download cache directory (default: data/modelscope_cache or data/huggingface_cache)",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Do not install the pinned collision-build packages",
    )
    args = parser.parse_args(argv)
    install_climb_assets(
        source=args.source,
        cache_dir=args.cache_dir,
        skip_deps=args.skip_deps,
    )


if __name__ == "__main__":
    main()
