"""Versioned manifests and unique output directories; never overwrite evidence."""

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import mujoco

SCHEMA = "o6_diagnostic_v1"
UNITS = {
    "time": "s",
    "length": "m",
    "angle": "rad",
    "force": "N",
    "moment": "N*m",
    "mass": "kg",
}


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} already contains a run; choose a new output directory"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def code_state():
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_changes": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).splitlines(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "diagnostic_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def manifest(path, kind, metadata):
    files = {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(
        path / "manifest.json",
        {
            "schema": SCHEMA,
            "kind": kind,
            "units": UNITS,
            "frame": "world",
            "wrench_origin": "wrist_mount unless explicitly COM/rung",
            **metadata,
            "files_sha256": files,
            "child_runs": {
                p.parent.name: {
                    "manifest": p.relative_to(path).as_posix(),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(path.glob("*/manifest.json"))
            },
        },
    )


def write_csv(path, rows, columns=None):
    columns = columns or sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]
