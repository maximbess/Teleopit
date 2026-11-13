"""Measured CRAP and AST duplication checks for the ladder/material changes."""

import argparse
import ast
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from radon.complexity import cc_visit

BASE = "cc0cd5393c670e7801bb7d242c700e6af4d47303"


def functions(source):
    return {
        node.lineno: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def complexity_blocks(source):
    result = {}

    def visit(block):
        if hasattr(block, "methods"):
            for method in block.methods:
                visit(method)
        else:
            result[block.lineno] = block
            for closure in getattr(block, "closures", []):
                visit(closure)

    for block in cc_visit(source):
        visit(block)
    return result


def quality(coverage_path, output):
    coverage = json.loads(coverage_path.read_text())["files"]
    stats = {name.replace("\\", "/"): value for name, value in coverage.items()}
    paths = [Path(p) for p in stats if p.startswith("teleopit/sim/")]
    reports, bodies = [], []
    for path in paths:
        source = path.read_text()
        old = subprocess.run(
            ["git", "show", BASE + ":" + path.as_posix()],
            capture_output=True,
            text=True,
            check=False,
        )
        original = (
            {ast.dump(n) for n in functions(old.stdout).values()}
            if old.returncode == 0
            else set()
        )
        blocks = complexity_blocks(source)
        executed = set(stats[path.as_posix()]["executed_lines"])
        statements = executed | set(stats[path.as_posix()]["missing_lines"])
        for line, node in functions(source).items():
            if ast.dump(node) in original:
                continue
            block = blocks[line]
            loc = set(range(line, node.end_lineno + 1)) & statements
            fraction = len(loc & executed) / len(loc) if loc else 1
            reports.append(
                {
                    "file": path.as_posix(),
                    "function": node.name,
                    "complexity": block.complexity,
                    "coverage": fraction,
                    "crap": block.complexity
                    + block.complexity**2 * (1 - fraction) ** 3,
                }
            )
            body = ast.Module(body=node.body, type_ignores=[])
            if sum(1 for _ in ast.walk(body)) >= 50:
                bodies.append(
                    (
                        path.as_posix() + ":" + node.name,
                        re.findall(r"\w+|[^\w\s]", ast.dump(body)),
                    )
                )
    duplicates = duplication(bodies)
    result = {
        "tools": "Radon + measured coverage; AST SequenceMatcher >=50 nodes (named crapper/dryer unavailable)",
        "reference": BASE,
        "functions": reports,
        "crap_ge_10": [r for r in reports if r["crap"] >= 10],
        "duplication_ge_0_8": duplicates,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "functions"}, indent=2))


def duplication(bodies):
    result = []
    for index, (name, tokens) in enumerate(bodies):
        for other, candidate in bodies[index + 1 :]:
            if (
                2 * min(len(tokens), len(candidate)) / (len(tokens) + len(candidate))
                < 0.8
            ):
                continue
            matcher = SequenceMatcher(None, tokens, candidate, autojunk=False)
            if matcher.quick_ratio() >= 0.8 and matcher.ratio() >= 0.8:
                result.append(
                    {"first": name, "second": other, "similarity": matcher.ratio()}
                )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality(args.coverage, args.output)
