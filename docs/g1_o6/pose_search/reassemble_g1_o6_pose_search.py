"""Reassemble verified evidence parts. Uses only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_MANIFEST="G1_O6_Pose_Search_2026-11-13.zip.parts.json"


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):result.update(block)
    return result.hexdigest()


def child(directory,name):
    if not name or Path(name).name!=name or name in (".",".."):
        raise ValueError("Manifest must use plain filenames")
    return directory/name


def verified_parts(directory, records):
    parts=[]
    for record in records:
        path=child(directory,record["name"])
        if path.stat().st_size!=record["bytes"] or digest(path)!=record["sha256"]:
            raise ValueError("Part checksum mismatch: "+path.name)
        parts.append(path)
    return parts


def assemble_parts(parts, output):
    total=hashlib.sha256()
    with output.open("xb") as target:
        for path in parts:
            with path.open("rb") as source:
                for block in iter(lambda:source.read(1024*1024),b""):
                    target.write(block)
                    total.update(block)
    return total.hexdigest()


def archive_matches(path, manifest):
    return path.stat().st_size==manifest["bytes"] and digest(path)==manifest["sha256"]


def reassemble(manifest_path, output=None):
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    directory=manifest_path.parent
    output=output or child(directory,manifest["archive_name"])
    if output.exists():
        if not archive_matches(output,manifest):
            raise FileExistsError("Output already exists and does not match; choose a new --output")
        print("Existing archive verified:",output)
        return output
    parts=verified_parts(directory,manifest["parts"])
    actual=assemble_parts(parts,output)
    if output.stat().st_size!=manifest["bytes"] or actual!=manifest["sha256"]:
        raise ValueError("Assembled archive checksum mismatch; preserve output for inspection")
    print("Archive SHA256 verified:",output,manifest["sha256"])
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",nargs="?",type=Path,default=Path(__file__).with_name(DEFAULT_MANIFEST))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    reassemble(args.manifest,args.output)
