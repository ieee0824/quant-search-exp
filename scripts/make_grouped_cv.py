#!/usr/bin/env python3
"""Build five deterministic, capture-grouped folds without copying source images.

Capture months are kept together across cameras. Undated gallery images stay in
one group unless a nearby camera sequence number maps unambiguously to a dated
month. The output contains no EXIF fields or image bytes.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEQUENCE = re.compile(r"(IMG_|0Q1A|_MG_|_45A|DSCF|545A|DSC_)(\d+)", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(paths: list[Path]) -> dict[Path, dict[str, str]]:
    result = subprocess.run(
        ["sips", "-g", "creation", *(str(path.resolve()) for path in paths)],
        capture_output=True,
        text=True,
        check=True,
    )
    found = {}
    current = None
    for line in result.stdout.splitlines():
        if line.startswith("/"):
            current = Path(line).resolve()
            found[current] = {}
        elif current is not None and ": " in line:
            key, value = line.strip().split(": ", 1)
            found[current][key] = value
    if set(found) != {path.resolve() for path in paths}:
        raise ValueError("sips did not return metadata for every image")
    return found


def sequence(path: Path) -> tuple[str, int] | None:
    match = SEQUENCE.search(path.name)
    return (match.group(1).lower(), int(match.group(2))) if match else None


def group_keys(paths: list[Path], exif: dict[Path, dict[str, str]]) -> dict[Path, str]:
    groups = {}
    known = []
    for path in paths:
        creation = exif[path.resolve()].get("creation", "<nil>")
        if re.match(r"\d{4}:\d{2}:\d{2}", creation):
            group = "month:" + creation[:7]
            groups[path] = group
            if item := sequence(path):
                known.append((item, group))
    for path in paths:
        if path in groups:
            continue
        item = sequence(path)
        nearby = {
            group
            for (prefix, number), group in known
            if item and prefix == item[0] and abs(number - item[1]) <= 50
        }
        if len(nearby) == 1:
            groups[path] = next(iter(nearby))
        elif path.parent.name == "train_inbox":
            groups[path] = "undated:inbox"
        else:
            groups[path] = "undated:gallery"
    # Undated inbox images in a gallery camera sequence must stay with the gallery.
    gallery_numbers = [sequence(path) for path in paths if path.parent.name != "train_inbox"]
    for path in paths:
        if groups[path] != "undated:inbox" or not (item := sequence(path)):
            continue
        if any(
            other and other[0] == item[0] and abs(other[1] - item[1]) <= 50
            for other in gallery_numbers
        ):
            groups[path] = "undated:gallery"
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inbox-manifest", type=Path)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("at least two folds required")
    paths = sorted(
        path
        for name in ("train", "val", "test", "train_inbox")
        for path in (args.data_root / name).iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if args.inbox_manifest:
        snapshot = json.loads(args.inbox_manifest.read_text())
        expected = {record["path"]: record["sha256"] for record in snapshot["images"]}
        paths = [
            path
            for path in paths
            if path.parent.name != "train_inbox"
            or str(path.relative_to(args.data_root)) in expected
        ]
        found = {str(path.relative_to(args.data_root)): sha256(path) for path in paths if path.parent.name == "train_inbox"}
        if found != expected:
            raise ValueError("inbox snapshot is missing images or differs from current files")
    exif = metadata(paths)
    groups = group_keys(paths, exif)
    members = defaultdict(list)
    for path, group in groups.items():
        members[group].append(path)
    loads = [0] * args.folds
    assignment = {}
    for group in sorted(members, key=lambda key: (-len(members[key]), key)):
        fold = min(range(args.folds), key=lambda index: (loads[index], index))
        assignment[group] = fold
        loads[fold] += len(members[group])
    records = []
    for path in paths:
        group = groups[path]
        records.append(
            {
                "path": str(path.relative_to(args.data_root)),
                "sha256": sha256(path),
                "group": hashlib.sha256(group.encode()).hexdigest()[:12],
                "fold": assignment[group],
            }
        )
    for fold in range(args.folds):
        for split in ("train", "eval"):
            destination = args.output_root / f"fold-{fold}" / split
            destination.mkdir(parents=True, exist_ok=True)
            expected = {}
            for index, (path, record) in enumerate(zip(paths, records, strict=True)):
                if (record["fold"] == fold) != (split == "eval"):
                    continue
                name = f"{index:03d}-{path.name}"
                expected[name] = path.resolve()
            for existing in destination.iterdir():
                if existing.name not in expected:
                    raise ValueError(f"unexpected file in staged directory: {existing}")
            for name, source in expected.items():
                link = destination / name
                if link.is_symlink():
                    if link.resolve() != source:
                        raise ValueError(f"incorrect staged link: {link}")
                elif link.exists():
                    raise ValueError(f"staged path is not a symlink: {link}")
                else:
                    link.symlink_to(source)
    manifest = {
        "schema_version": 1,
        "grouping": "capture month, nearby camera sequence for undated files, conservative undated groups",
        "folds": args.folds,
        "images": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"images={len(paths)} groups={len(members)} fold_sizes={loads}")
    print("group_sizes=" + str(sorted(Counter(groups.values()).values(), reverse=True)))
    print(f"manifest_sha256={sha256(args.manifest)}")


if __name__ == "__main__":
    main()
