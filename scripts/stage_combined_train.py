#!/usr/bin/env python3
"""Stage original and local images for a reproducible training comparison."""

import argparse
import hashlib
import json
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def images(directory: Path) -> list[Path]:
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no JPEG or PNG images in {directory}")
    return paths


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("additional", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    original = images(args.original)
    additional = images(args.additional)
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {f"00-old-{path.name}" for path in original}
    expected.update(f"10-new-{path.name}" for path in additional)
    unexpected = sorted(path.name for path in args.output.iterdir() if path.name not in expected)
    if unexpected:
        raise ValueError(f"unexpected files in {args.output}: {unexpected}")
    records = []
    seen = {}
    for group, paths in (("00-old", original), ("10-new", additional)):
        for source in paths:
            sha256 = digest(source)
            if sha256 in seen:
                raise ValueError(f"exact duplicate: {seen[sha256]} and {source}")
            seen[sha256] = source
            destination = args.output / f"{group}-{source.name}"
            target = source.resolve()
            if destination.is_symlink():
                if destination.resolve() != target:
                    raise ValueError(f"symlink points elsewhere: {destination}")
            elif destination.exists():
                raise ValueError(f"output is not a symlink: {destination}")
            else:
                destination.symlink_to(target)
            if group == "10-new":
                records.append(
                    {
                        "path": str(source.relative_to(args.additional.parent)),
                        "bytes": source.stat().st_size,
                        "sha256": sha256,
                    }
                )

    manifest = {
        "schema_version": 1,
        "source": "user-supplied local training images",
        "images": records,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"staged {len(original)} original + {len(additional)} additional images")
    print(f"additional manifest SHA-256: {digest(args.manifest)}")


if __name__ == "__main__":
    main()
