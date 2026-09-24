#!/usr/bin/env python3
"""Check a held-out image directory against the frozen training manifest."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dhash(path: Path) -> int:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=9:8:flags=area,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    if len(result.stdout) != 72:
        raise ValueError(f"cannot hash image: {path}")
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (result.stdout[y * 9 + x] > result.stdout[y * 9 + x + 1])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("final_dir", type=Path)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())["images"]
    train = [(args.data_root / record["path"], record["sha256"]) for record in records]
    final = sorted(path for path in args.final_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not final:
        raise ValueError("no final images")
    train_hashes = [(path, dhash(path)) for path, _ in train]
    exact = []
    near = []
    closest = []
    for path in final:
        sha = digest(path)
        exact.extend((str(path), str(other)) for other, old_sha in train if sha == old_sha)
        value = dhash(path)
        distances = sorted(((value ^ old_hash).bit_count(), str(other))
                           for other, old_hash in train_hashes)
        near.extend((str(path), other, distance) for distance, other in distances if distance <= 5)
        closest.append((str(path), *distances[0]))
    print(json.dumps({"train_images": len(train), "final_images": len(final),
                      "exact": exact, "dhash_distance_5_or_less": near,
                      "closest_dhash": closest}, indent=2))


if __name__ == "__main__":
    main()
