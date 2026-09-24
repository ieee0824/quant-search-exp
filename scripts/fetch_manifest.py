#!/usr/bin/env python3
"""Fetch the exact image bytes pinned by a manifest without changing the manifest."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def fetch(root, record):
    relative = Path(record["path"])
    source = urlparse(record["source_url"])
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or ".." in relative.parts
        or relative.parts[0] != record["split"]
        or source.scheme != "https"
        or not source.netloc
        or source.query
        or source.fragment
        or relative.name != f"{record['index']:02d}_{Path(source.path).name}"
    ):
        raise ValueError(f"invalid image URL or path: {relative}")
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == record["bytes"]:
        if digest(destination) == record["sha256"]:
            return f"verified {relative}"
    temporary = destination.with_name(destination.name + ".part")
    request = Request(record["source_url"], headers={"User-Agent": "quant-search-exp/0.1"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if temporary.stat().st_size != record["bytes"] or digest(temporary) != record["sha256"]:
            raise ValueError(f"download differs from pinned size or SHA-256: {relative}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return f"downloaded {relative}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["images"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, args.data_root, record) for record in records]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print(f"verified {len(records)} pinned images", flush=True)


if __name__ == "__main__":
    main()
