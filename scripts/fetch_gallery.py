#!/usr/bin/env python3
"""Download the full-size images linked by the blog gallery thumbnails."""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


GALLERY = "https://blog.ast.moe/gallery/"
ROOT = Path(__file__).resolve().parents[1] / "data"
MAX_BYTES = 200 * 1024 * 1024


class GalleryLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.anchor = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self.anchor = attrs.get("href")
        elif tag == "div" and "thumb-img" in attrs.get("class", "").split():
            if self.anchor:
                self.links.append(self.anchor)

    def handle_endtag(self, tag):
        if tag == "a":
            self.anchor = None


def request(url):
    return Request(url, headers={"User-Agent": "quant-search-exp/0.1 (gallery image training)"})


def gallery_urls():
    with urlopen(request(GALLERY), timeout=30) as response:
        page = response.read()
    parser = GalleryLinks()
    parser.feed(page.decode("utf-8"))
    urls = list(dict.fromkeys(urljoin(GALLERY, link) for link in parser.links))
    if not urls:
        raise RuntimeError("gallery has no thumbnail destination links")
    for url in urls:
        parts = urlparse(url)
        if (
            parts.scheme != "https"
            or parts.netloc != "blog.ast.moe"
            or not parts.path.startswith("/images/")
            or Path(parts.path).suffix.lower() not in {".jpg", ".jpeg"}
            or parts.query
        ):
            raise RuntimeError(f"unexpected gallery link: {url}")
    return urls, hashlib.sha256(page).hexdigest()


def valid_jpeg(path):
    if not path.is_file() or path.stat().st_size < 4:
        return False
    with path.open("rb") as stream:
        start = stream.read(2)
        stream.seek(-2, os.SEEK_END)
        end = stream.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def download(index, url):
    # Neighboring groups contribute one validation and one test image each.
    remainder = (index + 1) % 6
    split = "test" if remainder == 0 else "val" if remainder == 5 else "train"
    filename = f"{index + 1:02d}_{Path(urlparse(url).path).name}"
    destination = ROOT / split / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not valid_jpeg(destination):
        for previous_split in ("train", "val", "test"):
            previous = ROOT / previous_split / filename
            if previous != destination and valid_jpeg(previous):
                previous.replace(destination)
                break
    if not valid_jpeg(destination):
        temporary = destination.with_name(destination.name + ".part")
        for attempt in range(3):
            try:
                with urlopen(request(url), timeout=90) as response, temporary.open("wb") as out:
                    content_type = response.headers.get_content_type()
                    if content_type != "image/jpeg":
                        raise RuntimeError(f"unexpected content type: {content_type}")
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > MAX_BYTES:
                            raise RuntimeError(f"image exceeds {MAX_BYTES} bytes")
                        out.write(block)
                    expected = response.headers.get("Content-Length")
                    if expected and size != int(expected):
                        raise RuntimeError(f"incomplete download: {size} of {expected} bytes")
                if not valid_jpeg(temporary):
                    raise RuntimeError("download is not a complete JPEG")
                temporary.replace(destination)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return {
        "index": index + 1,
        "source_url": url,
        "split": split,
        "path": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main():
    urls, page_sha256 = gallery_urls()
    print(f"gallery: {len(urls)} original links", flush=True)
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, i, url): url for i, url in enumerate(urls)}
        for future in as_completed(futures):
            try:
                record = future.result()
                records.append(record)
                print(
                    f"{record['index']:02d} {record['split']:5} "
                    f"{record['bytes'] / 1048576:.1f} MiB {record['path']}",
                    flush=True,
                )
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
                print(f"FAILED {failures[-1]}", file=sys.stderr, flush=True)
    records.sort(key=lambda record: record["index"])
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "gallery_manifest.json").write_text(
        json.dumps(
            {"gallery": GALLERY, "page_sha256": page_sha256, "images": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {len(records)}/{len(urls)} images "
        f"({sum(r['bytes'] for r in records) / 1048576:.1f} MiB)",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{len(failures)} images failed; rerun to resume")


if __name__ == "__main__":
    main()
