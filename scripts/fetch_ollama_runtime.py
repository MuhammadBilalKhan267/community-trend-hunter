"""Fetch only CPU runtime files from Ollama's official Windows portable ZIP.

The release ZIP contains large optional GPU libraries. HTTP byte ranges and the
ZIP index let this task retain the CLI and complete base runtime without them.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import zipfile

VERSION = "v0.40.6"
ASSET = "ollama-windows-amd64.zip"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / ".tools" / "ollama-0.40.6-cpu"


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid remote ZIP position")
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        end = self.position + count - 1
        request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}", "User-Agent": "local-trend-hunter"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Official asset server did not honor the byte range; refusing a full archive download")
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected:
                raise RuntimeError("Unexpected ZIP byte range")
            result = response.read(count)
        if len(result) != count:
            raise RuntimeError("Incomplete ZIP byte range")
        self.position += len(result)
        self.transferred += len(result)
        return result


def main():
    request = Request(f"https://api.github.com/repos/ollama/ollama/releases/tags/{VERSION}", headers={"User-Agent": "local-trend-hunter"})
    with urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(item for item in release["assets"] if item["name"] == ASSET)
    url = asset["browser_download_url"]
    expected_url = f"https://github.com/ollama/ollama/releases/download/{VERSION}/{ASSET}"
    if url != expected_url:
        raise RuntimeError("Unexpected official asset URL")
    remote = RangeReader(url, asset["size"])
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"release": VERSION, "source": url, "mode": "CPU base runtime; optional GPU subdirectories omitted", "files": []}
    with zipfile.ZipFile(remote) as archive:
        files = [item for item in archive.infolist() if not item.is_dir() and
                 (item.filename == "ollama.exe" or PurePosixPath(item.filename).parent.as_posix() == "lib/ollama")]
        print(f"Fetching {len(files)} CPU/base files; compressed payload {sum(item.compress_size for item in files)/2**20:.1f} MiB", flush=True)
        for item in files:
            target = (DESTINATION / item.filename).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise RuntimeError("Unsafe archive path")
            data = archive.read(item)  # zipfile verifies each entry's CRC.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest["files"].append({"path": item.filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            if item.filename == "ollama.exe":
                print("Portable Ollama CLI fetched.", flush=True)
    manifest["download_bytes"] = remote.transferred
    (DESTINATION / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Runtime ready at {DESTINATION}; transferred {remote.transferred/2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
