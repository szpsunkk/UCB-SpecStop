"""Verify the public data archive without changing it."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest = json.loads((root / "data/SHA256SUMS.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise ValueError("Missing or modified archived files: " + ", ".join(failures))
    return len(manifest["files"])


if __name__ == "__main__":
    print("Verified %d archived files" % verify())
