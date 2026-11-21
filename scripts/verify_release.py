"""Check all versioned research artifacts against the release's SHA-256 inventory."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys


def verify(root):
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError("SHA256SUMS is missing")
    seen = set()
    problems = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or name in seen or len(digest) != 64):
            raise ValueError(f"Invalid manifest entry: {name}")
        seen.add(name)
        path = root / relative
        if not path.is_file():
            problems.append(f"Missing: {name}")
            continue
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            problems.append(f"Content differs: {name}")
    if not seen:
        raise ValueError("Empty manifest")
    if problems:
        raise ValueError("\n".join(problems))
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        checked = verify(args.root)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    print(f"Verified SHA-256 for {checked} files.")
