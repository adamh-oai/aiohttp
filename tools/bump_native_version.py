#!/usr/bin/env python3

import argparse
import re
import subprocess
from pathlib import Path


DIST_VERSION_RE = re.compile(
    r'^(?P<prefix>__dist_version__\s*=\s*f"\{__version__\}\+native)'
    r"(?P<number>\d+)"
    r'(?P<suffix>")$',
    re.MULTILINE,
)


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, check=True, cwd=cwd)


def bump_dist_version(init_path: Path) -> str:
    init_text = init_path.read_text()
    match = DIST_VERSION_RE.search(init_text)
    if match is None:
        raise RuntimeError(
            f"could not find derived __dist_version__ assignment in {init_path}"
        )

    next_number = int(match.group("number")) + 1
    next_text = DIST_VERSION_RE.sub(
        rf"\g<prefix>{next_number}\g<suffix>",
        init_text,
        count=1,
    )
    init_path.write_text(next_text)

    version_match = re.search(r'^__version__\s*=\s*"([^"]+)"$', init_text, re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"could not find __version__ assignment in {init_path}")
    return f"{version_match.group(1)}+native{next_number}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bump the native wheel suffix, tag the current jj change, and push it."
    )
    parser.add_argument(
        "--remote",
        default="fork",
        help="Git remote that should receive the release tag (default: fork)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    init_path = root / "aiohttp" / "__init__.py"
    version = bump_dist_version(init_path)
    tag = f"v{version}"
    message = f"bump version {version}\n\nCo-authored-by: Codex <noreply@openai.com>"

    run("jj", "desc", "-r", "@", "-m", message, cwd=root)
    run("jj", "tag", "set", "-r", "@", tag, cwd=root)
    run("jj", "git", "export", cwd=root)
    run("git", "push", args.remote, tag, cwd=root)

    print(version)


if __name__ == "__main__":
    main()
