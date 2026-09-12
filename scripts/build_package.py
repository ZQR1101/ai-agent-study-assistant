"""Build the release artifacts (sdist + wheel) for rulebook.

The wheel is self-contained: it bundles the built frontend into
``backend/static`` and ships the full ``.env.example`` as ``backend/env.example``.

Usage:
    python scripts/build_package.py            # npm build + python -m build
    python scripts/build_package.py --skip-npm # reuse an existing backend/static
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "backend" / "static"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_EXAMPLE_COPY = PROJECT_ROOT / "backend" / "env.example"
DIST_DIR = PROJECT_ROOT / "dist"


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def build_frontend(skip: bool) -> None:
    if skip:
        if STATIC_DIR.is_dir():
            print(f"[SKIP] Reusing existing frontend build: {STATIC_DIR}")
            return
        print("[ERROR] --skip-npm set but backend/static does not exist. Run npm run build first.")
        sys.exit(1)

    npm = shutil.which("npm")
    if npm is None:
        print("[ERROR] npm not found. Install Node.js 18+ or pass --skip-npm with an existing backend/static.")
        sys.exit(1)

    run([npm, "install"])
    run([npm, "run", "build"])
    if not STATIC_DIR.is_dir():
        print("[ERROR] Frontend build did not produce backend/static. Check vite.config.js.")
        sys.exit(1)


def stage_env_example() -> None:
    if ENV_EXAMPLE.is_file():
        shutil.copyfile(ENV_EXAMPLE, ENV_EXAMPLE_COPY)
        print(f"[OK] Staged {ENV_EXAMPLE_COPY.relative_to(PROJECT_ROOT)}")
    else:
        print("[WARN] .env.example not found; the wheel will not ship a full env template")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-npm", action="store_true", help="Skip the frontend build")
    args = parser.parse_args()

    build_frontend(args.skip_npm)
    stage_env_example()

    try:
        import build  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "build"])

    run([sys.executable, "-m", "build"])

    artifacts = sorted(DIST_DIR.glob("rulebook_review*"))
    print("\n[OK] Build complete:")
    for artifact in artifacts:
        print(f"  {artifact}")
    print("\nNext steps:")
    print("  1. Inspect the wheel:  python -m zipfile -l dist/<wheel>")
    print("  2. Test install:       pip install dist/<wheel> and run `rulebook check`")
    print("  3. Publish:            twine upload dist/*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
