"""Assemble the static site in ``app/dist``.

The result is a plain folder of files that can be served by any static host,
such as GitHub Pages. It is the same set of files the development server hands
out, with ``voronoi_mosaic.py`` copied in from the repository root.
"""

import shutil
from pathlib import Path

APP = Path(__file__).resolve().parent
DIST = APP / "dist"


def main():
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.copytree(APP / "static", DIST, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(APP.parent / "voronoi_mosaic.py", DIST)
    print(f"Wrote the static site to {DIST}")


if __name__ == "__main__":
    main()
