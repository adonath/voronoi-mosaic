"""Development server for the Voronoi mosaic web app.

The app is fully static: the mosaic is computed in the browser with Pyodide, so
no image is ever sent anywhere. This server only hands out the files, and it
serves ``voronoi_mosaic.py`` straight from the repository root, so the app
always runs the algorithm as it is currently checked out.

Run it with::

    uv run --group app python app/server.py

Use ``python app/build.py`` to assemble the same files for static hosting.
"""

from pathlib import Path

from flask import Flask, send_from_directory

APP = Path(__file__).resolve().parent
ROOT = APP.parent
STATIC = APP / "static"

app = Flask(__name__, static_folder=STATIC, static_url_path="")


@app.route("/")
def index():
    """Serve the app itself."""
    return send_from_directory(STATIC, "index.html")


@app.route("/voronoi_mosaic.py")
def module():
    """Serve the algorithm, which the worker imports into Pyodide."""
    return send_from_directory(ROOT, "voronoi_mosaic.py", mimetype="text/x-python")


if __name__ == "__main__":
    # the built site is a copy of the files that are already watched
    app.run(debug=True, exclude_patterns=[str(APP / "dist" / "*")])
