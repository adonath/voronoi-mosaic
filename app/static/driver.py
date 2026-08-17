"""Bridge between the browser and the Voronoi mosaic algorithm.

This module is imported inside Pyodide, in the web worker of the app. Pixels
come in and the rendered mosaic goes out through the Pyodide file system, which
keeps the raw image buffers out of the JavaScript to Python conversion.

The figures are created without ``pyplot``, so rendering a PNG needs no
matplotlib backend and works in the worker as it does on the command line.
"""

import io
import json
import logging

import numpy as np
from matplotlib.figure import Figure

import voronoi_mosaic

PIXELS_PATH = "/tmp/pixels.bin"
MOSAIC_PATH = "/tmp/mosaic.png"


class ProgressHandler(logging.Handler):
    """Forward the log messages of the algorithm to a callback."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        self.callback(record.getMessage())


def install_progress_handler(callback):
    """Report the progress of the algorithm through ``callback``."""
    logging.getLogger("voronoi_mosaic").addHandler(ProgressHandler(callback))


def warm_up():
    """Render an empty figure, to build the font cache before the first run."""
    Figure().savefig(io.BytesIO(), format="png")


def run_mosaic(width, height, params_json):
    """Render the mosaic of the RGBA pixels found in `PIXELS_PATH`.

    The parameters arrive as a JSON string, because strings and numbers are the
    values that cross the JavaScript to Python boundary without conversion.
    """
    params = json.loads(params_json)

    pixels = np.fromfile(PIXELS_PATH, dtype=np.uint8).reshape(height, width, 4)
    image = voronoi_mosaic.as_float_rgb(pixels)

    figure = voronoi_mosaic.make_mosaic(image, **params)
    figure.savefig(MOSAIC_PATH, dpi=params["dpi"])
