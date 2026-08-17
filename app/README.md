# Voronoi Mosaic Web App

A minimal web interface for the `voronoi-mosaic` command line tool.

The mosaic is computed in the browser with [Pyodide](https://pyodide.org), which runs the very same `voronoi_mosaic.py` as the command line tool, compiled to WebAssembly.
The uploaded image is decoded by the browser and handed to Pyodide directly, so it never reaches a server and nothing is stored anywhere.

## Running it locally

```bash
uv run --group app python app/server.py
```

Then open http://127.0.0.1:5000.

The server exists only to hand out the files during development.
It serves `voronoi_mosaic.py` from the repository root, so any edit to the algorithm shows up on the next reload.

## Deploying it

There is no backend to deploy.
Assemble the site and upload the folder to any static host:

```bash
python app/build.py
```

This writes `app/dist`, which contains the page, the worker and a copy of `voronoi_mosaic.py`.

## How it works

```
index.html / main.js          browser main thread
  |  decode the image, scale it to the maximum size
  |  postMessage(pixels, parameters)
  v
worker.js                     web worker, so the page stays responsive
  |  load Pyodide + numpy, scipy, matplotlib, click
  |  fetch and import voronoi_mosaic.py
  v
voronoi_mosaic.make_mosaic()  the same function the CLI calls
  |  returns a matplotlib figure, saved as PNG
  v
main.js                       preview and download link, from a blob URL
```

The first visit downloads about 27 MB of packages, dominated by scipy (14 MB) and matplotlib (7 MB).
The browser caches them, so later visits start in a second or two.
`click` is in that list only because `voronoi_mosaic` imports it for its command line interface, which the app never uses.

Progress is reported by attaching a logging handler to the `voronoi_mosaic` logger, so the per iteration errors the CLI prints show up in the status line of the page.

## Parameters

All options of the command line tool are exposed, and are documented in the [main README](../README.md#parameters).

The page adds one control that has no command line equivalent: **maximum size** is the long edge the image is scaled down to before it is processed.
The optimizer touches every pixel on every iteration, so this is what governs how long a run takes.
The default of 1200 keeps a run in the range of seconds; a full resolution photo can take minutes in WebAssembly.
