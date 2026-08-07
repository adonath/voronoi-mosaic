# Image Mosaic Generator using Voronoi Tesellation

[![CI](https://github.com/adonath/voronoi-mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/adonath/voronoi-mosaic/actions/workflows/ci.yml)

This repository implements a method to create image mosaics using Voronoi tessellation proposed by Yoshinori Dobashi and Toshiyuki Haga, Henry Johan,
and Tomoyuki Nishita (Eurographics Short Presentations, 2002, DOI: 10.2312/egs.20021036).

You can use it to create mosaics like this:


| Input Image | Voronoi Mosaic |
| -------- | ------- |
| ![Input Image](https://raw.githubusercontent.com/adonath/voronoi-mosaic/main/example-images/butterfly.jpg) | ![Voronoi Mosaic](https://raw.githubusercontent.com/adonath/voronoi-mosaic/main/example-images/butterfly-mosaic.jpg) |



## Installation

To install the package:
```bash
pip install .
```

Or to install it as a standalone command line tool:
```bash
uv tool install .
```

## Usage

Once installed, the mosaic can be created with:
```bash
voronoi-mosaic example-images/butterfly.jpg --output-path example-images/butterfly-mosaic.jpg
```

The same command is available as a module entry point:
```bash
python -m voronoi_mosaic example-images/butterfly.jpg --output-path example-images/butterfly-mosaic.jpg
```

The example mosaic shown above was created with the default parameters.

To get help on the parameters:
```bash
voronoi-mosaic --help
```

## Parameters

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--init-method` | `hex-grid` | How the initial cell centers are placed, either `hex-grid` or `random` |
| `--cellsize` | `15` | Average diameter of a cell in pixels, `hex-grid` only |
| `--jitter` | `5` | Width of the random offset applied to the lattice, in pixels, `hex-grid` only |
| `--npoints` | `1000` | Number of cells, `random` only |
| `--niter` | `10` | Number of optimization steps |
| `--seed` | `0` | Random seed for reproducibility |
| `--outline-color` | `black` | Any matplotlib color, or `none` to draw no outlines |
| `--background-color` | `white` | Any matplotlib color, shown in the gaps between the cells |
| `--pad` | `0.0` | Distance in pixels each cell is shrunk by, which opens up a gap between neighboring cells |
| `--radius` | `0.0` | Corner radius in pixels used to round the cells |
| `--output-path` | `mosaic.png` | Path to save the output image |
| `--dpi` | `300` | Resolution used to render the mosaic |

The `hex-grid` method places the cell centers on a hexagonal lattice with a spacing of `cellsize`, where each center is displaced by up to half the `jitter` value.
The `random` method instead samples `npoints` centers from the image, with a probability proportional to the brightness.

Each optimization step moves every cell center by at most one pixel, so larger cells need more steps to settle.
For `cellsize` values much above the default, `niter` should be increased accordingly.

The `pad` and `radius` options turn the sharp cells into rounded pebbles.
Each cell is first shrunk by `pad` towards its own center, which opens up a gap that shows the `background-color`, and its corners are then rounded with `radius`.
Both are given in pixels and are best chosen relative to `cellsize`, for example a `pad` of `1.5` and a `radius` of `4` for the default `cellsize` of `15`.
The rounding is limited to half an edge length, so a large `radius` turns small cells into ellipses rather than distorting them.

The output image has the same pixel size as the input image, up to rounding, independent of `dpi`.
Since line widths are given in points, `dpi` controls how thick the cell outlines appear.

## Development

To run the tests:
```bash
uv run pytest
```

### Releasing

Publishing a release on GitHub builds the package and uploads it to PyPI.
The release tag has to match the version in `pyproject.toml`, written as `v0.1.0` or `0.1.0`, otherwise the workflow stops before the upload.

This requires a one time setup, as the workflow authenticates without a stored token:

- a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for this repository on PyPI, with `publish.yml` as the workflow and `pypi` as the environment
- an environment named `pypi` in the repository settings, which is also the place to require a manual approval before the upload





