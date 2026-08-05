# Image Mosaic Generator using Voronoi Tesellation

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
| `--output-path` | `mosaic.png` | Path to save the output image |
| `--dpi` | `300` | Resolution used to render the mosaic |

The `hex-grid` method places the cell centers on a hexagonal lattice with a spacing of `cellsize`, where each center is displaced by up to half the `jitter` value.
The `random` method instead samples `npoints` centers from the image, with a probability proportional to the brightness.

Each optimization step moves every cell center by at most one pixel, so larger cells need more steps to settle.
For `cellsize` values much above the default, `niter` should be increased accordingly.

The output image has the same pixel size as the input image, up to rounding, independent of `dpi`.
Since line widths are given in points, `dpi` controls how thick the cell outlines appear.

## Development

To run the tests:
```bash
uv run pytest
```





