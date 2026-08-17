import logging

import click
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import KDTree, Voronoi

RANDOM_STATE = np.random.RandomState(409239)

# Luminance weights calibrated for CRT phosphors, see http://poynton.ca/PDFs/ColorFAQ.pdf
GREY_COEFFS = np.array([0.2125, 0.7154, 0.0721])

# The eight directions a Voronoi site is allowed to move into
SHIFTS = np.array(
    [[1, 0], [-1, 0], [1, 1], [-1, -1], [0, 1], [0, -1], [1, -1], [-1, 1]]
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def rgb_to_grey(image):
    """Compute the luminance of an RGB image."""
    return image @ GREY_COEFFS


def voronoi_centers_random(image, n_points, random_state=RANDOM_STATE):
    """Sample points according to the image brightness."""
    pdf = image / image.sum()
    coords = random_state.choice(np.arange(image.size), p=pdf.ravel(), size=n_points)
    y, x = np.unravel_index(coords, image.shape)
    x = random_state.uniform(x - 0.5, x + 0.5)
    y = random_state.uniform(y - 0.5, y + 0.5)
    return np.column_stack((x.flatten(), y.flatten()))


def voronoi_centers_hex_grid(
    width, height, cell_size, jitter, random_state=RANDOM_STATE
):
    """Generate a grid of points for a hexagonal lattice."""
    dx, dy = float(cell_size), cell_size * np.sqrt(3) / 2

    x, y = np.meshgrid(
        np.arange(cell_size, width - cell_size, dx),
        np.arange(cell_size, height - cell_size, dy),
    )

    x[1::2] += dx / 2  # offset every other row to obtain a hexagonal lattice

    x = x + random_state.uniform(-jitter / 2, jitter / 2, x.shape)
    y = y + random_state.uniform(-jitter / 2, jitter / 2, y.shape)
    return np.column_stack((x.flatten(), y.flatten()))


INIT_METHODS = ("hex-grid", "random")


def get_colors(points, image):
    """Get the image color at the position of each point."""
    x_idx = np.clip(points[:, 0].astype(int), 0, image.shape[1] - 1)
    y_idx = np.clip(points[:, 1].astype(int), 0, image.shape[0] - 1)
    return image[y_idx, x_idx]


def get_voronoi_tesselation(points):
    """Get Voronoi tessellation for given points.

    Four far away points are added, such that all cells of the actual points are
    bounded. They are placed relative to the extent of the points, to not degrade
    the numerical precision of the tessellation.
    """
    offset = 10 * np.ptp(points, axis=0)
    corners = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])

    points_vor = np.append(points, points.mean(axis=0) + corners * offset, axis=0)
    return Voronoi(points_vor, qhull_options="Qbb Qc")


def optimize_voronoi_cells(image, points, niter=5):
    """Optimize Voronoi cells to better fit the image.

    Each site is moved by one pixel into the direction which reduces the color
    error most. For a given direction the error is only evaluated on the pixels
    along the boundary between two cells, because those are the pixels which
    change ownership when either of the two sites moves: moving both the site of
    the cell in the direction of the shift and the site of the cell opposite to
    it hands the boundary pixels over to the latter.
    """
    height, width = image.shape[0], image.shape[1]

    index = np.arange(len(points))
    inner = (slice(1, -1), slice(1, -1))

    for idx in range(niter):
        labels = points_to_label_image(points, width=width, height=height)
        colors = get_colors(points, image)

        image_voronoi = colors_to_voronoi_image(colors, labels)
        error_null = np.square(image_voronoi - image).sum(axis=-1)

        log.info(f"Iteration {idx + 1}/{niter}, error: {error_null.sum():.2f}")

        errors = np.zeros((len(SHIFTS), len(points)))

        for idx_shift, (dx, dy) in enumerate(SHIFTS):
            neighbor = labels[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx]
            opposite = labels[1 - dy : height - 1 - dy, 1 - dx : width - 1 - dx]

            is_boundary = neighbor != opposite

            error = np.square(colors[opposite] - image[inner]).sum(axis=-1)
            delta = (error - error_null[inner])[is_boundary]

            for labels_shifted in [neighbor, opposite]:
                errors[idx_shift] += np.bincount(
                    labels_shifted[is_boundary], weights=delta, minlength=len(points)
                )

        idx_best = np.argmin(errors, axis=0)
        is_improved = errors[idx_best, index] < 0

        points[is_improved] += SHIFTS[idx_best[is_improved]]
        np.clip(points, 0, [width - 1, height - 1], out=points)

    return points


def drop_repeated_vertices(polygon):
    """Drop vertices which coincide with their predecessor.

    Cocircular sites, such as the ones of a regular grid, yield cells with
    repeated vertices. Those have no well defined edge direction and would
    also bias the center used to shrink the cell.
    """
    is_repeated = np.all(np.isclose(polygon, np.roll(polygon, 1, axis=0)), axis=1)
    return polygon[~is_repeated]


def shrink_polygon(polygon, pad):
    """Shrink a polygon by moving each vertex by ``pad`` towards its center.

    This is not a true polygon offset, but it is enough to open up a gap of
    roughly ``pad`` between neighboring cells. Vertices are never moved past
    the center, so the polygon cannot turn inside out for a large ``pad``.
    """
    center = polygon.mean(axis=0)
    vectors = polygon - center
    distance = np.linalg.norm(vectors, axis=1, keepdims=True)
    return center + np.maximum(distance - pad, 0) * vectors / distance


def round_polygon(polygon, radius):
    """Convert a polygon to a path with rounded corners.

    Each corner is cut back by ``radius`` along both of its edges and the two
    resulting points are joined by a quadratic Bezier curve with the corner
    itself as control point. The cut is limited to half an edge length, so
    that the two corners of an edge cannot overlap.

    Adapted from https://stackoverflow.com/a/72099748, CC BY-SA 4.0.
    """
    n = len(polygon)

    next_ = np.roll(polygon, -1, axis=0)
    edges = next_ - polygon
    lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    offsets = np.minimum(radius, 0.5 * lengths) * edges / lengths

    # start and end of the straight part of each edge
    start, end = polygon + offsets, next_ - offsets

    verts = np.empty((3 * n + 1, 2))
    verts[0] = start[0]
    verts[1::3] = end
    verts[2::3] = next_
    verts[3::3] = np.roll(start, -1, axis=0)

    codes = [Path.MOVETO] + n * [Path.LINETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


def cells_to_collection(vor, colors, outline_color, pad=0.0, radius=0.0, lw=0.5):
    """Convert Voronoi cells to a matplotlib PatchCollection.

    Each cell is first shrunk by ``pad`` pixels and its corners are then
    rounded with a radius of ``radius`` pixels.
    """
    patches, face_colors = [], []

    for idx, idx_region in enumerate(vor.point_region[: len(colors)]):
        region = vor.regions[idx_region]

        if -1 in region or len(region) < 3:
            continue

        polygon = drop_repeated_vertices(vor.vertices[region])

        if pad > 0:
            # a pad larger than the cell radius collapses the cell onto its center
            polygon = drop_repeated_vertices(shrink_polygon(polygon, pad=pad))

        if len(polygon) < 3:
            continue

        patches.append(PathPatch(round_polygon(polygon, radius=radius)))
        face_colors.append(colors[idx])

    if outline_color == "none":
        # avoid white seams between the cells from antialiasing
        outline_color = face_colors

    return PatchCollection(patches, fc=face_colors, ec=outline_color, lw=lw)


def colors_to_voronoi_image(values, labels):
    """Convert per cell values to a voronoi image."""
    return values[labels]


def points_to_label_image(points, width, height):
    """Convert points to a label image using a KDTree."""
    tree = KDTree(points)
    y, x = np.mgrid[0:height, 0:width]
    _, labels = tree.query(np.column_stack((x.ravel(), y.ravel())), k=1, workers=-1)
    return labels.reshape((height, width))


def plot_voronoi_mosaic(
    voronoi, image, dpi, colors, outline_color, background_color, pad, radius
):
    """Plot the Voronoi mosaic and return the figure.

    The figure is created directly, instead of through ``pyplot``, so that it
    is not registered in the global figure manager. This keeps the rendering
    free of global state and lets the caller decide when to drop the figure.
    """
    width, height = image.shape[1], image.shape[0]

    fig = Figure(figsize=(width / dpi, height / dpi), facecolor=background_color)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

    collection = cells_to_collection(
        vor=voronoi,
        colors=colors,
        outline_color=outline_color,
        pad=pad,
        radius=radius,
        lw=0.1,
    )
    ax.add_collection(collection)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig


def as_float_rgb(image):
    """Convert an image to RGB values in the range [0, 1], dropping any alpha."""
    image = image[:, :, :3]

    if image.dtype == np.uint8:
        image = image / 255.0

    return image


def make_mosaic(
    image,
    init_method="hex-grid",
    cellsize=15,
    jitter=5,
    npoints=1000,
    niter=10,
    seed=0,
    outline_color="black",
    background_color="white",
    pad=0.0,
    radius=0.0,
    dpi=300,
):
    """Create the Voronoi mosaic of an image and return it as a figure.

    ``image`` is expected as RGB values in the range [0, 1], as returned by
    `as_float_rgb`. The parameters match the options of the command line tool.
    """
    height, width, _ = image.shape

    random_state = np.random.RandomState(seed)

    if init_method == "random":
        centers = voronoi_centers_random(
            image=rgb_to_grey(image), n_points=npoints, random_state=random_state
        )
    elif init_method == "hex-grid":
        centers = voronoi_centers_hex_grid(
            width, height, cell_size=cellsize, jitter=jitter, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown initialization method: '{init_method}'")

    points = optimize_voronoi_cells(image=image, points=centers, niter=niter)
    voronoi = get_voronoi_tesselation(points=points)
    colors = get_colors(points=points, image=image)

    return plot_voronoi_mosaic(
        voronoi=voronoi,
        image=image,
        dpi=dpi,
        colors=colors,
        outline_color=outline_color,
        background_color=background_color,
        pad=pad,
        radius=radius,
    )


@click.command(context_settings={"show_default": True})
@click.argument("image-path", type=click.Path(exists=True))
@click.option(
    "--init-method",
    type=click.Choice(INIT_METHODS),
    default="hex-grid",
    help="Method to place the initial Voronoi cell centers.",
)
@click.option("--cellsize", default=15, help="Average diameter of a Voronoi cell.")
@click.option("--jitter", default=5, help="Width of the jitter applied to the lattice.")
@click.option("--npoints", default=1000, help="Number of Voronoi points.")
@click.option(
    "--niter", default=10, help="Number of Voronoi cell refinement iterations"
)
@click.option("--seed", default=0, help="Random seed for reproducibility.")
@click.option(
    "--outline-color", default="black", help="Color of the Voronoi cell outlines."
)
@click.option(
    "--background-color",
    default="white",
    help="Color shown in the gaps between the cells.",
)
@click.option("--pad", default=0.0, help="Distance in pixels each cell is shrunk by.")
@click.option(
    "--radius", default=0.0, help="Corner radius in pixels used to round the cells."
)
@click.option(
    "--output-path", default="mosaic.png", help="Path to save the output image."
)
@click.option("--dpi", default=300, help="DPI for the output image.")
def cli(
    image_path,
    init_method,
    cellsize,
    jitter,
    npoints,
    niter,
    seed,
    outline_color,
    background_color,
    pad,
    radius,
    output_path,
    dpi,
):
    """Create a Voronoi mosaic from an image."""
    log.info(f"Read image from {image_path}")
    image = as_float_rgb(imread(image_path))

    figure = make_mosaic(
        image=image,
        init_method=init_method,
        cellsize=cellsize,
        jitter=jitter,
        npoints=npoints,
        niter=niter,
        seed=seed,
        outline_color=outline_color,
        background_color=background_color,
        pad=pad,
        radius=radius,
        dpi=dpi,
    )

    log.info(f"Saving output to {output_path}")
    figure.savefig(output_path, dpi=dpi)


if __name__ == "__main__":
    cli()
