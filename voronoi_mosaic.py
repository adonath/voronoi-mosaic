import logging

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from scipy import ndimage as ndi
from scipy.spatial import KDTree, Voronoi
from skimage import color

RANDOM_STATE = np.random.RandomState(409239)
MAX_VALUE = 1e6

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def voronoi_centers_random(image, n_points, random_state=RANDOM_STATE):
    """Sample points according to the image brightness."""
    pdf = image / image.sum()
    coords = random_state.choice(np.arange(image.size), p=pdf.ravel(), size=n_points)
    x, y = np.unravel_index(coords, image.shape)
    x = random_state.uniform(x - 0.5, x + 0.5)
    y = random_state.uniform(y - 0.5, y + 0.5)
    return np.column_stack((x.flatten(), y.flatten()))


def voronoi_centers_hex_grid(
    width, height, cell_size, jitter, random_state=RANDOM_STATE
):
    """Generate a grid of points for a hexagonal lattice."""
    dy, dx = 2 * cell_size, cell_size / np.sqrt(3)
    ratio = np.sqrt(3) / 2  # cos(60°)
    y, x = np.meshgrid(np.arange(0, height / ratio, dy), np.arange(0, width, dx))

    y = y * ratio
    y[::2, :] += cell_size * ratio

    x = x + random_state.uniform(-jitter, jitter, x.shape)
    y = y + random_state.uniform(-jitter, jitter, y.shape)
    return np.column_stack((x.flatten(), y.flatten()))


INIT_METHODS = {
    "random": voronoi_centers_random,
    "hex-grid": voronoi_centers_hex_grid,
}


def get_mean_colors(points, image):
    """Get colors from the image at given points."""
    labels = points_to_label_image(points, width=image.shape[1], height=image.shape[0])
    index = np.arange(np.max(labels) + 1)

    colors = []

    for channel in range(image.shape[2]):
        mean = ndi.mean(image[:, :, channel], labels=labels, index=index)
        colors.append(mean)

    return np.column_stack(colors)


def get_colors(points, image):
    """Get colors"""
    x_idx = np.clip(points[:, 0].astype(int), 0, image.shape[1] - 1)
    y_idx = np.clip(points[:, 1].astype(int), 0, image.shape[0] - 1)
    return image[y_idx, x_idx]


def get_voronoi_tesselation(points):
    """Get Voronoi tessellation for given points."""
    points_vor = np.append(
        points,
        [
            [MAX_VALUE, MAX_VALUE],
            [-MAX_VALUE, MAX_VALUE],
            [MAX_VALUE, -MAX_VALUE],
            [-MAX_VALUE, -MAX_VALUE],
        ],
        axis=0,
    )
    voronoi = Voronoi(points_vor, qhull_options="Qbb Qc Qx")
    return voronoi


def optimize_voronoi_cells(image, points, niter=5, error_threshold=0.1):
    """Optimize Voronoi cells to better fit the image."""
    width, height = image.shape[1], image.shape[0]

    shifts = np.array(
        [[1, 0], [-1, 0], [1, 1], [-1, -1], [0, 1], [0, -1], [1, -1], [-1, 1]]
    )

    index = np.arange(len(points)) + 1

    image_padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="reflect")

    for idx in range(niter):
        labels = points_to_label_image(points, width=width, height=height)
        colors = np.nan_to_num(get_colors(points, image))
        vor = get_voronoi_tesselation(points)
        collection = cells_to_collection(vor, colors, outline_color="none")
        image_voronoi = collection_to_voronoi_image(
            collection, width=width, height=height
        )

        error_null = ndi.sum(
            np.square(image_voronoi - image), labels=labels[..., None], index=index
        )

        log.info(f"Iteration {idx + 1}/{niter}, error: {error_null.sum():.2f}")

        errors = []

        for shift in shifts:
            dx, dy = shift[1], shift[0]
            shifted = image_padded[1 + dy : 1 + height + dy, 1 + dx : 1 + width + dx]
            error = ndi.sum(
                np.square(image_voronoi - shifted),
                labels=labels[..., None],
                index=index,
            )
            errors.append(error_null - error)

        errors = np.array(errors)

        mask = (errors >= error_threshold).any(axis=0)

        points[mask] += shifts[np.argmax(errors, axis=0)[mask]]

    return points


def cells_to_collection(vor, colors, outline_color, lw=0.5):
    """Convert Voronoi cells to a matplotlib PatchCollection."""
    # TODO: add smoothing of the Voronoi cells, e.g.
    # https://stackoverflow.com/a/69247177/19802442 and https://stackoverflow.com/a/72099748/19802442

    patches = []

    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region:
            polygon = Polygon([vor.vertices[_] for _ in region])
            patches.append(polygon)

    return PatchCollection(patches, fc=colors, ec=outline_color, lw=lw)


def collection_to_voronoi_image(collection, width, height, dpi=300):
    """Convert a PatchCollection to a label image."""
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(collection)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    canvas.draw()
    values = np.asarray(canvas.buffer_rgba())[..., :3]
    return values / 255.0


def points_to_label_image(points, width, height):
    """Convert points to a label image using a KDTree."""
    tree = KDTree(points)
    y, x = np.mgrid[0:height, 0:width]
    _, labels = tree.query(np.column_stack((x.ravel(), y.ravel())), k=1)
    return labels.reshape((height, width)) + 1


def plot_voronoi_mosaic(voronoi, image, dpi, colors, outline_color):
    """Plot the Voronoi mosaic."""
    width, height = image.shape[1], image.shape[0]

    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

    collection = cells_to_collection(
        vor=voronoi, colors=colors, outline_color=outline_color, lw=0.1
    )
    ax.add_collection(collection)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")


@click.command()
@click.argument("image-path", type=click.Path(exists=True))
@click.option(
    "--init-method", type=click.Choice(list(INIT_METHODS.keys())), default="hex-grid"
)
@click.option("--cellsize", default=15, help="Size of each Voronoi cell.")
@click.option("--jitter", default=5, help="Jitter to apply to the Voronoi points.")
@click.option("--npoints", default=1000, help="Number of Voronoi points.")
@click.option(
    "--niter", default=10, help="Number of Voronoi cell refinement iterations"
)
@click.option("--seed", default=0, help="Random seed for reproducibility.")
@click.option(
    "--outline-color", default="black", help="Color of the Voronoi cell outlines."
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
    output_path,
    dpi,
):
    """Create a Voronoi mosaic from an image."""
    log.info(f"Read image from {image_path}")
    image = plt.imread(image_path)[:, :, :3] / 255.0

    height, width, _ = image.shape

    method = INIT_METHODS[init_method]

    random_state = np.random.RandomState(seed)

    if init_method == "random":
        image_grey = color.rgb2gray(image)
        centers = method(image=image_grey, n_points=npoints, random_state=random_state)
    elif init_method == "hex-grid":
        centers = method(
            width, height, cell_size=cellsize, jitter=jitter, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown initialization method: '{init_method}'")

    points = optimize_voronoi_cells(image=image, points=centers, niter=niter)
    voronoi = get_voronoi_tesselation(points=points)
    colors = get_colors(points=points, image=image)

    plot_voronoi_mosaic(
        voronoi=voronoi,
        image=image,
        dpi=dpi,
        colors=colors,
        outline_color=outline_color,
    )

    log.info(f"Saving output to {output_path}")
    plt.savefig(output_path, dpi=dpi)


if __name__ == "__main__":
    cli()
