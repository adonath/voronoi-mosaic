import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner
from numpy.testing import assert_allclose

from voronoi_mosaic import (
    cells_to_collection,
    cli,
    colors_to_voronoi_image,
    get_colors,
    get_voronoi_tesselation,
    optimize_voronoi_cells,
    points_to_label_image,
    voronoi_centers_hex_grid,
    voronoi_centers_random,
)

# a non square shape, such that transposed x and y coordinates are caught
HEIGHT, WIDTH = 12, 30


def get_total_error(image, points):
    """Total squared color error of the mosaic defined by the given points."""
    labels = points_to_label_image(points, width=image.shape[1], height=image.shape[0])
    colors = get_colors(points, image)
    return np.square(colors_to_voronoi_image(colors, labels) - image).sum()


@pytest.fixture
def image_edge():
    """An image of two uniform halves, separated by a vertical edge."""
    image = np.zeros((HEIGHT, WIDTH, 3))
    image[:, WIDTH // 2 :] = 1.0
    return image


def test_hex_grid_lattice():
    centers = voronoi_centers_hex_grid(width=100, height=60, cell_size=10, jitter=0)

    x, y = centers[:, 0], centers[:, 1]
    rows = np.unique(y)

    # rows are spaced by the height of an equilateral triangle
    assert_allclose(np.diff(rows), 10 * np.sqrt(3) / 2)

    row_first, row_second = np.sort(x[y == rows[0]]), np.sort(x[y == rows[1]])

    assert_allclose(np.diff(row_first), 10)
    # every other row is offset by half the cell size
    assert_allclose(row_second - row_first, 5)


def test_hex_grid_inside_image():
    centers = voronoi_centers_hex_grid(width=100, height=60, cell_size=10, jitter=4)

    assert np.all(centers[:, 0] >= 0) and np.all(centers[:, 0] < 100)
    assert np.all(centers[:, 1] >= 0) and np.all(centers[:, 1] < 60)


def test_hex_grid_jitter_bounded():
    kwargs = {"width": 100, "height": 60, "cell_size": 10}

    centers = voronoi_centers_hex_grid(jitter=0, **kwargs)
    centers_jitter = voronoi_centers_hex_grid(
        jitter=4, random_state=np.random.RandomState(0), **kwargs
    )

    # each center is displaced by at most half the jitter value
    assert np.all(np.abs(centers_jitter - centers) <= 2)


def test_random_centers():
    image = np.zeros((HEIGHT, WIDTH))
    image[2, 25] = 1.0

    centers = voronoi_centers_random(
        image, n_points=16, random_state=np.random.RandomState(0)
    )

    assert centers.shape == (16, 2)
    # all centers are drawn from the single bright pixel, in (x, y) order
    assert np.all(np.abs(centers[:, 0] - 25) <= 0.5)
    assert np.all(np.abs(centers[:, 1] - 2) <= 0.5)


def test_get_colors():
    image = np.zeros((HEIGHT, WIDTH, 3))
    image[2, 25] = [1.0, 0.5, 0.25]

    colors = get_colors(np.array([[25.0, 2.0], [0.0, 0.0]]), image)

    assert_allclose(colors[0], [1.0, 0.5, 0.25])
    assert_allclose(colors[1], [0.0, 0.0, 0.0])


def test_get_colors_outside_image():
    image = np.zeros((HEIGHT, WIDTH, 3))
    image[HEIGHT - 1, WIDTH - 1] = 1.0

    colors = get_colors(np.array([[WIDTH + 10.0, HEIGHT + 10.0]]), image)

    assert_allclose(colors[0], [1.0, 1.0, 1.0])


def test_points_to_label_image():
    points = np.array([[1.0, 1.0], [WIDTH - 2.0, 1.0]])

    labels = points_to_label_image(points, width=WIDTH, height=HEIGHT)

    assert labels.shape == (HEIGHT, WIDTH)
    # labels are zero based indices into the points
    assert np.all(labels[:, 0] == 0)
    assert np.all(labels[:, -1] == 1)
    assert set(np.unique(labels)) == {0, 1}


def test_colors_to_voronoi_image():
    labels = np.array([[0, 1], [1, 0]])
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    image = colors_to_voronoi_image(values, labels)

    assert image.shape == (2, 2, 3)
    assert_allclose(image[0, 0], [1.0, 0.0, 0.0])
    assert_allclose(image[0, 1], [0.0, 1.0, 0.0])


def test_voronoi_tesselation_cells_are_bounded():
    random_state = np.random.RandomState(0)
    points = random_state.uniform(0, 50, size=(32, 2))

    vor = get_voronoi_tesselation(points)

    for idx in range(len(points)):
        region = vor.regions[vor.point_region[idx]]
        assert -1 not in region
        assert len(region) >= 3


def test_cells_to_collection_color_alignment():
    random_state = np.random.RandomState(0)
    points = random_state.uniform(0, 50, size=(16, 2))
    colors = random_state.uniform(0, 1, size=(16, 3))

    collection = cells_to_collection(
        get_voronoi_tesselation(points), colors, outline_color="black"
    )

    paths = collection.get_paths()
    face_colors = collection.get_facecolor()

    assert len(paths) == len(points)

    for point, color in zip(points, colors):
        contains = [idx for idx, path in enumerate(paths) if path.contains_point(point)]
        # each point lies in exactly its own cell, which carries its color
        assert len(contains) == 1
        assert_allclose(face_colors[contains[0]][:3], color)


def test_cells_to_collection_without_outline():
    points = np.array([[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]])
    colors = np.eye(3)

    collection = cells_to_collection(
        get_voronoi_tesselation(points), colors, outline_color="none"
    )

    # outlines take the face color, to avoid white seams between the cells
    assert_allclose(collection.get_edgecolor(), collection.get_facecolor())


def test_cells_to_collection_coincident_points():
    points = np.array([[10.0, 10.0], [10.0, 10.0], [30.0, 10.0], [20.0, 30.0]])

    collection = cells_to_collection(
        get_voronoi_tesselation(points), np.eye(4, 3), outline_color="black"
    )

    assert len(collection.get_paths()) == len(collection.get_facecolor())


@pytest.mark.parametrize("axis", [0, 1])
def test_optimize_moves_boundary_onto_image_edge(axis):
    size, edge = 40, 20

    image = np.zeros((size, size, 3))

    if axis == 0:
        image[:, edge:] = 1.0
    else:
        image[edge:, :] = 1.0

    points = np.full((2, 2), 20.0)
    points[0, axis], points[1, axis] = 5.0, 30.0

    optimize_voronoi_cells(image, points, niter=10)

    # the two cells meet halfway between their centers, which has to end up on
    # the edge of the image, so that neither cell straddles it
    assert_allclose(points[:, axis].mean(), edge - 0.5)
    assert get_total_error(image, points) == 0.0


def test_optimize_reduces_error(image_edge):
    points = voronoi_centers_hex_grid(
        width=WIDTH,
        height=HEIGHT,
        cell_size=4,
        jitter=2,
        random_state=np.random.RandomState(0),
    )

    error_start = get_total_error(image_edge, points)
    optimize_voronoi_cells(image_edge, points, niter=5)

    assert get_total_error(image_edge, points) < error_start


def test_optimize_keeps_points_inside_image(image_edge):
    points = np.array([[0.0, 0.0], [WIDTH - 1.0, HEIGHT - 1.0], [0.0, HEIGHT - 1.0]])

    optimize_voronoi_cells(image_edge, points, niter=5)

    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= WIDTH - 1)
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] <= HEIGHT - 1)


@pytest.mark.parametrize("init_method", ["hex-grid", "random"])
def test_cli(tmp_path, init_method):
    image = np.random.RandomState(0).uniform(0, 1, size=(HEIGHT, WIDTH, 3))

    path_input, path_output = tmp_path / "input.png", tmp_path / "mosaic.png"
    plt.imsave(path_input, image)

    result = CliRunner().invoke(
        cli,
        [
            str(path_input),
            "--init-method",
            init_method,
            "--cellsize",
            "4",
            "--npoints",
            "16",
            "--niter",
            "2",
            "--output-path",
            str(path_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert path_output.exists()
    # the mosaic keeps the size of the input image, up to rounding
    assert_allclose(plt.imread(path_output).shape[:2], (HEIGHT, WIDTH), atol=1)
