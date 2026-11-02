"""mask.cell_shares and cell_bounds: the one overlap computation rasterise and momp.spatial share."""
import numpy as np
from shapely.geometry import Polygon, box

from momp.utils.mask import cell_bounds, cell_shares, rasterise

LAT = np.array([0.5, 1.5, 2.5])
LON = np.array([0.5, 1.5, 2.5, 3.5])


def test_shares_by_hand_and_outside_the_box_is_absent():
    rows, cols, share = cell_shares(box(0.7, 0.0, 2.3, 2.0), LAT, LON)
    # the window holds the cells that touch the shape's bounding box, the third row only along its edge
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [0, 1, 2]
    np.testing.assert_allclose(share, [[0.3, 1.0, 0.3]] * 2 + [[0, 0, 0]])
    assert cell_shares(box(10, 10, 11, 11), LAT, LON)[2].size == 0
    assert cell_shares(Polygon(), LAT, LON)[2].size == 0


def test_longitudes_are_wrapped_for_the_overlap():
    lon = np.array([358.5, 359.5, 360.5])
    rows, cols, share = cell_shares(box(-1.5, 0.0, 0.0, 1.0), np.array([0.5, 1.5]), lon)
    full = np.zeros((2, 3))
    full[np.ix_(rows, cols)] = share
    np.testing.assert_allclose(full, [[0.5, 1.0, 0.0], [0, 0, 0]])


def test_rasterise_overlap_is_a_threshold_of_the_shares():
    triangle = Polygon([(0, 0), (4, 0), (0, 3)])
    rows, cols, share = cell_shares(triangle, LAT, LON)
    full = np.zeros((3, 4))
    full[np.ix_(rows, cols)] = share
    for fraction in (0.2, 0.5, 1.0):
        np.testing.assert_array_equal(rasterise(triangle, LAT, LON, "overlap_fraction", fraction),
                                      full >= fraction - 1e-9)


def test_cell_bounds_of_a_grid_and_of_one_cell():
    lat_edges, lon_edges = cell_bounds(LAT, LON)
    assert lat_edges.tolist() == [0, 1, 2, 3] and lon_edges.tolist() == [0, 1, 2, 3, 4]
    lat_edges, lon_edges = cell_bounds(np.array([5.0]), np.array([1.0, 3.0]))
    assert lat_edges.tolist() == [4.0, 6.0] and lon_edges.tolist() == [0.0, 2.0, 4.0]
