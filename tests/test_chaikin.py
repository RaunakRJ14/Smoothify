"""Tests for the core Chaikin corner-cutting algorithm."""

import numpy as np
from shapely.geometry import LineString, Polygon

from smoothify.smoothify_core import _chaikin_corner_cutting


class TestChaikinCornerCutting:
    """Test suite for _chaikin_corner_cutting function."""

    def test_simple_square_polygon(self):
        """Test smoothing a simple square polygon."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = _chaikin_corner_cutting(square, num_iterations=1)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # After 1 iteration, should have 2x vertices (minus 1 for closed ring)
        assert len(smoothed.exterior.coords) > len(square.exterior.coords)

    def test_simple_linestring(self):
        """Test smoothing a simple linestring."""
        line = LineString([(0, 0), (5, 5), (10, 0)])
        smoothed = _chaikin_corner_cutting(line, num_iterations=1)

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Should have more points after smoothing
        assert len(smoothed.coords) > len(line.coords)

    def test_endpoints_preserved_linestring(self):
        """Test that LineString endpoints are preserved."""
        line = LineString([(0, 0), (5, 5), (10, 0)])
        smoothed = _chaikin_corner_cutting(line, num_iterations=3)

        # Endpoints should remain the same
        assert smoothed.coords[0] == line.coords[0]
        assert smoothed.coords[-1] == line.coords[-1]

    def test_multiple_iterations(self):
        """Test that more iterations produce more vertices."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        smoothed_1 = _chaikin_corner_cutting(square, num_iterations=1)
        smoothed_2 = _chaikin_corner_cutting(square, num_iterations=2)
        smoothed_3 = _chaikin_corner_cutting(square, num_iterations=3)

        assert isinstance(smoothed_1, Polygon)
        assert isinstance(smoothed_2, Polygon)
        assert isinstance(smoothed_3, Polygon)

        # More iterations = more vertices
        assert len(smoothed_1.exterior.coords) < len(smoothed_2.exterior.coords)
        assert len(smoothed_2.exterior.coords) < len(smoothed_3.exterior.coords)

    def test_polygon_remains_closed(self):
        """Test that polygon remains closed after smoothing."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = _chaikin_corner_cutting(square, num_iterations=2)

        assert isinstance(smoothed, Polygon)

        # First and last coordinates should be identical (closed ring)
        assert smoothed.exterior.coords[0] == smoothed.exterior.coords[-1]

    def test_triangle_polygon(self):
        """Test smoothing a triangle."""
        triangle = Polygon([(0, 0), (10, 0), (5, 10)])
        smoothed = _chaikin_corner_cutting(triangle, num_iterations=2)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should smooth the sharp corners
        assert len(smoothed.exterior.coords) > len(triangle.exterior.coords)

    def test_complex_linestring(self):
        """Test smoothing a more complex linestring."""
        line = LineString([(0, 0), (2, 2), (4, 0), (6, 2), (8, 0), (10, 2)])
        smoothed = _chaikin_corner_cutting(line, num_iterations=2)

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Endpoints preserved
        np.testing.assert_array_almost_equal(
            smoothed.coords[0], line.coords[0], decimal=10
        )
        np.testing.assert_array_almost_equal(
            smoothed.coords[-1], line.coords[-1], decimal=10
        )
