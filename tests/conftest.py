"""Pytest configuration and fixtures for Smoothify tests."""

import pytest
from shapely.geometry import LineString, Polygon


@pytest.fixture
def simple_square():
    """Fixture providing a simple square polygon."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def simple_triangle():
    """Fixture providing a simple triangle polygon."""
    return Polygon([(0, 0), (10, 0), (5, 10)])


@pytest.fixture
def simple_linestring():
    """Fixture providing a simple linestring."""
    return LineString([(0, 0), (5, 5), (10, 0)])


@pytest.fixture
def polygon_with_hole():
    """Fixture providing a polygon with a hole."""
    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
    return Polygon(exterior, [hole])


@pytest.fixture
def zigzag_linestring():
    """Fixture providing a zigzag linestring."""
    return LineString([(0, 0), (2, 2), (4, 0), (6, 2), (8, 0), (10, 2)])


@pytest.fixture
def large_square():
    """Fixture providing a large square polygon."""
    return Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
