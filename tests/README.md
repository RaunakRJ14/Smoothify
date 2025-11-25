# Smoothify Test Suite

Pytest test suite for Smoothify.

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=smoothify --cov-report=html

# Run specific test
uv run pytest tests/test_chaikin.py::TestChaikinCornerCutting::test_simple_square_polygon
```

## Test Files

- **test_chaikin.py** - Chaikin corner-cutting algorithm
- **test_smoothify_core.py** - Core utility functions
- **test_smoothify_api.py** - Main API
- **test_geometry_types.py** - Geometry type handling and edge cases
- **test_area_tolerance.py** - Area tolerance parameter tests
- **test_auto_segment_length.py** - Auto-detection of segment length
- **test_edge_cases_coverage.py** - Edge cases and error conditions
