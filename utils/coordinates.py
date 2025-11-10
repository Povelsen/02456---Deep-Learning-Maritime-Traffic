"""
Denmark Geographic Coordinates
=============================

Centralized coordinates for Denmark region to ensure consistency
across all sample data generation and visualization.
"""

DENMARK_BBOX = {
    'north': 58.0,
    'south': 54.5,
    'west': 7.5,
    'east': 15.5
}

DENMARK_COORDS = {
    'lat_min': 54.5,
    'lat_max': 58.0,
    'lon_min': 7.5,
    'lon_max': 15.5
}

def get_denmark_bbox():
    """Return Denmark bounding box as [north, west, south, east]"""
    return [DENMARK_BBOX['north'], DENMARK_BBOX['west'], 
            DENMARK_BBOX['south'], DENMARK_BBOX['east']]

def get_denmark_coordinate_ranges():
    """Return Denmark coordinate ranges for sample data generation"""
    return DENMARK_COORDS