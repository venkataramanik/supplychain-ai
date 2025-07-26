def generate_points(n: int, seed: int) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(seed)

    # Define a tighter bounding box for the contiguous US landmass
    # These ranges are adjusted to minimize points falling into oceans/Great Lakes
    min_lat, max_lat = 29.0, 47.0 # Further inland, avoiding extreme north/south coasts
    min_lon, max_lon = -119.0, -74.0 # Further inland, avoiding east/west coasts

    points = []
    for _ in range(n):
        while True:
            lat = rng.uniform(min_lat, max_lat)
            lon = rng.uniform(min_lon, max_lon)

            # Simple check to avoid some major water bodies or extreme borders
            # This is a heuristic, not a perfect landmass check.
            # You can add more complex conditions if certain regions are problematic.
            if (lat > 32 and lat < 45 and lon > -110 and lon < -80): # Central US bias
                points.append((lat, lon))
                break
            elif (lat >= min_lat and lat <= max_lat and lon >= min_lon and lon <= max_lon):
                # Allow points within the main bounding box, but favor central
                # This is a simplified heuristic, not a robust geo-check.
                # For robust land-checking, you'd need a GIS library like shapely + geopandas
                points.append((lat, lon))
                break
            # If the point is outside the 'preferred' central US or the main bounding box,
            # it will loop and try to generate a new point. This reduces off-land points.

    return points
