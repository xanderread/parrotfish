import numpy as np
import pandas as pd
import torch.nn as nn

def calculate_temporal_weights(target_year, sample_year):
    """
    Calculate temporal weights based on the formula: YearWeights = 3 - (abs(Tc-Ti))
    Where Tc is the target year and Ti is the sample year
    """
    weight = 3 - abs(target_year - sample_year)
    return max(0, weight)  # Return 0 if weight would be negative

def idw_interpolation(points, values, grid_points, power=2, smoothing=0, 
                     search_radius=40, min_points=10, max_points=15,
                     temporal_weights=None):
    """
    Perform IDW interpolation with temporal weighting and debugging
    """
    
    # Calculate distances between all grid points and sample points
    distances = np.zeros((len(grid_points), len(points)))
    for i in range(len(grid_points)):
        distances[i] = np.sqrt(np.sum((points - grid_points[i])**2, axis=1))
    
    # Initialize results array
    results = np.zeros(len(grid_points))
    valid_points = 0
    
    # Process each grid point
    for i in range(len(grid_points)):
        # Find points within search radius
        mask = distances[i] <= search_radius
        if temporal_weights is not None:
            mask = mask & (temporal_weights > 0)
        
        local_distances = distances[i][mask]
        local_values = values[mask]
        
        if len(local_distances) < min_points:
            results[i] = np.nan
            continue
            
        valid_points += 1
        
        # Sort by distance and limit to max_points
        sort_idx = np.argsort(local_distances)[:max_points]
        local_distances = local_distances[sort_idx]
        local_values = local_values[sort_idx]
        if temporal_weights is not None:
            local_temporal_weights = temporal_weights[mask][sort_idx]
        
        # Calculate weights
        if np.any(local_distances == 0):
            results[i] = local_values[local_distances == 0][0]
        else:
            weights = 1 / (local_distances ** power + smoothing)
            if temporal_weights is not None:
                weights *= local_temporal_weights
            
            weights /= np.sum(weights)
            results[i] = np.sum(weights * local_values)
    
    return results

def create_distribution_surface(df, target_year, bounds, resolution=0.20):
    """
    Create distribution surface for a specific year with debugging
    """
    # Filter data for 2 years before and after target year
    year_range = range(target_year - 2, target_year + 3)
    df_temporal = df[df['Year'].isin(year_range)].copy()
    
    # print(f"\nDebugging create_distribution_surface:")
    # print(f"Target year: {target_year}")
    # print(f"Number of points in 5-year window: {len(df_temporal)}")
    
    if len(df_temporal) == 0:
        print("WARNING: No data points found in the temporal window!")
        return None, None, None
    
    # Calculate temporal weights
    df_temporal['temporal_weight'] = df_temporal['Year'].apply(
        lambda x: calculate_temporal_weights(target_year, x)
    )
    
    # Create grid
    x_grid = np.arange(bounds[0], bounds[1], resolution)
    y_grid = np.arange(bounds[2], bounds[3], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # print(f"Grid dimensions: {len(x_grid)} x {len(y_grid)}")
    # print(f"Number of grid points: {len(grid_points)}")
    
    # Prepare input data
    points = df_temporal[['LON', 'LAT']].values
    values = np.cbrt(df_temporal['wtcpue'].values)  # Cube root transform
    temporal_weights = df_temporal['temporal_weight'].values
    
    # Check if points are within bounds
    points_in_bounds = ((points[:, 0] >= bounds[0]) & 
                       (points[:, 0] <= bounds[1]) & 
                       (points[:, 1] >= bounds[2]) & 
                       (points[:, 1] <= bounds[3]))
    
    if np.sum(points_in_bounds) == 0:
        print("WARNING: No points found within specified bounds!")
        return None, None, None
    
    # Perform interpolation
    result = idw_interpolation(
        points, values, grid_points,
        power=2,
        search_radius=40,
        min_points=10,
        max_points=15,
        temporal_weights=temporal_weights
    )
    
    # Reshape result to grid
    result = result.reshape(len(y_grid), len(x_grid))
    
    return result, x_grid, y_grid

# Prepare data
def preprocess_data(distribution_surfaces, interpolate_days):
    """
    Preprocess distribution data for daily prediction.
    :param distribution_surfaces: List of annual grids (year x grid).
    :param interpolate_days: Number of days per year (365).
    :return: Scaled and interpolated data.
    """
    # Interpolate annual data to daily
    interpolated_data = []
    for i in range(len(distribution_surfaces) - 1):
        start = distribution_surfaces[i]
        end = distribution_surfaces[i + 1]
        interpolated = np.linspace(start, end, interpolate_days)
        interpolated_data.append(interpolated)
    return np.vstack(interpolated_data)

def fetch_latest_month(file, bounds):
    """
    Calculate the last two distribution surfaces (done annually). Perform linear interpolation and save the last 30 days.
    """
    df = pd.read_csv(file, skiprows=3)
    final_years = sorted(df['Year'].unique())
    final_year = final_years[-2:]
    final_df = df[df['Year'].isin(final_year)]

    annual_surfaces = []
    x_grid = []
    y_grid = []
    for year in final_years:
        surface, x_grid, y_grid = create_distribution_surface(df, year, bounds)
        annual_surfaces.append(surface)
    
    grid_size = annual_surfaces[0].shape
    # Linear interpolation between the two surfaces
    interpolated_data = preprocess_data([surface.flatten() for surface in annual_surfaces], 365)

    # Return the last 30 days
    return interpolated_data[-30:], grid_size
