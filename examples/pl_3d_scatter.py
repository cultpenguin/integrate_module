#%%
import pandas as pd
import numpy as np
import pyvista as pv
import matplotlib.colors as colors

#%%
# Load the CSV file

df = pd.read_csv('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_4-4_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_Median.csv')
#df_std = pd.read_csv('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_4-4_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_Std.csv')

Nmax = 10000000
if Nmax < len(df):
    df = df.sample(n=Nmax, random_state=42)  # Randomly sample 100,000 points

print(f"Loaded {len(df)} points")

# Filter
d_min, d_max = 10, 50
d_min, d_max = .1, 1050
df =  df[(df['D'] >= d_min) & (df['D'] <= d_max)]

print(f"Loaded {len(df)} points")


# Extract X, Y, Z, and D columns
points = df[['X', 'Y', 'Z']].values
d_values = df['D'].values


# scale 
z_scale_factor = 5  # Adjust this value to change the z-axis scaling
# ...
points[:, 2] *= z_scale_factor

print(f"Points shape: {points.shape}")
print(f"D values shape: {d_values.shape}")

# Create a PyVista PolyData object
point_cloud = pv.PolyData(points)
print(f"Created PolyData with {point_cloud.n_points} points")

# Add the D values as a scalar array to the PolyData
point_cloud['D'] = d_values

# Transparancy
#transparency = 1 - 0.7 * (d_values - d_min) / (d_max - d_min)

# Create a plotter
plotter = pv.Plotter()

# Add the point cloud to the plotter, coloring by 'D' values
plotter.add_mesh(point_cloud, render_points_as_spheres=True, scalars='D', cmap='jet',
                  clim=[10, 500], 
                  log_scale=True,
                  point_size=4,
                  )

# Print the range of the plot
print(f"Plot range: {plotter.bounds}")

# Set a nice camera position
plotter.camera_position = 'xy'

# Show the plot
plotter.show()

# %%
