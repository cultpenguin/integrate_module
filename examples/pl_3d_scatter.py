#%%
import pandas as pd
import numpy as np
import pyvista as pv
import matplotlib.colors as colors

#pv.start_xvfb()
pv.set_plot_theme("document")
pv.set_jupyter_backend('static')
#"static", "client", "server", "trame", "html", "none"

# Your existing code here

#%%
# Load the CSV file
df = pd.read_csv('POST_FANGEL_AVG_PRIOR_UNIFORM_NL_4-4_log-uniform_N100000_TX07_20230828_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_point.csv')
#df = pd.read_csv('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_4-4_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_Median.csv')
#df_std = pd.read_csv('POST_DAUGAARD_AVG_PRIOR_UNIFORM_NL_4-4_log-uniform_N100000_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu100000_aT1_M1_Std.csv')

Nmax = 10000000
if Nmax < len(df):
    df = df.sample(n=Nmax, random_state=42)  # Randomly sample 100,000 points

print(f"Loaded {len(df)} points")

data_string = 'Median'

# Filter Value
doFilterValue = False
if doFilterValue:
    d_min, d_max = 50, 100
    #d_min, d_max = .1, 1050
    #df =  df[(df[data_string] >= d_min) & (df[data_string] <= d_max)]
    df =  df[(df[data_string] <= d_min) | (df[data_string] >= d_max)]

# Filter by line 
doFilterLine = True
if doFilterLine:
    line_min = 99
    line_max = 1001
    df =  df[(df['LINE'] >= line_min) & (df['LINE'] <= line_max)]
    #df =  df[(df['LINE'] <= line_max)]


print(f"Loaded {len(df)} points")




# Extract X, Y, Z, and D columns
points = df[['X', 'Y', 'Z']].values
d_values = df[data_string].values


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
point_cloud[data_string] = d_values

# Transparancy
#opacity = 1 - 0.7 * (d_values - d_min) / (d_max - d_min)

# Create a plotter
plotter = pv.Plotter()

# Add the point cloud to the plotter, coloring by 'D' values
plotter.add_mesh(point_cloud, render_points_as_spheres=True, scalars=data_string, cmap='jet',
                  clim=[10, 350], 
                  log_scale=True,
                  point_size=6,                                
                  )
# Add a mesh plot of the surface/measurement elevation

# Print the range of the plot
print(f"Plot range: {plotter.bounds}")

# Set a nice camera position
plotter.camera_position = 'xz'
# tilt the camera slightly
plotter.camera.roll += 10
plotter.camera.elevation += 20
#plotter.camera.pitch += 30

# show axes
plotter.show_axes()

# Show the plot
plotter.show()

# %%
