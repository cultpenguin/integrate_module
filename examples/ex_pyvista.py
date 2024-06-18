# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
import integrate as ig
import h5py

# %%
f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12_Nu1000000_aT1.h5'
f_post_h5 = 'POST_Fra20200930_202001001_1_AVG_export_gotaelv2_N1000000_fraastad_ttem_Nh280_Nf12_Nu100000_aT1.h5'
with h5py.File(f_post_h5,'r') as f_post:
    f_prior_h5 = f_post['/'].attrs['f5_prior']
    f_data_h5 = f_post['/'].attrs['f5_data']
    D_mean =  f_post['/M1/Mean'][:]

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)


# %%
import numpy as np
import pyvista as pv

# Example input data
num_locations = len(X)

# Define a vertical scaling factor
vertical_scaling_factor = 1.0

# Create a plotter
plotter = pv.Plotter()
plotter.show_axes()
plotter.show_grid()

# Number of locations to show
n_show = 800

# Select every 10th location
selected_loc = np.arange(0, num_locations, int(num_locations / n_show))
selected_loc = np.arange(n_show)
# Loop through each selected location and plot the 1D log at each location
for i in selected_loc:
    x = X[i]
    y = Y[i]

    # Define the depths (z-axis) for the vertical logs, reversed order
    z_values = ELEVATION[i] - np.arange(1, 91)
    scaled_z_values = z_values * vertical_scaling_factor
    
    # Get the 1D log data at the current X-Y location
    log_values = np.log10(D_mean[i, :])
    
    # Create a line for the log values
    points = np.column_stack((np.full_like(scaled_z_values, x), np.full_like(scaled_z_values, y), scaled_z_values))
    lines = pv.lines_from_points(points)
    
    # Convert lines to tubes
    tubes = lines.tube(radius=5.0)  # Adjust radius as needed
    
    # Interpolate the log values along the tube points
    tube_points = tubes.points[:, 2][::-1]
    reversed_z_values = scaled_z_values[::-1]
    reversed_log_values = log_values[::-1]
    
    tube_log_values = np.interp(tube_points, reversed_z_values, reversed_log_values)
    
    # Add the interpolated log values as scalars to the tubes
    tubes.point_data['log_values'] = tube_log_values

    # Create an opacity array based on the log values
    opacity = np.ones_like(tube_log_values)
    opacity[tube_log_values  < 1] = 0.0  # Adjust transparency for values below 4

    # Add the tubes to the plotter with the opacity array
    clim = [1, 3.5]
    #plotter.add_mesh(tubes, scalars='log_values', cmap='jet', opacity=opacity, scalar_bar_args={'title': 'Log Values'}, clim=clim)

    # Add the tubes to the plotter
    plotter.add_mesh(tubes, scalars='log_values', cmap='jet', scalar_bar_args={'title': 'Log Values'}, clim=clim)

# Show the plot
plotter.show()
