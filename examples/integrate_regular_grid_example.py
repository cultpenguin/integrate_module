#!/usr/bin/env python
# %% [markdown]
# # Regular Grid Inversion Example
#
# This example demonstrates the use of INTEGRATE with data on a regular grid
# compared to the original irregular TEM measurement locations.
#
# The workflow:
# 1. Load TEM data at irregular measurement locations
# 2. Create a regular grid covering the survey area
# 3. Interpolate data onto the regular grid with distance-weighted uncertainties
# 4. Perform rejection sampling inversion on both parameterizations
# 5. Compare results via profile extraction
#

# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import h5py

# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True

# %% [markdown]
# ## 1. Load TEM Data at Irregular Locations
#
# Load the DAUGAARD dataset which contains TEM measurements at irregular
# locations along survey lines.

# %%
# Load case data
case = 'DAUGAARD'
files = ig.get_case_data(case=case, loadType='prior_data')
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)

print(f"Using data file: {f_data_h5}")
print(f"Using GEX file: {file_gex}")

# Extract geometry from irregular TEM data
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

print(f"\nOriginal irregular grid:")
print(f"  Number of points: {len(X)}")
print(f"  X range: [{X.min():.1f}, {X.max():.1f}] m")
print(f"  Y range: [{Y.min():.1f}, {Y.max():.1f}] m")

# Plot original survey geometry
ig.plot_geometry(f_data_h5, pl='LINE', hardcopy=hardcopy)
plt.title('Original TEM Survey (Irregular Grid)')
if hardcopy:
    plt.savefig('REGULAR_GRID_original_geometry.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. Create Regular Grid
#
# Construct a regular grid with specified spacing that covers the survey area.

# %%
# Grid parameters
grid_spacing = 10.0  # meters between grid nodes
buffer = 100.0       # buffer around data extent

# Create regular grid
x_min, x_max = X.min() - buffer, X.max() + buffer
y_min, y_max = Y.min() - buffer, Y.max() + buffer

Xg_1d = np.arange(x_min, x_max + grid_spacing, grid_spacing)
Yg_1d = np.arange(y_min, y_max + grid_spacing, grid_spacing)
Xg, Yg = np.meshgrid(Xg_1d, Yg_1d)

# Flatten for processing
Xg_flat = Xg.flatten()
Yg_flat = Yg.flatten()

print(f"\nRegular grid:")
print(f"  Grid spacing: {grid_spacing} m")
print(f"  Grid dimensions: {len(Yg_1d)} x {len(Xg_1d)}")
print(f"  Total grid nodes: {len(Xg_flat)}")

# Plot both grids
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, c='blue', s=1, alpha=0.5, label='TEM measurements')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Original Irregular Grid')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(Xg_flat, Yg_flat, c='red', s=10, marker='s', alpha=0.3, label='Regular grid')
plt.scatter(X, Y, c='blue', s=1, alpha=0.5, label='TEM measurements')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title(f'Regular Grid (spacing={grid_spacing}m)')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
if hardcopy:
    plt.savefig('REGULAR_GRID_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Interpolate Data onto Regular Grid
#
# Interpolate observed data and uncertainties from irregular TEM locations
# to the regular grid. Scale uncertainties based on distance to nearest
# measurement point.

# %%
# Load original data
DATA = ig.load_data(f_data_h5)
d_obs_orig = DATA['d_obs'][0]  # Shape: (n_points, n_channels)
d_std_orig = DATA['d_std'][0]

print(f"\nOriginal data shape:")
print(f"  d_obs: {d_obs_orig.shape}")
print(f"  d_std: {d_std_orig.shape}")

# Distance-based uncertainty scaling parameters
d_min = 40.0         # Distance below which std is unchanged
d_max = 80.0        # Distance above which std is scaled by max factor
std_gain_factor = 100.0  # Maximum scaling factor for distant points

# Build spatial tree for nearest neighbor search
tree = cKDTree(np.column_stack([X, Y]))

# Find distance to nearest measurement for each grid point
distances, nearest_idx = tree.query(np.column_stack([Xg_flat, Yg_flat]))

print(f"\nDistance statistics:")
print(f"  Min distance: {distances.min():.1f} m")
print(f"  Max distance: {distances.max():.1f} m")
print(f"  Mean distance: {distances.mean():.1f} m")

# Compute distance-based scaling factor
# Linear interpolation between d_min and d_max
distance_scale = np.ones_like(distances)
mask_mid = (distances > d_min) & (distances <= d_max)
mask_far = distances > d_max

distance_scale[mask_mid] = 1.0 + (std_gain_factor - 1.0) * (distances[mask_mid] - d_min) / (d_max - d_min)
distance_scale[mask_far] = std_gain_factor

print(f"\nUncertainty scaling:")
print(f"  Points within {d_min}m: {np.sum(distances <= d_min)} (scale = 1.0)")
print(f"  Points {d_min}-{d_max}m: {np.sum(mask_mid)} (scale = 1.0 to {std_gain_factor})")
print(f"  Points beyond {d_max}m: {np.sum(mask_far)} (scale = {std_gain_factor})")

# Interpolate ELEVATION using nearest neighbor
print(f"\nInterpolating ELEVATION (nearest neighbor)...")
ELEVATION_grid = ELEVATION[nearest_idx]
print(f"  ELEVATION range: [{ELEVATION_grid.min():.2f}, {ELEVATION_grid.max():.2f}] m")

# Create LINE indices based on X-coordinate grid position
# Each vertical column in the grid gets the same line number
X_unique = np.unique(Xg_1d)
LINE_grid = np.zeros(len(Xg_flat), dtype=int)
for i, x_val in enumerate(Xg_flat):
    # Find which X-grid line this point belongs to
    LINE_grid[i] = np.argmin(np.abs(X_unique - x_val))

print(f"  LINE indices: [{LINE_grid.min()}, {LINE_grid.max()}] ({len(X_unique)} unique lines)")

# Interpolate observed data (linear interpolation)
n_channels = d_obs_orig.shape[1]
d_obs_grid = np.zeros((len(Xg_flat), n_channels))
d_std_grid = np.zeros((len(Xg_flat), n_channels))

# Remove points with all NaN
valid_points = ~np.all(np.isnan(d_obs_orig), axis=1)
X_valid = X[valid_points]
Y_valid = Y[valid_points]
d_obs_valid = d_obs_orig[valid_points]
d_std_valid = d_std_orig[valid_points]

print(f"\nInterpolating {n_channels} data channels...")

for i_channel in range(n_channels):
    # Get valid data for this channel
    channel_valid = ~np.isnan(d_obs_valid[:, i_channel])

    if np.sum(channel_valid) > 3:  # Need at least 3 points for interpolation
        # Interpolate observed data
        d_obs_grid[:, i_channel] = griddata(
            (X_valid[channel_valid], Y_valid[channel_valid]),
            d_obs_valid[channel_valid, i_channel],
            (Xg_flat, Yg_flat),
            method='linear',
            fill_value=np.nan
        )

        # Interpolate base uncertainty
        d_std_base = griddata(
            (X_valid[channel_valid], Y_valid[channel_valid]),
            d_std_valid[channel_valid, i_channel],
            (Xg_flat, Yg_flat),
            method='linear',
            fill_value=np.nan
        )

        # Apply distance-based scaling
        d_std_grid[:, i_channel] = d_std_base * distance_scale
    else:
        d_obs_grid[:, i_channel] = np.nan
        d_std_grid[:, i_channel] = np.nan

print(f"✓ Interpolation complete")

# %% Plot distance scaling
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(Xg_flat, Yg_flat, c=distances, s=20, cmap='viridis')
plt.colorbar(scatter, label='Distance to nearest TEM point (m)')
plt.scatter(X, Y, c='red', s=1, alpha=0.5, label='TEM measurements')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Distance to Nearest Measurement')
plt.axis('equal')
plt.legend()

plt.subplot(1, 2, 2)
scatter = plt.scatter(Xg_flat, Yg_flat, c=distance_scale, s=2, cmap='hot_r', vmin=1, vmax=std_gain_factor)
plt.colorbar(scatter, label='Uncertainty scaling factor')
plt.scatter(X, Y, c='red', s=0.1, label='TEM measurements')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Uncertainty Scaling Factor')
plt.axis('equal')
plt.legend()

plt.tight_layout()
if hardcopy:
    plt.savefig('REGULAR_GRID_distance_scaling.png', dpi=300, bbox_inches='tight')
plt.show()

# %%  Only use data less d_max m from a measurement
i_use = np.ones((len(Xg_flat),1))
# set i_use = 0 when distances> d_max
i_use[distances > d_max] = 0

# %% [markdown]
# ## 4. Save Regular Grid Data
#
# Save the interpolated data to HDF5 format using integrate's write function.
# This includes geometry information (UTMX, UTMY, ELEVATION, LINE) in a single call.

# %%
f_data_regular_h5 = 'DATA_regular.h5'

# Write regular grid data with geometry (all in one call)
ig.write_data_gaussian(
    d_obs_grid,
    D_std=d_std_grid,
    f_data_h5=f_data_regular_h5,
    UTMX=Xg_flat,
    UTMY=Yg_flat,
    ELEVATION=ELEVATION_grid,  # Interpolated from nearest neighbor
    LINE=LINE_grid,            # X-coordinate index [0, 1, 2, ...]
    id=1,
    i_use = i_use,
    showInfo=1
)

print(f"\n✓ Regular grid data saved to: {f_data_regular_h5}")

# Visualize interpolated ELEVATION and LINE using plot_geometry
ig.plot_geometry(f_data_regular_h5, pl='ELEVATION', hardcopy=hardcopy)
plt.title('ELEVATION (Nearest Neighbor Interpolation)')
if hardcopy:
    plt.savefig('REGULAR_GRID_elevation.png', dpi=300, bbox_inches='tight')
plt.show()

ig.plot_geometry(f_data_regular_h5, pl='LINE', hardcopy=hardcopy)
plt.title(f'LINE Indices (X-coordinate based: 0 to {LINE_grid.max()})')
if hardcopy:
    plt.savefig('REGULAR_GRID_line.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare data coverage
ig.plot_data(f_data_h5, useLog=0, hardcopy=False)
plt.suptitle('Original Irregular Grid Data', y=1.02)
plt.show()

ig.plot_data(f_data_regular_h5, useLog=0, hardcopy=False)
plt.suptitle(f'Regular Grid Data (spacing={grid_spacing}m)', y=1.02)
plt.show()

# %% [markdown]
# ## 5. Setup Prior Model
#
# Create or load prior model for inversion. This will be used for both
# irregular and regular grid inversions.

# %%
# Use existing prior if available, otherwise generate new one
try:
    f_prior_data_h5 = 'daugaard_merged.h5'  # From case data
    print(f"Using existing prior+data file: {f_prior_data_h5}")
except:
    # Generate new prior
    N = 200000
    f_prior_h5 = ig.prior_model_layered(
        N=N,
        lay_dist='chi2',
        NLAY_deg=4,
        RHO_min=1,
        RHO_max=3000,
        f_prior_h5='PRIOR_regular_grid.h5'
    )
    print(f"Generated prior: {f_prior_h5}")

    # Compute prior data
    f_prior_data_h5 = ig.prior_data_gaaem(
        f_prior_h5,
        file_gex,
        doMakePriorCopy=False,
        parallel=parallel
    )

print(f"Using prior+data: {f_prior_data_h5}")

# %% [markdown]
# ## 6. Perform Rejection Sampling on Both Grids
#
# Run integrate_rejection on both irregular and regular grid data
# using the same prior model.

# %%
# Inversion parameters
N_use = 2000000  # Number of prior samples to use
nr = 1000         # Number of posterior samples per point
autoT = True     # Automatic temperature estimation

# Invert irregular grid data
print("\n" + "="*70)
print("INVERTING IRREGULAR GRID DATA")
print("="*70)

f_post_irregular_h5 = ig.integrate_rejection(
    f_prior_data_h5,
    f_data_h5,
    f_post_h5='POST_irregular.h5',
    N_use=N_use,
    nr=nr,
    autoT=autoT,
    parallel=parallel,
    showInfo=1,
    updatePostStat=True
)

print(f"\n✓ Irregular grid inversion complete: {f_post_irregular_h5}")

# Invert regular grid data
print("\n" + "="*70)
print("INVERTING REGULAR GRID DATA")
print("="*70)

f_post_regular_h5 = ig.integrate_rejection(
    f_prior_data_h5,
    f_data_regular_h5,
    f_post_h5='POST_regular.h5',
    N_use=N_use,
    nr=nr,
    autoT=autoT,
    parallel=parallel,
    showInfo=1,
    updatePostStat=True
)

print(f"\n✓ Regular grid inversion complete: {f_post_regular_h5}")

# %% [markdown]
# ## 7. Extract Profile for Comparison
#
# Define a profile line and extract indices from both grids for comparison.

# %%
# Define profile line endpoints (modify as needed for your area)
# Using example from Daugaard case
profile_start = [544000, 6174500]  # [X_start, Y_start]
profile_end = [543550, 6176500]    # [X_end, Y_end]

profile_start = [542800, 6175600]  # [X_start, Y_start]
profile_end =   [545500, 6175600]    # [X_end, Y_end]

y_pro  = (np.min(Yg) + np.max(Yg)) / 2
profile_start = [np.min(Xg), y_pro]
profile_end =   [np.max(Xg), y_pro]



buffer = 10.0  # Buffer distance for point selection
buffer = 5.0  # Buffer distance for point selection

# Extract profile from irregular grid
X_irr, Y_irr, _, _ = ig.get_geometry(f_post_irregular_h5)
Xl_irr = np.array([profile_start[0], profile_end[0]])
Yl_irr = np.array([profile_start[1], profile_end[1]])

indices_irr, distances_irr, _ = ig.find_points_along_line_segments(
    X_irr, Y_irr, Xl_irr, Yl_irr, tolerance=buffer
)

print(f"\nProfile extraction:")
print(f"  Profile start: [{profile_start[0]:.1f}, {profile_start[1]:.1f}]")
print(f"  Profile end: [{profile_end[0]:.1f}, {profile_end[1]:.1f}]")
print(f"  Irregular grid points on profile: {len(indices_irr)}")

# Extract profile from regular grid
X_reg, Y_reg, _, _ = ig.get_geometry(f_post_regular_h5)
Xl_reg = np.array([profile_start[0], profile_end[0]])
Yl_reg = np.array([profile_start[1], profile_end[1]])

indices_reg, distances_reg, _ = ig.find_points_along_line_segments(
    X_reg, Y_reg, Xl_reg, Yl_reg, tolerance=buffer
)

print(f"  Regular grid points on profile: {len(indices_reg)}")

# Plot profile locations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_irr, Y_irr, c='lightgray', s=1, alpha=0.5)
plt.plot(X_irr[indices_irr], Y_irr[indices_irr], 'b-', linewidth=2, label='Profile')
plt.plot([profile_start[0], profile_end[0]], [profile_start[1], profile_end[1]], 'r--', linewidth=1, label='Profile line')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Irregular Grid - Profile Location')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_reg, Y_reg, c='lightgray', s=10, alpha=0.3, marker='s')
plt.plot(X_reg[indices_reg], Y_reg[indices_reg], 'r-', linewidth=2, label='Profile')
plt.plot([profile_start[0], profile_end[0]], [profile_start[1], profile_end[1]], 'k--', linewidth=1, label='Profile line')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Regular Grid - Profile Location')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
if hardcopy:
    plt.savefig('REGULAR_GRID_profile_locations.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Plot Profile Comparison
#
# Display resistivity profiles from both parameterizations side by side.

# %%
# Get colormap and limits for resistivity
cmap, clim = ig.get_colormap_and_limits('resistivity')

# Plot irregular grid profile
print("\nPlotting irregular grid profile...")
ig.plot_profile(
    f_post_irregular_h5,
    im=2,
    ii=indices_irr,
    gap_threshold=50,
    xaxis='x',
    cmap=cmap,
    clim=clim,
    hardcopy=hardcopy
)
plt.suptitle('Resistivity Profile - Irregular Grid (TEM Locations)', y=0.98)
if hardcopy:
    plt.savefig('REGULAR_GRID_profile_irregular.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot regular grid profile
print("Plotting regular grid profile...")
ig.plot_profile(
    f_post_regular_h5,
    im=2,
    ii=indices_reg,
    gap_threshold=grid_spacing*1.5,
    xaxis='x',
    cmap=cmap,
    clim=clim,
    hardcopy=hardcopy
)
plt.suptitle(f'Resistivity Profile - Regular Grid ({grid_spacing}m spacing)', y=0.98)
if hardcopy:
    plt.savefig('REGULAR_GRID_profile_regular.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Compare Temperature
#
# Compare the temperature and evidence values between the two parameterizations.

# %%
# Plot temperature comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

with h5py.File(f_post_irregular_h5, 'r') as f:
    T_irr = f['/T'][:]
axes[0].scatter(X_irr, Y_irr, c=T_irr, s=10, cmap='hot')
axes[0].set_title('Temperature - Irregular Grid')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].axis('equal')
plt.colorbar(axes[0].collections[0], ax=axes[0], label='Temperature')

with h5py.File(f_post_regular_h5, 'r') as f:
    T_reg = f['/T'][:]
axes[1].scatter(X_reg, Y_reg, c=T_reg, s=10, cmap='hot', marker='s')
axes[1].set_title(f'Temperature - Regular Grid ({grid_spacing}m)')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Y (m)')
axes[1].axis('equal')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Temperature')

plt.tight_layout()
if hardcopy:
    plt.savefig('REGULAR_GRID_temperature_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTemperature statistics:")
print(f"  Irregular grid: mean={np.nanmean(T_irr):.2f}, median={np.nanmedian(T_irr):.2f}")
print(f"  Regular grid: mean={np.nanmean(T_reg):.2f}, median={np.nanmedian(T_reg):.2f}")

# %% [markdown]
# ## 10. Summary
#
# This example demonstrated:
# - Creating a regular grid from irregular TEM survey data
# - Interpolating data with distance-weighted uncertainty scaling
# - Performing rejection sampling inversion on both grids
# - Comparing results via profile extraction
#
# **Key observations:**
# - Regular grids provide uniform spatial coverage
# - Uncertainty scaling accounts for interpolation quality
# - Both approaches should yield similar results where data coverage is good
# - Regular grids may show smoother features due to interpolation

print("\n" + "="*70)
print("EXAMPLE COMPLETE")
print("="*70)
print("\nGenerated files:")
print(f"  Regular grid data: {f_data_regular_h5}")
print(f"  Irregular posterior: {f_post_irregular_h5}")
print(f"  Regular posterior: {f_post_regular_h5}")
print("\n✓ Regular grid inversion example completed successfully!")

# %%

f_post_arr = [f_post_regular_h5,f_post_irregular_h5]
for f_post_h5 in f_post_arr:
    ig.plot_feature_2d(f_post_h5,im=1,iz=20,key='Mean', s=18, uselog=1)
    plt.show()
    
for f_post_h5 in f_post_arr:
   
    ig.plot_feature_2d(f_post_h5,im=2,iz=10,key='Mode', s=18, uselog=0)
    plt.show()