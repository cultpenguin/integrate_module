#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Synthetic Case Study example
# An example using inverting data obtained from synthetic reference model
#
# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # # # # #%load_ext autoreload
    # # # # # # # # # # #%autoreload 2
    pass

import integrate as ig
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
hardcopy=True


# %% [markdown]
# ## Create The reference model and data

# %%
# Create reference model

# select the type of reference model
case = 'wedge'
case = '3layer'

z_max = 60
#rho = [720,10,520]
dx=.2

# New feature: Use rho_1, rho_2, rho_3 arrays to define resistivity variations
# Example 1: Constant resistivity in each layer
# rho_1 = [120]  # Layer 1: constant 120 Ohm-m
# rho_2 = [10]   # Layer 2: constant 10 Ohm-m
# rho_3 = [120]  # Layer 3: constant 120 Ohm-m

# Example 2: Linear variation from left to right
# rho_1 = [120, 10]  # Layer 1: 120 (left) to 10 (right)
# rho_2 = [10, 80]   # Layer 2: 10 (left) to 80 (right)
# rho_3 = [120, 10]  # Layer 3: 120 (left) to 10 (right)

# Example 3: Three-point variation (left, middle, right)
rho_1 = [10, 200, 10]  # Layer 1: 120 -> 10 -> 120
rho_2 = [50, 200, 800]    # Layer 2: 10 -> 80 -> 10
rho_3 = [120, 120, 120] # Layer 3: constant 120

rho_min = min(rho_1 + rho_2 + rho_3)
rho_max = max(rho_1 + rho_2 + rho_3)
clim = [0.8*rho_min, 1.2*rho_max]
clim_log10 = np.log10(clim)  # log10 scale for plotting

if case.lower() == 'wedge':
    # Make Wedge MODEL - using new rho_1, rho_2, rho_3 interface
    M_ref, x_ref, z_ref, M_ref_lith = ig.synthetic_case(
        case='Wedge',
        wedge_angle=10,
        dx=dx,
        z_max=z_max,
        dz=.5,
        x_max=200,
        z1=15,
        rho_1=rho_1,
        rho_2=rho_2,
        rho_3=rho_3
    )
elif case.lower() == '3layer':
    # Make 3 layer MODEL - using new rho_1, rho_2, rho_3 interface
    M_ref, x_ref, z_ref, M_ref_lith = ig.synthetic_case(
        case='3layer',
        dx=dx,
        x_max=200,
        x_range=10,
        rho_1=rho_1,
        rho_2=rho_2,
        rho_3=rho_3
    )

# Create reference data
f_data_h5 = 'DATA_%s_%d.h5' % (case,z_max)    
thickness = np.diff(z_ref)
# Get an exampele of a GEX file
file_gex = ig.get_case_data(case='DAUGAARD', filelist=['TX07_20231016_2x4_RC20-33.gex'])[0]
D_ref = ig.forward_gaaem(C=1./M_ref, thickness=thickness, file_gex=file_gex)

# Initialize random number generator to sample from noise model!
rng = np.random.default_rng()
d_std = 0.06
d_std_base = 1e-12
D_std = d_std * D_ref + d_std_base
D_noise = rng.normal(0, D_std, D_ref.shape)
D_obs = D_ref + D_noise
#D_obs = D_ref

# Write to hdf5 file
# Add option to reomve existing file before writing!
f_data_h5 = ig.save_data_gaussian(D_obs, D_std = D_std, f_data_h5 = f_data_h5, id=1, showInfo=1)
#check_data(f_data_h5)

# %%
# Save reference model to hdf5 file
model_name = '%s_reference_model_RE%d_BE%d_N%d' % (case,100*d_std,np.log10(d_std_base),D_obs.shape[0])
f_ref_h5 = '%s.h5' % model_name
with h5py.File(f_ref_h5, 'w') as f_ref:
    f_ref.create_dataset('M_ref', data=M_ref)
    f_ref.create_dataset('M_ref_lith', data=M_ref_lith)
    f_ref.create_dataset('D_ref', data=D_ref)
    f_ref.create_dataset('D_std', data=D_std)
    f_ref.create_dataset('x_ref', data=x_ref)
    f_ref.create_dataset('z_ref', data=z_ref)
    f_ref.attrs['description'] = 'Reference model for %s case' % case   


# %%
# Plot the model and data
fig = plt.figure(figsize=(8, 10))
xx_ref, zz_ref = np.meshgrid(x_ref, z_ref)

# Plot resistivity model with log10 scale
plt.subplot(3,1,1)
im1 = plt.pcolor(xx_ref, zz_ref, np.log10(M_ref.T), cmap='jet')
plt.gca().invert_yaxis()
plt.axis('equal')
cbar1 = plt.colorbar(im1, label='log10(Resistivity [Ohm-m])')
# Add ticks at nice round numbers
log_rho_min = np.floor(np.log10(M_ref.min()))
log_rho_max = np.ceil(np.log10(M_ref.max()))
log_ticks = np.arange(log_rho_min, log_rho_max + 1, 0.5)
cbar1.set_ticks(log_ticks)
cbar1.set_ticklabels([f'{10**v:.0f}' if v == int(v) else f'{10**v:.1f}' for v in log_ticks])
plt.title('Reference Model - Resistivity (log10 scale)')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.clim(clim_log10)

# Plot lithology model (new feature!)
plt.subplot(3,1,2)
cmap_lith = plt.cm.get_cmap('tab10', 3)
im2 = plt.pcolor(xx_ref, zz_ref, M_ref_lith.T, cmap=cmap_lith, vmin=1, vmax=3)
plt.gca().invert_yaxis()
plt.axis('equal')
cbar2 = plt.colorbar(im2, label='Layer Number', ticks=[1, 2, 3])
cbar2.set_label('Layer Number')
plt.title('Reference Model - Lithology/Layer Numbers')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

# Plot reference data
plt.subplot(3,1,3)
plt.semilogy(x_ref, D_obs)
plt.title('Reference Data')
plt.xlabel('Distance (m)')
plt.ylabel('Response')
plt.grid(True, alpha=0.3)

plt.tight_layout()
if hardcopy:
    plt.savefig(f'{model_name}_ref.png', dpi=150)
plt.show()

ig.plot_data(f_data_h5)



# %% [markdown]
# ## Create prior model and data

# %%
N=1000000 # sample size 
RHO_dist='log-uniform'
#RHO_dist='uniform'
RHO_min=0.8*clim[0]
RHO_max=1.25*clim[1]
RHO_min=5
RHO_max=1000
NLAY_min=2
NLAY_max=3
NLAY_deg=3
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='chi2', z_max = z_max, 
                                    NLAY_deg=NLAY_deg,
                                    RHO_dist=RHO_dist, RHO_min=RHO_min, RHO_max=RHO_max, save_sparse=False, f_prior_h5 = 'PRIOR_%s_N%d.h5' % (model_name,N))

ig.plot_prior_stats(f_prior_h5)

# %%
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

# %%  plot prior and observed data to chech that the prior data span the same range as the observed data
ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy) 

# %% [markdown]
# ## Perform inversion

# %%
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, 
                                    parallel=parallel, 
                                    Ncpu=8,
                                    nr=2000,

                                    )

# %%
ig.plot_profile(f_post_h5, i1=0, i2=1000, hardcopy=hardcopy,  clim = clim)
ig.plot_profile(f_post_h5, i1=0, i2=1000, hardcopy=hardcopy,  im=2)

# %%
ig.plot_data_prior_post(f_post_h5, i_plot=0, hardcopy=hardcopy)


# %% [markdown]
# ## Compare reference model to posterior median

# %%
# Read 'M1/Median' from f_post_h5
with h5py.File(f_post_h5, 'r') as f_post:
    #M_mode = f_post['/M3/Mode'][:]
    M_median = f_post['/M1/Median'][:]
    M_mean = f_post['/M1/Mean'][:]
    M_mean = f_post['/M1/Mean'][:]
    M_std = f_post['/M1/Std'][:]

with h5py.File(f_prior_h5,'r') as f_prior:
    # REad 'x' feature from f_prior
    z =  f_prior['/M1'].attrs['x']

xx, zz = np.meshgrid(x_ref, z)

# Make a figure with two subplots, each with plt.pcolor(xx,zz,M_median.T) and, plt.pcolor(xx_ref,zz_ref,M_ref.T), and use the same colorbar and x.axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Use log10 scale for color limits
clim_comparison = np.log10([rho_min*0.8, rho_max*1.2])

# First subplot - ref model
c1 = ax1.pcolor(xx_ref, zz_ref, np.log10(M_ref.T), clim=clim_comparison, cmap='jet')
ax1.invert_yaxis()
#ax1.axis('equal')
cbar1 = fig.colorbar(c1, ax=ax1, label='log10(Resistivity [Ohm-m])')
# Format colorbar with actual resistivity values
log_ticks = np.linspace(clim_comparison[0], clim_comparison[1], 6)
cbar1.set_ticks(log_ticks)
cbar1.set_ticklabels([f'{10**v:.0f}' if 10**v >= 10 else f'{10**v:.1f}' for v in log_ticks])
ax1.set_title('Prior Reference %s Model (log10 scale)' % case)

# Second subplot - Median
c2 = ax2.pcolor(xx, zz, np.log10(M_median.T), clim=clim_comparison, cmap='jet')
ax2.invert_yaxis()
#ax2.axis('equal')
cbar2 = fig.colorbar(c2, ax=ax2, label='log10(Resistivity [Ohm-m])')
cbar2.set_ticks(log_ticks)
cbar2.set_ticklabels([f'{10**v:.0f}' if 10**v >= 10 else f'{10**v:.1f}' for v in log_ticks])
ax2.set_title('Posterior Median Model (log10 scale)')
# add a contour plot of xx_ref, zz_ref, M_ref.T on top of current figure
ax2.contour(xx_ref, zz_ref, M_ref.T, colors='k', linewidths=1)

# Third subplot - Std
c3 = ax3.pcolor(xx, zz, M_std.T, clim=[0,0.4], cmap='gray_r')
ax3.invert_yaxis()
#ax3.axis('equal')
cbar3 = fig.colorbar(c3, ax=ax3, label='Std Dev (linear)')
ax3.set_title('Posterior Std')
# add a contour plot of xx_ref, zz_ref, M_ref.T on top of current figure
ax3.contour(xx_ref, zz_ref, M_ref.T, colors='r', linewidths=.5)

# change aspect ratio of the figure to 2:1
ax1.set_aspect(.5)
ax2.set_aspect(.5)
ax3.set_aspect(.5)

plt.tight_layout()
plt.savefig('%s_z%d_rho%d-%d-%d_Ndeg%d_N%d' % (model_name,z_max, rho_1[0],rho_1[1],rho_1[2],NLAY_deg, N))
plt.show()

# %%
