#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Synthetic Wedge Study example
#
# %% Imports
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # # # #%load_ext autoreload
    # # # # # # # # # #%autoreload 2
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

# %% 
def integrate_syntetic_case(case='Wedge', **kwargs):

    showInfo = kwargs.get('showInfo', 0)

    if case.lower() == 'wedge':
        # Create synthetic 
        
        x_max = kwargs.get('x_max', 1000)
        rho = kwargs.get('rho', [100,200,120])
        wedge_angle = kwargs.get('wedge_angle', 1)
        dx = kwargs.get('dx', 1000./x_max)
        dz = kwargs.get('dz', 1)
        z_max = kwargs.get('z_max', 90)
        z1 = kwargs.get('z1', z_max/10)

        if showInfo>0:
            print('Creating synthetic case with wedge angle=%f' % ẃedge_angle)

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho[0]
        # set M=rho[3] of all iz> (z==z1)
        iz = np.where(z>=z1)[0]
        M[:,iz] = rho[2]
        for ix in range(nx):
            wedge_angle_rad = np.deg2rad(wedge_angle)
            z2 = z1 + x[ix]*np.tan(wedge_angle_rad)            
            #find iz where  (z>=z1) and (z<=z2)
            iz = np.where((z>=z1) & (z<=z2))[0]
            #print(z[iz[0]])
            M[ix,iz] = rho[1]

        return M, x, z



# Make Wedge MODEL
rho = [120,10,120]
M_ref, x_ref, z_ref = integrate_syntetic_case(case='Wedge', wedge_angle=10, z_max=60, dz=.5, x_max=100, dx=.1, z1=15, rho = rho)
thickness = np.diff(z_ref)

# Make Weghe DATA
file_gex = 'TX07_20231016_2x4_RC20-33.gex'
D_ref = ig.forward_gaaem(C=1./M_ref, thickness=thickness, file_gex=file_gex)

# Plot the model and data
# Compute xx and zz meshgrids properly formatted for use with imshow
plt.figure()
plt.subplot(2,1,1)
xx_ref, zz_ref = np.meshgrid(x_ref, z_ref)
plt.pcolor(xx_ref,zz_ref,M_ref.T)
plt.gca().invert_yaxis()
plt.axis('equal')
plt.subplot(2,1,2)
plt.semilogy(D_ref.T);

#%% SAVE DATA
d_std = 0.03
d_std_base = 1e-12
D_std = d_std * D_ref + d_std_base
rng = np.random.default_rng()
D_noise = rng.normal(0, D_std, D_ref.shape)
D_obs = D_ref + D_noise


UTMX = np.atleast_2d(x_ref).T
UTMY = UTMX*0
LINE = UTMX*0
ELEVATION = UTMX*0

id = 1

D_str = 'D%d' % id

f_data_h5 = 'data_wedge.h5'
with h5py.File(f_data_h5, 'w') as f:
    f.create_dataset('UTMX', data=UTMX) 
    f.create_dataset('UTMY', data=UTMY)
    f.create_dataset('LINE', data=LINE)
    f.create_dataset('ELEVATION', data=ELEVATION)
    f.create_dataset('/%s/d_obs' % D_str, data=D_obs)
    f.create_dataset('/%s/d_std' % D_str, data=D_std)
    f.create_dataset('/%s/d_ref' % D_str, data=D_ref)
    #f.create_dataset('/%s/Cd' % D_str, data=Cd)
    # wrote attribute noise_model
    f['/%s/' % D_str].attrs['noise_model'] = 'gaussian'
    f['/%s/' % D_str].attrs['is_log'] = 0

ig.plot_data(f_data_h5)

#%% make prior
N=100000
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', z_max = 60, 
                                    NLAY_min=3, NLAY_max=3, 
                                    RHO_dist='log-uniform', RHO_min=5, RHO_max=150)

ig.plot_prior_stats(f_prior_h5)

#%% MAKE PRIOR DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=0)

ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy) 

# %% INVERT 
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, parallel=parallel, Ncpu=8)

# %% Plot some stats
#ig.plot_T_EV(f_post_h5, pl='EV')
    
ig.plot_profile(f_post_h5, i1=0, i2=1000, hardcopy=hardcopy,  clim = [5, 220])

#%% 
ig.plot_data_prior_post(f_post_h5, i_plot=0, hardcopy=hardcopy)
ig.plot_data_prior_post(f_post_h5, i_plot=len(x_ref)-1, hardcopy=hardcopy)




# %%
# Read 'M1/Median' from f_post_h5
with h5py.File(f_post_h5, 'r') as f_post:
    M_median = f_post['/M1/Median'][:]
    
# %%
