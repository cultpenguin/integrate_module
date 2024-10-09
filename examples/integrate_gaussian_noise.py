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
        # Create synthetic wedhge model
        
        # variables
        x_max = kwargs.get('x_max', 1000)
        dx = kwargs.get('dx', 1000./x_max)
        z_max = kwargs.get('z_max', 90)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/10)
        rho = kwargs.get('rho', [100,200,120])
        wedge_angle = kwargs.get('wedge_angle', 1)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

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

    elif case.lower() == '3layer':
        # Create synthetic 3 layer model

        # variables
        x_max = kwargs.get('x_max', 100)
        x_range = kwargs.get('x_range', x_max/4)
        dx = kwargs.get('dx', 1)
        z_max = kwargs.get('z_max', 90)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/10)
        z_thick = kwargs.get('z_thick', 20)
        

        rho1_1 = kwargs.get('rho1_1', 100)
        rho1_2 = kwargs.get('rho1_2', 2*rho1_1)
        rho2_1 = kwargs.get('rho1_2', 200)
        rho2_2 = kwargs.get('rho1_2', 0.5*rho2_1)
        rho3 = kwargs.get('rho3', 120)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho1_1
        iz = np.where(z>=z1)[0]
        M[:,iz] = rho3
        for ix in range(nx):
            rho1 = rho1_1 + (rho1_2 - rho1_1) * x[ix]/x_max
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            M[ix,:] = rho1
            z2 = z1 + z_thick*0.5*(1+np.cos(x[ix]/(x_range)*np.pi))
            iz = np.where((z>=z1) & (z<=z2))[0]
            #print(z[iz[0]])
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            M[ix,iz] = rho2

        return M, x, z

case = 'wedge'
case = '3layer'
z_max = 60
rho = [120,10,120]
if case.lower() == 'wedge':
    # Make Wedge MODEL
    M_ref, x_ref, z_ref = integrate_syntetic_case(case='Wedge', wedge_angle=10, z_max=z_max, dz=.5, x_max=100, dx=.1, z1=15, rho = rho)
elif case.lower() == '3layer':
    # Make 3 layer MODEL
    M_ref, x_ref, z_ref = integrate_syntetic_case(case='3layer', rho1_1 = rho[0], rho2_1 = rho[1], rho3=rho[2], x_max = 100, x_range = 10)
    M_ref, x_ref, z_ref = integrate_syntetic_case(case='3layer', dx=.1, z1 = 20, z_thick=30, z_max = 60)

# Make DATA
thickness = np.diff(z_ref)
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

#%% SAVE DATA --> Make function that saves data
d_std = 0.01
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

f_data_h5 = 'data_%s_n%d.h5' % (case,d_std*100)
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

#ig.plot_data(f_data_h5)
plt.semilogy(x_ref,D_obs);

#%% make prior
N=50*100000
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', z_max = z_max, 
                                    NLAY_min=3, NLAY_max=3, 
                                    RHO_dist='uniform', RHO_min=0.5*min(rho), RHO_max=2*max(rho))

ig.plot_prior_stats(f_prior_h5)

#%% MAKE PRIOR DATA
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

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
