#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE Daugaard Case Study with three eology-resistivity prior models.
#
# This notebook contains an example of inverison of the DAUGAARD tTEM data using three different geology-resistivity prior models

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
    # # # # # # # #%load_ext autoreload
    # # # # # # # #%autoreload 2
    pass

import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import h5py
hardcopy=True
import time


#%% The new version of integrate_rejection using multidata
updatePostStat =False
N_use = 15000
f_prior_h5='prior.h5'
f_prior_h5='prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N50000_Nh280_Nf12.h5'
f_prior_h5='prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N50000_Nh280_Nf12.h5'
f_prior_h5='prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N50000_Nh280_Nf12.h5'
f_prior_h5='prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
#f_data_h5='DAUGAARD_AVG_inout.h5'

# get numer of cpu's
import multiprocessing
Ncpu = multiprocessing.cpu_count()/2
Ncpu = 12
ip_range = []
#ip_range=np.arange(0,11000,10)   
f_post_h5 = 'post.h5'


#%% TEST NEW
t0=time.time()
T, E, i_use = ig.integrate_rejection_multi(f_post_h5=f_post_h5,
                            f_prior_h5=f_prior_h5, 
                            f_data_h5=f_data_h5, 
                            N_use=N_use, 
                            id_use=[1],
                            autoT=1,
                            ip_range=ip_range,
                            Ncpu=Ncpu,
                            updatePostStat=updatePostStat,
                            showInfo=1                                                        
                            )
t1=time.time()-t0

#%%  TEST OLD
t0=time.time()  
f_post_h5_old = ig.integrate_rejection(f_prior_h5, f_data_h5, 
                                N_use = N_use, 
                                parallel=1, 
                                updatePostStat=updatePostStat, 
                                showInfo=1,
                                Nproc = Ncpu)
t2=time.time()-t0

print('TIMIMG New:', t1, 'Old:', t2)


#%% 
ip=100
ig.plot_data_prior_post(f_post_h5, i_plot = ip)
ig.plot_data_prior_post(f_post_h5_old, i_plot = ip)




#%% 
ig.plot_T_EV(f_post_h5, hardcopy=hardcopy,  pl='T')
ig.plot_T_EV(f_post_h5_old, hardcopy=hardcopy,  pl='T')

#%%
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
with h5py.File(f_post_h5,'r') as f_post:
    T = f_post['T'][:]
    i_use = f_post['i_use'][:]
with h5py.File(f_post_h5_old,'r') as f_post:
    T_old = f_post['T'][:]
    i_use_old = f_post['i_use'][:]

plt.figure(figsize=(10,5))
plt.plot(T, 'o', label='New')
plt.plot(T_old, 'x', label='Old')
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.scatter(X,Y,c=T, cmap='jet',s=6, vmax=100)
plt.axis('equal')
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(X,Y,c=T-T_old, cmap='jet',s=2)
plt.axis('equal')
plt.colorbar()
plt.show()



#%%
'''
ig.integrate_posterior_stats(f_post_h5_old)
ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)
ig.plot_profile(f_post_h5_old, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)
'''
ig.integrate_posterior_stats(f_post_h5)
with h5py.File(f_post_h5,'r') as f_post:
    M3_P = f_post['M3/P'][:]
    M3_Mode = f_post['M3/Mode'][:]
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.scatter(X,Y,c=M3_P[:,0], cmap='hot_r',s=2, vmin=0, vmax=1)
plt.title('P(inside valley)')
plt.subplot(2,2,2)
plt.scatter(X,Y,c=M3_P[:,1], cmap='hot_r',s=2, vmin=0, vmax=1)
plt.title('P(outside valley)')
plt.colorbar()


plt.tight_layout()
# %%
