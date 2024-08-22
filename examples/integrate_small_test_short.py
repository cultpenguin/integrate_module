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
N_use = 1000000
f_prior_h5='prior.h5'
f_data_h5='DAUGAARD_AVG_inout.h5'
Ncpu = 32
ip_range = []
#ip_range=np.arange(0,11000,10)   
f_post_h5 = 'post.h5'
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

#%%
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
plt.scatter(X,Y, c=E, s=5, cmap='jet', vmin=-100, vmax=0)
plt.colorbar()
#ig.plot_T_EV(f_post_h5, pl='EV')
#plt.plot(T,'.')
T[5000:5020]
#%%  TEST OLD
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5, 
                                N_use = N_use, 
                                parallel=1, 
                                updatePostStat=updatePostStat, 
                                showInfo=1,
                                Nproc = Ncpu)



#%%
ig.integrate_posterior_stats(f_post_h5)
ig.plot_profile(f_post_h5, i1=0, i2=2000, cmap='jet', hardcopy=hardcopy)



# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
# read Mode from M3 in f_post_h5
with h5py.File(f_post_h5, 'r') as f:
    M3_mode = f['/M3/Mode'][:]
    M3_entropy = f['/M3/Entropy'][:]
    M3_P = f['/M3/P'][:]
    M2_entropy = f['/M2/Entropy'][:]
    
#plt.scatter(X,Y, c=np.mean(M2_entropy, axis=1), s=1)
plt.scatter(X,Y, c=M3_P[:,1], s=1, vmin=0, vmax=1)
plt.colorbar()    



# %%
ig.plot_T_EV(f_post_h5, pl='T')
# %%
with h5py.File(f_post_h5, 'r') as f_post:
    EV = f_post['/EV'][:]

# %%

# %% direct computation of the likelihood
ip_range = []
ip_range=np.arange(100,1000)   
i_use, T, EV, ip_range = integrate_rejection_range(f_prior_h5='prior.h5', 
                                     f_data_h5='DAUGAARD_AVG_inout.h5', 
                                     N_use=4000, 
                                     id_use=[1,2],
                                     ip_range=ip_range,
                                     autoT=0,
                                     T_base = 1000,
                                     )

plt.figure()
plt.plot(EV,label='EV')
plt.plot(T,label='T')
plt.legend()
plt.show()

# %%
