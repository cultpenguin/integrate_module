#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE 21232123Daugaard Case Study -- Prior Hypothesis testing
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
#files = ig.get_case_data(case='DAUGAARD', loadAll=True) # Load data and prior+data realizations

f_prior_h5='prior.h5'
f_prior_h5='prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_N50000_Nh280_Nf12.h5'
f_prior_h5_in='prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5_out='prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'
f_prior_h5='prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5'

updatePostStat =False
N_use = 4000000
#N_use = 2000000
#N_use = 1000000
N_use = 500000
#N_use = 100000
#N_use = 10000
#N_use = 1000
f_data_h5='DAUGAARD_AVG_inout.h5'

# get numer of cpu's
import multiprocessing
Ncpu = multiprocessing.cpu_count()/2
Ncpu = 8
ip_range = []
#ip_range=np.arange(0,11000,10)   
f_post_h5 = 'post_inout_N%d.h5'% (N_use)
hardcopy = True


#%%
ig.plot_data_xy(f_data_h5, hardcopy=hardcopy)
ig.plot_data(f_data_h5, hardcopy=hardcopy, plType='plot')
ig.plot_data(f_data_h5, hardcopy=hardcopy, plType='imshow')

#%% TEST NEW
t0=time.time()
f_post_h5 = ig.integrate_rejection_multi(f_prior_h5, 
                            f_data_h5, 
                            N_use=N_use, 
                            id_use=[1],
                            autoT=1,
                            ip_range=ip_range,
                            Ncpu=Ncpu,
                            updatePostStat=updatePostStat,
                            )



t1=time.time()-t0
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)

#%% Plot probability of prior hypothesis
#ig.integrate_posterior_stats(f_post_h5)
import geopandas as gpd

# Read the shapefile and # Extract X and Y from gdf
gdf = gpd.read_file('Begravet dal.shp')
line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
line1=np.array(line_coords[0])
line2=np.array(line_coords[1])

X, Y, LINE, ELEVATION = ig.get_geometry(f_post_h5)
with h5py.File(f_post_h5,'r') as f_post:
    M3_P = f_post['M3/P'][:]
    M3_Mode = f_post['M3/Mode'][:]

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.scatter(X,Y,c=M3_P[:,0], cmap='seismic_r',s=2, vmin=0, vmax=1)
plt.axis('equal')
plt.grid()
plt.title('P(inside valley)')
plt.colorbar()
plt.subplot(2,2,2)
plt.scatter(X,Y,c=M3_P[:,1], cmap='seismic_r',s=2, vmin=0, vmax=1)
plt.grid()
plt.title('P(outside valley)')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.savefig('P_hypothesis_N%d.png' % (N_use))


plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.scatter(X,Y,c=M3_P[:,0], cmap='seismic_r',s=2, vmin=0, vmax=1)
plt.axis('equal')
plt.grid()
plt.title('P(inside valley)')
plt.colorbar()
plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
plt.plot(line1[:,0],line1[:,1],'k-',linewidth=2)
plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
plt.plot(line2[:,0],line2[:,1],'k-',linewidth=2)
plt.subplot(2,2,2)
plt.scatter(X,Y,c=M3_P[:,1], cmap='seismic_r',s=2, vmin=0, vmax=1)
plt.grid()
plt.title('P(outside valley)')
plt.axis('equal')
plt.colorbar()
plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
plt.plot(line1[:,0],line1[:,1],'k-',linewidth=2)
plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
plt.plot(line2[:,0],line2[:,1],'k-',linewidth=2)
plt.tight_layout()
plt.savefig('P_hypothesis_N%d_flemming.png' % (N_use))





#%%
try:
    import geopandas as gpd

    # Read the shapefile and # Extract X and Y from gdf
    gdf = gpd.read_file('Begravet dal.shp')
    line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
    line1=np.array(line_coords[0])
    line2=np.array(line_coords[1])

    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.scatter(X,Y,c=M3_P[:,0], cmap='seismic_r',s=2, vmin=0, vmax=1)
    plt.axis('equal')
    plt.grid()
    plt.title('P(inside valley)')
    plt.colorbar()
    # plot allthe locations in gdf as scatter poiints without removing the cirent fiogure
    plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
    plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
    plt.show()
except:
    pass

#%%
ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)

# %% Nowe
t=[]
f_post_list=[]
EV=[]
for f_prior_h5 in [f_prior_h5_in, f_prior_h5_out]:
    t0=time.time()
    f_post_h5_out = ig.integrate_rejection_multi(f_post_h5='%s_%s' % ('post',f_prior_h5),
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
    f_post_list.append(f_post_h5_out)
    t.append(time.time()-t0)
    ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)

    with h5py.File(f_post_h5_out,'r') as f_post:
        EVs = f_post['EV'][:]
        EV.append(EVs)
        

#%%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

nd=len(X)
nev=len(EV)

EV_mul = np.zeros((nev,nd))

for iev in range(len(EV)):
    # Read '/EV' from f_post_h5
    EV_mul[iev]=EV[iev]

#% Normalize EV

for T_EV in [1,2,4,8,16,32,256]:

    EV_P = 0*EV_mul
    E_max = np.max(EV_mul, axis=0)

    for iev in range(nev):
        EV_P[iev] = np.exp(EV_mul[iev]-E_max)

    # Use annealing to flaten prob    
    EV_P = EV_P**(1/T_EV)

    EV_P_sum = np.sum(EV_P,axis=0)
    for iev in range(nev):
        EV_P[iev] = EV_P[iev]/EV_P_sum

    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.scatter(X, Y, c=EV_P[0], cmap='seismic_r', s=3, vmin=0, vmax=1)
    plt.tight_layout()
    plt.axis('equal')
    plt.grid()
    plt.colorbar()
    plt.title('In Valleys')
    plt.subplot(2,2,2)
    plt.scatter(X, Y, c=EV_P[1], cmap='seismic_r', s=3, vmin=0, vmax=1)
    plt.tight_layout()
    plt.axis('equal')
    plt.grid()
    plt.colorbar()
    plt.title('Out of valleys')

    plt.savefig('P_hypothesis_mulrun_EV%d_N%d.png' % (T_EV,N_use))

    try:

        import geopandas as gpd

        # Read the shapefile and # Extract X and Y from gdf
        gdf = gpd.read_file('Begravet dal.shp')
        line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
        line1=np.array(line_coords[0])
        line2=np.array(line_coords[1])

        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.scatter(X, Y, c=EV_P[0], cmap='seismic_r', s=3, vmin=0, vmax=1)
        plt.tight_layout()
        plt.axis('equal')
        plt.grid()
        plt.colorbar()
        plt.title('In Valleys')
        plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
        plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
        plt.subplot(2,2,2)
        plt.scatter(X, Y, c=EV_P[1], cmap='seismic_r', s=3, vmin=0, vmax=1)
        plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
        plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
        plt.axis('equal')
        plt.grid()
        plt.colorbar()
        plt.tight_layout()
        plt.title('Out of valleys')

        plt.show()
        plt.savefig('P_hypothesis_mulrun_EV%d_N%d_flemming.png' % (T_EV,N_use))

    except:
        pass
# %% Plot profiles
for i in range(len(f_post_list)):
    ig.integrate_posterior_stats(f_post_list[i], usePrior=True)
    ig.plot_profile(f_post_list[i], i1=0, i2=2000, hardcopy=hardcopy)
    ig.integrate_posterior_stats(f_post_list[i], usePrior=False)
    ig.plot_profile(f_post_list[i], i1=0, i2=2000, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_list[i], i_plot = 310, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_list[i], i_plot = 760, hardcopy=hardcopy)


# %%
