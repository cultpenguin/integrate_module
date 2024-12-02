#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE on ESBJERG data

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
    # # # # # # #%load_ext autoreload
    # # # # # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time

plt.ion()

# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True
# %% Get tTEM data from DAUGAARD
N=200000
case = 'ESBJERG'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

ig.plot_geometry(f_data_h5, pl='LINE')
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

f_data_new_h5 = 'DATA.h5'
os.system('cp %s %s' % (f_data_h5,f_data_new_h5))
f_data_h5= f_data_new_h5
ig.check_data(f_data_h5, showInfo=1)
ig.plot_data(f_data_h5)
plt.show()

# %% [markdown]
# ### SETUP INTEGRATE

# SET PRIOR
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=500)
#f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', NLAY_min=1, NLAY_max=8, RHO_min=1, RHO_max=500)
# From GEUS
#f_prior_h5 = 'prior_Esbjerg_claysand_N200000_dmax90'
#f_prior_h5 = 'prior_Esbjerg_piggy_N200000.h5'
filelist = ['prior_Esbjerg_claysand_N200000_dmax90.h5','prior_Esbjerg_piggy_N200000.h5']
geus_files = ig.get_case_data(case=case, filelist=filelist)
f_prior_h5 = geus_files[0]
ig.integrate_update_prior_attributes(f_prior_h5)
# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

# %% [markdown]
# ### 1b. Then, a corresponding sample of $\rho(\mathbf{d})$, will be generated

# %% Compute prior EM DATA 
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, parallel=parallel, showInfo=0, N=N)
f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=2, doMakePriorCopy=True)
ig.plot_data_prior(f_prior_data_h5,f_data_h5,nr=1000,alpha=1, ylim=[1e-13,1e-5], hardcopy=hardcopy)


# %% Well Observations
# create dict called well_obs, with well_obs[1]['name'] = 'W1'
well_obs = []
W1 = {'name': 'Roust01',
               'UTMX': 474659, 
               'UTMY': 6156777,
               'z_top':[0,9.7,10.8],
               'z_bot':[9.7,10.8,15],
               'lith':[1,2,1]}
#               'z_top':[0,9.7,15.8],
#               'z_bot':[9.7,15.8,25],
#               'lith':[2,1,2]}

W2 = {'name': 'Sakds01',
                'UTMX': 474093, 
                'UTMY': 6151995,
                'z_top':[0,2,2.5,3.2],
                'z_bot':[2,2.5,3.2,15],
                'lith':[2,1,2,1]}   

W3 = {'name': 'DGU nr. 121.993',
                'UTMX': 473769, 
                'UTMY': 6155215,
                'z_top':[0,18,21,23,26,32,100,103],
                'z_bot':[18,21,23,26,32,100,103,250],
                'lith':[1,2,1,2,1,2,1,2]}
well_obs.append(W1)
well_obs.append(W2)
well_obs.append(W3)

# %% 
makeMulTest=False
if makeMulTest:
    for r_data in [200,1000,50000]:
        for r_dis in [1, 4, 1000]:
        #for r_dis in [200,1000,50000]:
            for iw in range(len(well_obs)):
                x_well = well_obs[iw]['UTMX']
                y_well = well_obs[iw]['UTMY']            
                w_combined, w_dis, w_data, i_use = ig.get_weight_from_position(f_data_h5, x_well, y_well, r_data=r_data, r_dis=r_dis)
                w = w_combined
                i_use = np.where(w>0.001)[0]
                plt.figure()
                plt.plot(X,Y,'k.')
                plt.scatter(X[i_use], Y[i_use], c=w[i_use], cmap='jet', s=.1, zorder=3, vmin=0, vmax=1)
                plt.plot(well_obs[iw]['UTMX'],well_obs[iw]['UTMY'],'ro')
                plt.text(well_obs[iw]['UTMX'],well_obs[iw]['UTMY'],well_obs[iw]['name'], color='red')
                plt.title(well_obs[iw]['name'])
                plt.xlabel('UTMX')
                plt.ylabel('UTMY')
                plt.grid()
                plt.axis('equal')
                plt.colorbar()
                plt.savefig('well_%d_rdis%d_rdata%d' % (iw, r_dis, r_data))
#w_combined, w_dis, w_data = ig.get_weight_from_position(f_data_h5, x_well, y_well, r_data = 4, r_dis=200)

#%% 
# Setup 'observed data'
DATA = ig.load_data(f_data_h5)
id=0
d_obs = DATA['d_obs'][id]
nd = d_obs.shape[0]
#d_std = DATA['d_std'][id]

def P_distance_weight(P, P_prior, w=1):
    P_post = (1-w)*P_prior + w*P
    return P_post

showInfo = 0 
with h5py.File(f_prior_data_h5, 'r') as f:
    #print(f['M2'].attrs.keys())
    x = f['M2'].attrs['x']
    class_id = f['M2'].attrs['class_id']
nm = len(x)
nclass = len(class_id)

P_prior = np.ones((nclass,nm))/nclass

for iw in range(3): #len(well_obs)):
    x_well = well_obs[iw]['UTMX']
    y_well = well_obs[iw]['UTMY']   
    r_data = 250
    r_dis = 4
    w, w_dis, w_data, i_use = ig.get_weight_from_position(f_data_h5, x_well, y_well, -1, r_data, r_dis, doPlot=True)

    P_obs = P_prior.copy()
    # Get P_obs
    for i in range(len(well_obs[iw]['z_top'])):
        z_top = well_obs[iw]['z_top'][i]
        z_bot = well_obs[iw]['z_bot'][i]
        # find index of z_top<x<z_bot
        j = np.where( (x > z_top) & (x < z_bot) )[0]
        if showInfo>0:
            print("z_top: %f, z_bot: %f, j: %s" % (z_top, z_bot, j))
        lith = well_obs[iw]['lith'][i]
        # find index ic of class_id == lith
        ic = np.where(class_id == lith)[0][0]
        P1 = 0.9
        P_obs[:,j] = (1-P1)/(nclass-1)
        P_obs[ic,j] = P1
        
    # Compute Observed data weighed by distance
    d_obs = np.zeros((nd,nclass,nm))
    i_use = np.zeros((nd,1))
    for i in range(nd):
        if np.isnan(w[i]):
            w[i]=0
        if w[i]>0.2: #0.05:
            i_use[i]=1
        P_post = P_distance_weight(P_obs, P_prior, w[i])
        
        d_obs[i] = P_post

    print('Using %s of %d data' % (np.sum(i_use), nd))        
    #ig.write_data_multinomial(d_obs, f_data_h5=f_data_h5, i_use=i_use, id=iw+1+1)
    ig.write_data_multinomial(d_obs, f_data_h5=f_data_h5, i_use=i_use, id=iw+1+1, id_use=2)

ig.write_data_gaussian

plt.figure()
plt.plot(w)
plt.show()


#%% TEST load data
DATA = ig.load_data(f_data_h5, id_arr=[1,2,3,4])
print(DATA['noise_model'])
#print(DATA['i_use'])
print(DATA['id_use'])

#%% COMPUTE LIKELIOOF for tTEM data
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
dis = (well_obs[1]['UTMX']-X)**2 + (well_obs[1]['UTMY']-Y)**2
i_min_dis = np.argmin(dis)

D_all = ig.load_prior_data(f_prior_data_h5)[0]
D1 = D_all[0]
D2 = D_all[1]

D2[0][0:10]=2
D2[0][10:16]=1
D2[0][16:29]=2

DOBS1 = ig.load_data(f_data_h5, id_arr=[1])
DOBS2 = ig.load_data(f_data_h5, id_arr=[2])

# read d_obs and d_std from f_data_h5
i=0
ip=ip_range = [23]
ip=i_min_dis


d_obs1 = DOBS1['d_obs'][0][ip]
d_std1 = DOBS1['d_std'][0][ip]
logL1 = ig.likelihood_gaussian_diagonal(D1, d_obs1, d_std1)

d_obs2 = DOBS2['d_obs'][0][ip]
d_obs2 = np.squeeze(d_obs2)
logL2 = ig.likelihood_multinomial(D2, d_obs2, class_id)

print(logL1.shape)
print(logL2.shape)

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

print(logL2[0])


# %% INVERT AND PLOT
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, id_use=[2,3,4], showInfo=1, updatePostStat=False, ip_range=np.arange(100), parallel=False)

#%% TEST INVERSION
'''
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                f_data_h5, 
                                showInfo=1, 
                                Ncpu=8,
                                id_use=[2,3,4],
                                updatePostStat=True)

ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, i1=i_min_dis-400, i2=i_min_dis+400, im=2, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, i1=i_min_dis-400, i2=i_min_dis+400, im=2, hardcopy=hardcopy)
#ig.plot_profile(f_post_h5, im=2, i1=1, i2=5000, hardcopy=hardcopy)

with h5py.File(f_post_h5,'r') as f_post:
    T=f_post['/T'][:].T
    EV=f_post['/EV'][:].T
plt.figure()
plt.scatter(X,Y,c=EV, s=.1)
plt.axis('equal')
plt.colorbar()
plt.title('Evidence')
'''
# %% READY FOR INVERSION
id_use_arr = []
id_use_arr.append([1])
id_use_arr.append([2])
id_use_arr.append([3])
id_use_arr.append([4])
id_use_arr.append([2,3,4])
id_use_arr.append([1,2,3,4])

N_use = N
for i in range(len(id_use_arr)):
    
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                    f_data_h5, 
                                    showInfo=1, 
                                    Ncpu=8,
                                    id_use=id_use_arr[i])
    
    ig.plot_data_prior_post(f_post_h5, i_plot=100, hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
    ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
    ig.plot_profile(f_post_h5, i_plot=10000, i2=14000, im=1, hardcopy=hardcopy)
    ig.plot_profile(f_post_h5, i_plot=10000, i2=14000, im=2, hardcopy=hardcopy)

    ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=1,iz=20, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=1,iz=40, key='Median', uselog=1, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5,im=2,iz=10, key='Mode', uselog=0, cmap='jet', s=1, hardcopy=hardcopy)
    plt.show()
    