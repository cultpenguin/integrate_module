#!/usr/bin/env python
# %% [markdown]
# # Weights based in distance and data similarity
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
    # # # # #%load_ext autoreload
    # # # # #%autoreload 2
    pass
# %%
import numpy as np
import integrate as ig
import matplotlib.pyplot as plt
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

# %% Get tTEM data from DAUGAARD
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
f_data_h5 = 'DAUGAARD_AVG.h5'
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)



# %%
# Load data and ploti
DATA = ig.load_data(f_data_h5)
d_obs = DATA['d_obs'][0]
d_std = DATA['d_std'][0]

# get geometry
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

# %% SELECT a specic data point


# %% 

i_ref = 6000-1
#i_ref = 1560

x_well = X[i_ref]
y_well = Y[i_ref]
r_dis = 1000
r_data = 4
w_combined, w_dis, w_data, i_use = ig.get_weight_from_position(f_data_h5, x_well, y_well, r_data=r_data, r_dis=r_dis, doPlot=True)   
   

#%%
r_data_arr = [1,2,5,10]
r_dis_arr = [3000,500,1000]
r_data_arr = [1,3,10]
r_dis_arr = [1000,400,200]


x0 =np.min(X) + 0.2*(np.max(X)-np.min(X))
y0 =np.min(Y)
y1 =np.max(Y)-50
y0 = y0+0.5*(y1-y0)

y_arr = np.arange(y0, y1, 25)

#i_ref_arr = np.arange(1000,1100)
#for i_ref in i_ref_arr: 
for r_data in r_data_arr:
    for r_dis in r_dis_arr:

        j=0
        for yy in y_arr:
            j=j+1
            dis = np.sqrt((X-x0)**2 + (Y-yy)**2)
            i_ref = np.argmin(dis)
            x_well = X[i_ref]
            y_well = Y[i_ref]

            print('i_ref=%d, x=%f, y=%f' % (i_ref, X[i_ref], Y[i_ref]))

            #plFile = 'weight_%04d_%d_%d' % (i_ref,r_data,r_dis)
            plFile = 'weight_%04d_%d_%d' % (j,r_data,r_dis)

            w_combined, w_dis, w_data, i_use = ig.get_weight_from_position(f_data_h5, x_well, y_well, r_data=r_data, r_dis=r_dis, doPlot=True, plFile=plFile)   


        # ffmpeg -framerate 30 -i weight_%04d_3_2000.png -c:v libx264 -pix_fmt yuv420p output.mp4
        # run ffmpeg command in current folder
        import os 
        try:
            cmd = 'ffmpeg -y -framerate 10 -i weight_%%04d_%d_%d.png -c:v libx264 -pix_fmt yuv420p weight_movie_%d_%d_n%d.mp4' % (r_data,r_dis,r_data,r_dis,j)
            print(cmd)
            os.system(cmd)    
        except:
            print('Error in ffmpeg command')
            pass

#%%
# select gates to use 
# find the number of data points for each gate that has non-nan values
n_not_nan = np.sum(~np.isnan(d_obs), axis=0)
n_not_nan_freq = n_not_nan/d_obs.shape[0]
# use the data for which n_not_nan_freq>0.8
i_use = np.where(n_not_nan_freq>0.8)[0]

# select gates to use, manually
i_use = [5,17,18,19,20]

d_ref = np.log10(d_obs[i_ref,i_use])
d_test = np.log10(d_obs[:,i_use])
dd = np.abs(d_test - d_ref)
sum_dd = np.sum(dd, axis=1)

w_data = np.exp(-1*sum_dd**2/r_data**2)

# COmpute the distance from d_ref to all other points
dis = np.sqrt((X-X[i_ref])**2 + (Y-Y[i_ref])**2)
w_dis = np.exp(-1*dis**2/r_dis**2)

w_combined = w_data * w_dis

fig, axs = plt.subplots(3, 1, figsize=(5, 10))

# Plot w_data
axs[0].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[0].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc0 = axs[0].scatter(X, Y, c=w_data, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc0, ax=axs[0])
axs[0].set_title('w_data')
axs[0].axis('equal')

# Plot w_dis
axs[1].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[1].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc1 = axs[1].scatter(X, Y, c=w_dis, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc1, ax=axs[1])
axs[1].set_title('w_dis')
axs[1].axis('equal')

# Plot w_combined
axs[2].plot(X, Y, '.', ms=1, zorder=1, color='lightgray')
axs[2].plot(X[i_ref], Y[i_ref], 'k.', ms=6, zorder=4)
sc2 = axs[2].scatter(X, Y, c=w_combined, cmap='hot_r', s=.4, zorder=3)
fig.colorbar(sc2, ax=axs[2])
axs[2].set_title('w_combined')
axs[2].axis('equal')

plt.tight_layout()

plt.savefig('Daugard_weights_%04d.png' % (i_ref))
plt.show()






# %%
