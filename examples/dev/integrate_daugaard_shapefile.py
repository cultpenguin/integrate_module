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

#%% Load the necessary libraries

import integrate as ig
import numpy as np
import matplotlib.pyplot as plt

#%% Get data from Daugaard including shapefiles
files = ig.get_case_data(case='DAUGAARD', loadType='shapefiles')
f_data_h5=files[0]
# GET X, Y from DAUGAARD_AVG.h5
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

#%%
import geopandas as gpd

# Read the shapefile and # Extract X and Y from gdf
gdf = gpd.read_file('Begravet dal.shp')
line_coords = gdf[gdf.geometry.type == 'LineString'].geometry.apply(lambda geom: list(geom.coords))
line1=np.array(line_coords[0])
line2=np.array(line_coords[1])
#%%

plt.figure(figsize=(10,8))
plt.scatter(X,Y,c=ELEVATION, s=2)
plt.axis('equal')
plt.grid()
plt.title('Elevation with buried valleys')

plt.axis('equal')
plt.colorbar()
plt.plot(line1[:,0],line1[:,1],'y-',linewidth=6)
plt.plot(line1[:,0],line1[:,1],'k-',linewidth=2)
plt.plot(line2[:,0],line2[:,1],'y-',linewidth=6)
plt.plot(line2[:,0],line2[:,1],'k-',linewidth=2)
plt.tight_layout()
#plt.savefig('P_hypothesis_N%d_flemming.png' % (N_use))


