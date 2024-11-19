#!/usr/bin/env python
# %% [markdown]
# # Merging multiple data files from the same survey (ESBJERG)

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
    # # #%load_ext autoreload
    # # #%autoreload 2
    pass
# %%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt
import h5py
# check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)

#%% 
#
# GEX filer passer til dage som følger:
# ”TX07_20230906_2x4_RC20-33.gex”: 20230921, 20230922, 20230925, 20230926
# “TX07_20231016_2x4_RC20-33.gex”: 20231026, 20231027
# “TX07_20231127_2x4x1_RC20_33.gex”: 20240109
# “TX07_20240125_2x4_RC20-33.gex”: 20240313

f_gex1 ='TX07_20230906_2x4_RC20-33.gex'
f_data1 = []
f_data1.append('20230921_AVG_export.h5')
f_data1.append('20230922_AVG_export.h5')
f_data1.append('20230925_AVG_export.h5')
f_data1.append('20230926_AVG_export.h5')

f_gex2 ='TX07_20231016_2x4_RC20-33.gex'
f_data2 = []
f_data2.append('20231026_AVG_export.h5')
f_data2.append('20231027_AVG_export.h5')

f_gex3 = 'TX07_20231127_2x4x1_RC20_33.gex'
f_data3 = []
f_data3.append('20240109_AVG_export.h5')

f_gex4 = 'TX07_20240125_2x4_RC20-33.gex'
f_data4 = []
f_data4.append('20240313_AVG_export.h5')

#%% 


ig.merge_data(f_data1, f_gex1)
ig.merge_data(f_data2, f_gex2)
ig.merge_data(f_data3, f_gex3)
ig.merge_data(f_data4, f_gex4)

#%% 
# Combine f_data1, f_data2, f_data3, f_data4 into a single list
f_data = f_data1 + f_data2 + f_data3 + f_data4
f_data_merged_h5 = ig.merge_data(f_data1+f_data2+f_data3, f_gex1, f_data_merged_h5='ESBJERG_ALL.h5')
f_data_merged4_h5 = ig.merge_data(f_data4, f_gex4, f_data_merged_h5='ESBJERG_4.h5')


#%%
Xc, Yc, LINEc, ELEVATIONc = ig.get_geometry(f_data_merged_h5)

#%% 

plt.figure()
plt.subplot(1, 3, 1)    
plt.scatter(Xc, Yc, c=ELEVATIONc, cmap='viridis', s=4)
plt.colorbar()
plt.subplot(1, 3, 2)    
plt.scatter(Xc, Yc, c=LINEc, cmap='viridis', s=4)
plt.colorbar()
plt.subplot(1, 3, 3)    
plt.plot(LINEc)
plt.show()

#%% Make a scatter plot of Xc,Yc with colors representing the LINEc and size representing the ELEVATIONc/10
plt.figure()
plt.scatter(Xc, Yc, c=LINEc, cmap='viridis', s=ELEVATIONc/20)
plt.colorbar()
plt.axis('equal')
plt.grid()
plt.show()

