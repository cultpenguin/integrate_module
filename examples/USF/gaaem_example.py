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
    # # # # # #%load_ext autoreload
    # # # # # #%autoreload 2
    pass

#%% 

import matplotlib;
import matplotlib.pyplot as plt;

import time;
import random;
from gatdaem1d import tdlib;
from gatdaem1d import TDAEMSystem;
from gatdaem1d import Earth;
from gatdaem1d import Geometry;
from gatdaem1d import Response;

import numpy as np
import integrate as ig 



#%% 


#%%
# A reference model
conductivity = [1/1200., 1/20.0]
#resistivity = [100, 10, 1000]
thickness    = [10]



data_lm = np.array([
    [1.09820e-05, 3.85493e-07, 3.00, 1, 9.61766e-06, 1.25398e-05],
    [1.34004e-05, 2.34489e-07, 3.00, 1, 1.17357e-05, 1.53013e-05],
    [1.63515e-05, 1.43604e-07, 3.00, 1, 1.43201e-05, 1.86710e-05],
    [1.99524e-05, 8.77662e-08, 3.01, 1, 1.74736e-05, 2.27827e-05],
    [2.43463e-05, 5.37711e-08, 3.02, 1, 2.13217e-05, 2.77999e-05],
    [2.97078e-05, 3.28685e-08, 3.06, 1, 2.60171e-05, 3.39220e-05]
])

# Load data as numpy array
data_hm = np.array([
    [2.247E-05, 2.397538E-07],
    [2.805E-05, 1.514288E-07],
    [3.501E-05, 9.496128E-08],
    [4.371E-05, 5.911992E-08],
    [5.456E-05, 3.655617E-08], 
    [6.811E-05, 2.245690E-08],
    [8.502E-05, 1.371055E-08],
    [1.061E-04, 8.321895E-09],
    [1.325E-04, 5.023145E-09],
    [1.654E-04, 3.015578E-09],
    [2.064E-04, 1.801749E-09],
    [2.577E-04, 1.071005E-09],
    [3.217E-04, 6.325993E-10],
    [4.015E-04, 3.713836E-10],
    [5.012E-04, 2.165452E-10],
    [6.257E-04, 1.252587E-10],
    [7.811E-04, 7.175696E-11],
    [9.750E-04, 4.063551E-11],
    [1.217E-03, 2.269755E-11],
    [1.519E-03, 1.248344E-11],
    [1.897E-03, 6.746935E-12],
    [2.367E-03, 3.575940E-12]
])

#t=data_lm[:,0]
#d=data_lm[:,1]




#%% GA-AEM forward
#Construct the AEM system class instance
#stmfile = "../../examples/SkyTEM-BHMAR-2009/stmfiles/Skytem-HM.stm";
#stmfile  = 'Skytem-LM.stm'
#stmfile  = 'Skytem-HM.stm'
#stmfile  = 'TX07_20230906_2x4_RC20-33-P-tTEM42_LM.stm'
#stmfile  = 'TX07_20230906_2x4_RC20-33-P-tTEM42_HM.stm'
stmfile = 'FINAL_sTEMprofiler_test_HM.stm'
t=data_hm[:,0]
d=data_hm[:,1]
stmfile = 'FINAL_sTEMprofiler_test_LM.stm'
t=data_lm[:,0]
d=data_lm[:,1]



S = TDAEMSystem(stmfile)

fig1 = plt.figure(1)
print('-- Waveform')
#S.waveform.print(); 
S.waveform_windows_plot(fig1)
plt.show()

#%%
#Set the conductivity and thicknesses
E = Earth(conductivity,thickness);
print('-- Model')
E.print();

#Set the system geometry
G = Geometry(tx_height=0,txrx_dx = -13, txrx_dz     = .1)
print('-- Geometry')
G.print();

#Do a forward model
fm = S.forwardmodel(G,E); 
print('-- Forward response')
fm.print();



#Plot the responses
plt.figure()
plt.loglog(S.windows.centre,-fm.SZ,'-k',linewidth=2,label='Forward model')
plt.loglog(t,d,'-r',linewidth=2,label='Data')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V/A)')
plt.legend()
plt.show()




# %%
GA = -fm.SZ 
AI = d

r = (GA - AI )/AI

plt.figure()
plt.plot(S.windows.centre, r, '-k', linewidth=2, label='Residual')
plt.ylim(-0.1, 0.1)
# %% Yse integrate
# forceconductivity to be 2d nuympy array
conductivity = np.atleast_2d(conductivity)
thickness = np.atleast_2d(thickness)
stmfiles= []
stmfiles.append('FINAL_sTEMprofiler_test_LM.stm')
stmfiles.append('FINAL_sTEMprofiler_test_HM.stm')
# GAAEM, need both STM files ANG Gemoetry information.. 
#
# WHEN READING a GEX FILE, one should get btoh STM FILES and a GEOMOETRY variable
# WHEN READING an USF FILE, one should get beoth STM FILES and a GEOMOETRY variable (for use with GA-AEM)


D = ig.forward_gaaem(conductivity, thickness=thickness, stmfiles=stmfiles, parallel=False)




## %%
