
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
# Check if parallel computations can be performed
parallel = ig.use_parallel(showInfo=1)
hardcopy = True 
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from load_data_XYZ import load_data_XYZ 
from load_data_XYZ_mod_dat import load_data_XYZ as load_data_XYZ_mod_dat
from load_data_XYZ_avg_export import load_data_XYZ as load_data_XYZ_avg_export

doPlot=True
    

# %%  load data
file_gex = '202500404_DEN_Diamond_Soeballe_mergedGates_SR.gex'
file_stm = '202500404_DEN_Diamond_Soeballe_mergedGates_SR.STM'

GENERAL, CHANNELS = ig.read_gex2(file_gex)
ig.describe_gex2(GENERAL, CHANNELS)

#%% AVG_EXPORT
useAVG = True
if useAVG:
    f_xyz=[]
    f_xyz.append('Pro20250401_1_AVG_export.xyz')
    f_xyz.append('Pro20250403_1_AVG_export.xyz')
    f_xyz.append('Pro20250407_1_AVG_export.xyz')
    D_list = []
    D_std_list = []
    ALT_list = []
    XYZ_list = []
    LINE_list = []

    for i in range(len(f_xyz)):
        print('Reading %s' % (f_xyz[i]))
        D, D_std, ALT, XYZ, LINE = load_data_XYZ_avg_export(f_xyz[i])
        D_list.append(D)
        D_std_list.append(D_std)
        ALT_list.append(ALT)
        XYZ_list.append(XYZ)
        LINE_list.append(LINE)
    # concatenate the list of numpy array in D_list to D
    D = np.concatenate(D_list, axis=0)
    #scale = 17.68*223
    scale = GENERAL['TxLoopArea']*CHANNELS[0]['TxApproximateCurrent'];
    D=D*scale
    # The STD is provided as percentage error. We convert into absolute error
    D_std_rel = np.concatenate(D_std_list, axis=0)
    D_std = D * D_std_rel

    ALT = np.atleast_2d(np.concatenate(ALT_list, axis=0))
    XYZ = np.atleast_2d(np.concatenate(XYZ_list, axis=0))
    LINE = np.atleast_2d(np.concatenate(LINE_list, axis=0))
    LINE = np.concatenate(LINE_list, axis=0)
    f_data_h5 = f_xyz[0].replace('.xyz', '.h5')
    f_data_h5 = 'DATA_avg.h5'
    ig.write_data_gaussian(D, D_std = D_std, f_data_h5=f_data_h5, 
                           file_gex=file_gex, showInfo=0, 
                           UTMX=XYZ[:,0], 
                           UTMY=XYZ[:,1],
                           ELEVATION=XYZ[:,2],
                           LINE=LINE,
                           name='Diamond Data',
                           delete_if_exist=True,                        
    )

    ig.write_data_gaussian(ALT.T, D_std = ALT.T*0+1, f_data_h5=f_data_h5, 
                        id=2,
                        showInfo=0, 
                        name='Altitude',
    )

    if doPlot:
        ig.plot_geometry(f_data_h5, pl='LINE')
        ig.plot_geometry(f_data_h5, pl='ELEVATION')
        ig.plot_data(f_data_h5, Dkey='D1', uselog=True)
        ig.plot_data(f_data_h5, Dkey='D2', uselog=False)

    DD = ig.load_data(f_data_h5)


#%% MOD.DAT
useMOD = False
if useMOD:
    f_xyz=[]
    f_xyz.append('SCI1_I01_MOD_dat.xyz')
    i=0;
    D, D_std_rel, ALT, XYZ, LINE = load_data_XYZ_mod_dat(f_xyz[i])
    D_std = D * (D_std_rel-1)
   

    # f_data_h5 should be the same as f_xyz[0], but with h5 extension
    f_data_h5 = f_xyz[i].replace('.xyz', '.h5')
    f_data_h5 = 'DATA_mod.h5'

    ig.write_data_gaussian(D, D_std = D_std, f_data_h5=f_data_h5,
                        id=1, 
                        file_gex=file_gex, showInfo=2, 
                        UTMX=XYZ[:,0], 
                        UTMY=XYZ[:,1],
                        ELEVATION=XYZ[:,2],
                        LINE=LINE,
                        name='Diamond Data',
                        delete_if_exist=True                        
    )

    ig.write_data_gaussian(ALT, D_std = ALT*0+1, f_data_h5=f_data_h5, 
                        id=2,
                        showInfo=2,
                        name='Altitude',
    )
    
    if doPlot:
        ig.plot_geometry(f_data_h5, pl='LINE')
        ig.plot_geometry(f_data_h5, pl='ELEVATION')
        ig.plot_data(f_data_h5, Dkey='D1', uselog=True)
        ig.plot_data(f_data_h5, Dkey='D2', uselog=False)
        
    DD = ig.load_data(f_data_h5)

    # %%
    if doPlot:
        ig.plot_data_xy(f_data_h5, Dkey='D1', cmap='jet', data_channel=10, uselog=True)
        #ig.plot_data_xy(f_data_h5, Dkey='D1', cmap='jet', data_channel=25, uselog=True)
        ig.plot_data_xy(f_data_h5, Dkey='D2', cmap='jet')

#%% 

# %%
#ig.plot_data(f_data_h5)  

# %%
# Select how many prior model realizations (N) should be generated
import integrate as ig
import numpy as np
N=10000
RHO_min=1
RHO_max=3000
f_prior_h5 = ig.prior_model_layered(
    N=N,
    lay_dist='uniform',         # Uniform distribution of layer numbers
    NLAY_min=2,                 # Minimum 3 layers
    NLAY_max=6,                 # Maximum 6 layers
    RHO_dist='log-uniform',     # Log-uniform resistivity distribution
    RHO_min=RHO_min,
    RHO_max=RHO_max,
    f_prior_h5='PRIOR_layered_uniform_log-uniform_N%d.h5' % N,
    showInfo=1
)


print('%s is used to hold prior realizations' % (f_prior_h5))

# Simulate N heights and update f_prior_h5 with these data
ALT_prior = np.random.uniform(low=10, high=50, size=(N,1))
ALT_prior = np.random.uniform(low=np.min(ALT.flatten()), high=np.max(ALT.flatten()), size=(N,1))

#ALT_prior = np.random.normal(loc=25, scale=5, size=(N,1))
im_height = ig.save_prior_model(f_prior_h5, ALT_prior, name='Altitude')
print('Altitude prior model saved in %s as /M%d' % (f_prior_h5,im_height))

ig.plot_prior_stats(f_prior_h5)

#ig.copy_hdf5_file(f_prior_h5, 'PRIOR_2.h5',showInfo=2)
#
M,idx = ig.load_prior_model(f_prior_h5)

M_ALT = M[im_height-1]

# %%
#f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)
from gatdaem1d import TDAEMSystem;
from gatdaem1d import Earth;
from gatdaem1d import Geometry;

rho = M[0][0]
cond = 1/rho
#thick=np.array([20,20])
thick=np.ones(cond.shape[0]-1)
stmfiles=[file_stm]


S = TDAEMSystem(stmfiles[0])
G = Geometry(tx_height=M_ALT[0], txrx_dx = -2.86, txrx_dz = -1.0)
G.print()
E = Earth(cond, thick)  # Create Earth model with the last defined conductivity and thickness
E.print()
fm = S.forwardmodel(G,E)  # Forward model response
d_sim =  -fm.SZ
#ig.forward_gaaem(C,thickness,stmfiles=stmfiles)

# %% 
d_sim_2 = ig.forward_gaaem(C=cond, 
                    thickness=thick, 
                    stmfiles=stmfiles,
                    file_gex=None,
                    tx_height=M_ALT,
                    txrx_dx = -2.86, 
                    txrx_dz = -1.0, showInfo = 11)


#%% 
print(GENERAL['RxCoilPosition'])
txrx_dx= GENERAL['RxCoilPosition'][0][0]
txrx_dy= GENERAL['RxCoilPosition'][0][1]
txrx_dz= -1*GENERAL['RxCoilPosition'][0][2]
print('txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (txrx_dx, txrx_dy, txrx_dz))
#%% 

f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, f_data_h5='DATA_sim.h5', 
                    im_height=im_height, 
                    stmfiles=stmfiles, 
                    showInfo=1, 
                    txrx_dx = txrx_dx,
                    txrx_dy = txrx_dy,
                    txrx_dz = txrx_dz,
                    parallel=False)

f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=im_height)



ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=1, id_data=1)
ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=2, id_data=2)

#%%
D_obs = ig.load_data(f_data_h5)

D_prior, idx = ig.load_prior_data(f_prior_data_h5)
d_sim_3 = D_prior[0]

#plt.plot(D_obs['d_obs'][1])
#plt.plot(D_prior[1],'r-')

#%% 

plt.semilogy(d_sim,'k-',linewidth=4)
plt.semilogy(d_sim_2[0],'r*')
plt.semilogy(d_sim_3[0],'g-',linewidth=1)
plt.show()

#%% %%%% INVERT
f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                   f_data_h5, 
                                   f_post_h5 = 'POST.h5', 
                                   showInfo=1, 
                                   parallel=parallel,
                                   id_use=[1,2])

# %%
ig.plot_profile(f_post_h5, i1=1, i2=4000, im=1, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, i1=1, i2=4000, im=4, hardcopy=hardcopy)
#%%
ig.plot_T_EV(f_post_h5, pl='LOGL_mean',hardcopy=hardcopy)

# %% 
try:
    ig.plot_feature_2d(f_post_h5,im=1,iz=30, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
    plt.show()
except:
    pass

# %%
