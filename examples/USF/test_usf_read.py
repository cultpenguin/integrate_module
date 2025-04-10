#%%
import integrate as ig
import numpy as np
import matplotlib.pyplot as plt


# %%

file_path = "USF_Files_calc/L001_S005_2025_0108_094849.usf"
USF = ig.read_usf(file_path)

directory='USF_Files_calc'
directory='USF_Files_Unknown'
D_obs, D_rel_err, usf_list=ig.read_usf_mul(directory=directory)

plt.semilogy(D_obs)
plt.title('Observed Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (V)')
plt.grid()

f_data_h5 = ig.write_data_gaussian(D_obs=np.log10(D_obs), D_std = D_rel_err, id=1, is_log = 1, f_data_h5=directory+'.h5')
f_data_h5 = ig.write_data_gaussian(D_obs=D_obs, D_std=D_obs*D_rel_err/100, id=2, is_log = 2, f_data_h5=directory+'.h5')

# %%

stm_files= []
stm_files.append('FINAL_sTEMprofiler_test_HM.stm')
stm_files.append('FINAL_sTEMprofiler_test_LM.stm')


# %%
# %% A. CONSTRUCT PRIOR MODEL OR USE EXISTING
N=100
# Layered model
f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=4, RHO_min=1, RHO_max=3000)

# Plot some summary statistics of the prior model
ig.plot_prior_stats(f_prior_h5)

# %%
stmfiles= []
stmfiles.append('FINAL_sTEMprofiler_test_LM.stm')
stmfiles.append('FINAL_sTEMprofiler_test_HM.stm')

f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, stmfiles=stmfiles, showInfo=0, file_gex='g.gex')
# %%
