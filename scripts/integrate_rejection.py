"""
integrate_rejection.py

This script implements a localized rejections sampler provided HDF5 datasets. 

Usage: integrate_rejection.py [-h] [--autoT AUTOT] [--N_use N_USE] [--ns NS] [--parallel PARALLEL] [--updatePostStat 1]
                              [f_prior_h5] [f_data_h5]

Arguments:
    f_prior_h5: str, optional (default='DJURSLAND_P01_N0010000_NB-13_NR03_PRIOR.h5')
        The name of the HDF5 file containing the prior models and data.

    f_data_h5: str, optional (default='tTEM-Djursland.h5')
        The name of the HDF5 file containing the observational data.

    --autoT: int, optional (default=1)
        A flag indicating whether to automatically adjust the temperature parameter during the rejection sampling.

    --N_use: int, optional (default=1000000)
        The maximum number of data points to use from the prior data.

    --ns: int, optional (default=400)
        The number of samples to draw from the posterior distribution.

    --parallel: int, optional (default=1)
        A flag indicating whether to process the data in parallel using multiple processes.

    --updatePostStat: int, optional (default=1)
        A flag determining where poster statistics is computed [1] or not [0]


Output:
    A HDF5 file named "{f_prior_h5[:-9]}_POST_Nu{N_use}_aT{autoT}.h5" containing the indices realizations of the posterior.

"""

#%%

import h5py
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import argparse
from tqdm import tqdm

import integrate as ig

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def process(is_):
    import integrate as ig
    d_obs = h5py.File(f_data_h5, 'r')['/D1/d_obs'][is_,:]
    d_std = h5py.File(f_data_h5, 'r')['/D1/d_std'][is_,:]
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    logL = np.zeros(N_use)
    for i in range(N_use):
        dd = (d_sim[i,i_use] - d_obs)**2
        logL[i] = -.5*np.sum(dd/d_var)

    if autoT == 1:
        T = ig.logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    i_use, P_acc = ig.lu_post_sample_logl(logL, ns, T)
    EV=maxlogL + np.log(np.nansum(np.exp(logL-maxlogL))/len(logL))
    return i_use, T, EV, is_


#%%

# Simulate command-line arguments, when run from a notebook
if is_notebook():
    import sys
    sys.argv = ['', '']

# create the parser
parser = argparse.ArgumentParser(description='Process some inputs.')

# add arguments
parser.add_argument('f_prior_h5',  nargs='?', type=str, default='DJURSLAND_P01_N2000000_NB-13_NR03_PRIOR.h5', help='optional filename to process')
parser.add_argument('f_data_h5',  nargs='?', type=str, default='tTEM-Djursland.h5', help='optional filename to process')
parser.add_argument('--autoT', type=int, default=1, help='optional autoT argument')
parser.add_argument('--N_use', type=int, default=2000000, help='optional N_use argument')
parser.add_argument('--ns', type=int, default=400, help='optional ns argument')
parser.add_argument('--parallel', type=int, default=1, help='optional ns argument')
parser.add_argument('--updatePostStat', type=int, default=1, help='optional update posterior statistics')

# parse arguments
args = parser.parse_args()
f_prior_h5 = args.f_prior_h5
f_data_h5 = args.f_data_h5
autoT = args.autoT
N_use = args.N_use
ns = args.ns
parallel = args.parallel
updatePostStat = args.updatePostStat

print('Running: integrate_rejection.py %s %s --autoT %d --N_use %d --ns %d --parallel %d --updatePostStat %d' % (f_prior_h5,f_data_h5,autoT,N_use,ns,parallel,updatePostStat))

#%% Check that hdf5 files exists
import os.path
if not os.path.isfile(f_prior_h5):
    print('File %s does not exist' % f_prior_h5)
    exit()  
if not os.path.isfile(f_data_h5):
    print('File %s does not exist' % f_data_h5)
    exit()


#%% 

f_post_h5=False 

with h5py.File(f_data_h5, 'r') as f:
    d_obs = f['/D1/d_obs']
    nd = d_obs.shape[1]
    nsoundings = d_obs.shape[0]

data_str = '/D1'
with h5py.File(f_prior_h5, 'r') as f:
    d_sim = f[data_str]
    N = d_sim.shape[0]
N_use = min([N_use, N])

with h5py.File(f_prior_h5, 'r') as f:
    d_sim = f[data_str][:, :N_use]

if not f_post_h5:
    f_post_h5 = f"{f_prior_h5[:-9]}_POST_Nu{N_use}_aT{autoT}.h5"

print('nsoundings:%d, N_use:%d, nd:%d' % (nsoundings,N_use,nd))
print('Writing results to ',f_post_h5)

# remaining code...

date_start = str(datetime.now())
t_start = datetime.now()
i_use_all = np.zeros((ns,nsoundings), dtype=int)
POST_T = np.ones(nsoundings) * np.nan
POST_EV = np.ones(nsoundings) * np.nan


if parallel:
    # % PARALLEL
    if __name__ == "__main__":
        
        with Pool() as p:
            result_iterator = p.imap_unordered(process, range(nsoundings))
            results = list(tqdm(result_iterator, total=nsoundings))

        for res in results:
            i_use, T, EV, is_ = res
            POST_T[is_] = T
            POST_EV[is_] = EV
            i_use_all[:,is_] = i_use

else:
    # SEQUENTIAL
    for is_ in tqdm(range(nsoundings)):
        i_use, T, EV, i = process(is_)
        POST_T[is_] = T
        POST_EV[is_] = EV
        i_use_all[:,is_] = i_use
 
date_end = str(datetime.now())
t_end = datetime.now()
t_elapsed = (t_end - t_start).total_seconds()
t_per_sounding = t_elapsed / nsoundings

print('Average temperature, T=%3.1f ' % (np.nanmean(POST_T)))
print('Time elapsed: %5.1f s, for %d soundings' % (t_elapsed,nsoundings))
print('Time per sounding: %4.3f ms' % (t_per_sounding*1000))

with h5py.File(f_post_h5, 'w') as f:
    f.create_dataset('i_use', data=i_use_all.T)
    f.create_dataset('T', data=POST_T.T)
    f.create_dataset('EV', data=POST_EV.T)
    f.attrs['date_start'] = date_start
    f.attrs['date_end'] = date_end
    f.attrs['inv_time'] = t_elapsed
    f.attrs['f5_prior'] = f_prior_h5
    f.attrs['f5_data'] = f_data_h5
    f.attrs['N_use'] = N_use

#%% Update posterior statistics
if updatePostStat:
    ig.integrate_posterior_stats(f_post_h5)
        
# %%
