import h5py
import numpy as np
import os.path
import subprocess
from sys import exit
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Pool
from multiprocessing import shared_memory
from functools import partial
import time
    
def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or IPython shell.

    Returns:
        bool: True if running in a Jupyter notebook or IPython shell, False otherwise.
    """
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


def use_parallel(**kwargs):
    import os
    showInfo = kwargs.get('showInfo', 0)
    
    parallel = True
    if is_notebook():
        # Then it is always OK to use parallel processing
        if showInfo>0:
            print('Notebook detected. Parallel processing is OK')
        parallel = True
    
    else:
        # if os is Linux, when default is spawn, then parallel processing is OK
        if os.name == 'posix':
            if showInfo>0:
                print('Posix system detected. Parallel processing is OK')        
            parallel = True
        else:
            if showInfo>0:
                print('Non posix system detected. Parallel processing is not OK')        
                print('If parallel processing is needed, make sure to embed you primary script in a :if __main__ == "__main__": block')        
            parallel = False

    return parallel

    


def logl_T_est(logL, N_above=10, P_acc_lev=0.2):
    """
    Estimate a temperature (T_est) based on a given logarithmic likelihood (logL), 
    a number (N_above), and an acceptance level (P_acc_lev).

    :param logL: An array of logarithmic likelihoods.
    :type logL: numpy.ndarray
    :param N_above: The number of elements above which to consider in the sorted logL array. Default is 10.
    :type N_above: int, optional
    :param P_acc_lev: The acceptance level for the calculation. Default is 0.2.
    :type P_acc_lev: float, optional
    :return: The estimated temperature. It's either a positive number or infinity.
    :rtype: float

    note: The function sorts the logL array in ascending order after normalizing the data by subtracting the maximum value from each element.
    It then removes any NaN values from the sorted array.
    If the sorted array is not empty, it calculates T_est based on the N_above+1th last element in the sorted array and the natural logarithm of P_acc_lev.
    If the sorted array is empty, it sets T_est to infinity.
    """
    sorted_logL = np.sort(logL - np.nanmax(logL))
    sorted_logL = sorted_logL[~np.isnan(sorted_logL)]
    
    if sorted_logL.size > 0:
        logL_lev = sorted_logL[-N_above-1]
        T_est = logL_lev / np.log(P_acc_lev)
        T_est = np.nanmax([1, T_est])
    else:
        T_est = np.inf

    return T_est


def lu_post_sample_logl(logL, ns=1, T=1):
    """
    Perform LU post-sampling log-likelihood calculation.

    :param logL: Array of log-likelihood values.
    :type logL: array-like
    :param ns: Number of samples to generate. Defaults to 1.
    :type ns: int, optional
    :param T: Temperature parameter. Defaults to 1.
    :type T: float, optional

    :return: A tuple containing the generated samples and the acceptance probabilities.
    :rtype: tuple
        - i_use_all (numpy darray): Array of indices of the selected samples.
        - P_acc (numpy array): Array of acceptance probabilities.
    """

    N = len(logL)
    P_acc = np.exp((1/T) * (logL - np.nanmax(logL)))
    P_acc[np.isnan(P_acc)] = 0

    Cum_P = np.cumsum(P_acc)
    Cum_P = Cum_P / np.nanmax(Cum_P)
    dp = 1 / N
    p = np.array([i * dp for i in range(1, N+1)])

    i_use_all = np.zeros(ns, dtype=int)
    for is_ in range(ns):
        r = np.random.rand()
        i_use = np.where(Cum_P > r)[0][0]
        i_use_all[is_] = i_use
    
    return i_use_all, P_acc

def integrate_update_prior_attributes(f_prior_h5, **kwargs):
    """
    Update the 'is_discrete' attribute of datasets in an HDF5 file.

    This function iterates over all datasets in the provided HDF5 file. 
    If a dataset's name starts with 'M', the function checks if the dataset 
    has an 'is_discrete' attribute. If not, it checks if the dataset appears 
    to represent discrete data by sampling the first 1000 elements and checking 
    how many unique values there are. If there are fewer than 20 unique values, 
    it sets 'is_discrete' to 1; otherwise, it sets 'is_discrete' to 0. 
    The 'is_discrete' attribute is then added to the dataset.

    :param f_prior_h5: The path to the HDF5 file to process.
    :type f_prior_h5: str
    """
    
    showInfo = kwargs.get('showInfo', 0)
    
    # Check that hdf5 files exists
    if not os.path.isfile(f_prior_h5):
        print('File %s does not exist' % f_prior_h5)
        exit()  

    with h5py.File(f_prior_h5, 'a') as f:  # open file in append mode
        for name, dataset in f.items():
            if showInfo>0:
                print("integrate_update_prior_attributes: Checking %s" % (name))
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'x' in dataset.attrs:
                    pass
                else:
                    if 'z' in dataset.attrs:
                        dataset.attrs['x'] = dataset.attrs['z']
                    else:
                        if 'M1' in f.keys():
                            if 'x' in f['/M1'].attrs.keys():
                                f[name].attrs['x'] = f['/M1'].attrs['x']
                                print('Setting %s/x = /M1/x ' % name)
                            else:
                                print('No x attribute found in %s' % name)    
                
                if 'is_discrete' in dataset.attrs:
                    if (showInfo>0):
                        print('%s: %s.is_discrete=%d' % (f_prior_h5,name,dataset.attrs['is_discrete']))
                else:
                    # Check if M is discrete
                    M_sample = dataset[:1000]  # get the first 1000 elements
                    class_id = np.unique(M_sample)
                    print(class_id)
                    if len(class_id) < 20:
                        is_discrete = 1
                        dataset.attrs['class_id'] = class_id
                        ## convert class_id to an array of strings and save it as an attribute if the attribute does not 
                        ## already exist
                        if 'class_name' not in dataset.attrs:
                            dataset.attrs['class_name'] = np.array([str(x) for x in class_id])
                        
                    else:
                        is_discrete = 0

                    if (showInfo>0):
                        print(f'Setting is_discrete={is_discrete}, for {name}')
                    dataset.attrs['is_discrete'] = is_discrete

                if dataset.attrs['is_discrete']==1:
                    if not ('class_id' in dataset.attrs):
                        M_sample = dataset[:1000]  # get the first 1000 elements
                        class_id = np.unique(M_sample)
                        dataset.attrs['class_id'] = class_id
                    if not ('class_name' in dataset.attrs):
                        # Convert class_id to an array of strings and save it as an attribute if the attribute does not
                        class_id = dataset.attrs['class_id']
                        dataset.attrs['class_name'] = [str(x) for x in class_id]


def integrate_posterior_stats(f_post_h5='POST.h5', **kwargs):
    """
    Compute posterior statistics for datasets in an HDF5 file.

    This function computes various statistics for datasets in an HDF5 file based on the posterior samples.
    The statistics include mean, median, standard deviation for continuous datasets, and mode, entropy, and class probabilities for discrete datasets.
    The computed statistics are stored in the same HDF5 file.

    :param f_post_h5: The path to the HDF5 file to process.
    :type f_post_h5: str
    :param usePrior: Flag indicating whether to use the prior samples. Default is False.
    :type usePrior: bool
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    """
    import h5py
    import numpy as np
    import integrate
    import scipy as sp
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if showInfo<0:
        disableTqdm=True
    else:
        disableTqdm=False
    usePrior = kwargs.get('usePrior', False)

    #f_post_h5='DJURSLAND_P01_N0100000_NB-13_NR03_POST_Nu50000_aT1.h5'
    # Check if f_prior_h5 attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_prior' in f.attrs:
            f_prior_h5 = f.attrs['f5_prior']
        else:
            raise ValueError(f"'f5_prior' attribute does not exist in {f_post_h5}")

    integrate.integrate_update_prior_attributes(f_prior_h5, **kwargs)


    # Load 'i_use' data from the HDF5 file
    try:
        with h5py.File(f_post_h5, 'r') as f:
            i_use = f['i_use'][:]
    except KeyError:
        print(f"Could not read 'i_use' from {f_post_h5}")
        #return
    
    if usePrior:
        with h5py.File(f_prior_h5, 'r') as f_prior:
            N = f_prior['/M1'].shape[0]
            nr=i_use.shape[1]
            nd=i_use.shape[0]
            # compute i_use of  (nd,nr), with random integer numbers between 0 and N-1
            i_use = np.random.randint(0, N, (nd,nr))
    
    # Process each dataset in f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior, h5py.File(f_post_h5, 'a') as f_post:
        for name, dataset in f_prior.items():
                
            if name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 0:
                if showInfo>0:
                    print('%s: CONTINUOUS' % name)

                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                m_post = np.zeros((nm, nr))

                M_logmean = np.zeros((nsounding,nm))
                M_mean = np.zeros((nsounding,nm))
                M_std = np.zeros((nsounding,nm))
                M_median = np.zeros((nsounding,nm))

                if showInfo>0:
                    print('nm=%d, nsounding=%d, nr=%d' % (nm, nsounding, nr))
                    print('M_mean.shape=%s' % str(M_mean.shape))

                # Create datasets
                for stat in ['Mean', 'Median', 'Std','LogMean']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))

                #if dataset.size <= 1e6:  # arbitrary threshold for loading all data into memory
                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1, disable=disableTqdm, desc='poststat'):
                    ir = np.int64(i_use[iid,:])
                    m_post = M_all[ir,:]

                    m_logmean = np.exp(np.mean(np.log(m_post), axis=0))
                    m_mean = np.mean(m_post, axis=0)
                    m_median = np.median(m_post, axis=0)
                    m_std = np.std(np.log10(m_post), axis=0)

                    M_logmean[iid,:] = m_logmean
                    M_mean[iid,:] = m_mean
                    M_median[iid,:] = m_median
                    M_std[iid,:] = m_std

                f_post['/%s/%s' % (name,'LogMean')][:] = M_logmean
                f_post['/%s/%s' % (name,'Mean')][:] = M_mean
                f_post['/%s/%s' % (name,'Median')][:] = M_median
                f_post['/%s/%s' % (name,'Std')][:] = M_std

            elif name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 1:
                
                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                nsounding, nr = i_use.shape
                nm = dataset.shape[1]
                # Get number of classes for name    
                class_id = f_prior[name].attrs['class_id']                
                n_classes = len(class_id)
                
                if showInfo>0:
                    print('%s: DISCRETE, N_classes =%d' % (name,n_classes))    

                M_mode = np.zeros((nsounding,nm))
                M_entropy = np.zeros((nsounding,nm))
                M_P= np.zeros((nsounding,n_classes,nm))

                # Create datasets in h5 file
                for stat in ['Mode', 'Entropy']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))
                for stat in ['Mode', 'P']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,n_classes,nm))

                M_all = dataset[:]

                for iid in tqdm(range(nsounding), mininterval=1, disable=disableTqdm, desc='poststat'):

                    # Get the indices of the rows to use
                    ir = np.int64(i_use[iid,:])
                    
                    m_post = M_all[ir,:]
                    
                    # Compute the class probability
                    n_count = np.zeros((n_classes,nm))
                    for ic in range(n_classes):
                        n_count[ic,:]=np.sum(class_id[ic]==m_post, axis=0)/nr    
                    M_P[iid,:,:] = n_count

                    # Compute the mode
                    M_mode[iid,:] = class_id[np.argmax(n_count, axis=0)]

                    # Compute the entropy
                    M_entropy[iid,:]=sp.stats.entropy(n_count, base=n_classes)

                f_post['/%s/%s' % (name,'Mode')][:] = M_mode
                f_post['/%s/%s' % (name,'Entropy')][:] = M_entropy
                f_post['/%s/%s' % (name,'P')][:] = M_P


            else: 
                if (showInfo>0):
                    print('%s: NOT RECOGNIZED' % name.upper())
                
            
                

    return None


def sample_from_posterior(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
    r"""
    Sample from the posterior distribution.

    Parameters:
    - is\_ (int): Index of data f_data_h5.
    - d_sim (ndarray): Simulated data.
    - f_data_h5 (str): Filepath of the data file (default: 'tTEM-Djursland.h5').
    - N_use (int): Number of samples to use (default: 1000000).
    - autoT (int): Flag indicating whether to estimate temperature (default: 1).
    - ns (int): Number of samples to draw from the posterior (default: 400).

    Returns:
    - i_use (ndarray): Indices of the samples used.
    - T (float): Temperature.
    - EV (float): Expected value.
    - is\_ (int): Index of the posterior sample.
    """
    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs'][is_,:]
        d_std = f['/D1/d_std'][is_,:]
    
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    dd = (d_sim[:, i_use] - d_obs)**2
    #logL = -.5*np.sum(dd/d_var, axis=1)
    logL = np.sum(-0.5 * dd / d_var, axis=1)

    # Compute the annealing temperature
    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    
    # Compute the evidence
    exp_logL = np.exp(logL - maxlogL)
    EV = maxlogL + np.log(np.nansum(exp_logL)/len(logL))
    return i_use, T, EV, is_




#def sample_from_posterior_chunk(is_,d_sim,f_data_h5, N_use,autoT,ns):
#    return sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns) 

#%% integrate_prior_data: updates PRIOR strutcure with DATA
def prior_data(f_prior_in_h5, f_forward_h5, id=1, im=1, doMakePriorCopy=0, parallel=True):
    # Check if at least two inputs are provided
    if f_prior_in_h5 is None or f_forward_h5 is None:
        print(f'{__name__}: Use at least two inputs to')
        help(__name__)
        return ''

    # Open HDF5 files
    with h5py.File(f_forward_h5, 'r') as f:
        # Check type=='TDEM'
        if 'type' in f.attrs:
            data_type = f.attrs['type']
        else:
            data_type = 'TDEM'

    f_prior_h5 = ''
    if data_type.lower() == 'tdem':
        # TDEM
        with h5py.File(f_forward_h5, 'r') as f:
            if 'method' in f.attrs:
                method = f.attrs['method']
            else:
                print(f'{__name__}: "TDEM/{method}" not supported')
                return

        if method.lower() == 'ga-aem':
            f_prior_h5, id, im = integrate_prior_data_gaaem(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
        else:
            print(f'{__name__}: "TDEM/{method}" not supported')
            return
    elif data_type.lower() == 'identity':
        f_prior_h5, id, im = integrate_prior_data_identity(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
    else:
        print(f'{__name__}: "{data_type}" not supported')
        return

    # update prior data with an attribute defining the prior
    with h5py.File(f_prior_h5, 'a') as f:
        f.attrs[f'/D{id}'] = 'f5_forward'

    return f_prior_h5


'''
Forward simulation
'''

def forward_gaaem(C=np.array(()), thickness=np.array(()), GEX={}, file_gex='', stmfiles=[], showtime=False, **kwargs):
    """
    Perform forward modeling using the **GAAEM** method.

    :param C: Conductivity array, defaults to np.array(())
    :type C: numpy.ndarray, optional
    :param thickness: Thickness array, defaults to np.array(())
    :type thickness: numpy.ndarray, optional
    :param GEX: GEX dictionary, defaults to {}
    :type GEX: dict, optional
    :param file_gex: Path to GEX file, defaults to ''
    :type file_gex: str, optional
    :param stmfiles: List of STM files, defaults to []
    :type stmfiles: list, optional
    :param showtime: Flag to display execution time, defaults to False
    """
    
    from gatdaem1d import Earth;
    from gatdaem1d import Geometry;
    # Next should probably only be loaded if the DLL is not allready loaded!!!
    from gatdaem1d import TDAEMSystem; # loads the DLL!!
    import integrate as ig
    import time 
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=False

    doCompress = kwargs.get('doCompress', True)

    if (len(stmfiles)>0) and (file_gex != '') and (len(GEX)==0):
        # GEX FILE and STM FILES
        if (showInfo)>0:
            print('Using submitted GEX file (%s)' % (file_gex))
        GEX =   ig.read_gex(file_gex)
    elif (len(stmfiles)==0) and (file_gex != '') and (len(GEX)==0):
        # ONLY GEX FILE
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (len(stmfiles)>0) and (file_gex == '') and (len(GEX)>0):
        # Using GEX dict and STM FILES
        a = 1
    elif (len(GEX)>0) and (len(stmfiles)>1):
        # using the GEX file in stmfiles
        print('Using submitted GEX and STM files')
    elif (len(GEX)>0) and (len(stmfiles)==0):
        # using GEX file and writing STM files
        print('Using submitted GEX and writing STM files')
        stmfiles = ig.write_stm_files(GEX, **kwargs)
    elif (len(GEX)==0) and (len(stmfiles)>1):
        if (file_gex == ''):
            print('Error: file_gex not provided')
            return -1
        else:
            print('Converting STM files to GEX')
            GEX =   ig.read_gex(file_gex)
    elif (len(GEX)>0) and (len(stmfiles)==0):
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (file_gex != ''):
        a=1
        #stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    else:   
        print('Error: No GEX or STM files provided')
        return -1

    if (showInfo>0):
        print('Using GEX file: ', GEX['filename'])

    nstm=len(stmfiles)
    if (showInfo>0):
        for i in range(len(stmfiles)):
            print('Using MOMENT:', stmfiles[i])

    if C.ndim==1:
        nd=1
        nl=C.shape[0]
    else:
        nd,nl=C.shape

    nt = thickness.shape[0]
    if nt != (nl-1):
        raise ValueError('Error: thickness array (nt=%d) does not match the number of layers minus 1(nl=%d)' % (nt,nl))

    if (showInfo>0):
        print('nd=%s, nl=%d,  nstm=%d' %(nd,nl,nstm))

    # SETTING UP t1=time.time()
    t1=time.time()
    #S=[]
    #for i in range(nstm):
    #    S.append = TDAEMSystem(stmfiles[i])
    
    S_LM = TDAEMSystem(stmfiles[0])
    if nstm>1:
        S_HM = TDAEMSystem(stmfiles[1])
        S=[S_LM, S_HM]
    else:
        S=[S_LM]
    t2=time.time()
    t_system = 1000*(t2-t1)
    if showtime:
        print("Time, Setting up systems = %4.1fms" % t_system)

    # Setting up geometry
    GEX = ig.read_gex(file_gex)
    txrx_dx = float(GEX['General']['RxCoilPosition1'][0])-float(GEX['General']['TxCoilPosition1'][0])
    txrx_dy = float(GEX['General']['RxCoilPosition1'][1])-float(GEX['General']['TxCoilPosition1'][1])
    txrx_dz = float(GEX['General']['RxCoilPosition1'][2])-float(GEX['General']['TxCoilPosition1'][2])
    tx_height = -float(GEX['General']['TxCoilPosition1'][2])
    #G = Geometry(tx_height=tx_height, txrx_dx = -txrx_dx, txrx_dz = -txrx_dz)
    #G = Geometry(tx_height=.01, txrx_dx = -12.62, txrx_dz = +2.16)
    G = Geometry(tx_height=tx_height, txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
    if (showInfo>0):
        print('tx_height=%f, txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (tx_height, txrx_dx, txrx_dy, txrx_dz))
    
    ng0 = GEX['Channel1']['NoGates']-GEX['Channel1']['RemoveInitialGates'][0]
    if nstm>1:
        ng1 = GEX['Channel2']['NoGates']-GEX['Channel2']['RemoveInitialGates'][0]
    else:
        ng1 = 0

    
    ng = int(ng0+ng1)
    D = np.zeros((nd,ng))

    # Compute forward data
    t1=time.time()
    for i in tqdm(range(nd), mininterval=1, disable=disableTqdm, desc='gatdaem1d'):
        if C.ndim==1:
            # Only one model
            conductivity = C
        else:
            conductivity = C[i]

        #doCompress=True
        if doCompress:
            i_change=np.where(np.diff(conductivity) != 0 )[0]+1
            n_change = len(i_change)
            conductivity_compress = np.zeros(n_change+1)+conductivity[0]
            thickness_compress = np.zeros(n_change)
            for il in range(n_change):
                conductivity_compress[il+1] = conductivity[i_change[il]]
                if il==0:
                    thickness_compress[il]=np.sum(thickness[0:i_change[il]])
                else:   
                    i1=i_change[il-1]
                    i2=i_change[il]
                    #print("i1: %d, i2: %d" % (i1, i2))
                    thickness_compress[il]=np.sum(thickness[i1:i2]) 
            E = Earth(conductivity_compress,thickness_compress)
        else:   
            E = Earth(conductivity,thickness)

        fm0 = S[0].forwardmodel(G,E)
        d = -fm0.SZ
        if nstm>1:
            fm1 = S[1].forwardmodel(G,E)
            d1 = -fm1.SZ
            d = np.concatenate((d,d1))    

        D[i] = d    

        '''
        fm_lm = S_LM.forwardmodel(G,E)
        fm_hm = S_HM.forwardmodel(G,E)
        # combine -fm_lm.SZ and -fm_hm.SZ
        d = np.concatenate((-fm_lm.SZ,-fm_hm.SZ))
        d_ref = D[i]
        '''
        
    t2=time.time()
    if showtime:
        print("Time = %4.1fms per model and %d model tests" % (1000*(t2-t1)/nd, nd))

    return D


def forward_gaaem_chunk(C_chunk, thickness, stmfiles, file_gex, Nhank, Nfreq, **kwargs):
    """
    Perform forward modeling using the GAAEM method on a chunk of data.

    :param C_chunk: The chunk of data to be processed.
    :type C_chunk: numpy.ndarray
    :param thickness: The thickness of the model.
    :type thickness: float
    :param stmfiles: A list of STM files.
    :type stmfiles: list
    :param file_gex: The path to the GEX file.
    :type file_gex: str
    :param Nhank: The number of Hankel functions.
    :type Nhank: int
    :param Nfreq: The number of frequencies.
    :type Nfreq: int
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :return: The result of the forward modeling.
    :rtype: numpy.ndarray
    """
    # pause for random time
    # time.sleep(np.random.rand()*10)
    return forward_gaaem(C=C_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)


# %% PRIOR DATA GENERATORS

def prior_data_gaaem(f_prior_h5, file_gex, N=0, doMakePriorCopy=True, im=1, id=1, Nhank=280, Nfreq=12, is_log=False, parallel=True, **kwargs):
    """
    Generate prior data for the ga-aem method.

    :param f_prior_h5: Path to the prior data file in HDF5 format.
    :type f_prior_h5: str
    :param file_gex: Path to the file containing geophysical exploration data.
    :type file_gex: str
    :param N: Number of soundings to consider (default: 0).
    :type N: int
    :param doMakePriorCopy: Flag indicating whether to make a copy of the prior file (default: True).
    :type doMakePriorCopy: bool
    :param im: Index of the model (default: 1).
    :type im: int
    :param id: Index of the data (default: 1).
    :type id: int
    :param Nhank: Number of Hankel transform quadrature points (default: 280).
    :type Nhank: int
    :param Nfreq: Number of frequencies (default: 12).
    :type Nfreq: int
    :param parallel: Flag indicating whether multiprocessing is used (default: True).
    :type parallel: bool
    :param Ncpu: Number of cpus/threads used (default: 0 - all).
    :type Ncpu: int
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict
    :param Ncpu: Number of CPUs to use (default: 0->all).
    :type Ncpu: int
    
    :return: Filename of the HDF5 file containing the updated prior data.
    :rtype: str
    """
    import integrate as ig

    type = 'TDEM'
    method = 'ga-aem'
    showInfo = kwargs.get('showInfo', 0)
    Ncpu = kwargs.get('Ncpu', 0)
    # of 'Nproc' is set in kwargs use it 
    if 'Nproc' in kwargs:
        Ncpu = kwargs.get('Nproc', 0)
    Ncpu = kwargs.get('Ncpu', 0)

    # Force open/close of hdf5 file
    with h5py.File(f_prior_h5, 'r') as f:
        # open and close
        pass

    with h5py.File(f_prior_h5, 'a') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    if not os.path.isfile(file_gex):
        print("ERRROR: file_gex=%s does not exist in the current folder." % file_gex)

    if doMakePriorCopy:
        
        if N < N_in:
            f_prior_data_h5 = '%s_%s_N%d_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex)[0], N, Nhank, Nfreq)
        else:
            f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_gex)[0], Nhank, Nfreq)
            
        
        if (showInfo>0):
            print("Creating a copy of %s" % (f_prior_h5))
            print("                as %s" % (f_prior_data_h5))
        if (showInfo>0):
                print('  using N=%d of N_in=%d data' % (N,N_in))
        
        # make a copy of the prior file
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N,showInfo=showInfo)
            
    else:
        f_prior_data_h5 = f_prior_h5

    
    Mname = '/M%d' % im
    Dname = '/D%d' % id
    f_prior = h5py.File(f_prior_data_h5, 'a')

    # Get thickness
    if 'x' in f_prior[Mname].attrs:
        z = f_prior[Mname].attrs['x']
    else:
        z = f_prior[Mname].attrs['z']
    thickness = np.diff(z)

    # Get conductivity
    if Mname in f_prior.keys():
        C = 1 / f_prior[Mname][:]
    else:
        print('Could not load %s from %s' % (Mname, f_prior_data_h5))

    N = f_prior[Mname].shape[0]
    t1 = time.time()
    if not parallel:
        if (showInfo>-1):
            print("prior_data_gaaem: Using 1 thread /(sequential).")
        # Sequential
        D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)
        if is_log:
            D = np.log10(D)
    else:
    
        # Make sure STM files are only written once!!! (need for multihreading)
        # D = ig.forward_gaaem(C=C[0:1,:], thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        stmfiles, GEX = ig.gex_to_stm(file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Parallel
        if Ncpu < 1 :
            Ncpu =  int(multiprocessing.cpu_count()/2)
            Ncpu =  int(multiprocessing.cpu_count())
        if (showInfo>-1):
            print("prior_data_gaaem: Using %d parallel threads." % (Ncpu))

        # 1: Define a function to compute a chunk
        ## OUTSIDE
        # 2: Create chunks
        C_chunks = np.array_split(C, Ncpu)
        C_chunks = np.array_split(C, Ncpu)

        # 3: Compute the chunks in parallel
        forward_gaaem_chunk_partial = partial(forward_gaaem_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Create a multiprocessing pool and compute D for each chunk of C
        
        # Use spawn context for cross-platform compatibility
        #ctx = multiprocessing.get_context("spawn")
        #with ctx.Pool(processes=Nproc) as p:
        #    D_chunks = p.map(forward_gaaem_chunk_partial, C_chunks)
        
        with Pool() as p:
            D_chunks = p.map(forward_gaaem_chunk_partial, C_chunks)
        
        #useIterative=0
        #if useIterative==1:
        #    D_chunks = []
        #    for C_chunk in C_chunks:    
        #        D_chunk = ig.forward_gaaem(C=C_chunk, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        #        D_chunks.append(D_chunk)

        # 4: Combine the chunks into D
        #print('Concatenating D_chunks')
        D = np.concatenate(D_chunks)
        #print("D.shape", D.shape)
        if is_log:
            D = np.log10(D)


        # D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)

    t2 = time.time()
    t_elapsed = t2 - t1
    if (showInfo>-1):
        print('Time elapsed: %5.1f s, for %d soundings. %4.3f ms/sounding. %4.1fit/s' % (t_elapsed, N, 1000*t_elapsed/N,N/t_elapsed))
    
    # Write D to f_prior['/D1']
    f_prior[Dname] = D

    # Add method, type, file_ex, and im as attributes to '/D1'
    f_prior[Dname].attrs['method'] = method
    f_prior[Dname].attrs['type'] = type
    f_prior[Dname].attrs['im'] = im
    f_prior[Dname].attrs['Nhank'] = Nhank
    f_prior[Dname].attrs['Nfreq'] = Nfreq

    f_prior.close()

    return f_prior_data_h5


def prior_data_identity(f_prior_h5, id=0, im=1, N=0, doMakePriorCopy=False, **kwargs):
    '''
    Generate data D%id from model M%im in the prior file f_prior_h5 as an identity of M%im.

    :param f_prior_h5: Path to the prior data file in HDF5 format.
    :type f_prior_h5: str
    :param id: Index of the data (default: 0). if id=0, the next available data id is used
    :type id: int
    :param im: Index of the model (default: 1).
    :type im: int
    :param N: Number of soundings to consider (default: 0).
    :type N: int
    :param doMakePriorCopy: Flag indicating whether to make a copy of the prior file (default: False).
    :type doMakePriorCopy: bool
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    '''
    import integrate as ig
    import time
    
    type = 'idenity'
    method = '--'
    showInfo = kwargs.get('showInfo', 0)
    forceDeleteExisting = kwargs.get('forceDeleteExisting', True)


    # check keys for the data with max id form 'D1', 'D2', 'D3', ...
    if id==0:
        with h5py.File(f_prior_h5, 'a') as f_prior:
            id = 1
            for id_test in range(15):
                key = '/D%d' % id_test
                if key in f_prior.keys():
                    print('Checking key EXISTS: %s' % key)
                    id = id_test+1
                else:                    
                    pass
            print('using id = %d' % id)    
        
    
    with h5py.File(f_prior_h5, 'a') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    print('N=%d, N_in=%d' % (N,N_in))
    if doMakePriorCopy:
        if N < N_in:
            f_prior_data_h5 = '%s_N%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], N, im, id)
        else:
            f_prior_data_h5 = '%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], im, id)
        if (showInfo>0):
            print("Creating a copy of %s as %s" % (f_prior_h5, f_prior_data_h5))
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N)
        
    else:
        f_prior_data_h5 = f_prior_h5

    Mname = '/M%d' % im
    Dname = '/D%d' % id

    # copy f_prior[Mname] to Dname
    print('Copying %s to %s in filename=%s' % (Mname, Dname, f_prior_data_h5))

    # f_prior = h5py.File(f_prior_data_h5, 'r+')
    with h5py.File(f_prior_data_h5, 'a') as f:
        D = f[Mname]
        # check if Dname exists, if so, delete it
        if Dname in f.keys():
            if forceDeleteExisting:
                print('Key %s allready exists -- DELETING !!!!' % Dname)
                del f[Dname]
            else:
                print('Key %s allready exists - doing nothing' % Dname)
                return f_prior_data_h5
        
        dataset = f.create_dataset(Dname, data=D)  # 'i4' represents 32-bit integers
        dataset.attrs['description'] = 'Identiy of %s' % Mname
        dataset.attrs['f5_forward'] = 'none'
        dataset.attrs['with_noise'] = 0
        #f_prior.close()
    
    
    return f_prior_data_h5

# %% PRIOR MODEL GENERATORS
def prior_model_layered(lay_dist='uniform', dz = 1, z_max = 90, 
                        NLAY_min=3, NLAY_max=6, NLAY_deg=6, 
                        RHO_dist='log-uniform', RHO_min=0.1, RHO_max=100, RHO_MEAN=100, RHO_std=80, 
                        N=100000, **kwargs):
    """
    Generate a prior model with layered structure.

    :param lay_dist: Distribution of the number of layers. Options are 'chi2' and 'uniform'. Default is 'chi2'.
    :type lay_dist: str
    :param NLAY_min: Minimum number of layers. Default is 3.
    :type NLAY_min: int
    :param NLAY_max: Maximum number of layers. Default is 6.
    :type NLAY_max: int
    :param NLAY_deg: Degrees of freedom for chi-square distribution. Only applicable if lay_dist is 'chi2'. Default is 6.
    :type NLAY_deg: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', and 'lognormal'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param RHO_min: Minimum resistivity value. Default is 0.1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 100.
    :type RHO_max: float
    :param RHO_MEAN: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 100.
    :type RHO_MEAN: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param N: Number of prior models to generate. Default is 100000.
    :type N: int

    :return: Filepath of the saved prior model.
    :rtype: str
    """
    
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
 
    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1
        
    if lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        f_prior_h5 = 'PRIOR_UNIFORM_NL_%d-%d_%s_N%d.h5' % (NLAY_min, NLAY_max, RHO_dist, N)

    elif lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        f_prior_h5 = 'PRIOR_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)

    # Force NLAY to be a 2 dimensional numpy array
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    M_rho = np.zeros((N, nz))

    # save to hdf5 file
    
    #% simulate the number of layers as in integer
    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_layered'):
    
        i_boundaries = np.sort(np.random.choice(nz, NLAY[i]-1, replace=False))        

        ### simulate the resistivity in each layer
        if RHO_dist=='log-normal':
            rho_all=np.random.lognormal(mean=np.log10(RHO_MEAN), sigma=np.log10(RHO_std), size=NLAY[i])
        elif RHO_dist=='normal':
            rho_all=np.random.normal(mean=RHO_MEAN, sigma=RHO_std, size=NLAY[i])
        elif RHO_dist=='log-uniform':
            rho_all=np.exp(np.random.uniform(np.log(RHO_min), np.log(RHO_max), NLAY[i]))
        elif RHO_dist=='uniform':
            rho_all=np.random.uniform(RHO_min, RHO_max, NLAY[i])

        rho = np.zeros(nz)+rho_all[0]
        for j in range(len(i_boundaries)):
            rho[i_boundaries[j]:] = rho_all[j+1]

        M_rho[i]=rho        

    if (showInfo>0):
        print("Saving prior model to %s" % f_prior_h5)
    
    with h5py.File(f_prior_h5, 'w') as f_prior:
        f_prior.create_dataset('/M1', data=M_rho)
        f_prior['/M1'].attrs['name']='Resistivity'
        f_prior['/M1'].attrs['is_discrete'] = 0
        f_prior['/M1'].attrs['z'] = z
        f_prior['/M1'].attrs['x'] = z
        f_prior.create_dataset('/M2', data=NLAY)
        f_prior['/M2'].attrs['name'] = 'Number of layers'
        f_prior['/M2'].attrs['is_discrete'] = 0
        f_prior['/M2'].attrs['z'] = z
        f_prior['/M2'].attrs['x'] = z
    
    return f_prior_h5

def prior_model_workbench_direct(N=100000, RHO_dist='log-uniform', z1=0, z_max= 100, 
                          nlayers=0, p=2, NLAY_min=3, NLAY_max=6,
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100, **kwargs):
    """
    Generate a prior model with increasingly thick layers.
    ALl models have the same number of layers!
    See also: prior_model_workbench
 
    :param N: Number of prior models to generate. Default is 100000.
    :type N: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param z1: Minimum depth value. Default is 0.
    :type z1: float
    :param z2: Maximum depth value. Default is 100.
    :type z2: float
    :param nlayers: Number of layers. Default is 30.
    :type nlayers: int
    :param p: Power parameter for thickness increase. Default is 2.
    :type p: int
    :param RHO_min: Minimum resistivity value. Default is 1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 300.
    :type RHO_max: float
    :param RHO_mean: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 180.
    :type RHO_mean: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param chi2_deg: Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 'chi2'. Default is 100.
    :type chi2_deg: int

    :return: Filepath of the saved prior model.
    :rtype: str
    """

    showInfo = kwargs.get('showInfo', 0)
    if nlayers<1:
        nlayers = 30
    

    f_prior_h5 = 'PRIOR_WB%d_N%d_%s' % (nlayers,N,RHO_dist)
    
    print('nlayers=%d, N=%d' % (nlayers,N))
    print('NLAY_min=%d, NLAY_max=%d' % (NLAY_min,NLAY_max))

    z2=z_max
    z= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

    nz = len(z)
    
    if RHO_dist=='uniform':
        M_rho = np.random.uniform(low=RHO_min, high = RHO_max, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='log-uniform':
        M_rho = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(N, nz)))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='normal':
        M_rho = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='log-normal':
        M_rho = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(N, nz))
        f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='chi2':
        M_rho = np.random.chisquare(df = chi2_deg, size=(N, nz))
        f_prior_h5 = '%s_deg%d.h5' % (f_prior_h5,chi2_deg)

    #f_prior_h5 = f_prior_h5 + '.h5'

    if (showInfo>0):
        print("Saving prior model to %s" % f_prior_h5)

    with h5py.File(f_prior_h5, 'w') as f_prior:
        f_prior.create_dataset('/M1', data=M_rho)
        f_prior['/M1'].attrs['name']='Resistivity'
        f_prior['/M1'].attrs['is_discrete'] = 0
        f_prior['/M1'].attrs['z'] = z
        f_prior['/M1'].attrs['x'] = z
        
    # return the full filepath to f_prior_h5
    return f_prior_h5


def prior_model_workbench(N=100000, p=2, z1=0, z_max= 100, dz=1,
                          lay_dist='uniform', nlayers=0, NLAY_min=3, NLAY_max=6, NLAY_deg=5,
                          RHO_dist='log-uniform', 
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100, **kwargs):
    """
    Generate a prior model with increasingly thick layers
 
    :param lay_dist: Distribution of the number of layers. Options are 'chi2' and 'uniform'. Default is 'chi2'.
    :type lay_dist: str:param N: Number of prior models to generate. Default is 100000.
    :type N: int
    :param RHO_dist: Distribution of resistivity within each layer. Options are 'log-uniform', 'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    :type RHO_dist: str
    :param z1: Minimum depth value. Default is 0.
    :type z1: float
    :param z2: Maximum depth value. Default is 100.
    :type z2: float
    :param nlayers: Number of layers. Default is 30.
    :type nlayers: int
    :param NLAY_min: Minimum number of layers. Default is 3.
    :type NLAY_min: int
    :param NLAY_max: Maximum number of layers. Default is 6.
    :type NLAY_max: int
    :param NLAY_deg: Degrees of freedom for chi-square distribution. Only applicable if lay_dist is 'chi2'. Default is 6.
    :type NLAY_deg: int
    :param p: Power parameter for thickness increase. Default is 2.
    :type p: int
    :param RHO_min: Minimum resistivity value. Default is 1.
    :type RHO_min: float
    :param RHO_max: Maximum resistivity value. Default is 300.
    :type RHO_max: float
    :param RHO_mean: Mean resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 180.
    :type RHO_mean: float
    :param RHO_std: Standard deviation of resistivity value. Only applicable if RHO_dist is 'normal' or 'lognormal'. Default is 80.
    :type RHO_std: float
    :param chi2_deg: Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is 'chi2'. Default is 100.
    :type chi2_deg: int

    :return: Filepath of the saved prior model.
    :rtype: str
    """
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if nlayers>0:
        NLAY_min = nlayers
        NLAY_max = nlayers

    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1

    
    if lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)    
        f_prior_h5 = 'PRIOR_WB_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)
    elif lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        if NLAY_min == NLAY_max:
            nlayers = NLAY_min
            f_prior_h5 = 'PRIOR_WB_UNIFORM_%d_N%d_%s' % (nlayers,N,RHO_dist)
        else:   
            f_prior_h5 = 'PROPR_WB_UNIFORM_%d-%d_N%d_%s' % (NLAY_min,NLAY_max,N,RHO_dist)
        


    # Force NLAY to be a 2 dimensional numpy array (for when exporting to HDF5)
    NLAY = NLAY[:, np.newaxis]
    
    z_min = 0
    z = np.arange(z_min, z_max, dz)
    nz= len(z)
    print('z_min, z_max, dz, nz = %g, %g, %g, %d' % (z_min, z_max, dz, nz))
    M_rho = np.zeros((N, nz))

    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_workbench'):
        nlayers = NLAY[i][0]
        #print(nlayers)
        z2=z_max
        z_single= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

        if RHO_dist=='uniform':
            M_rho_single = np.random.uniform(low=RHO_min, high = RHO_max, size=(1, nz))
        elif RHO_dist=='log-uniform':
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nz)))
        elif RHO_dist=='normal':
            M_rho_single = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(1, nz))
        elif RHO_dist=='log-normal':
            M_rho_single = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(1, nz))
        elif RHO_dist=='chi2':
            M_rho_single = np.random.chisquare(df = chi2_deg, size=(1, nz))

        for j in range(nlayers):
            ind = np.where(z>=z_single[j])[0]
            M_rho[i,ind]= M_rho_single[0,j]

    if (showInfo>0):
        print("Saving prior model to %s" % f_prior_h5)

    print(f_prior_h5)
    with h5py.File(f_prior_h5, 'w') as f_prior:
        f_prior.create_dataset('/M1', data=M_rho)
        f_prior['/M1'].attrs['name'] = 'Resistivity'
        f_prior['/M1'].attrs['is_discrete'] = 0
        f_prior['/M1'].attrs['z'] = z
        f_prior['/M1'].attrs['x'] = z
        f_prior.create_dataset('/M2', data=NLAY)
        f_prior['/M2'].attrs['name'] = 'Number of layers'
        f_prior['/M2'].attrs['is_discrete'] = 0
        f_prior['/M2'].attrs['z'] = np.array([0])
        f_prior['/M2'].attrs['x'] = np.array([0])
    
    # return the full filepath to f_prior_h5
    return f_prior_h5



def posterior_cumulative_thickness(f_post_h5, im=2, icat=[0], usePrior=False, **kwargs):
    """
    Calculate the posterior cumulative thickness based on the given inputs.

    :param f_post_h5: Path to the input h5 file.
    :type f_post_h5: str
    :param im: Index of model parameter number, M[im].
    :type im: int
    :param icat: List of category indices.
    :type icat: list
    :param usePrior: Flag indicating whether to use prior.
    :type usePrior: bool
    :param kwargs: Additional keyword arguments.
    :returns: 
        - thick_mean (ndarray): Array of mean cumulative thickness.
        - thick_median (ndarray): Array of median cumulative thickness.
        - thick_std (ndarray): Array of standard deviation of cumulative thickness.
        - class_out (list): List of class names.
        - X (ndarray): Array of X values.
        - Y (ndarray): Array of Y values.
    :rtype: tuple
    """

    import h5py
    import integrate as ig

    if isinstance(icat, int):
        icat = np.array([icat])

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im
    with h5py.File(f_prior_h5,'r') as f_prior:
        if not Mstr in f_prior.keys():
            print('No %s found in %s' % (Mstr, f_prior_h5))
            return -1
        if not f_prior[Mstr].attrs['is_discrete']:
            print('M%d is not discrete' % im)
            return -1



    with h5py.File(f_prior_h5,'r') as f_prior:
        try:
            z = f_prior[Mstr].attrs['z'][:].flatten()
        except:
            z = f_prior[Mstr].attrs['x'][:].flatten()
        is_discrete = f_prior[Mstr].attrs['is_discrete']
        if 'clim' in f_prior[Mstr].attrs.keys():
            clim = f_prior[Mstr].attrs['clim'][:].flatten()
        else:
            # if clim set in kwargs, use it, otherwise use default
            if 'clim' in kwargs:
                clim = kwargs['clim']
            else:
                clim = [.1, 2600]
                clim = [10, 500]
        if 'class_id' in f_prior[Mstr].attrs.keys():
            class_id = f_prior[Mstr].attrs['class_id'][:].flatten()
        else:   
            print('No class_id found')
        if 'class_name' in f_prior[Mstr].attrs.keys():
            class_name = f_prior[Mstr].attrs['class_name'][:].flatten()
        else:
            class_name = []
        n_class = len(class_name)
        if 'cmap' in f_prior[Mstr].attrs.keys():
            cmap = f_prior[Mstr].attrs['cmap'][:]
        else:
            cmap = plt.cm.hot(np.linspace(0, 1, n_class)).T
        from matplotlib.colors import ListedColormap

    with h5py.File(f_post_h5,'r') as f_post:
        #P=f_post[Mstr+'/P'][:]
        i_use = f_post['/i_use'][:]

    ns,nr=i_use.shape

    if usePrior:
        for i in range(ns):
            i_use[i,:]=np.arange(nr)
        

    f_prior = h5py.File(f_prior_h5,'r')
    M_prior = f_prior[Mstr][:]
    f_prior.close()
    nz = M_prior.shape[1]

    thick_mean = np.zeros((ns))
    thick_median = np.zeros((ns))
    thick_std = np.zeros((ns))


    thick = np.diff(z)

    for i in range(ns):

        jj = i_use[i,:].astype(int)-1
        m_sample = M_prior[jj,:]
            
        cum_thick = np.zeros((nr))
        for ic in range(len(icat)):

            
            # the number of values of i_cat in the sample

            i_match = (m_sample == class_id[icat[ic]]).astype(int)
            i_match = i_match[:,0:nz-1]
            
            n_cat = np.sum(m_sample==icat[ic], axis=0)
        
            cum_thick = cum_thick + np.sum(i_match*thick, axis=1)

        thick_mean[i] = np.mean(cum_thick)
        thick_median[i] = np.median(cum_thick)
        thick_std[i] = np.std(cum_thick)

    class_out = class_name[icat]

    return thick_mean, thick_median, thick_std, class_out, X, Y


'''
THIS IS THE NEW MULTI DATA IMPLEMENTATION
'''

def load_prior(f_prior_h5, N_use=0, idx = [], Randomize=False):
    if len(idx)==0:
        D, idx = load_prior_data(f_prior_h5, N_use=N_use, Randomize=Randomize)
    else:
        D, idx = load_prior_data(f_prior_h5, idx=idx, Randomize=Randomize)
    M, idx = load_prior_model(f_prior_h5, idx=idx, Randomize=Randomize)
    return D, M, idx

def load_prior_model(f_prior_h5, im_use=[], idx=[], N_use=0, Randomize=False):
    import h5py
    import numpy as np

    if len(im_use)==0:
        Nmt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='M':
                    Nmt = Nmt+1
        if len(im_use)==0:
            im_use = np.arange(1,Nmt+1) 
    
    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/M1'].shape[0]
        if N_use == 0:
            N_use = N    
        
        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)      
                print('using N_use=len(idx)=%d' % N_use)
                
        M = [f_prior[f'/M{id}'][:][idx] for id in im_use]
    
    
    return M, idx



def load_prior_data(f_prior_h5, id_use=[], idx=[], N_use=0, Randomize=False):
    import h5py
    import numpy as np

    if len(id_use)==0:        
        Ndt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='D':
                    Ndt = Ndt+1
        if len(id_use)==0:
            id_use = np.arange(1,Ndt+1) 

    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]
        if N_use == 0:
            N_use = N    
        if N_use>N:
            N_use = N

        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)      
                print('using N_use=len(idx)=%d' % N_use)


        D = [f_prior[f'/D{id}'][:][idx] for id in id_use]
    return D, idx

def load_data(f_data_h5, id_use=[1]):
    import h5py
    with h5py.File(f_data_h5, 'r') as f_data:
        noise_model = [f_data[f'/D{id}'].attrs.get('noise_model', 'none') for id in id_use]
        d_obs = [f_data[f'/D{id}/d_obs'][:] for id in id_use]
        d_std = [f_data[f'/D{id}/d_std'][:] if 'd_std' in f_data[f'/D{id}'] else None for id in id_use]
        Cd = [f_data[f'/D{id}/Cd'][:] if 'Cd' in f_data[f'/D{id}'] else None for id in id_use]

    DATA = {}
    DATA['noise_model'] = noise_model
    DATA['d_obs'] = d_obs
    DATA['d_std'] = d_std
    DATA['Cd'] = Cd
    DATA['id_use'] = id_use        
    # return noise_model, d_obs, d_std, Cd, id_use
    return DATA


# Create shared memory arrays
def create_shared_memory(arrays):
    shared_memories = []
    for array in arrays:
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        # Copy the data into shared memory
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        # Store the shared memory name, array shape, and dtype
        shared_memories.append((shm.name, array.shape, array.dtype))
    return shared_memories

# Function to reconstruct the list D from shared memory
def reconstruct_shared_arrays(shared_memory_refs):
    reconstructed_arrays = []
    for shm_name, shape, dtype in shared_memory_refs:
        shm = shared_memory.SharedMemory(name=shm_name)
        array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        reconstructed_arrays.append(array.copy())  # Copy the data to avoid leaving a view
        shm.close()
    return reconstructed_arrays

# Cleanup function for shared memory
def cleanup_shared_memory(shared_memory_refs):
    for shm_name, _, _ in shared_memory_refs:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()

# START OF REJECTION SAMPLING

def integrate_rejection(f_prior_h5='prior.h5', 
                              f_data_h5='DAUGAAD_AVG_inout.h5',
                              f_post_h5='',                              
                              N_use=100000000000, 
                              id_use=[], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              Nchunks=0,
                              Ncpu=0,
                              parallel=True,
                              use_N_best=0,  
                              **kwargs):
    from datetime import datetime   
    #from multiprocessing import Pool
    #import multiprocessing
    #import integrate as ig
    #import numpy as np
    #import h5py

    # get optional arguments
    showInfo = kwargs.get('showInfo', 1)
    updatePostStat = kwargs.get('updatePostStat', True)
    # If set, Nproc will be used as the number of processors
    Ncpu = kwargs.get('Nproc', Ncpu)
    if Ncpu < 1 :
        Ncpu =  int(multiprocessing.cpu_count())
    
    # Set default f_post_h5 filename if not set    
    if len(f_post_h5)==0:
        f_post_h5 = "POST_%s_%s_Nu%d_aT%d.h5" % (os.path.splitext(f_data_h5)[0],os.path.splitext(f_prior_h5)[0],N_use,autoT)

    # Check that f_post_h5 allready exists, and warn the user   
    if os.path.isfile(f_post_h5):
        if (showInfo>0):    
            print('File %s allready exists' % f_post_h5)
            print('Overwriting...')    

    
    # Load observed data from f_data_h5
    DATA = load_data(f_data_h5)
    Ndt = len(DATA['d_obs']) # Number of data types
    if len(id_use)==0:
        id_use = np.arange(1,Ndt+1) 
    # Perhaps load only the data types that are used
    DATA = load_data(f_data_h5, id_use=id_use)
    
     # Load the prior data from the h5 files
    #D = load_prior_data(f_prior_h5, id_use = id_use, N_use = N_use, Randomize=True)[0]
    D, idx = load_prior_data(f_prior_h5, id_use = id_use, N_use = N_use, Randomize=True)
    # M, idx = load_prior_model(f_prior_h5, idx=idx, N_use=N_use, Randomize=True)

    # Get sample size N from f_prior_h5
    N = D[0].shape[0]
    if N_use>N:
        N_use = N

    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]
    
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)
    Ndp_invert = len(ip_range)
            
        
    if Ncpu ==1:
        parallel = False

    
    if showInfo>0:
        print('<--INTEGRATE_REJECTION-->')
        print('f_prior_h5=%s\nf_data_h5=%s\nf_post_h5=%s' % (f_prior_h5, f_data_h5, f_post_h5))
    
    if showInfo>1:
        print('Number of data points: %d (available), %d (used). Nchunks=%s, Ncpu=%d,use_N_best=%d' % (Ndp,Ndp_invert,Nchunks,Ncpu,use_N_best))    
        print('N_use = %d' % (N_use))
        print('use_N_best=%d' % use_N_best)
        print('Number of data types: %d' % Ndt)
        print('Using these data types: %d' % id_use)
    
    
    
    # set i_use_all to be a 2d Matrie of size (nump,nr) of random integers in range(N)
    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    # 'posterior' evience - mean posterior likelihood TODO
    EV_post_all  = np.zeros(Ndp)*np.nan

    date_start = str(datetime.now())
    t_start = datetime.now()
    
    # PERFORM INVERSION PERHAPS IN PARALLEL

    if parallel:
        # Split the ip_range into Nchunks
        if Nchunks==0:
            if parallel:
                Nchunks = Ncpu
            else:   
                Nchunks = 1
        ip_chunks = np.array_split(ip_range, Nchunks) 

        if showInfo>1:
            print('Ncpu = %d\nNchunks=%d' % (Ncpu, Nchunks))

        i_use_all, T_all, EV_all = integrate_posterior_main(
            ip_chunks=ip_chunks,
            D=D, 
            DATA = DATA,
            idx = idx,  
            N_use=N_use,
            id_use=id_use,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            Ncpu=Ncpu,
            use_N_best=use_N_best            
        )


    else:

        #for i_chunk in range(len(ip_chunks)):        
        #    ip_range = ip_chunks[i_chunk]
        #    if showInfo>0:
        #        print('Chunk %d/%d, ndp=%d' % (i_chunk+1, len(ip_chunks), len(ip_range)))

            i_use, T, EV, ip_range = integrate_rejection_range(D=D, 
                                        DATA = DATA,
                                        idx = idx,                                   
                                        N_use=N_use, 
                                        id_use=id_use,
                                        ip_range=ip_range,
                                        autoT=autoT,
                                        T_base = T_base,
                                        nr=nr,
                                        **kwargs
                                        )
        
            for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]

    # WHere T_all is Inf set it to Nan
    T_all[T_all==np.inf] = np.nan
    EV_all[EV_all==np.inf] = np.nan
    print('All done')
    date_end = str(datetime.now())
    t_end = datetime.now()
    t_elapsed = (t_end - t_start).total_seconds()
    t_per_sounding = t_elapsed / Ndp_invert
    if (showInfo>-1):
        print('T_av=%3.1f, Time=%5.1fs/%d soundings ,%4.1fms/sounding, %3.1fit/s' % (np.nanmean(T_all),t_elapsed,Ndp_invert,t_per_sounding*1000,Ndp_invert/t_elapsed))

    # SAVE THE RESULTS to f_post_h5
    with h5py.File(f_post_h5, 'w') as f_post:
        f_post.create_dataset('i_use', data=i_use_all)
        f_post.create_dataset('T', data=T_all)
        f_post.create_dataset('EV', data=EV_all)
        #f_post.create_dataset('ip_range', data=ip_range)
        f_post.attrs['date_start'] = date_start
        f_post.attrs['date_end'] = date_end
        f_post.attrs['inv_time'] = t_elapsed
        f_post.attrs['f5_prior'] = f_prior_h5
        f_post.attrs['f5_data'] = f_data_h5
        f_post.attrs['N_use'] = N_use

    if updatePostStat:
        integrate_posterior_stats(f_post_h5, **kwargs)

    #return f_post_h5 T_all, EV_all, i_use_all
    return f_post_h5



def integrate_rejection_range(D, 
                              DATA, 
                              idx = [],
                              N_use=1000, 
                              id_use=[1,2], 
                              ip_range=[], 
                              nr=400,
                              autoT=1,
                              T_base = 1,
                              **kwargs):

    from tqdm import tqdm
    import numpy as np
    import h5py
    import time
    import integrate as ig

    # get optional arguments
    use_N_best = kwargs.get('use_N_best', 0)
    #print("use_N_best=%d" % use_N_best)
    showInfo = kwargs.get('showInfo', 0)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=False
    
    useRandomData = kwargs.get('useRandomData', True)
    #useRandomData = kwargs.get('useRandomData', False)
    

    # Get number of data points
    Ndp = DATA['d_obs'][0].shape[0]
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)

    nump=len(ip_range)
    if showInfo>1:
        print('Number of data points to invert: %d' % nump)
    i_use_all = np.zeros((nump, nr), dtype=np.int32)
    T_all = np.zeros(nump)*np.nan
    EV_all = np.zeros(nump)*np.nan
    
    
    # Get the lookup sample size
    N = D[0].shape[0]
    
    if N_use>N:
        N_use = N

    if len(idx)==0:
        idx = np.arange(N_use)
    
    #i=0
    
    noise_model = DATA['noise_model']
    #print(noise_model)
    
    # THIS IS THE ACTUAL INVERSION!!!!
    for j in tqdm(range(len(ip_range)), miniters=10, disable=disableTqdm, desc='rejection'):
        ip = ip_range[j] # This is the index of the data point to invert
        t=[]
        N = D[0].shape[0]
        NDsets = len(id_use)
        L = np.zeros((NDsets, N))

        for i in range(len(D)):
            t0=time.time()
            id = id_use[i]
            DS = '/D%d' % id
            if noise_model[i]=='gaussian':
                d_obs = DATA['d_obs'][i][ip]
                
                if DATA['Cd'][0] is not None:                    
                    # if Cd is 3 dimensional, take the first slice
                    if len(DATA['Cd'][0].shape) == 3:
                        Cd = DATA['Cd'][0][ip]
                    else:
                        Cd = DATA['Cd'][0][:]

                    L_single = likelihood_gaussian_full(D[i], d_obs, Cd, N_app = use_N_best)
                    
                elif DATA['d_std'][0] is not None:
                    d_std = DATA['d_std'][i][ip]
                    L_single = likelihood_gaussian_diagonal(D[i], d_obs, d_std, use_N_best)

                else:
                    print('No d_std or Cd in %s' % DS)

                L[i] = L_single
                t.append(time.time()-t0)
            elif noise_model[i]=='multinomial':
                d_obs = DATA['d_obs'][i][ip]
                
                class_id = [1,2]

                useVetorized = True
                if useVetorized:
                    D_ind = np.zeros(D[i].shape[0], dtype=int)
                    D_ind[:] = np.searchsorted(class_id, D[i].squeeze())
                    L_single = np.log(d_obs[D_ind])
                else:
                    D_ind = np.zeros(D[id].shape[0], dtype=int)
                    for i in range(D_ind.shape[0]):
                        for j in range(len(class_id)):
                            if D[id][i]==class_id[j]:
                                D_ind[i]=j
                                break
                    L_single = np.zeros(D[id].shape[0])

                    for i in range(D_ind.shape[0]):
                        L_single[i] = np.log(d_obs[D_ind[i]])

                L[i] = L_single           
                t.append(time.time()-t0)

            else: 
                # noise model not regcognized
                # L_single = -1
                pass

        
        t0=time.time()


        # NOw we have all the likelihoods for all data types. Copmbine them into ooe
        L_single = L
        L = np.sum(L_single, axis=0)
        #plt.plot(L.T)


        # AUTO ANNEALE
        t0=time.time()
        #autoT=1
        # Compute the annealing temperature
        if autoT == 1:
            T = ig.logl_T_est(L)
        else:
            T = T_base        
        # maxlogL = np.nanmax(logL)
        t.append(time.time()-t0)

        # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
        
        P_acc = np.exp((1/T) * (L - np.nanmax(L)))
        P_acc[np.isnan(P_acc)] = 0

        # Select the index of P_acc propportion to the probabilituy given by P_acc
        t0=time.time()
        try:
            i_use = np.random.choice(N, nr, p=P_acc/np.sum(P_acc))
        except:
            print('Error in np.random.choice for ip=%d' % ip)   
            i_use = np.random.choice(N, nr)
        
        if useRandomData:
            # get the correct index of the subset used
            i_use = idx[i_use]
            

        # Unfortunately this code originally used matlab style codeing for i_use, 
        # this we need to add 1 to the index
        #i_use = i_use+1            

        t.append(time.time()-t0)        

    
        # find the number of unique indexes
        n_unique = len(np.unique(i_use))


        # Compute the evidence
        maxlogL = np.nanmax(L)
        exp_logL = np.exp(L - maxlogL)
        EV = maxlogL + np.log(np.nansum(exp_logL)/len(L))

        t.append(time.time()-t0)

        pltDegug = 0
        if pltDegug>0:
            import matplotlib.pyplot as plt
            plt.semilogy(d_obs, 'k', linewidth=4)
            plt.semilogy(D[0][i_use].T, 'r', linewidth=1)
            plt.show()
            print(D[0][10])

        i_use_all[j] = i_use
        T_all[j] = T
        EV_all[j] = EV

        if showInfo>2:
            for i in range(len(t)):
                if i<len(D):
                    print(' Time id%d: %f - %s' % (i,t[i],noise_model[i]))
                else:
                    print(' Time id%d, sampling: %f' % (i,t[i]))
            print('Time total: %f' % np.sum(t))
        
    return i_use_all, T_all, EV_all, ip_range



def integrate_posterior_main(ip_chunks, D, DATA, idx, N_use, id_use, autoT, T_base, nr, Ncpu, use_N_best):
    #import integrate as ig
    from multiprocessing import Pool, shared_memory

    shared_memory_refs = create_shared_memory(D)
    
    #with Pool(Ncpu) as p:
    with Pool(Ncpu) as p:
        # New implementation with shared memory
        results = p.map(integrate_posterior_chunk, [(i, ip_chunks, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best) for i in range(len(ip_chunks))])
        # Old implementation where D was copied to each process
        #results = p.map(integrate_posterior_chunk, [(i, ip_chunks, D, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best) for i in range(len(ip_chunks))])

    # Cleanup shared memory
    cleanup_shared_memory(shared_memory_refs)

    # Get sample size N from f_prior_h5
    N = D[0].shape[0]  

    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]

    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    
    for i, (i_use, T, EV, ip_range) in enumerate(results):
        for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]

    return i_use_all, T_all, EV_all



def integrate_posterior_chunk(args):
    #import integrate as ig
    
    # New implementation with shared memory
    i_chunk, ip_chunks, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    # Old implementation where D was copied to each process
    #i_chunk, ip_chunks, D, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    D=reconstruct_shared_arrays(shared_memory_refs)
    
    ip_range = ip_chunks[i_chunk]

    #print(f'Chunk {i_chunk+1}/{len(ip_chunks)}, ndp={len(ip_range)}')

    i_use, T, EV, ip_range = integrate_rejection_range(
        D,
        DATA,
        idx,
        N_use=N_use,
        id_use=id_use,
        ip_range=ip_range,
        autoT=autoT,
        T_base=T_base,
        nr=nr,     
        use_N_best=use_N_best, 
    )

    return i_use, T, EV, ip_range


# END OF REJECTION SAMPLING


def select_subset_for_inversion(dd, N_app):
    """
    Select a subset of indices for inversion based on the sum of absolute values.

    This function calculates the sum of absolute values along the specified axis
    for each row in the input array `dd`. It then selects the indices of the 
    `N_app` smallest sums.

    Parameters
    ----------
    dd : numpy.ndarray
        A 2D array of data from which to select the subset.
    N_app : int
        The number of indices to select based on the smallest sums.

    Returns
    -------
    idx : numpy.ndarray
        An array of indices corresponding to the `N_app` smallest sums.
    nsum : numpy.ndarray
        An array containing the sum of absolute values for each row in `dd`.

    Notes
    -----
    This function uses `np.nansum` to ignore NaNs in the summation and 
    `np.argpartition` for efficient selection of the smallest sums.
    """
    nsum = np.nansum(np.abs(dd), axis=1)
    idx = np.argpartition(nsum, N_app)[:N_app]
    return idx, nsum


def likelihood_gaussian_diagonal(D, d_obs, d_std, N_app=0):
    """
    Compute the Gaussian likelihood for a diagonal covariance matrix.
    This function calculates the likelihood of observed data `d_obs` given 
    a set of predicted data `D` and standard deviations `d_std` assuming 
    a Gaussian distribution with a diagonal covariance matrix.
    Parameters
    ----------
    D : numpy.ndarray
        Predicted data array of shape (n_samples, n_features).
    d_obs : numpy.ndarray
        Observed data array of shape (n_features,).
    d_std : numpy.ndarray
        Standard deviation array of shape (n_features,).
    Returns
    -------
    numpy.ndarray
        Likelihood array of shape (n_samples,).
    """
    
    # Compute the likelihood
    # Sequential
    #L = np.zeros(D.shape[0])
    #for i in range(D.shape[0]):
    #    L[i] = -0.5 * np.nansum(dd[i]**2 / d_std**2)
    # Vectorized
    dd = D - d_obs
    
    if N_app > 0:
       L = np.ones(D.shape[0])*-1e+15
       idx = select_subset_for_inversion(dd, N_app)[0] 
       #L_small = -0.5 * np.nansum(dd[idx]**2 / d_std**2, axis=1)
       L_small = likelihood_gaussian_diagonal(D[idx], d_obs, d_std,0)
       L[idx]=L_small
       
    else:
        L = -0.5 * np.nansum(dd**2 / d_std**2, axis=1)

    return L

def likelihood_gaussian_full(D, d_obs, Cd, N_app=0, checkNaN=True, useVectorized=False):
    """
    Calculate the Gaussian likelihood for a given dataset.
    Parameters
    ----------
    D : numpy.ndarray
        The model predictions, with shape (n_samples, n_features).
    d_obs : numpy.ndarray
        The observed data, with shape (n_features,).
    Cd : numpy.ndarray
        The covariance matrix of the observed data, with shape (n_features, n_features).
    checkNaN : bool, optional
        If True, the function will handle NaN values in `d_obs` by ignoring them in the calculations. 
        Default is True.
    Returns
    -------
    numpy.ndarray
        The Gaussian likelihood for each sample, with shape (n_samples,).
    TODO
        We need to check that this works when D has NAN value.. (and Why does it ever?)
    """
    
    if checkNaN:
        # find index of non-nan values in d_obs or non-nan values in np.sum(Cd, axis=0)
        #ind = np.where(~np.isnan(d_obs))[0]
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)))[0]
        # Exclude also all data for which one Nan Is available.. This is probably not ideal
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)) & ~np.isnan(np.sum(D, axis=0)) )[0]
        dd = D[:,ind] - d_obs[ind]
        iCd = np.linalg.inv(Cd[np.ix_(ind, ind)])
    else:    
        dd = D - d_obs
        iCd = np.linalg.inv(Cd)
        
    if N_app > 0:
        #print('N_app=%d' % N_app)
        L = np.ones(D.shape[0])*-1e+15
        idx = select_subset_for_inversion(dd, N_app)[0] 
        if useVectorized:
            L_small = -.5 * np.einsum('ij,ij->i', dd[idx] @ iCd, dd[idx])
        else:
            L_small = np.zeros(idx.shape[0])
            for i in range(idx.shape[0]):
                L_small[i] = -.5 * np.nansum(dd[idx[i]].T @ iCd @ dd[idx[i]])
        L[idx] = L_small
    
        return L
    
    if useVectorized:
        # vectorized    
        L = -.5 * np.einsum('ij,ij->i', dd @ iCd, dd)        
    else:   
        # non-vectorized
        L = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            L[i] = -.5 * np.nansum(dd[i].T @ iCd @ dd[i])
        
    return L


# %% Synthetic data

def synthetic_case(case='Wedge', **kwargs):
    """
    Generate synthetic geological models for different cases.
    Parameters
    ----------
    case : str, optional
        The type of synthetic case to generate. Options are 'Wedge' and '3Layer'. Default is 'Wedge'.
    **kwargs : dict, optional
        Additional parameters for the synthetic case generation.
        Common parameters:
        - showInfo : int, optional
            If greater than 0, print information about the generated case. Default is 0.
        Parameters for 'Wedge' case:
        - x_max : int, optional
            Maximum x-dimension size. Default is 1000.
        - dx : float, optional
            Step size in the x-dimension. Default is 1000./x_max.
        - z_max : int, optional
            Maximum z-dimension size. Default is 90.
        - dz : float, optional
            Step size in the z-dimension. Default is 1.
        - z1 : float, optional
            Depth at which the wedge starts. Default is z_max/10.
        - rho : list of float, optional
            Density values for different layers. Default is [100, 200, 120].
        - wedge_angle : float, optional
            Angle of the wedge in degrees. Default is 1.
        Parameters for '3Layer' case:
        - x_max : int, optional
            Maximum x-dimension size. Default is 100.
        - x_range : float, optional
            Range in the x-dimension for the cosine function. Default is x_max/4.
        - dx : float, optional
            Step size in the x-dimension. Default is 1.
        - z_max : int, optional
            Maximum z-dimension size. Default is 60.
        - dz : float, optional
            Step size in the z-dimension. Default is 1.
        - z1 : float, optional
            Depth at which the first layer ends. Default is z_max/3.
        - z_thick : float, optional
            Thickness of the second layer. Default is z_max/2.
        - rho1_1 : float, optional
            Density at the start of the first layer. Default is 120.
        - rho1_2 : float, optional
            Density at the end of the first layer. Default is 10.
        - rho2_1 : float, optional
            Density at the start of the second layer. Default is rho1_2.
        - rho2_2 : float, optional
            Density at the end of the second layer. Default is rho1_1.
        - rho3 : float, optional
            Density of the third layer. Default is 120.
    Returns
    -------
    M : numpy.ndarray
        The generated synthetic model.
    x : numpy.ndarray
        The x-coordinates of the model.
    z : numpy.ndarray
        The z-coordinates of the model.
    """
    
    showInfo = kwargs.get('showInfo', 0)

    if case.lower() == 'wedge':
        # Create synthetic wedhge model
        
        # variables
        x_max = kwargs.get('x_max', 1000)
        dx = kwargs.get('dx', 1000./x_max)
        z_max = kwargs.get('z_max', 90)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/10)
        rho = kwargs.get('rho', [100,200,120])
        wedge_angle = kwargs.get('wedge_angle', 1)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho[0]
        # set M=rho[3] of all iz> (z==z1)
        iz = np.where(z>=z1)[0]
        M[:,iz] = rho[2]
        for ix in range(nx):
            wedge_angle_rad = np.deg2rad(wedge_angle)
            z2 = z1 + x[ix]*np.tan(wedge_angle_rad)            
            #find iz where  (z>=z1) and (z<=z2)
            iz = np.where((z>=z1) & (z<=z2))[0]
            #print(z[iz[0]])
            M[ix,iz] = rho[1]

        return M, x, z

    elif case.lower() == '3layer':
        # Create synthetic 3 layer model

        # variables
        x_max = kwargs.get('x_max', 100)
        x_range = kwargs.get('x_range', x_max/4)
        dx = kwargs.get('dx', 1)
        z_max = kwargs.get('z_max', 60)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/3)
        z_thick = kwargs.get('z_thick', z_max/2)
        

        rho1_1 = kwargs.get('rho1_1', 120)
        rho1_2 = kwargs.get('rho1_2', 10)
        rho2_1 = kwargs.get('rho1_2', rho1_2)
        rho2_2 = kwargs.get('rho2_2', rho1_1)
        rho3 = kwargs.get('rho3', 120)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,ẃedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        M = np.zeros((nx,nz))+rho3
        iz1 = np.where(z<=z1)[0]
        for ix in range(nx):
            rho1 = rho1_1 + (rho1_2 - rho1_1) * x[ix]/x_max
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            M[ix,iz1] = rho1
            z2 = z1 + z_thick*0.5*(1+np.cos(x[ix]/(x_range)*np.pi))            
            rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
            iz2 = np.where((z>=z1) & (z<=z2))[0]
            M[ix,iz2] = rho2

        return M, x, z
