"""
INTEGRATE I/O Module - Data Input/Output and File Management

This module provides comprehensive input/output functionality for the INTEGRATE
geophysical data integration package. It handles reading and writing of HDF5 files,
data format conversions, and management of prior/posterior data structures.

Key Features:
    - HDF5 file I/O for prior models, data, and posterior results
    - Support for multiple geophysical data formats (GEX, STM, USF)
    - Automatic data validation and format checking
    - File conversion utilities between different formats
    - Data merging and aggregation functions
    - Checksum verification and file integrity checks

Main Functions:
    - load_*(): Functions for loading prior models, data, and results
    - save_*(): Functions for saving prior models and data arrays
    - read_*(): File format readers (GEX, USF, etc.)
    - write_*(): File format writers and converters
    - merge_*(): Data and posterior merging utilities

File Format Support:
    - HDF5: Primary data storage format
    - GEX: Geometry and survey configuration files
    - STM: System transfer function files
    - USF: Field measurement files
    - CSV: Export format for GIS integration

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import os
import numpy as np
import h5py
import re
from typing import Dict, List, Union, Any

def load_prior(f_prior_h5, N_use=0, idx = [], Randomize=False):
    """
    Load prior data from an HDF5 file.

    :param f_prior_h5: Path to the prior HDF5 file.
    :type f_prior_h5: str
    :param N_use: Number of samples to use. Default is 0, which means all samples.
    :type N_use: int, optional
    :param idx: List of indices to use. If empty, all indices are used.
    :type idx: list, optional
    :param Randomize: Flag indicating whether to randomize the order of the samples. Default is False.
    :type Randomize: bool, optional
    :return: Dictionary containing the loaded prior data.
    :rtype: dict
    """
    if len(idx)==0:
        D, idx = load_prior_data(f_prior_h5, N_use=N_use, Randomize=Randomize)
    else:
        D, idx = load_prior_data(f_prior_h5, idx=idx, Randomize=Randomize)
    M, idx = load_prior_model(f_prior_h5, idx=idx, Randomize=Randomize)
    return D, M, idx



def load_prior_model(f_prior_h5, im_use=[], idx=[], N_use=0, Randomize=False):
    """
    Load prior model data from an HDF5 file.
    
    This function loads model parameter arrays from the prior structure with flexible
    indexing and sampling options.
    
    :param f_prior_h5: Path to the HDF5 file containing the prior model data
    :type f_prior_h5: str
    :param im_use: List of model indices to use. If empty, all models are used
    :type im_use: list, optional
    :param idx: List of indices to select from the data. If empty, indices are generated based on N_use and Randomize
    :type idx: list, optional
    :param N_use: Number of samples to use. If 0, all samples are used
    :type N_use: int, optional
    :param Randomize: If True, indices are randomized. If False, sequential indices are used
    :type Randomize: bool, optional
    
    :returns: Tuple containing (M, idx) where M is list of arrays containing selected data for each model and idx is array of indices used to select the data
    :rtype: tuple
    
    :raises ValueError: If the length of idx is not equal to N_use
    """
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

def save_prior_model(f_prior_h5, M_new, 
                     im=None, 
                     force_replace=False,
                     delete_if_exist=False,   
                     **kwargs):
    """
    Save new prior model data to an HDF5 file.
    
    This function saves model parameter arrays to the prior structure with automatic
    or manual model identifier assignment. Supports replacing existing models and
    optional file deletion for clean saves.
    
    :param f_prior_h5: Path to the HDF5 file where the prior model data will be saved
    :type f_prior_h5: str
    :param M_new: The new prior model data array to be saved
    :type M_new: numpy.ndarray
    :param im: Model identifier to use as the key. If None, auto-generates next available ID
    :type im: int, optional
    :param force_replace: If True, replaces existing model data with the same ID
    :type force_replace: bool, optional
    :param delete_if_exist: If True, deletes the entire file before saving new data
    :type delete_if_exist: bool, optional
    :param kwargs: Additional arguments including showInfo for verbosity control
    :type kwargs: dict
    
    :returns: The model identifier used for saving the data
    :rtype: int
    
    .. note::
        Model data is stored in HDF5 groups named '/M1', '/M2', etc. The function
        automatically determines the next available ID if im is None.
    """
    import h5py
    import numpy as np
    import os

    showInfo = kwargs.get('showInfo', 0)
    # if f_prior_h5 exists, delete it
    if delete_if_exist:
        
        # Assuming f_prior_h5 already contains the filename
        if os.path.exists(f_prior_h5):
            os.remove(f_prior_h5)
            if showInfo>1:
                print("File %s has been deleted." % f_prior_h5)
        else:
            print("File %s does not exist." % f_prior_h5)
            pass

        
    if im is None:
        Nmt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='M':
                    Nmt = Nmt+1
        im = Nmt+1
    
    key = '/M%d' % im
    if showInfo>1:
        print("Saving new prior model '%s' to file: %s " % (key,f_prior_h5))

    # Delete the 'key' if it exists
    with h5py.File(f_prior_h5, 'a') as f_prior:
        if key in f_prior:
            print("Deleting prior model '%s' from file: %s " % (key,f_prior))
            if force_replace:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_replace=True to overwrite." % key)
                return False

    # Make sure the data is 2D using atleast_2d
    if M_new.ndim<2:
        M_new = np.atleast_2d(M_new.flatten()).T

    # Write the new data
    with h5py.File(f_prior_h5, 'a') as f_prior:
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(M_new.dtype, np.floating):
            M_new_32 = M_new.astype(np.float32)
            f_prior.create_dataset(key, data=M_new_32, compression='gzip', compression_opts=9)
        elif np.issubdtype(M_new.dtype, np.integer):
            M_new_32 = M_new.astype(np.int32)
            f_prior.create_dataset(key, data=M_new_32, compression='gzip', compression_opts=9)
        else:
            f_prior.create_dataset(key, data=M_new, compression='gzip', compression_opts=9)

        # if 'name' is not set in kwargs, set it to 'XXX'
        if 'name' not in kwargs:
            kwargs['name'] = 'Model %d' % (im)
        if 'is_discrete' not in kwargs:
            kwargs['is_discrete'] = 0
        if 'x' not in kwargs:
            kwargs['x'] = np.arange(M_new.shape[1])

        # if kwargs is set print keys
        if showInfo>2:
            for kwargkey in kwargs:
                print('save_prior_model: key=%s, value=%s' % (kwargkey, kwargs[kwargkey]))


        # if kwarg has keyy 'method' then write it to the file as att
        if 'x' in kwargs:
             f_prior[key].attrs['x'] = kwargs['x']
        if 'name' in kwargs:
             f_prior[key].attrs['name'] = kwargs['name']
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'is_discrete' in kwargs:
            f_prior[key].attrs['is_discrete'] = kwargs['is_discrete']
        if 'class_id' in kwargs:
            f_prior[key].attrs['class_id'] = kwargs['class_id']
        if 'class_name' in kwargs:
            f_prior[key].attrs['class_name'] = kwargs['class_name']
        if 'clim' in kwargs:
            f_prior[key].attrs['clim'] = kwargs['clim']
        if 'cmap' in kwargs:
            f_prior[key].attrs['cmap'] = kwargs['cmap']

        if showInfo>1:
            print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))



def load_prior_data(f_prior_h5, id_use=[], idx=[], N_use=0, Randomize=False):
    """
    Load prior data from an HDF5 file.
    
    This function loads forward modeled data arrays from the prior structure,
    supporting selective loading by data identifier, sample indices, and size limits.
    The data can be optionally randomized for sampling purposes.
    
    :param f_prior_h5: Path to the prior HDF5 file containing data arrays
    :type f_prior_h5: str
    :param id_use: List of data identifiers to load. If empty, loads all available data types
    :type id_use: list, optional
    :param idx: List of sample indices to load. If empty, uses N_use or all samples
    :type idx: list, optional
    :param N_use: Number of samples to load. If 0, loads all available samples
    :type N_use: int, optional
    :param Randomize: If True, randomizes the order of loaded samples
    :type Randomize: bool, optional
    
    :returns: Tuple containing (D, idx) where D is list of data arrays and idx is array of used indices
    :rtype: tuple
    
    .. note::
        Data arrays are expected to be stored in HDF5 groups named '/D1', '/D2', etc.
        The function automatically detects available data types if id_use is empty.
    """
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

def save_prior_data(f_prior_h5, D_new, id=None, force_delete=False, **kwargs):
    """
    Save new prior data arrays to an HDF5 file.
    
    This function saves forward modeled data arrays to the prior structure with automatic
    or manual data identifier assignment. Supports replacing existing data arrays.
    
    :param f_prior_h5: Path to the HDF5 file where the prior data will be saved
    :type f_prior_h5: str
    :param D_new: The new prior data array to be saved
    :type D_new: numpy.ndarray
    :param id: Data identifier to use as the key. If None, auto-generates next available ID
    :type id: int, optional
    :param force_delete: If True, deletes existing data with the same ID before saving
    :type force_delete: bool, optional
    :param kwargs: Additional arguments including showInfo for verbosity control
    :type kwargs: dict
    
    :returns: The data identifier used for saving the data
    :rtype: int
    
    .. note::
        Data arrays are stored in HDF5 groups named '/D1', '/D2', etc. The function
        automatically determines the next available ID if id is None.
    """
    import h5py
    import numpy as np

    if id is None:
        Ndt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='D':
                    Ndt = Ndt+1
        id = Ndt+1
    
    key = '/D%d' % id
    print("Saving new prior data '%s' to file: %s " % (key,f_prior_h5))

    # Delete the 'key' if it exists
    with h5py.File(f_prior_h5, 'a') as f_prior:
        if key in f_prior:
            print("Deleting prior data '%s' from file: %s " % (key,f_prior))
            if force_delete:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_delete=True to overwrite." % key)
                return False

    # Write the new data
    with h5py.File(f_prior_h5, 'a') as f_prior:
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(D_new.dtype, np.floating):
            D_new_32 = D_new.astype(np.float32)
            f_prior.create_dataset(key, data=D_new_32, compression='gzip', compression_opts=9)
        else:
            f_prior.create_dataset(key, data=D_new, compression='gzip', compression_opts=9)
        print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))
        # if kwarg has keyy 'method' then write it to the file as att
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'type' in kwargs:
            f_prior[key].attrs['type'] = kwargs['type']
        if 'im' in kwargs:
            f_prior[key].attrs['im'] = kwargs['im']
        if 'Nhank' in kwargs:
            f_prior[key].attrs['Nhank'] = kwargs['Nhank']
        if 'Nfreq' in kwargs:
            f_prior[key].attrs['Nfreq'] = kwargs['Nfreq']
        if 'f5_forward' in kwargs:
            f_prior[key].attrs['f5_forward'] = kwargs['f5_forward']
        if 'with_noise' in kwargs:
            f_prior[key].attrs['with_noise'] = kwargs['with_noise']

    return id


def load_data(f_data_h5, id_arr=[1], **kwargs):
    """
    Load observational data from an HDF5 file.
    
    This function loads observed electromagnetic data including measurements, uncertainties,
    covariance matrices, and associated metadata from structured HDF5 files. Supports
    multiple data types and automatic handling of missing data components.
    
    :param f_data_h5: Path to the HDF5 file containing the observational data
    :type f_data_h5: str
    :param id_arr: List of dataset identifiers to load from the file
    :type id_arr: list of int, optional
    :param kwargs: Additional arguments including showInfo for verbosity control
    :type kwargs: dict
    
    :returns: Dictionary containing loaded data with the following keys:
        
        - 'noise_model': Noise model type for each dataset (list of str)
        - 'd_obs': Observed data measurements (list of numpy.ndarray)
        - 'd_std': Standard deviations of observations (list of numpy.ndarray or None)
        - 'Cd': Full covariance matrices (list of numpy.ndarray or None)
        - 'id_arr': Dataset identifiers that were loaded (list of int)
        - 'i_use': Data point usage indicators (list of numpy.ndarray)
        - 'id_use': Dataset usage identifiers (list of int or numpy.ndarray)
    :rtype: dict
    
    .. note::
        Missing data components (d_std, Cd, i_use, id_use) are automatically handled:
        missing id_use defaults to sequential IDs, missing i_use defaults to all ones,
        missing d_std and Cd remain as None.
    """

    showInfo = kwargs.get('showInfo', 1)
    
    import h5py
    with h5py.File(f_data_h5, 'r') as f_data:
        noise_model = [f_data[f'/D{id}'].attrs.get('noise_model', 'none') for id in id_arr]
        d_obs = [f_data[f'/D{id}/d_obs'][:] for id in id_arr]
        d_std = [f_data[f'/D{id}/d_std'][:] if 'd_std' in f_data[f'/D{id}'] else None for id in id_arr]
        Cd = [f_data[f'/D{id}/Cd'][:] if 'Cd' in f_data[f'/D{id}'] else None for id in id_arr]
        i_use = [f_data[f'/D{id}/i_use'][:] if 'i_use' in f_data[f'/D{id}'] else None for id in id_arr]
        id_use = [f_data[f'/D{id}/id_use'][()] if 'id_use' in f_data[f'/D{id}'] and f_data[f'/D{id}/id_use'].shape == () else f_data[f'/D{id}/id_use'][:] if 'id_use' in f_data[f'/D{id}'] else None for id in id_arr]
        
    for i in range(len(id_arr)):
        if id_use[i] is None:
            id_use[i] = i+1
        if i_use[i] is None:
            i_use[i] = np.ones((len(d_obs[i]),1))

        
    DATA = {}
    DATA['noise_model'] = noise_model
    DATA['d_obs'] = d_obs
    DATA['d_std'] = d_std
    DATA['Cd'] = Cd
    DATA['id_arr'] = id_arr        
    DATA['i_use'] = i_use        
    DATA['id_use'] = id_use        
    # return noise_model, d_obs, d_std, Cd, id_arr


    if showInfo>0:
        print('Loaded data from %s' % f_data_h5)
        for i in range(len(id_arr)):
            print('Data type %d: id_use=%d, %11s, Using %5d/%5d data' % (id_arr[i], id_use[i], noise_model[i], np.sum(i_use[i]), len(i_use[i])))

    return DATA


## def ###################################################

#def write_stm_files(GEX, Nhank=140, Nfreq=6, Ndig=7, **kwargs):
def write_stm_files(GEX, **kwargs):
    """
    Write STM (System Transfer Matrix) files based on the provided GEX system data file.

    :param GEX: The GEX data containing the system information.
    :type GEX: dict
    :param kwargs: Additional keyword arguments for customization.
    :type kwargs: dict
    :return: A list of file paths for the generated STM files.
    :rtype: list
    """
    system_name = GEX['General']['Description']

    # Parse kwargs
    Nhank = kwargs.get('Nhank', 280)
    Nfreq = kwargs.get('Nfreq', 12)
    Ndig = kwargs.get('Ndig', 7)
    showInfo = kwargs.get('showInfo', 0)
    WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'AreaUnderCurve')
    #WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'BoxCar')


    NumAbsHM = kwargs.get('NumAbsHM', Nhank)
    NumAbsLM = kwargs.get('NumAbsLM', Nhank)
    NumFreqHM = kwargs.get('NumFreqHM', Nfreq)
    NumFreqLM = kwargs.get('NumFreqLM', Nfreq)
    DigitFreq = kwargs.get('DigitFreq', 4E6)
    stm_dir = kwargs.get('stm_dir', os.getcwd())
    file_gex = kwargs.get('file_gex', '')

    windows = GEX['General']['GateArray']

    LastWin_LM = int(GEX['Channel1']['NoGates'][0])
    LastWin_HM = int(GEX['Channel2']['NoGates'][0])

    SkipWin_LM = int(GEX['Channel1']['RemoveInitialGates'][0])
    SkipWin_HM = int(GEX['Channel2']['RemoveInitialGates'][0])

    windows_LM = windows[SkipWin_LM:LastWin_LM, :] + GEX['Channel1']['GateTimeShift'][0] + GEX['Channel1']['MeaTimeDelay'][0]
    windows_HM = windows[SkipWin_HM:LastWin_HM, :] + GEX['Channel2']['GateTimeShift'][0] + GEX['Channel2']['MeaTimeDelay'][0]

    #windows_LM = GEX['Channel1']['GateFactor'][0] * windows_LM
    #windows_HM = GEX['Channel2']['GateFactor'][0] * windows_HM
    #windows_LM = windows_LM/GEX['Channel1']['GateFactor'][0] 
    #windows_HM = windows_HM/GEX['Channel2']['GateFactor'][0] 
    
    NWin_LM = windows_LM.shape[0]
    NWin_HM = windows_HM.shape[0]

    # PREPARE WAVEFORMS
    LMWF = GEX['General']['WaveformLM']
    HMWF = GEX['General']['WaveformHM']


    LMWFTime1 = LMWF[0, 0]
    LMWFTime2 = LMWF[-1, 0]

    HMWFTime1 = LMWF[0, 0]
    HMWFTime2 = LMWF[-1, 0]

    LMWF_Period = 1. / GEX['Channel1']['RepFreq'][0]
    HMWF_Period = 1. / GEX['Channel2']['RepFreq'][0]

    # Check if full waveform is defined
    LMWF_isfull = (LMWFTime2 - LMWFTime1) == LMWF_Period
    HMWF_isfull = (HMWFTime2 - HMWFTime1) == HMWF_Period

    

    if not LMWF_isfull:
        LMWF = np.vstack((LMWF, [LMWFTime1 + LMWF_Period, 0]))

    if not HMWF_isfull:
        HMWF = np.vstack((HMWF, [HMWFTime1 + HMWF_Period, 0]))

    # Make sure the output folder exists
    if not os.path.isdir(stm_dir):
        os.mkdir(stm_dir)

    if len(file_gex) > 0:
        p, gex_f = os.path.split(file_gex)
        # get filename without extension
        gex_f = os.path.splitext(gex_f)[0]
        gex_str = gex_f + '_'
        # Remove next line when working OK
        gex_str = gex_f + '-P-'
    else:
        gex_str = ''

    LM_name = os.path.join(stm_dir, gex_str + system_name + '_LM.stm')
    HM_name = os.path.join(stm_dir, gex_str + system_name + '_HM.stm')

    stm_files = [LM_name, HM_name]
    if (showInfo>0):
        print('writing LM to %s'%(LM_name))
        print('writing HM to %s'%(HM_name))

    # WRITE LM AND HM FILES
    with open(LM_name, 'w') as fID_LM:
        fID_LM.write('System Begin\n')
        
        fID_LM.write('\tName = %s\n' % (GEX['General']['Description']))        
        fID_LM.write("\tType = Time Domain\n\n")

        fID_LM.write("\tTransmitter Begin\n")
        fID_LM.write("\t\tNumberOfTurns = 1\n")
        fID_LM.write("\t\tPeakCurrent = 1\n")
        fID_LM.write("\t\tLoopArea = 1\n")
        fID_LM.write("\t\tBaseFrequency = %f\n" % GEX['Channel1']['RepFreq'][0])
        fID_LM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_LM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_LM, GEX['General']['WaveformLM'], fmt='%23.6e', delimiter=' ')
        fID_LM.write("\t\tWaveFormCurrent End\n")
        fID_LM.write("\tTransmitter End\n\n")
        
        fID_LM.write("\tReceiver Begin\n")
        fID_LM.write("\t\tNumberOfWindows = %d\n" % NWin_LM)
        fID_LM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_LM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_LM, windows_LM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_LM, windows_LM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_LM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel1']['TiBLowPassFilter']

        fID_LM.write('\t\tLowPassFilter Begin\n')
        fID_LM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_LM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_LM.write('\t\tLowPassFilter End\n\n')
        
        fID_LM.write('\tReceiver End\n\n')
        
        fID_LM.write('\tForwardModelling Begin\n')
        #fID_LM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_LM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_LM.write('\t\tOutputType = dB/dt\n')
        fID_LM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_LM.write('\t\tXOutputScaling = 0\n')
        fID_LM.write('\t\tYOutputScaling = 0\n')
        fID_LM.write('\t\tZOutputScaling = 1\n')
        fID_LM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_LM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqLM)
        fID_LM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsLM)
        fID_LM.write('\tForwardModelling End\n\n')

        fID_LM.write('System End\n')

    with open(HM_name, 'w') as fID_HM:
        fID_HM.write('System Begin\n')
        fID_HM.write('\tName = %s\n' % (GEX['General']['Description']))
        fID_HM.write("\tType = Time Domain\n\n")

        fID_HM.write("\tTransmitter Begin\n")
        fID_HM.write("\t\tNumberOfTurns = 1\n")
        fID_HM.write("\t\tPeakCurrent = 1\n")
        fID_HM.write("\t\tLoopArea = 1\n")
        fID_HM.write("\t\tBaseFrequency = %f\n" % GEX['Channel2']['RepFreq'][0])
        fID_HM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_HM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_HM, GEX['General']['WaveformHM'], fmt='%23.6e', delimiter=' ')
        fID_HM.write("\t\tWaveFormCurrent End\n")
        fID_HM.write("\tTransmitter End\n\n")

        fID_HM.write("\tReceiver Begin\n")
        fID_HM.write("\t\tNumberOfWindows = %d\n" % NWin_HM)
        fID_HM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_HM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_HM, windows_HM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_HM, windows_HM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_HM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel2']['TiBLowPassFilter']
        
        fID_HM.write('\t\tLowPassFilter Begin\n')
        fID_HM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_HM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_HM.write('\t\tLowPassFilter End\n\n')
        
        fID_HM.write('\tReceiver End\n\n')

        fID_HM.write('\tForwardModelling Begin\n')
        #fID_HM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_HM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_HM.write('\t\tOutputType = dB/dt\n')
        fID_HM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_HM.write('\t\tXOutputScaling = 0\n')
        fID_HM.write('\t\tYOutputScaling = 0\n')
        fID_HM.write('\t\tZOutputScaling = 1\n')
        fID_HM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_HM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqHM)
        fID_HM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsHM)
        fID_HM.write('\tForwardModelling End\n\n')

        fID_HM.write('System End\n')

    return stm_files


def read_gex(file_gex, **kwargs):
    """
    Read a GEX file and parse its contents into a dictionary.

    :param str file_gex: The path to the GEX file.
    :param kwargs: Additional keyword arguments for customization.
    :type kwargs: dict
    :keyword int Nhank: The number of Hankel transform abscissae for both low and high frequency windows.
    :keyword int Nfreq: The number of frequencies per decade for both low and high frequency windows.
    :keyword int Ndig: The number of digits for waveform digitizing frequency.
    :keyword int showInfo: Flag to control the display of information. Default is 0.
    :return: A dictionary containing the parsed contents of the GEX file.
    :rtype: dict
    :raises FileNotFoundError: If the specified GEX file does not exist.
    """
    showInfo = kwargs.get('showInfo', 0)
    
    GEX = {}
    GEX['filename']=file_gex
    comment_counter = 1
    current_key = None

    # Check if file_gex exists
    if not os.path.exists(file_gex):
        raise FileNotFoundError(f"Error: file {file_gex} does not exist")

    with open(file_gex, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('/'):
                GEX[f'comment{comment_counter}'] = line[1:].strip()
                comment_counter += 1
            elif line.startswith('['):
                current_key = line[1:-1]
                GEX[current_key] = {}
            else:
                key_value = line.split('=')
                if len(key_value) == 2:
                    key, value = key_value[0].strip(), key_value[1].strip()
                    
                    try:                        
                        GEX[current_key][key] = np.fromstring(value, sep=' ')
                    #except ValueError:
                    except:
                        GEX[current_key][key] = value

                    if len(GEX[current_key][key])==0:
                        # value is probably a string
                        GEX[current_key][key]=value


    # WaveformLM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformLMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformLMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformLM'] = np.vstack(waveform_values)
    
    for key in waveform_keys:
        del GEX['General'][key]

    # WaveformHM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformHMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformHMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformHM']=np.vstack(waveform_values)

    for key in waveform_keys:
        del GEX['General'][key]

    # GateArray
    gate_keys = [key for key in GEX['General'].keys() if 'GateTime' in key]
    gate_keys.sort(key=lambda x: int(x.replace('GateTime', '')))

    gate_values = [GEX['General'][key] for key in gate_keys]
    GEX['General']['GateArray']=np.vstack(gate_values)

    for key in gate_keys:
        del GEX['General'][key]

    return GEX
    


# gex_to_stm: convert a GEX file to a set of STM files
def gex_to_stm(file_gex, **kwargs):
    """
    Convert a GEX file to STM files.

    :param file_gex: The path to the GEX file or a GEX dictionary.
    :type file_gex: str or dict
    :param kwargs: Additional keyword arguments to be passed to the write_stm_files function.
    :return: A tuple containing the STM files and the GEX dictionary.
    :rtype: tuple
    :raises TypeError: If the file_gex argument is not a string or a dictionary.
    :notes:
        - If the file_gex argument is a string, it is assumed to be the path to the GEX file, which will be read using the read_gex function.
        - If the file_gex argument is a dictionary, it is assumed to be a GEX dictionary.
        - The write_stm_files function is called to generate the STM files based on the GEX data.
    """
    if isinstance(file_gex, str):
        GEX = read_gex(file_gex)
        stm_files = write_stm_files(GEX, file_gex=file_gex, **kwargs)
    else:
        GEX = file_gex
        stm_files = write_stm_files(GEX, file_gex=GEX['filename'], **kwargs)

    return stm_files, GEX


def get_gex_file_from_data(f_data_h5, id=1):
    """
    Retrieves the 'gex' attribute from the specified HDF5 file.

    :param str f_data_h5: The path to the HDF5 file.
    :param int id: The ID of the dataset within the HDF5 file. Defaults to 1.
    :return: The value of the 'gex' attribute if found, otherwise an empty string.
    :rtype: str
    """
    with h5py.File(f_data_h5, 'r') as f:
        dname = '/D%d' % id
        if 'gex' in f[dname].attrs:
            file_gex = f[dname].attrs['gex']
        else:
            print('"gex" attribute not found in %s:%s' % (f_data_h5,dname))
            file_gex = ''
    return file_gex


def get_geometry(f_data_h5):
    """
    Retrieve geometry information from an HDF5 file.

    :param f_data_h5: The path to the HDF5 file. If a posterior HDF5 file is passed, the corresponding data file is read from the 'h5_data' attribute
    :type f_data_h5: str
    
    :returns: A tuple containing the X, Y, LINE, and ELEVATION arrays
    :rtype: tuple
    
    :raises IOError: If the HDF5 file cannot be opened or read
    
    .. note::
        Example usage:
        
        >>> X, Y, LINE, ELEVATION = get_geometry('/path/to/file.h5')
    """

    # if f_data_h5 has a feature called 'f5_prior' then use that file
    with h5py.File(f_data_h5, 'r') as f_data:
        if 'f5_data' in f_data.attrs:
            f_data_h5 = f_data.attrs['f5_data']
            print('Using f5_data_h5: %s' % f_data_h5)

    with h5py.File(f_data_h5, 'r') as f_data:
        X = f_data['/UTMX'][:].flatten()
        Y = f_data['/UTMY'][:].flatten()
        LINE = f_data['/LINE'][:].flatten()
        ELEVATION = f_data['/ELEVATION'][:].flatten()

    return X, Y, LINE, ELEVATION


def post_to_csv(f_post_h5='', Mstr='/M1'):
    """
    Convert a post-processing HDF5 file to a CSV file containing XYZ data.

    :param f_post_h5: Path to the post-processing HDF5 file. If not provided, the last used file will be used.
    :type f_post_h5: str
    :param Mstr: The dataset path within the HDF5 file. Default is '/M1'.
    :type Mstr: str

    :return: Path to the generated CSV file.
    :rtype: str

    :raises KeyError: If the specified dataset path does not exist in the HDF5 file.
    :raises FileNotFoundError: If the specified HDF5 file does not exist.
    """
    
    # TODO: Would be nice if also the LINE number was exported (to allow filter by LINE)
    # Perhaps this function should be split into two functions, 
    #   one for exporting the grid data and one for exporting the point data.
    # Also, split into a function the generates the points scatter data, and one that stores them as a csv file


    import pandas as pd
    import integrate as ig

    #Mstr = '/M1'
    # if f_post_h5 is Null then use the last f_post_h5 file

    if len(f_post_h5)==0:
        f_post_h5 = 'POST_PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5'

    f_post =  h5py.File(f_post_h5, 'r')
    f_prior_h5 = f_post.attrs['f5_prior']
    f_prior =  h5py.File(f_prior_h5, 'r')
    f_data_h5 = f_post.attrs['f5_data']
    if 'x' in f_prior[Mstr].attrs.keys():
        z = f_prior[Mstr].attrs['x']
    else:
        z = f_prior[Mstr].attrs['z']    
    is_discrete = f_prior[Mstr].attrs['is_discrete']

    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    # Check that Dstr exist in f_poyt_h5
    if Mstr not in f_post:
        print("ERROR: %s not in %s" % (Mstr, f_post_h5))
        sys.exit(1)

    D_mul = []
    D_name = []
    if is_discrete:
        D_mul.append(f_post[Mstr+'/Mode'])
        D_name.append('Mode')
        D_mul.append(f_post[Mstr+'/Entropy'])
        D_name.append('Entropy')
    else:
        D_mul.append(f_post[Mstr+'/Median'])
        D_name.append('Median')
        D_mul.append(f_post[Mstr+'/Mean'])
        D_name.append('Mean')
        D_mul.append(f_post[Mstr+'/Std'])
        D_name.append('Std')
    

    # replicate z[1::] to be a 2D matric of zie ndx89
    ZZ = np.tile(z[1::], (D_mul[0].shape[0], 1))

    #
    df = pd.DataFrame(data={'X': X, 'Y': Y, 'Line': LINE, 'ELEVATION': ELEVATION})

    dataframes = [df]

    for i in range(len(D_mul)):
        D = D_mul[i][:]
        
        for j in range(D.shape[1]):
            temp_df = pd.DataFrame(D[:,j], columns=[D_name[i]+'_'+str(j)])
            dataframes.append(temp_df)

    for j in range(ZZ.shape[1]):
        temp_df = pd.DataFrame(ZZ[:,j], columns=['zbot_'+str(j)])
        dataframes.append(temp_df)

    df = pd.concat(dataframes, axis=1)
    f_post_csv='%s_%s.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
    #f_post_csv='%s.csv' % (os.path.splitext(f_post_h5)[0])
    #f_post_csv = f_post_h5.replace('.h5', '.csv')
    print('Writing to %s' % f_post_csv)
    df.to_csv(f_post_csv, index=False)

    
    #%% Store point data sets of varianle in D_name
    # # Save a file with columns, x, y, z, and the median.
    print("----------------------------------------------------")
    D_mul_out = []
    for icat in range(len(D_name)):
        #icat=0
        Vstr = D_name[icat]
        print('Creating point data set: %s'  % Vstr)
        D=f_post[Mstr+'/'+Vstr]
        nd,nz=D.shape
        n = nd*nz

        Xp = np.zeros(n)
        Yp = np.zeros(n)
        Zp = np.zeros(n)
        LINEp = np.zeros(n)
        Dp = np.zeros(n)
        
        for i in range(nd):
            for j in range(nz):
                k = i*nz+j
                Xp[k] = X[i]
                Yp[k] = Y[i]
                Zp[k] = ELEVATION[i]-z[j]
                LINEp[k] = LINE[i]
                Dp[k] = D[i,j]        
        D_mul_out.append(Dp)

    if is_discrete:
        df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1] })
    else:
        df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1], D_name[2]: D_mul_out[2] })
    
    f_csv = '%s_%s_point.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
    print('- saving to : %s'  % f_csv)

    df.to_csv(f_csv, index=False)

    
    #%% CLOSE
    f_post.close()
    f_prior.close()

    return f_post_csv, f_csv


'''
HDF% related functions
'''
def copy_hdf5_file(input_filename, output_filename, N=None, loadToMemory=True, compress=True, **kwargs):
    """
    Copy the contents of an HDF5 file to another HDF5 file.

    :param input_filename: The path to the input HDF5 file.
    :type input_filename: str
    :param output_filename: The path to the output HDF5 file.
    :type output_filename: str
    :param N: The number of elements to copy from each dataset. If not specified, all elements will be copied.
    :type N: int, optional
    :param loadToMemory: Whether to load the entire dataset to memory before slicing. Default is True.
    :type loadToMemory: bool, optional
    :param compress: Whether to compress the output dataset. Default is True.
    :type compress: bool, optional

    :return: None
    """
    showInfo = kwargs.get('showInfo', 0)
    # Open the input file
    if showInfo>0:
        print('Trying to copy %s to %s' % (input_filename, output_filename))
    with h5py.File(input_filename, 'r') as input_file:
        # Create the output file
        with h5py.File(output_filename, 'w') as output_file:
            # Copy each group/dataset from the input file to the output file
            #i_use = np.sort(np.random.choice(400000,10,replace=False))
            i_use = []
            for name in input_file:
                if showInfo>0:
                    print('Copying %s' % name)
                if isinstance(input_file[name], h5py.Dataset):                    
                    # If N is specified, only copy the first N elements

                    if len(i_use)==0:
                        N_in = input_file[name].shape[0]
                        if N is None:
                            N=N_in
                        if N>N_in:
                            N=N_in
                        if N==N_in:                            
                            i_use = np.arange(N)
                        else:
                            i_use = np.sort(np.random.choice(N_in,N,replace=False))

                    if N<20000:
                        loadToMemory=False

                    # Read full dataset into memory
                    if loadToMemory:
                        # Load all data to memory, before slicing
                        if showInfo>0:
                            print('Loading %s to memory' % name)
                        data_in = input_file[name][:]    
                        data = data_in[i_use]
                    else:
                        # Read directly from HDF5 file   
                        data = input_file[name][i_use]

                    # Create new dataset in output file with compression
                    # Convert floating point data to 32-bit precision
                    if data.dtype.kind == 'f':
                        data = data.astype(np.float32)
                        
                    if compress:
                        #output_dataset = output_file.create_dataset(name, data=data, compression="lzf")
                        output_dataset = output_file.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                    else:
                        output_dataset = output_file.create_dataset(name, data=data)
                    # Copy the attributes of the dataset
                    for key, value in input_file[name].attrs.items():                        
                        output_dataset.attrs[key] = value
                else:
                    input_file.copy(name, output_file)

            # Copy the attributes of the input file to the output file
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value

        return output_filename

def hdf5_scan(file_path):
    """
    Scans an HDF5 file and prints information about datasets (including their size) and attributes.

    Args:
        file_path (str): The path to the HDF5 file.

    """
    import h5py
    with h5py.File(file_path, 'r') as f:
        def print_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Data type: {obj.dtype}")
                if obj.attrs:
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Group):
                if obj.attrs:
                    print(f"Group: {name}")
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")

        f.visititems(print_info)




def file_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.

    :param file_path: The path to the file.
    :type file_path: str
    :return: The MD5 checksum of the file.
    :rtype: str
    """
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def download_file(url, download_dir, use_checksum=False, **kwargs):
    """
    Download a file from a URL to a specified directory.

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param use_checksum: Whether to verify the file checksum after download.
    :type use_checksum: bool
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the file already exists locally
    if os.path.exists(file_path):
        if showInfo>0:
            print(f'File {file_name} already exists. Skipping download.')
        return

    # Check if the remote file exists
    if showInfo>1:
        print('Checking if file exists on the remote server...')
    head_response = requests.head(url)
    if head_response.status_code != 200:
        if showInfo>-1:
            print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Download and save the file
    print(f'Downloading {file_name}')
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {file_name}')

    # Check if checksum verification is enabled
    if use_checksum:
        # Calculate the MD5 checksum of the downloaded file
        downloaded_checksum = file_checksum(file_path)

        # Get the remote file checksum
        remote_checksum = head_response.headers.get('Content-MD5')

        # Compare checksums
        if downloaded_checksum != remote_checksum:
            print(f'Checksum verification failed for {file_name}. Downloaded file may be corrupted.')
            os.remove(file_path)
        else:
            print(f'Checksum verification successful for {file_name}.')
    else:
        pass
        # print(f'Checksum verification disabled for {file_name}.')

def download_file_old(url, download_dir, **kwargs):
    """
    Download a file from a URL to a specified directory (old version).

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the remote file exists
    head_response = requests.head(url)
    if head_response.status_code != 200:
        print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Check if the file already exists locally
    if os.path.exists(file_path):
        # Get the local file checksum
        local_checksum = file_checksum(file_path)

        # Download the remote file to a temporary location to compare checksums
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        remote_temp_path = os.path.join(download_dir, f'temp_{file_name}')
        with open(remote_temp_path, 'wb') as temp_file:
            temp_file.write(response.content)

        # Get the remote file checksum
        remote_checksum = file_checksum(remote_temp_path)

        # Compare checksums
        if local_checksum == remote_checksum:
            print(f'File {file_name} already exists and is identical. Skipping download.')
            os.remove(remote_temp_path)
            return
        else:
            print(f'File {file_name} exists but is different. Downloading new version.')
            os.remove(remote_temp_path)

    # Download and save the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    print(f'Downloading {file_name}')
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {file_name}')


def get_case_data(case='DAUGAARD', loadAll=False, loadType='', filelist=[], **kwargs):
    """
    Get case data for a specific case.

    :param case: The case name. Default is 'DAUGAARD'. Options are 'DAUGAARD', 'GRUSGRAV', 'FANGEL', 'HALD', 'ESBJERG', and 'OERUM.
    :type case: str
    :param loadAll: Whether to load all files for the case. Default is False.
    :type loadAll: bool
    :param loadType: The type of files to load. Options are '', 'prior', 'prior_data', 'post', and 'inout'.
    :type loadType: str
    :param filelist: A list of files to load. Default is an empty list.
    :type filelist: list
    :param kwargs: Additional keyword arguments.
    :return: A list of file names for the case.
    :rtype: list
    """
    showInfo = kwargs.get('showInfo', 0)

    if showInfo>-1:
        print('Getting data for case: %s' % case)

    if case=='DAUGAARD':

        if len(filelist)==0:
            filelist.append('DAUGAARD_AVG.h5')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('README_DAUGAARD')

        if loadAll:
            filelist.append('DAUGAARD_RAW.h5')
            filelist.append('TX07_20230731_2x4_RC20-33.gex')
            filelist.append('TX07_20230828_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('tTEM_20230727_AVG_export.h5')
            filelist.append('tTEM_20230814_AVG_export.h5')
            filelist.append('tTEM_20230829_AVG_export.h5')
            filelist.append('tTEM_20230913_AVG_export.h5')
            filelist.append('tTEM_20231109_AVG_export.h5')
            filelist.append('DAUGAARD_AVG_inout.h5')

        if (loadAll or loadType=='shapefiles'):            
            #filelist.append('Begravet dal.zip')
            filelist.append('Begravet dal.shp')
            filelist.append('Begravet dal.shx')
            #filelist.append('Erosion øvre.zip')
            filelist.append('Erosion øvre.shp')
            filelist.append('Erosion øvre.shx')
            
        
        if (loadAll or loadType=='prior'):            
            filelist.append('prior_detailed_general_N2000000_dmax90.h5')
        
        if (loadAll or loadType=='prior_data' or loadType=='post'):            
            filelist.append('prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            
        if (loadAll or loadType=='post'):
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
                    
        if (loadAll or loadType=='inout'):
            filelist.append('prior_detailed_invalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')    
            filelist.append('prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

    elif case=='ESBJERG':
        
        if (loadAll or loadType=='gex'):  
            filelist.append('TX07_20230906_2x4_RC20-33_merged.h5')  
            filelist.append('TX07_20231016_2x4_RC20-33_merged.h5')
            filelist.append('TX07_20231127_2x4x1_RC20_33_merged.h5')
            filelist.append('TX07_20240125_2x4_RC20-33_merged.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')
        
        if (loadAll or loadType=='premerge'):
            filelist.append('20230921_AVG_export.h5')
            filelist.append('20230922_AVG_export.h5')
            filelist.append('20230925_AVG_export.h5')
            filelist.append('20230926_AVG_export.h5')
            filelist.append('20231026_AVG_export.h5')
            filelist.append('20231027_AVG_export.h5')
            filelist.append('20240109_AVG_export.h5')
            filelist.append('20240313_AVG_export.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')
            
        

        if (loadAll or loadType=='ESBJERG_ALL' or len(filelist)==0):
            filelist.append('ESBJERG_ALL.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('README_ESBJERG')
   
        if (loadAll or loadType=='prior' or len(filelist)==0):
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90.h5')
            filelist.append('prior_Esbjerg_piggy_N2000000.h5')
            
        if (loadAll or loadType=='priordata' or len(filelist)==0):
            filelist.append('prior_Esbjerg_piggy_N2000000_TX07_20230906_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')


    elif case=='GRUSGRAV':

        filelist = []
        filelist.append('GRUSGRAV_AVG.h5')
        filelist.append('TX07_20230425_2x4_RC20_33.gex')
        filelist.append('README_GRUSGRAV')                    

        if (loadAll or loadType=='prior'):            
            filelist.append('DJURSLAND_P01_N1000000_NB-13_NR03_PRIOR.h5') 
            filelist.append('DJURSLAND_P03_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P13_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P40_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P02_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P12_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P34_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P60_N1000000_NB-13_NR03_PRIOR.h5')    
            
    elif case=='FANGEL':

        filelist = []
        filelist.append('FANGEL_AVG.h5')
        filelist.append('TX07_20230828_2x4_RC20-33.gex')
        filelist.append('README_FANGEL')

    elif case=='HALD':

        filelist = []
        filelist.append('HALD_AVG.h5')
        filelist.append('TX07_20230731_2x4_RC20-33.gex')
        filelist.append('README_HALD')
        if loadAll:
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('HALD_RAW.h5')
            filelist.append('tTEM_20230801_AVG_export.h5')
            filelist.append('tTEM_20230815_AVG_export.h5')
            filelist.append('tTEM_20230905_AVG_export.h5')
            filelist.append('tTEM_20231018_AVG_export.h5')

    elif case=='OERUM':
        filelist.append('OERUM_AVG.h5')
        filelist.append('TX07_20240802_2x4_RC20-39.gex')
        filelist.append('README_OERUM')
        if loadAll:
            filelist.append('OERUM_RAW.h5')
            filelist.append('20240827_AVG_export.h5')
            filelist.append('20240828_AVG_export.h5')
            filelist.append('20240903_AVG_export.h5')
            filelist.append('20240827_RAW_export.h5')
            filelist.append('20240828_RAW_export.h5')
            filelist.append('20240903_RAW_export.h5')
                  

    elif case=='HJOELLUND':
        filelist.append('HJOELLUND_AVG.h5')
        filelist.append('TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex')
        filelist.append('README_HJOELLUND')
        if loadAll:
            filelist.append('HJOELLUND_RAW.h5')

    elif case=='HADERUP':
        filelist.append('HADERUP_MEAN_ALL.h5')
        filelist.append('TX07_Haderup_mean.gex')
        filelist.append('README_HADERUP')
        #if loadAll:
        #    filelist.append('HADERUP_RAW.h5')
                  

    else:
        
        filelist = []
        print('Case %s not found' % case)


    urlErda = 'https://anon.erda.au.dk/share_redirect/dxOLKDtoul'
    urlErdaCase = '%s/%s' % (urlErda,case)
    for remotefile in filelist:
        #print(remotefile)
        remoteurl = '%s/%s' % (urlErdaCase,remotefile)
        #remoteurl = 'https://anon.erda.au.dk/share_redirect/dxOLKDtoul/%s/%s' % (case,remotefile)
        download_file(remoteurl,'.',showInfo=showInfo)
    if showInfo>-1:
        print('--> Got data for case: %s' % case)

    return filelist



def write_data_gaussian(D_obs, D_std = [], d_std=[], Cd=[], id=1, is_log = 0, f_data_h5='data.h5', **kwargs):
    """
    Write Gaussian noise data to an HDF5 file.
    
    This function writes observed data and its associated Gaussian noise standard deviations
    to an HDF5 file. It also creates necessary datasets if they do not exist and handles
    optional attributes for electromagnetic data processing.
    
    :param D_obs: Observed data array
    :type D_obs: numpy.ndarray
    :param D_std: Standard deviation of the observed data. If not provided, calculated using d_std
    :type D_std: list, optional
    :param d_std: Default standard deviation multiplier. Used if D_std is not provided
    :type d_std: list, optional
    :param Cd: Covariance data. If provided, it is written to the file
    :type Cd: list, optional
    :param id: Identifier for the dataset group
    :type id: int, optional
    :param is_log: Flag indicating if the data is in logarithmic scale
    :type is_log: int, optional
    :param f_data_h5: Path to the HDF5 file
    :type f_data_h5: str, optional
    :param kwargs: Additional keyword arguments
    :type kwargs: dict
    
    :returns: Path to the HDF5 file
    :rtype: str
    
    .. note::
        **Additional Parameters (kwargs):**
        
        - showInfo (int): Level of verbosity for printing information. Default is 0.
        - f_gex (str): Name of the GEX file associated with the data. Default is empty string.
        
        **Behavior:**
        
        - If D_std is not provided, it is calculated as d_std * D_obs
        - The function ensures that datasets 'UTMX', 'UTMY', 'LINE', and 'ELEVATION' exist
        - If a group with name 'D{id}' exists, it is removed before adding new data
        - Writes attributes 'noise_model' and 'is_log' to the dataset group
    """
    
    showInfo = kwargs.get('showInfo', 0)
    f_gex = kwargs.get('f_gex', '')

    if len(D_std)==0:
        if len(d_std)==0:
            d_std = 0.01
        D_std = np.abs(d_std * D_obs)

    D_str = 'D%d' % id

    ns,nd=D_obs.shape
    
    with h5py.File(f_data_h5, 'a') as f:
        # check if '/UTMX' exists and create it if it does not
        if 'UTMX' not in f:
            if showInfo>0:
                print('Creating %s:/UTMX' % f_data_h5) 
            UTMX = np.atleast_2d(np.arange(ns)).T
            f.create_dataset('UTMX' , data=UTMX) 
        if 'UTMY' not in f:
            if showInfo>0:
                print('Creating %s:/UTMY' % f_data_h5)
            UTMY = f['UTMX'][:]*0
            f.create_dataset('UTMY', data=UTMY)
        if 'LINE' not in f:
            if showInfo>0:
                print('Creating %s:/LINE' % f_data_h5)
            LINE = f['UTMX'][:]*0+1
            f.create_dataset('LINE', data=LINE)
        if 'ELEVATION' not in f:
            if showInfo>0:
                print('Creating %s:/ELEVATION' % f_data_h5)
            ELEVATION = f['UTMX'][:]*0
            f.create_dataset('ELEVATION', data=ELEVATION)

    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
            del f[D_str]

    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        f.create_dataset('/%s/d_obs' % D_str, data=D_obs)
        # Write either Cd or d_std
        if len(Cd) == 0:
            f.create_dataset('/%s/d_std' % D_str, data=D_std)
        else:
            f.create_dataset('/%s/Cd' % D_str, data=Cd)

        # wrote attribute noise_model
        f['/%s/' % D_str].attrs['noise_model'] = 'gaussian'
        f['/%s/' % D_str].attrs['is_log'] = is_log
        if len(f_gex)>0:
            f['/%s/' % D_str].attrs['gex'] = f_gex
    
    return f_data_h5

def write_data_multinomial(D_obs, i_use=None, id=[],  id_use=None, f_data_h5='data.h5', **kwargs):
    """
    Writes observed data to an HDF5 file in a specified group with a multinomial noise model.

    :param D_obs: The observed data array to be written to the file.
    :type D_obs: numpy.ndarray
    :param id: The ID of the group to write the data to. If not provided, the function will find the next available ID.
    :type id: list, optional
    :param id_use: The ID of PRIOR data the refer to this data. If not set, id_use=id
    :type id_use: list, optional
    :param f_data_h5: The path to the HDF5 file where the data will be written. Default is 'data.h5'.
    :type f_data_h5: str, optional
    :param kwargs: Additional keyword arguments.
    :return: The path to the HDF5 file where the data was written.
    :rtype: str
    """
    showInfo = kwargs.get('showInfo', 0)

    if np.ndim(D_obs)==1:
        D_obs = np.atleast_2d(D_obs).T

    # f_data_h5 is a HDF% file grousp "/D1/", "/D2". 
    # FInd the is with for the maximum '/D*' group
    if not id:
        with h5py.File(f_data_h5, 'a') as f:
            for id in range(1, 100):
                D_str = 'D%d' % id
                if D_str not in f:
                    break
        if showInfo>0:
            print('Using id=%d' % id)


    D_str = 'D%d' % id

    if showInfo>0:
        print("Trying to write %s to %s" % (D_str,f_data_h5))

    ns,nclass,nm=D_obs.shape

    if i_use is None:
        i_use = np.ones((ns,1))
    if np.ndim(D_obs)==1:
        i_use = np.atleast_2d(i_use).T
    
    if id_use is None:
        id_use = id
        
    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
                del f[D_str]


    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        f.create_dataset('/%s/d_obs' % D_str, data=D_obs)
        f.create_dataset('/%s/i_use' % D_str, data=i_use)
        
        f.create_dataset('/%s/id_use' % D_str, data=id_use)
            

        # write attribute noise_model as 'multinomial'
        f['/%s/' % D_str].attrs['noise_model'] = 'multinomial'
        
    return f_data_h5


def check_data(f_data_h5='data.h5', **kwargs):
    """
    Check and update INTEGRATE data in an HDF5 file.
    
    This function validates and ensures the presence of essential geometry datasets
    (UTMX, UTMY, LINE, ELEVATION) in an HDF5 file. If any datasets are missing,
    it creates them using provided values or sensible defaults based on existing data.
    
    :param f_data_h5: Path to the HDF5 file to check and update
    :type f_data_h5: str, optional
    :param kwargs: Additional keyword arguments for dataset values and configuration
    :type kwargs: dict
    
    :returns: None (modifies the HDF5 file in place)
    :rtype: None
    
    :raises KeyError: If the 'D1/d_obs' dataset is not found in the file and 'UTMX' is not provided
    
    .. note::
        **Supported Keyword Arguments:**
        
        - showInfo (int): Verbosity level. If greater than 0, prints information messages. Default is 0.
        - UTMX (array-like): Array of UTMX coordinate values. If not provided, attempts to read from file or generates defaults.
        - UTMY (array-like): Array of UTMY coordinate values. Default is zeros array with same length as UTMX.
        - LINE (array-like): Array of survey line identifiers. Default is ones array with same length as UTMX.
        - ELEVATION (array-like): Array of elevation values. Default is zeros array with same length as UTMX.
        
        **Behavior:**
        
        - If UTMX is not provided, function attempts to determine array length from existing 'D1/d_obs' dataset
        - Missing datasets are created with appropriate default values
        - Existing datasets are preserved and not overwritten
    """

    showInfo = kwargs.get('showInfo', 0)

    if showInfo>0:
        print('Checking INTEGRATE data in %s' % f_data_h5)  

    UTMX = kwargs.get('UTMX', [])
    if len(UTMX)==0:
        with h5py.File(f_data_h5, 'r') as f:
            if 'UTMX' in f:
                UTMX = f['UTMX'][:]
            else:
                ns = f['D1/d_obs'].shape[0] 
                print('UTMX not found in %s' % f_data_h5)
                UTMX = np.atleast_2d(np.arange(ns)).T    
            f.close()

    UTMY = kwargs.get('UTMY', UTMX*0)
    LINE = kwargs.get('LINE', UTMX*0+1)
    ELEVATION = kwargs.get('ELEVATION', UTMX*0)

    with h5py.File(f_data_h5, 'a') as f:
        # check if '/UTMX' exists and create it if it does not
        if 'UTMX' not in f:
            if showInfo>0:
                print('Creating UTMX')            
            f.create_dataset('UTMX', data=UTMX) 
        if 'UTMY' not in f:
            if showInfo>0:
                print('Creating UTMY')            
            f.create_dataset('UTMY', data=UTMY)
        if 'LINE' not in f:
            if showInfo>0:
                print('Creating LINE')
            f.create_dataset('LINE', data=LINE)
        if 'ELEVATION' not in f:
            if showInfo>0:
                print('Creating ELEVATION')
            f.create_dataset('ELEVATION', data=ELEVATION)

            f.close()




def merge_data(f_data, f_gex='', delta_line=0, f_data_merged_h5='', **kwargs):
    """
    Merge multiple data files into a single HDF5 file.

    :param f_data: List of input data files to merge.
    :type f_data: list
    :param f_gex: Path to geometry exchange file, by default ''.
    :type f_gex: str, optional
    :param delta_line: Line number increment for each merged file, by default 0.
    :type delta_line: int, optional
    :param f_data_merged_h5: Output merged HDF5 file path, by default derived from f_gex.
    :type f_data_merged_h5: str, optional
    :param kwargs: Additional keyword arguments.
    :return: Filename of the merged HDF5 file.
    :rtype: str
    :raises ValueError: If f_data is not a list.
    """
    
    import h5py
    import numpy as np
    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)

    if len(f_data_merged_h5) == 0:
        f_data_merged_h5 = f_gex.split('.')[0] + '_merged.h5'
    

    # CHeck the f_data is a list. If so return a error
    if not isinstance(f_data, list):
        raise ValueError('f_data must be a list of strings')

    nd = len(f_data)
    if showInfo:
        print('Merging %d data sets to %s ' % (nd, f_data_merged_h5))
    
    f_data_h5 = f_data[0]
    if showInfo>1:
        print('.. Merging ', f_data_h5)    
    Xc, Yc, LINEc, ELEVATIONc = ig.get_geometry(f_data_h5)
    Dc = ig.load_data(f_data_h5, showInfo=showInfo-1)
    d_obs_c = Dc['d_obs']
    d_std_c = Dc['d_std']
    noise_model = Dc['noise_model']

    for i in range(1, len(f_data)):
        f_data_h5 = f_data[i]                   
        if showInfo>1:
            print('.. Merging ', f_data_h5)    
        X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
        D = ig.load_data(f_data_h5, showInfo=showInfo)

        # append data
        Xc = np.append(Xc, X)
        Yc = np.append(Yc, Y)
        LINEc = np.append(LINEc, LINE+i*delta_line)
        ELEVATIONc = np.append(ELEVATIONc, ELEVATION)
        
        for id in range(len(d_obs_c)):
            #print(id)
            try:
                d_obs_c[id] = np.vstack((d_obs_c[id], np.atleast_2d(D['d_obs'][id])))        
                d_std_c[id] = np.vstack((d_std_c[id], np.atleast_2d(D['d_std'][id])))
            except:
                if showInfo>-1:
                    print("!!!!! Could not merge %s" % f_data_h5)

    Xc = np.atleast_2d(Xc).T
    Yc = np.atleast_2d(Yc).T
    LINEc = np.atleast_2d(LINEc).T
    ELEVATIONc = np.atleast_2d(ELEVATIONc).T

    with h5py.File(f_data_merged_h5, 'w') as f:
        f.create_dataset('UTMX', data=Xc)
        f.create_dataset('UTMY', data=Yc)
        f.create_dataset('LINE', data=LINEc)
        f.create_dataset('ELEVATION', data=ELEVATIONc)

    for id in range(len(d_obs_c)):
        write_data_gaussian(d_obs_c[id], D_std = d_std_c[id], noise_model = noise_model, f_data_h5=f_data_merged_h5, id=id+1, f_gex = f_gex)

    return f_data_merged_h5




## 

def merge_posterior(f_post_h5_files, f_data_h5_files, f_post_merged_h5=''):
    """
    Merge multiple posterior HDF5 files and their corresponding data files.
    
    This function combines multiple posterior sampling results and their associated
    observational data files into single merged files for comprehensive analysis.
    Useful for aggregating results from different survey areas or time periods.
    
    :param f_post_h5_files: List of file paths to the posterior HDF5 files to be merged
    :type f_post_h5_files: list of str
    :param f_data_h5_files: List of file paths to the data HDF5 files corresponding to the posterior files
    :type f_data_h5_files: list of str
    :param f_post_merged_h5: File path for the merged posterior HDF5 file. If empty, generates default name
    :type f_post_merged_h5: str, optional
    
    :returns: Tuple containing (merged_posterior_file_path, merged_data_file_path)
    :rtype: tuple
    
    :raises ValueError: If the length of f_data_h5_files is not the same as f_post_h5_files
    
    .. note::
        **File Naming:**
        
        - If f_post_merged_h5 is not provided, uses format: 'POST_merged_N{number_of_files}.h5'
        - Data file uses format: 'DATA_merged_N{number_of_files}.h5'
        
        **Dependencies:**
        
        - Requires the merge_data function to be available for merging observational data
        - Posterior files must have compatible structure for merging
        
        **Merging Process:**
        
        - Combines posterior sampling results from multiple files
        - Merges corresponding observational data
        - Maintains data integrity and structure consistency
    """
    import h5py
    import integrate as ig

    nf = len(f_post_h5_files)
    # Check that legth of f_data_h5_files is the same as f_post_h5_files
    if len(f_data_h5_files) != nf:
        raise ValueError('Length of f_data_h5_files must be the same as f_post_h5_files')

    if len(f_post_merged_h5) == 0:
        f_post_merged_h5 = 'POST_merged_N%d.h5' % nf

    f_data_merged_h5 = 'DATA_merged_N%d.h5' % nf

    f_data_merged_h5 = ig.merge_data(f_data_h5_files, f_data_merged_h5=f_data_merged_h5)


    for i in range(len(f_post_h5_files)):
        #  get 'i_sample' from the merged file
        f_post_h5 = f_post_h5_files[i]
        with h5py.File(f_post_h5, 'r') as f:
            i_use_s = f['i_use'][:]
            T_s = f['T'][:]
            EV_s = f['EV'][:]
            f_prior_h5 = f['/'].attrs['f5_prior']
            f_data_h5 = f['/'].attrs['f5_data']
            if i == 0:
                i_use = i_use_s
                T = T_s
                EV = EV_s 
            else:
                i_use = np.concatenate((i_use,i_use_s))
                T = np.concatenate((T,T_s))
                EV = np.concatenate((EV,EV_s))

    # Write the merged data to             
    with h5py.File(f_post_merged_h5, 'w') as f:
        f.create_dataset('i_use', data=i_use)
        f.create_dataset('T', data=T)
        f.create_dataset('EV', data=EV)
        f.attrs['f5_prior'] = f_prior_h5
        f.attrs['f5_data'] = f_data_merged_h5
        # ALSOE WRITE AN ATTRIBUET 'f5_data_mul' to the merged file
        #f.attrs['f5_data_files'] = f_data_h5_files


    return f_post_merged_h5, f_data_merged_h5




def read_usf(file_path: str) -> Dict[str, Any]:
    """
    Read a Universal Sounding Format (USF) file and parse its contents.
    
    Args:
        file_path: Path to the USF file
        
    Returns:
        Dictionary containing all parsed parameters from the USF file
    """
    # Initialize result dictionary
    usf_data = {}
    # Current sweep being processed
    current_sweep = None
    # List to store all sweeps
    sweeps = []
    # Flag to indicate if we're reading data points
    reading_points = False
    # Store data points for current sweep
    data_points = []
    # Store the dummy value
    dummy_value = None
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    
    # Process each line in the file
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Process variable declarations in comment lines (//XXX: YYY)
        if line.startswith('//') and ': ' in line and not line.startswith('//USF:'):
            # Extract variable name and value
            var_match = re.match(r"//([^:]+):\s*(.+)", line)
            if var_match:
                var_name, var_value = var_match.groups()
                var_name = var_name.strip()
                var_value = var_value.strip()
                
                # Process dummy value
                if var_name == 'DUMMY':
                    try:
                        dummy_value = float(var_value)
                    except ValueError:
                        dummy_value = var_value
                    usf_data[var_name] = dummy_value
                else:
                    # Try to convert to numeric if possible
                    try:
                        usf_data[var_name] = float(var_value)
                    except ValueError:
                        usf_data[var_name] = var_value
        
        # Process lines starting with a single '/'
        elif line.startswith('/') and not line.startswith('//'):
            # Check if it's an END marker
            if line == '/END':
                # This doesn't actually end the data reading - it just marks the end of the sweep header
                # We'll now be expecting a header line followed by data points
                reading_points = True
                continue
                
            # Check if it's a SWEEP_NUMBER marker
            if line.startswith('/SWEEP_NUMBER:'):
                # If we already have a sweep, add it to our list
                if current_sweep is not None:
                    sweeps.append(current_sweep)
                
                # Start a new sweep
                current_sweep = {}
                reading_points = False
                data_points = []
                
                # Extract sweep number
                sweep_match = re.match(r"/SWEEP_NUMBER:\s*(\d+)", line)
                if sweep_match:
                    sweep_number = int(sweep_match.group(1))
                    current_sweep['SWEEP_NUMBER'] = sweep_number
                continue
            
            # Check if it's a POINTS marker
            if line.startswith('/POINTS:'):
                points_match = re.match(r"/POINTS:\s*(\d+)", line)
                if points_match and current_sweep is not None:
                    current_sweep['POINTS'] = int(points_match.group(1))
                continue
                
            # Process other parameters
            param_match = re.match(r"/([^:]+):\s*(.+)", line)
            if param_match:
                param_name, param_value = param_match.groups()
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # Check if this is TX_RAMP which contains a complex list
                if param_name == 'TX_RAMP':
                    values = []
                    pairs = param_value.split(',')
                    for i in range(0, len(pairs), 2):
                        if i+1 < len(pairs):
                            try:
                                time_val = float(pairs[i].strip())
                                amp_val = float(pairs[i+1].strip())
                                values.append((time_val, amp_val))
                            except ValueError:
                                pass
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                # Check if parameter contains multiple values
                elif ',' in param_value:
                    values = []
                    for val in param_value.split(','):
                        val = val.strip()
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(val)
                    
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                    else:
                        usf_data[param_name] = values
                else:
                    # Try to convert to numeric if possible
                    try:
                        value = float(param_value)
                        if current_sweep is not None:
                            current_sweep[param_name] = value
                        else:
                            usf_data[param_name] = value
                    except ValueError:
                        if current_sweep is not None:
                            current_sweep[param_name] = param_value
                        else:
                            usf_data[param_name] = param_value
            
            # Check if we should start reading data points
            if line == '/CHANNEL: 1' or line == '/CHANNEL: 2':
                reading_points = True
                channel_match = re.match(r"/CHANNEL:\s*(\d+)", line)
                if channel_match and current_sweep is not None:
                    current_sweep['CHANNEL'] = int(channel_match.group(1))
                continue
                
        # Process data points
        elif reading_points and current_sweep is not None:
            # Check for the header line that comes after /END
            if line.strip().startswith('TIME,'):
                # Store the header names for this data block
                headers = [h.strip() for h in line.split(',')]
                current_sweep['DATA_HEADERS'] = headers
                
                # Initialize arrays for each data column
                for header in headers:
                    current_sweep[header] = []
                
                continue
                
            # Parse data point values
            values = line.split(',')
            if len(values) >= 6:
                try:
                    # Add each value to the corresponding array
                    for i, val in enumerate(values):
                        if i < len(headers):
                            # Try to convert to appropriate type
                            try:
                                if headers[i] == 'QUALITY':
                                    current_sweep[headers[i]].append(int(val.strip()))
                                else:
                                    current_sweep[headers[i]].append(float(val.strip()))
                            except ValueError:
                                current_sweep[headers[i]].append(val.strip())
                                
                except (ValueError, IndexError, NameError) as e:
                    # Skip problematic lines
                    pass
    
    # Add the last sweep if there is one
    if current_sweep is not None:
        sweeps.append(current_sweep)
    
    # Add sweeps to the result
    usf_data['SWEEP'] = sweeps


    # Extract d_obs as an array of usf_data['SWEEP'][0]['VOLTAGE'],usf_data['SWEEP'][1]['VOLTAGE'] ...
    # and store it a single 1D numpy array
    d_obs = np.concatenate([sweep['VOLTAGE'] for sweep in usf_data['SWEEP']])
    d_obs = np.array(d_obs)
    usf_data['d_obs'] = d_obs
    d_rel_err = np.concatenate([sweep['ERROR_BAR'] for sweep in usf_data['SWEEP']])
    d_rel_err = np.array(d_rel_err)
    usf_data['d_rel_err'] = d_rel_err
    time = np.concatenate([sweep['TIME'] for sweep in usf_data['SWEEP']])
    time = np.array(time)   
    usf_data['time'] = time
    # Add usf_data['id'] that is '0' for SWEEP1 and '1' for SWEEP2  etc
    # so, usf_data['id'] = [0,0,0,0,1,1,1,1,1] for 2 sweeps with 4 and 5 data points
    usf_data['id'] = np.concatenate([[i] * sweep['POINTS'] for i, sweep in enumerate(usf_data['SWEEP'])])
    usf_data['id'] = 1+np.array(usf_data['id'])
    # Add usf_data['dummy'] that is the dummy value
    usf_data['dummy'] = dummy_value
    # Add usf_data['file_name'] that is the file name
    usf_data['file_name'] = file_path.split('/')[-1]
    # Add usf_data['file_path'] that is the file path
    usf_data['file_path'] = file_path
    
    
    return usf_data


def test_read_usf(file_path: str) -> None:
    """
    Test function to read a USF file and print some key values.
    
    Args:
        file_path: Path to the USF file
    """
    usf = read_usf(file_path)
    
    print(f"DUMMY: {usf.get('DUMMY')}")
    print(f"SWEEPS: {usf.get('SWEEPS')}")
    
    for i, sweep in enumerate(usf.get('SWEEP', [])):
        print(f"\nSWEEP {i}:")
        print(f"CURRENT: {sweep.get('CURRENT')}")
        print(f"FREQUENCY: {sweep.get('FREQUENCY')}")
        print(f"POINTS: {sweep.get('POINTS')}")
        
        if 'TIME' in sweep and len(sweep['TIME']) > 0:
            print(f"First TIME value: {sweep['TIME'][0]}")
            print(f"First VOLTAGE value: {sweep['VOLTAGE'][0]}")
            print(f"Number of data points: {len(sweep['TIME'])}")
            print(f"Data headers: {sweep.get('DATA_HEADERS', [])}")
    



    return usf


def read_usf_mul(directory: str = ".", ext: str = ".usf") -> List[Dict[str, Any]]:
    """
    Read all USF files in a specified directory and return a list of USF data structures.
    
    Args:
        directory: Path to the directory containing USF files (default: current directory)
        ext: File extension to look for (default: ".usf")
        
    Returns:
        tuple containing:
            - np.ndarray: Array of observed data (d_obs) from all USF files
            - np.ndarray: Array of relative errors (d_rel_err) from all USF files
            - List[Dict[str, Any]]: List of USF data structures, each representing a single USF file

    """
    import os
    import glob
    from typing import List, Dict, Any

    # Make sure the extension starts with a period
    if not ext.startswith('.'):
        ext = '.' + ext
    
    # Get all matching files in the directory
    file_pattern = os.path.join(directory, f"*{ext}")
    usf_files = sorted(glob.glob(file_pattern))
    
    if not usf_files:
        print(f"No files with extension '{ext}' found in '{directory}'")
        return []
    
    # List to hold all USF data structures
    usf_list = []


    D_obs = []
    D_rel_err = []
    # Process each file
    for file_path in usf_files:
        try:
            # Read the USF file
            usf_data = read_usf(file_path)
            
            # Add the file name to the USF data structure
            usf_data['FILE_NAME'] = os.path.basename(file_path)

            D_obs.append(usf_data['d_obs'])
            D_rel_err.append(usf_data['d_rel_err'])

            # Add to the list
            usf_list.append(usf_data)
            
            print(f"Successfully read: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    D_obs = np.array(D_obs)
    D_rel_err = np.array(D_rel_err)

    print(f"Read {len(usf_list)} out of {len(usf_files)} files.")
    return D_obs, D_rel_err, usf_list




