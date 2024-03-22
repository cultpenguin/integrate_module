import os
import numpy as np
import h5py

#def write_stm_files(GEX, Nhank=140, Nfreq=6, Ndig=7, **kwargs):
def write_stm_files(GEX, **kwargs):
    """
    Write STM (System Transfer Matrix) files based on the provided GEX (Geophysical EXchange) data.

    Parameters:
    - GEX (dict): The GEX data containing the system information.
    - Nhank (int): The number of Hankel transform abscissae for both low and high frequency windows.
    - Nfreq (int): The number of frequencies per decade for both low and high frequency windows.
    - Ndig (int): The number of digits for waveform digitizing frequency.
    - file_gex (str): The path to the GEX file.
    - **kwargs: Additional keyword arguments for customization.

    Returns:
    - stm_files (list): A list of file paths for the generated STM files.
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

    Args:
        file_gex (str): The path to the GEX file.
        - Nhank (int): The number of Hankel transform abscissae for both low and high frequency windows.
        - Nfreq (int): The number of frequencies per decade for both low and high frequency windows.
        - Ndig (int): The number of digits for waveform digitizing frequency.
        - **kwargs: Additional keyword arguments for customization.

    Returns:
        dict: A dictionary containing the parsed contents of the GEX file.

    """
    showInfo = kwargs.get('showInfo', 0)
    
    GEX = {}
    GEX['filename']=file_gex
    comment_counter = 1
    current_key = None

    # Chec if file_gex exists
    if not os.path.exists(file_gex):
        print(f"Error: file {file_gex} does not exist")
        exit(1)

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

    Args:
        file_gex (str or dict): The path to the GEX file or a GEX dictionary.
        **kwargs: Additional keyword arguments to be passed to the write_stm_files function.

    Returns:
        tuple: A tuple containing the STM files and the GEX dictionary.

    Raises:
        TypeError: If the file_gex argument is not a string or a dictionary.

    Notes:
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

    Args:
        f_data_h5 (str): The path to the HDF5 file.
        id (int, optional): The ID of the dataset within the HDF5 file. Defaults to 1.

    Returns:
        str: The value of the 'gex' attribute if found, otherwise an empty string.
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
    with h5py.File(f_data_h5,'r') as f_data:
        X = f_data['/UTMX'][:].flatten()
        Y = f_data['/UTMY'][:].flatten()
        LINE = f_data['/LINE'][:].flatten()
        ELEVATION = f_data['/ELEVATION'][:].flatten()


    return X, Y, LINE, ELEVATION


def post_h5_to_xyz(f_post_h5='', Mstr='/M1'):

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

    f_post.close()
    f_prior.close()

    return f_post_csv
