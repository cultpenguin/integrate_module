"""
INTEGRATE Plotting Module - Visualization and Analysis Tools

This module provides comprehensive visualization capabilities for the INTEGRATE
geophysical data integration package. It creates publication-quality plots for
data analysis, prior/posterior visualization, and results interpretation.

Key Features:
    - 1D profile plots for layered earth models
    - 2D spatial mapping and interpolation
    - Data-model comparison visualizations
    - Statistical plots (uncertainty, probability maps)
    - Geometry and survey layout visualization
    - Customizable plotting styles and colormaps

Main Function Categories:
    - plot_profile_*(): 1D vertical profile plotting
    - plot_data_*(): Observational data visualization
    - plot_geometry(): Survey geometry and layout
    - plot_feature_2d(): 2D spatial parameter mapping
    - plot_T_EV(): Temperature and evidence visualization
    - plot_*_stats(): Statistical analysis plots

Plot Types:
    - Discrete categorical models (geological units)
    - Continuous parameter models (resistivity, conductivity)
    - Uncertainty and probability distributions
    - Data fit and residual analysis
    - Cumulative thickness and property maps

Output Formats:
    - Interactive matplotlib figures
    - High-resolution PNG/PDF export
    - Customizable figure sizing and DPI

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import os
import numpy as np
import h5py
import integrate as ig
import matplotlib.pyplot as plt


def plot_posterior_cumulative_thickness(f_post_h5, im=2, icat=[0], property='median', usePrior=False, **kwargs):
    """
    Plot posterior cumulative thickness for specified categories.

    Creates a scatter plot showing the spatial distribution of cumulative thickness
    statistics (median, mean, standard deviation, or relative standard deviation)
    for selected geological categories from posterior sampling results.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    im : int, optional
        Model index for thickness calculation (default is 2).
    icat : int or list of int, optional
        Category index or list of category indices to include in thickness calculation
        (default is [0]).
    property : {'median', 'mean', 'std', 'relstd'}, optional
        Statistical property to plot (default is 'median'):
        - 'median': median thickness
        - 'mean': mean thickness  
        - 'std': standard deviation of thickness
        - 'relstd': relative standard deviation (std/median)
    usePrior : bool, optional
        Whether to use prior data instead of posterior data (default is False).
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plot as PNG file (default False)
        - s : float, scatter point size (default 10)

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    Notes
    -----
    The function calls ig.posterior_cumulative_thickness() to compute thickness
    statistics and creates a 2D scatter plot with equal aspect ratio and colorbar.
    Output files are saved with descriptive names when hardcopy=True.
    """

    if isinstance(icat, int):
        icat = np.array([icat])

    out = ig.posterior_cumulative_thickness(f_post_h5, im=2, icat=icat, usePrior=usePrior, **kwargs)
    if not isinstance(out, tuple):
        # Then output failed
        return

    thick_mean = out[0] 
    thick_median = out[1] 
    thick_std = out[2] 
    class_names = out[3] 
    X = out[4] 
    Y = out[5] 

    # set hardcopy to True as kwarg if not already set
    kwargs.setdefault('hardcopy', False)
    kwargs.setdefault('s', 10)
    s = kwargs['s']

    fig = plt.figure(figsize=(8, 8))
    if property == 'median':
        plt.scatter(X, Y, c=thick_median, cmap='jet', s=s)
    elif property == 'mean':
        plt.scatter(X, Y, c=thick_mean, cmap='jet', s=s)
    elif property == 'std':
        plt.scatter(X, Y, c=thick_std, cmap='jet', s=s)
    elif property == 'relstd':
        thick_std_rel = thick_std / thick_median
        plt.scatter(X, Y, c=thick_std_rel, cmap='gray_r', s=s, vmin=0, vmax=2)

    plt.colorbar().set_label('Thickness (m)')
    title_txt = 'Cumulative Thickness - %s - %s ' % (property, class_names)
    if usePrior:
        title_txt = title_txt + ' - Prior'
    plt.title(title_txt)
    plt.grid()
    plt.axis('equal')

    if kwargs['hardcopy']:
        # get filename without extension
        icat_str = '-'.join([str(i) for i in icat])
        f_png = '%s_im%d_ic%s_%s' % (os.path.splitext(f_post_h5)[0], im, icat_str, property)
        if usePrior:
            f_png = f_png + '_prior'
        plt.savefig(f_png + '.png')
        # plt.show()

    return fig

def plot_feature_2d(f_post_h5, key='', i1=1, i2=1e+9, im=1, iz=0, uselog=1, title_text='', hardcopy=False, cmap=[], clim=[], **kwargs):
    """
    Create 2D spatial scatter plot of model parameter features.

    Generates a 2D scatter plot showing the spatial distribution of a specific
    model parameter feature from posterior sampling results. Supports both
    logarithmic and linear scaling with customizable colormaps and limits.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    key : str, optional
        Dataset key within the model group to plot. If empty string, uses
        the first available key in the model group (default is '').
    i1 : int, optional
        Starting data point index for plotting (1-based indexing, default is 1).
    i2 : float, optional
        Ending data point index for plotting. If larger than data size,
        uses all available data (default is 1e+9).
    im : int, optional
        Model index to plot from (e.g., 1 for M1, 2 for M2, default is 1).
    iz : int, optional
        Feature/layer index within the model parameter array (default is 0).
    uselog : int, optional
        Apply logarithmic normalization to color scale (1=True, 0=False, default is 1).
    title_text : str, optional
        Additional text to append to the plot title (default is '').
    hardcopy : bool, optional
        Save the plot as a PNG file (default is False).
    cmap : list or str, optional
        Colormap specification. If empty list, uses colormap from prior file
        attributes (default is []).
    clim : list, optional
        Color scale limits as [min, max]. If empty list, uses limits from
        prior file attributes (default is []).
    **kwargs : dict
        Additional keyword arguments passed to matplotlib scatter function.
        Common options include showInfo for debug output level.

    Returns
    -------
    None
        Function creates and displays the plot but does not return a value.

    Notes
    -----
    The function automatically retrieves geometry data and colormap/limit
    information from linked prior and data files. Plot files are saved with
    descriptive names including indices and feature information when hardcopy=True.
    """
    from matplotlib.colors import LogNorm

    showInfo = kwargs.get('showInfo', 0)

    #kwargs.setdefault('hardcopy', False)
    dstr = '/M%d' % im
    
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    

    with h5py.File(f_prior_h5,'r') as f_prior:
        if 'name' in f_prior[dstr].attrs:
            name = f_prior[dstr].attrs['name']
        else:    
            name = dstr

        
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    
    if showInfo>1:
        print("f_prior_h5 = %d" % f_prior_h5)
    clim_ref, cmap_ref = ig.get_clim_cmap(f_prior_h5,dstr)
    if len(cmap)==0:
        cmap = cmap_ref
    if len(clim)==0:
        clim = clim_ref

    if showInfo>2:
        print("clim=%s" % str(clim))
        print("cmap=%s" % str(cmap))
    
    nd = X.shape[0]
    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd

    if i2<i1:
        i2=i1+1

    if len(key)==0:
        with h5py.File(f_post_h5,'r') as f_post:
            key = list(f_post[dstr].keys())[0]
        print("No key was given. Using the first key found: %s" % key)

    if showInfo>0:
        print("Plotting Feature %d from %s/%s" % (iz, dstr,key))

    with h5py.File(f_post_h5,'r') as f_post:

        if dstr in f_post:
            if key in f_post[dstr].keys():
                D = f_post[dstr][key][:,iz][:]
                # plot this KEY
                plt.figure(1, figsize=(20, 10))
                if uselog:
                    plt.scatter(X[i1:i2],Y[i1:i2],c=D[i1:i2],
                                cmap = cmap,
                                norm=LogNorm(),
                                **kwargs)      
                else:        
                    plt.scatter(X[i1:i2],Y[i1:i2],c=D[i1:i2],
                                cmap = cmap,
                                **kwargs)      
                plt.grid()
                plt.xlabel('X')                
                plt.colorbar()
                plt.title("%s/%s[%d,:] %s %s" %(dstr,key,iz,title_text,name))
                plt.axis('equal')
                plt.clim(clim)
                
                if hardcopy:
                    f_png = '%s_%d_%d_%d_%s%02d_feature.png' % (os.path.splitext(f_post_h5)[0],i1,i2,im,key,iz)
                    plt.savefig(f_png)
                #plt.show()
                
            else:
                print("Key %s not found in %s" % (key, dstr))
    return


def plot_T_EV(f_post_h5, i1=1, i2=1e+9, s=5, T_min=1, T_max=100, pl='all', hardcopy=False, **kwargs):
    """
    Plot temperature and evidence field values from posterior sampling results.

    Creates 2D spatial scatter plots showing the distribution of temperature (T),
    evidence (EV), and number of data points across the survey area. Temperature
    indicates sampling efficiency while evidence shows data fit quality.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    i1 : int, optional
        Starting data point index for plotting (1-based indexing, default is 1).
    i2 : float, optional
        Ending data point index for plotting. If larger than data size,
        uses all available data (default is 1e+9).
    s : int, optional
        Size of scatter plot markers in points (default is 5).
    T_min : int, optional
        Minimum temperature value for color scale normalization (default is 1).
    T_max : int, optional
        Maximum temperature value for color scale normalization (default is 100).
    pl : {'all', 'T', 'EV', 'ND'}, optional
        Type of plot to generate (default is 'all'):
        - 'all': plot all three types
        - 'T': temperature field only
        - 'EV': evidence field only  
        - 'ND': number of data points only
    hardcopy : bool, optional
        Save plots as PNG files with descriptive names (default is False).
    **kwargs : dict
        Additional keyword arguments passed to matplotlib scatter function.

    Returns
    -------
    None
        Function creates and displays plots but does not return values.

    Notes
    -----
    Temperature values are displayed on log10 scale. Evidence values are
    clamped to reasonable ranges (1st to 99th percentile) for better visualization.
    The number of data plot shows non-NaN data count per location.
    """

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    clim=(T_min,T_max)

    with h5py.File(f_post_h5,'r') as f_post:
        T=f_post['/T'][:].T
        EV=f_post['/EV'][:].T
        try:
            T_mul=f_post['/T_mul'][:]
        except:
            T_mul=[]

        try:
            EV_mul=f_post['/EV_mul'][:]
        except:
            EV_mu=[]

    nd = X.shape[0]
    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd

    if i2<i1:
        i2=i1+1
    
    if (pl=='all') or (pl=='T'):
        plt.figure(1, figsize=(20, 10))
        plt.scatter(X[i1:i2],Y[i1:i2],c=np.log10(T[i1:i2]),s=s,cmap='jet',**kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.clim(np.log10(clim))      
        plt.colorbar(label='log10(T)')
        plt.title('Temperature')
        plt.axis('equal')
        if hardcopy:
            # get filename without extension        
            f_png = '%s_%d_%d_T.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
            plt.savefig(f_png)
            plt.show()

    if (pl=='all') or (pl=='EV'):
        # get the 99% percentile of EV values
        EV_max = np.percentile(EV,99)
        EV_max = 0
        EV_min = np.percentile(EV,1)
        EV_min = -100
        
        #if 'vmin' not in kwargs:
        #    kwargs['vmin'] = EV_min
        #if 'vmax' not in kwargs:
        #    kwargs['vmax'] = EV_max
        #print('EV_min=%f, EV_max=%f' % (EV_min, EV_max))
        plt.figure(2, figsize=(20, 10))
        plt.scatter(X[i1:i2],Y[i1:i2],c=EV[i1:i2],s=s,cmap='jet_r', vmin = EV_min, vmax=EV_max, **kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.title('log(EV)')
        plt.axis('equal')
        if hardcopy:
            # get filename without extension
            f_png = '%s_%d_%d_EV.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
            plt.savefig(f_png)
            plt.show()
    if (pl=='all') or (pl=='ND'):
        # 
        f_data = h5py.File(f_data_h5,'r')
        ndata,ns = f_data['/%s' % 'D1']['d_obs'].shape
        # find number of nan values on d_obs
        non_nan = np.sum(~np.isnan(f_data['/%s' % 'D1']['d_obs']), axis=1)
        #print(non_nan)

        plt.figure(3, figsize=(20, 10))
        plt.scatter(X[i1:i2],Y[i1:i2],c=non_nan[i1:i2],s=s,cmap='jet', **kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Number of Data')
        plt.title('N data')
        plt.axis('equal')
        if hardcopy:
            # get filename without extension
            f_png = '%s_%d_%d_ND.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
            plt.savefig(f_png)
            plt.show()
            

    return


def plot_geometry(f_data_h5, i1=0, i2=0, ii=np.array(()), s=5, pl='all', hardcopy=False, **kwargs):
    """
    Plot survey geometry data from INTEGRATE HDF5 files.

    Creates 2D scatter plots showing the spatial distribution of survey lines,
    elevation data, and data point indices. Useful for visualizing survey
    layout and data coverage.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing geometry data. Can be either a data
        file or posterior file (function automatically detects and uses correct file).
    i1 : int, optional
        Starting index for data points to plot (0-based indexing, default is 0).
    i2 : int, optional
        Ending index for data points to plot. If 0, uses all available data
        (default is 0).
    ii : numpy.ndarray, optional
        Specific array of indices to plot. If provided, overrides i1 and i2
        (default is empty array).
    s : int, optional
        Size of scatter plot markers in points (default is 5).
    pl : {'all', 'LINE', 'ELEVATION', 'id'}, optional
        Type of geometry plot to generate (default is 'all'):
        - 'all': plot all geometry types
        - 'LINE': survey line numbers only
        - 'ELEVATION': elevation data only
        - 'id': data point indices only
    hardcopy : bool, optional
        Save plots as PNG files with descriptive names (default is False).
    **kwargs : dict
        Additional keyword arguments passed to matplotlib scatter function.

    Returns
    -------
    None
        Function creates and displays plots but does not return values.

    Notes
    -----
    The function automatically extracts geometry data (X, Y, LINE, ELEVATION)
    and creates equal-aspect plots with appropriate colorbars and grid lines.
    All data points are shown as light gray background with selected points highlighted.
    """
    import h5py
    # Test if f_data_h5 is in fact f_post_h5 type file
    with h5py.File(f_data_h5,'r') as f_data:
        if 'f5_prior' in f_data['/'].attrs:
            f_data_h5 = f_data['/'].attrs['f5_data']
    print('f_data_h5=%s' % f_data_h5)        
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    
    nd = X.shape[0]

    if len(ii)==0:
        if i1==0:
            i1=0
        if i2==0:
            i2=nd
        if i2<i1:
            i2=i1+1
        if i1<1: 
            i1=0
        if i2>nd-1:
            i2=nd
        ii = np.arange(i1,i2)


    tit = f_png = '%s_%d_%d.png' % (os.path.splitext(f_data_h5)[0],i1,i2)

    if (pl=='all') or (pl=='LINE'):
        plt.figure(1, figsize=(20, 10))
        plt.plot(X,Y,'.',color='lightgray', zorder=-1, markersize=1)
        plt.scatter(X[ii],Y[ii],c=LINE[ii],s=s,cmap='jet',**kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='LINE')
        plt.title('%s - LINE' % tit)
        plt.axis('equal')
        if hardcopy:
            # get filename without extension        
            f_png = '%s_%d_%d_LINE.png' % (os.path.splitext(f_data_h5)[0],i1,i2)
            plt.savefig(f_png)
        plt.show()
    

    if (pl=='all') or (pl=='ELEVATION'):
        plt.figure(1, figsize=(20, 10))
        plt.scatter(X[ii],Y[ii],c=ELEVATION[ii],s=s,cmap='jet',**kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='ELEVATION')
        plt.title('ELEVATION')
        plt.axis('equal')
        if hardcopy:
            # get filename without extension        
            f_png = '%s_%d_%d_ELEVATION.png' % (os.path.splitext(f_data_h5)[0],i1,i2)
            plt.savefig(f_png)
        plt.show()

    if (pl=='all') or (pl=='id'):
        plt.figure(1, figsize=(20, 10))
        plt.scatter(X[ii],Y[ii],c=ii,s=s,cmap='jet',**kwargs)  
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='id')
        plt.title('id')
        plt.axis('equal')
        if hardcopy:
            # get filename without extension        
            f_png = '%s_%d_%d_id.png' % (os.path.splitext(f_data_h5)[0],i1,i2)
            plt.savefig(f_png)

    return



def plot_profile(f_post_h5, i1=1, i2=1e+9, im=0, **kwargs):
    """
    Plot 1D profiles from posterior sampling results.
    
    This function creates vertical profile plots showing the posterior distribution
    of model parameters as a function of depth or model layer. Automatically
    detects model type (discrete or continuous) and calls appropriate plotting function.
    
    :param f_post_h5: Path to the HDF5 file containing posterior sampling results
    :type f_post_h5: str
    :param i1: Starting index for the data points to plot (1-based indexing)
    :type i1: int, optional
    :param i2: Ending index for the data points to plot (1-based indexing)
    :type i2: int, optional
    :param im: Model identifier to plot. If 0, automatically detects and plots all models
    :type im: int, optional
    :param kwargs: Additional plotting arguments passed to discrete/continuous plotting functions
    :type kwargs: dict
    
    :returns: None (creates matplotlib plots)
    :rtype: None
    
    .. note::
        The function automatically computes posterior statistics if not present in the file.
        For discrete models, calls plot_profile_discrete(). For continuous models,
        calls plot_profile_continuous().
    """

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    # Check if M1 exist in f_post_h5
    updatePostStat = False
    with h5py.File(f_post_h5,'r') as f_post:
        if '/M1' not in f_post:
            print('No posterior stats found in %s - computing them now' % f_post_h5)
            updatePostStat = True
    if updatePostStat:
            ig.integrate_posterior_stats(f_post_h5)
            
    if (im==0):
        print('Plot profile for all model parameters')

        with h5py.File(f_prior_h5,'r') as f_prior:
            for key in f_prior.keys():
                im = int(key[1:])
                try:
                    if key[0]=='M':
                        plot_profile(f_post_h5, i1, i2, im=im, **kwargs)
                except:
                    print('Error in plot_profile for key=%s' % key)
        return 
    
    
    Mstr = '/M%d' % im
    with h5py.File(f_prior_h5,'r') as f_prior:
        is_discrete = f_prior[Mstr].attrs['is_discrete']    
    #print(Mstr)
    #print(is_discrete)
    if is_discrete:
        plot_profile_discrete(f_post_h5, i1, i2, im, **kwargs)
    elif not is_discrete:
        plot_profile_continuous(f_post_h5, i1, i2, im, **kwargs)


def plot_profile_discrete(f_post_h5, i1=1, i2=1e+9, im=1, **kwargs):
    """
    Create vertical profile plots for discrete categorical model parameters.

    Generates a 4-panel plot showing discrete model parameter distributions
    with depth, including mode, entropy, and combined mode-entropy views,
    plus temperature and evidence curves. Designed for geological unit
    classification results.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    i1 : int, optional
        Starting data point index for profile plotting (1-based indexing, default is 1).
    i2 : float, optional
        Ending data point index for profile plotting. If larger than data size,
        uses all available data (default is 1e+9).
    im : int, optional
        Model index to plot (e.g., 1 for M1, 2 for M2, default is 1).
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plot as PNG file (default False)
        - txt : str, additional text for filename
        - showInfo : int, level of debug output (0=none, >0=verbose)
        - clim : list, color scale limits for discrete classes

    Returns
    -------
    None
        Function creates and displays the plot but does not return values.

    Notes
    -----
    The plot consists of four vertically stacked panels:
    1. Mode: most probable class at each depth/location
    2. Entropy: uncertainty measure (0=certain, 1=maximum uncertainty) 
    3. Mode with transparency: mode colored by entropy (transparent=uncertain)
    4. Temperature and evidence curves
    
    Class names and colors are automatically retrieved from prior file attributes.
    Depth coordinates are computed relative to surface elevation.
    """
    from matplotlib.colors import LogNorm

    kwargs.setdefault('hardcopy', False)
    txt = kwargs.get('txt','')
    showInfo = kwargs.get('showInfo', 0)
    
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im

    if showInfo>0:
        print("Plotting profile %s from %s" % (Mstr, f_post_h5))

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
            cmap = plt.cm.jet(np.linspace(0, 1, n_class)).T
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap.T)            

    if not is_discrete:
        print("%s refers to a continuous model. Use plot_profile_continuous instead" % Mstr)

    with h5py.File(f_post_h5,'r') as f_post:
        Mode=f_post[Mstr+'/Mode'][:].T
        Entropy=f_post[Mstr+'/Entropy'][:].T
        P=f_post[Mstr+'/P'][:]
        T=f_post['/T'][:].T
        EV=f_post['/EV'][:].T
        try:
            EV=f_post['/EV_mul'][:]
        except:
            a=1

    nm = Mode.shape[0]
    if nm<=1:
        print('Only nm=%d, model parameters. no profile will be plot' % (nm))
        return 1

    nd = LINE.shape[0]
    id = np.arange(nd)
    # Create a meshgrid from X and Y
    XX, ZZ = np.meshgrid(X,z)
    YY, ZZ = np.meshgrid(Y,z)
    ID, ZZ = np.meshgrid(id,z)

    ID = np.sort(ID, axis=0)
    ZZ = np.sort(ZZ, axis=0)

    # compute the depth from the surface plus the elevation
    for i in range(nd):
        ZZ[:,i] = ELEVATION[i]-ZZ[:,i]

    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd-1

    # Get center of grid cells
    IID = ID[:,i1:i2]
    IIZ = ZZ[:,i1:i2]
    # IID, IIZ is the center of the cell. Create new grids, DDc, ZZc, that hold the the cordńers if the grids. 
    # DDc should have cells of size 1, while ZZc should be the same as ZZ but with a row added at the bottom that is the same as the last row of ZZ plus 100
    DDc = np.zeros((IID.shape[0]+1,IID.shape[1]+1))
    ZZc = np.zeros((IID.shape[0]+1,IID.shape[1]+1))
    DDc[:-1,:-1] = IID - 0.5
    DDc[:-1,-1] = IID[:,-1] + 0.5
    DDc[-1,:] = DDc[-2,:] + 1

    ZZc[:-1,:-1] = IIZ
    ZZc[-1,:] = ZZc[-2,:] + 1
  
    # ii is a numpy array from i1 to i2
    # ii = np.arange(i1,i2)

    # Create a figure with 3 subplots sharing the same Xaxis!
    fig, ax = plt.subplots(4,1,figsize=(20,10), gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    # MODE
    im1 = ax[0].pcolormesh(DDc, ZZc, Mode[:,i1:i2], 
            cmap=cmap,            
            shading='auto')
    im1.set_clim(clim[0]-.5,clim[1]+.5)        

    ax[0].set_title('Mode')
    # /fix set the ticks to be 1 to n_class, and use class_name as tick labels
    cbar1 = fig.colorbar(im1, ax=ax[0], label='label')
    cbar1.set_ticks(np.arange(n_class)+1)
    cbar1.set_ticklabels(class_name)
    cbar1.ax.invert_yaxis()

    # ENTROPY
    im2 = ax[1].pcolormesh(DDc, ZZc, Entropy[:,i1:i2],
            cmap='hot_r', 
            shading='auto')
    im2.set_clim(0,1)
    ax[1].set_title('Entropy')
    fig.colorbar(im2, ax=ax[1], label='Entropy')

    # MODE with transparency set using entropy
    im3 = ax[2].pcolormesh(DDc, ZZc, Mode[:,i1:i2],
            cmap=cmap, 
            shading='auto',
            alpha=1-Entropy[:,i1:i2])
    im3.set_clim(clim[0]-.5,clim[1]+.5)
    ax[2].set_title('Mode with transparency')
    #fig.colorbar(im3, ax=ax[2], label='label')
    cbar3 = fig.colorbar(im3, ax=ax[2], label='label')
    cbar3.set_ticks(np.arange(n_class)+1)
    cbar3.set_ticklabels(class_name)
    cbar3.ax.invert_yaxis()

    ## T and V
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    
    im4 = ax[3].semilogy(ID[0,i1:i2],T[i1:i2], 'k', label='T')
    plt.semilogy(ID[0,i1:i2],-EV[i1:i2], 'r', label='-log(EV)')
    plt.tight_layout()
    ax[3].set_xlim(ID[0,i1], ID[0,i2])
    ax[3].set_ylim(0.99, 200)
    ax[3].legend(loc='upper right')
    plt.grid(True)

    # Create an invisible colorbar for the last subplot
    cbar4 = fig.colorbar(im3, ax=ax[3])
    cbar4.solids.set(alpha=0)
    cbar4.outline.set_visible(False)
    cbar4.ax.set_yticks([])  # Hide the colorbar ticks
    cbar4.ax.set_yticklabels([])  # Hide the colorbar ticks labels


    # get filename without extension
    if kwargs['hardcopy']:
        f_png = '%s_%d_%d_profile_%s%s.png' % (os.path.splitext(f_post_h5)[0],i1,i2,Mstr[1:],txt)
        plt.savefig(f_png)
    plt.show()

    return

def plot_profile_continuous(f_post_h5, i1=1, i2=1e+9, im=1, **kwargs):
    """
    Create vertical profile plots for continuous model parameters.

    Generates a 4-panel plot showing continuous parameter distributions with depth,
    including mean/median values, standard deviation, and temperature/evidence curves.
    Supports transparency based on uncertainty for better visualization.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    i1 : int, optional
        Starting data point index for profile plotting (1-based indexing, default is 1).
    i2 : float, optional
        Ending data point index for profile plotting. If larger than data size,
        uses all available data (default is 1e+9).
    im : int, optional
        Model index to plot (e.g., 1 for M1, 2 for M2, default is 1).
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plot as PNG file (default False)
        - cmap : str or colormap, color scheme for plotting (default 'jet')
        - key : {'Mean', 'Median'}, statistic to plot (default 'Median')
        - alpha : float, transparency scaling factor relative to standard deviation.
          alpha=0 means no transparency, alpha=0.8 means full transparency when std>0.8
        - txt : str, additional text for filename
        - showInfo : int, level of debug output (0=none, >0=verbose)
        - clim : list, color scale limits [min, max]

    Returns
    -------
    None
        Function creates and displays the plot but does not return values.

    Notes
    -----
    The plot layout adapts based on data dimensionality:
    
    For multi-layer models (nm > 1):
    - Panel 1: Mean or median values with logarithmic color scale
    - Panel 2: Standard deviation with grayscale colormap  
    - Panel 3: Temperature and evidence curves
    
    For single-parameter models (nm = 1):
    - Panel 1: Line plot with mean ± 2*std confidence bounds
    - Panel 2: Temperature and evidence curves
    
    Transparency can be applied based on uncertainty levels when alpha > 0.
    Depth coordinates are computed relative to surface elevation.
    """
    from matplotlib.colors import LogNorm

    kwargs.setdefault('hardcopy', False)
    kwargs.setdefault('cmap', 'jet')
    
    alpha = kwargs.get('alpha',0.0)
    key = kwargs.get('key','Median')
    txt = kwargs.get('txt','')
    showInfo = kwargs.get('showInfo', 0)
    
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    with h5py.File(f_prior_h5,'r') as f_prior:
        if 'name' in f_prior['/M%d' % im].attrs:
            name = f_prior['/M%d' % im].attrs['name']
        else:
            name='M%d' % im
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im

    if showInfo>0:
        print("Plotting profile %s from %s" % (Mstr, f_post_h5))

    with h5py.File(f_prior_h5,'r') as f_prior:
        if 'z' in f_prior[Mstr].attrs.keys():
            z = f_prior[Mstr].attrs['z'][:].flatten()
        elif 'x' in f_prior[Mstr].attrs.keys():
            z = f_prior[Mstr].attrs['x'][:].flatten()
        else:
            z=np.array(0)
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
        if 'cmap' in f_prior[Mstr].attrs.keys():
            cmap = f_prior[Mstr].attrs['cmap'][:]
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cmap.T)
        else:
            cmap = kwargs['cmap']

        if showInfo>1:
            print(cmap)
            print(clim)

    if is_discrete:
        print("%s refers to a discrete model. Use plot_profile_discrete instead" % Mstr)


    with h5py.File(f_post_h5,'r') as f_post:
        Mean=f_post[Mstr+'/Mean'][:].T
        Median=f_post[Mstr+'/Median'][:].T
        Std=f_post[Mstr+'/Std'][:].T
        T=f_post['/T'][:].T
        EV=f_post['/EV'][:].T
        try:
            EV=f_post['/EV_mul'][:]
        except:
            a=1

    # Compute alpha matrix 'A' such that 
    # any values with Std<alpha are fully solid
    # any values with Std>2*alpha are transparent
    # linear interpolation between 0 and 1 elsewhere
    A = np.zeros(Std.shape)+alpha
    if alpha>0:
        A = (Std-alpha)/(2*alpha)
        A[A<0] = 0
        A[A>1] = 1
        A=1-A
    
    nm = Mean.shape[0]
    if nm<=1:
        pass
        #print('Only nm=%d, model parameters. no profile will be plot' % (nm))
        #return 1

    # Check for out of range
    nd = LINE.shape[0]
    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd-1
    id = np.arange(nd)
    
    if nm>=1:
        # Create a meshgrid from X and Y
        XX, ZZ = np.meshgrid(X,z)
        YY, ZZ = np.meshgrid(Y,z)
        ID, ZZ = np.meshgrid(id,z)

        ID = np.sort(ID, axis=0)
        ZZ = np.sort(ZZ, axis=0)

        # compute the depth from the surface plus the elevation
        for i in range(nd):
            ZZ[:,i] = ELEVATION[i]-ZZ[:,i]




        # Get center of grid cells
        IID = ID[:,i1:i2]
        IIZ = ZZ[:,i1:i2]
        # IID, IIZ is the center of the cell. Create new grids, DDc, ZZc, that hold the the cordńers if the grids. 
        # DDc should have cells of size 1, while ZZc should be the same as ZZ but with a row added at the bottom that is the same as the last row of ZZ plus 100
        DDc = np.zeros((IID.shape[0]+1,IID.shape[1]+1))
        ZZc = np.zeros((IID.shape[0]+1,IID.shape[1]+1))
        DDc[:-1,:-1] = IID - 0.5
        DDc[:-1,-1] = IID[:,-1] + 0.5
        DDc[-1,:] = DDc[-2,:] + 1

        ZZc[:-1,:-1] = IIZ
        ZZc[-1,:] = ZZc[-2,:] + 1
    
    # Create a figure with 3 subplots sharing the same Xaxis!
    fig, ax = plt.subplots(4,1,figsize=(20,10), gridspec_kw={'height_ratios': [3, 3, 3, 1]})
    
    # Set ax[0] to be invisible
    ax[0].axis('off')

    if (nm>1)&(key=='Mean'):
        isp=1
        # MEAN
        im1 = ax[isp].pcolormesh(DDc, ZZc, Mean[:,i1:i2], 
                cmap=cmap,            
                shading='auto',
                norm=LogNorm())
        im1.set_clim(clim[0],clim[1])        
        # if transp>0, set alpha
        if alpha>0:
            im1.set_alpha(A[:,i1:i2])
        ax[isp].set_title('Mean %s' % name)
        fig.colorbar(im1, ax=ax[isp], label='%s' % name)
    
    if (nm>1)&(key=='Median'):
        isp=1
        # MEDIAN
        im2 = ax[isp].pcolormesh(DDc, ZZc, Median[:,i1:i2], 
                cmap=cmap,            
                shading='auto',
                norm=LogNorm())  # Set color scale to logarithmic
        im2.set_clim(clim[0],clim[1])        
        if alpha>0:
            im2.set_alpha(A[:,i1:i2])
        ax[isp].set_title('Median %s' % name)
        fig.colorbar(im2, ax=ax[isp], label='%s' % name) 

    if nm>1:
        isp=2
        # STD
        import matplotlib
        im3 = ax[isp].pcolormesh(DDc, ZZc, Std[:,i1:i2], 
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red"]), 
                    shading='auto')
        im3.set_clim(0,1)
        ax[isp].set_title('Std %s' % name)
        fig.colorbar(im3, ax=ax[isp], label='Standard deviation (Ohm.m)')
    else:
        isp=2
        
        im3 = ax[2].plot(id[i1:i2],Mean[:,i1:i2].T, 'k', label='Mean')
        ax[2].plot(id[i1:i2],Mean[:,i1:i2].T+2*Std[:,i1:i2].T, 'k:', label='P97.5')
        ax[2].plot(id[i1:i2],Mean[:,i1:i2].T-2*Std[:,i1:i2].T, 'k:', label='P2.5')
        
        ax[2].plot(id[i1:i2],Median[:,i1:i2].T, 'r', label='Median')
        # add legend
        ax[2].legend(loc='upper right')
        # add grd on
        ax[2].grid(True)
        # hide ax[0]

        # set axis on ax[3] to be the same as on ax[4]
        ax[2].set_xlim(i1,i2)
        ax[2].set_title(name)
        ax[0].axis('off')
        ax[1].axis('off')
        
    ## T and V
    #ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    
    im4 = ax[3].semilogy(ID[0,i1:i2],T[i1:i2], 'k', label='T')
    plt.semilogy(ID[0,i1:i2],-EV[i1:i2], 'r', label='-log(EV)')
    plt.tight_layout()
    ax[3].set_xlim(ID[0,i1], ID[0,i2])
    ax[3].set_ylim(0.99, 200)
    ax[3].legend(loc='upper right')
    plt.grid(True)

    if nm>1:
        # Create an invisible colorbar for the last subplot
        cbar4 = fig.colorbar(im3, ax=ax[3])
        cbar4.solids.set(alpha=0)
        cbar4.outline.set_visible(False)
        cbar4.ax.set_yticks([])  # Hide the colorbar ticks
        cbar4.ax.set_yticklabels([])  # Hide the colorbar ticks labels


    # get filename without extension
    if kwargs['hardcopy']:
        f_png = '%s_%d_%d_profile_%s%s.png' % (os.path.splitext(f_post_h5)[0],i1,i2,Mstr[1:],txt)
        plt.savefig(f_png)
    plt.show()

    return

def plot_data_xy(f_data_h5, pl_type='line', **kwargs):
    """
    Create 2D spatial plot of survey geometry and elevation data.

    Generates a scatter plot showing the spatial distribution of survey data
    points with color-coding for survey lines, elevation, or both. Useful for
    visualizing survey layout and topographic variations.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing geometry data.
    pl_type : {'line', 'elevation', 'all'}, optional
        Type of geometry plot to generate (default is 'line'):
        - 'line': color by survey line numbers only
        - 'elevation': color by elevation data only
        - 'all': show both line and elevation information
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plot as PNG file (default False)

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    Notes
    -----
    Coordinates are automatically scaled to kilometers for better readability.
    The plot uses equal aspect ratio and includes grid lines and appropriate
    colorbars for the selected data type. Figure size is automatically adjusted
    based on the aspect ratio of the survey area.
    """
    #import integrate as ig
    import matplotlib.pyplot as plt
    
    kwargs.setdefault('hardcopy', False)
    
    # Get 'f_prior' and 'f_data' from the selected file 
    # and display them in the sidebar
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    # Get the ratio between the width   and the height of the plot from X and Y
    ratio = (X.max()-X.min())/(Y.max()-Y.min())
    fig, ax = plt.subplots(figsize=(12, 12/ratio))
    ax.set_title('GEOMETRY')
    if (pl_type=='all')|(pl_type=='elevation'):
        cbar1 = plt.colorbar(ax.scatter(X/1000, Y/1000, c=ELEVATION, s=20, cmap='gray'))
        cbar1.set_label('Elevation (m)')
    if (pl_type=='all')|(pl_type=='line'):
        cbar2 = plt.colorbar(ax.scatter(X/1000, Y/1000, c=LINE, s=.1, cmap='jet'))
        cbar2.set_label('LINE')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    # add equal axis
    ax.axis('equal')
    ax.grid()

    if kwargs['hardcopy']:
        f_png = '%s_xy_%s.png' % (os.path.splitext(f_data_h5)[0],pl_type)
        plt.savefig(f_png)
    

    return fig

def plot_data(f_data_h5, i_plot=[], Dkey=[], plType='imshow', **kwargs):
    """
    Plot observational data from an HDF5 file.
    
    This function creates visualizations of electromagnetic data including time-series plots,
    2D image displays, and other data representations. Supports multiple data types and
    plotting styles for comprehensive data analysis.
    
    :param f_data_h5: Path to the HDF5 file containing observational data
    :type f_data_h5: str
    :param i_plot: Indices of data points to plot. If empty, plots all available data
    :type i_plot: list or array-like, optional
    :param Dkey: Data keys/identifiers to plot. If empty, uses all available datasets
    :type Dkey: str or list, optional
    :param plType: Plotting method - 'imshow' for 2D image display, 'plot' for line plots
    :type plType: str, optional
    :param kwargs: Additional plotting arguments including hardcopy, figsize, colormap options
    :type kwargs: dict
    
    :returns: None (creates matplotlib plots)
    :rtype: None
    
    .. note::
        The function automatically handles different data formats and creates appropriate
        visualizations based on data dimensions and type. Supports saving plots to file
        when hardcopy=True is specified in kwargs.

    :raises: None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import h5py

    # Check if the data file f_data_h5 exists
    if not os.path.exists(f_data_h5):
        print("plot_data: File %s does not exist" % f_data_h5)
        return


    f_data = h5py.File(f_data_h5,'r')

    if len(Dkey)==0:
        nd = 0
        Dkeys = []
        for key in f_data.keys():
            if key[0]=='D':
                print("plot_data: Found data set %s" % key)
                Dkeys.append(key)
            nd += 1
        Dkey=Dkeys[0]
        print("plot_data: Using data set %s" % Dkey)
 
    noise_model = f_data['/%s' % Dkey].attrs['noise_model']
    if noise_model == 'gaussian':
        noise_model = 'Gaussian'
        d_obs = f_data['/%s' % Dkey]['d_obs'][:]
        d_std = f_data['/%s' % Dkey]['d_std'][:]


        ndata,ns = f_data['/%s' % Dkey]['d_obs'].shape
        # set i_plot as an array from 0 to ndata
        if len(i_plot)==0:
            i_plot = np.arange(ndata)
            #i_plot = 1000+np.arange(5000)

        # remove all values in i_plot that are larger than the number of data
        i_plot = i_plot[i_plot<ndata]
        # remove all values in i_plot that are smaller than 0
        i_plot = i_plot[i_plot>=0]
        
        # reaplce values larger than 1 with nan in d_std
        d_std[d_std>1] = np.nan

        # find number of nan values on d_obs
        non_nan = np.sum(~np.isnan(d_obs), axis=1)

        # Calculate the extent
        xlim = [i_plot.min(), i_plot.max()]
        extent = [xlim[0], xlim[1], 0, d_obs.shape[1]]

        # plot figure with data

        fig, ax = plt.subplots(4,1,figsize=(10,12), gridspec_kw={'height_ratios': [3, 3, 3, 1]})

        if plType=='plot':
            im1 = ax[0].semilogy(d_obs[i_plot,:], linewidth=.5)
            im2 = ax[1].semilogy(d_std[i_plot,:], linewidth=.5)
            im3 = ax[2].semilogy((d_obs[i_plot,:]/d_std[i_plot,:]), linewidth=.5)
            ax[0].set_xlim(xlim)
            ax[1].set_xlim(xlim)
            ax[2].set_xlim(xlim)
            ax[2].set_ylim([.5, 50])
            ax[0].set_ylabel('d_obs')
            ax[1].set_ylabel('d_std')
            ax[2].set_ylabel('S/N (d_obs/d_std)')

        elif plType=='imshow':            
            im1 = ax[0].imshow(d_obs[i_plot,:].T, aspect='auto', cmap='jet_r', norm=matplotlib.colors.LogNorm(), extent=extent)
            im2 = ax[1].imshow(d_std[i_plot,:].T, aspect='auto', cmap='hot_r', norm=matplotlib.colors.LogNorm(), extent=extent)
            im3 = ax[2].imshow((d_obs[i_plot,:]/d_std[i_plot,:]).T, aspect='auto', vmin = 0.5, vmax = 50, extent=extent)

            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.colorbar(im3, ax=ax[2])
        
            ax[0].set_ylabel('gate number')
            ax[1].set_ylabel('gate number')
            ax[2].set_ylabel('gate number')
            ax[0].set_title('d_obs: observed data')
            ax[1].set_title('d_std: standard deviation')
            #ax[2].set_title('d_std/d_obs: relative standard deviation')
            
        
        im4 = ax[3].plot(i_plot,non_nan[i_plot], 'k.', markersize=.5)
        ax[3].set_ylabel('Number of data')
        ax[3].set_xlim(xlim)

        if plType=='imshow':            
            # Create an invisible colorbar for the last subplot
            cbar4 = fig.colorbar(im3, ax=ax[3])
            cbar4.solids.set(alpha=0)
            cbar4.outline.set_visible(False)
            cbar4.ax.set_yticks([])  # Hide the colorbar ticks
            cbar4.ax.set_yticklabels([])  # Hide the colorbar ticks labels

        ax[-1].set_xlabel('Index')
        
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()

        plt.suptitle('Data set %s' % Dkey)
        plt.tight_layout()
    else:
        print("plot_data: Unknown noise model: %s" % noise_model)
        
    # set plot in kwarg to True if not allready set
    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s_%s.png' % (os.path.splitext(f_data_h5)[0],Dkey,plType))




def plot_data_prior(f_prior_data_h5,
                    f_data_h5, 
                    nr=1000,
                    id=1,
                    id_data = None,
                    d_str='d_obs', 
                    alpha=0.5,
                    ylim=None, 
                    **kwargs):
    """
    Compare observed data with prior model predictions.

    Creates a logarithmic plot showing prior data realizations (model predictions)
    overlaid with observed data. Useful for validating forward models and
    assessing prior-data compatibility before inversion.

    Parameters
    ----------
    f_prior_data_h5 : str
        Path to the HDF5 file containing prior data realizations from forward modeling.
    f_data_h5 : str
        Path to the HDF5 file containing observed data.
    nr : int, optional
        Maximum number of prior realizations to plot (default is 1000).
    id : int, optional
        Data set identifier for prior data (default is 1).
    id_data : int, optional
        Data set identifier for observed data. If None, uses same as id (default is None).
    d_str : str, optional
        Data array key within the dataset (default is 'd_obs').
    alpha : float, optional
        Transparency level for prior realization lines, range 0-1 (default is 0.5).
    ylim : tuple or list, optional
        Y-axis limits as (ymin, ymax). If None, uses automatic scaling (default is None).
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plot as PNG file (default True)

    Returns
    -------
    bool
        True if plotting was successful.

    Notes
    -----
    Prior realizations are plotted as thin black lines with specified transparency.
    Observed data is plotted as thin red lines. The number of plotted realizations
    is limited by both the nr parameter and available data. Random sampling is
    used when more realizations are available than requested.
    """
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    if id_data is None:
        id_data = id
    
    cols=['wheat','black','red']

    f_data = h5py.File(f_data_h5)
    f_prior_data = h5py.File(f_prior_data_h5)
    
    plt.figure(figsize=(7,6))
    # PLOT PRIOR REALIZATIONS
    dh5_str = 'D%d' % (id)
    if dh5_str in f_prior_data:
        f_prior_data[dh5_str]  
        npr = f_prior_data[dh5_str].shape[0]
        
        nr = np.min([nr,npr])
        # select nr random sample of d_obs
        i_use = np.sort(np.random.choice(npr, nr, replace=False))
        D = f_prior_data[dh5_str][i_use]
        
        plt.semilogy(D.T,'-',alpha=alpha, linewidth=0.1, color=cols[1], label='\rho(d)') 
        
    else:   
        print('%s not in f_prior_data' % dh5_str)

    # PLOT OBSERVED DATA
    print('id_data = %d' % id_data)
    dh5_str = 'D%d/%s' % (id_data,d_str)

    # check that dh5_str is in f_data
    if dh5_str in f_data:
        d_obs = f_data[dh5_str][:]
        ns, nd = f_data[dh5_str].shape
        nr = np.min([nr,ns])    
        # select nr random sample of d_obs
        i_use_d = np.sort(np.random.choice(ns, nr, replace=False))

        plt.semilogy(d_obs[i_use_d,:].T,'-',alpha=alpha, linewidth=0.1,label='d_obs', color=cols[2])
        
    else:
        print('%s not in f_data'% dh5_str)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid()
    plt.xlabel('Data #')
    plt.ylabel('Data Value')
    plt.tight_layout()
    # Add legend but, only 'A' and 'B'
    #plt.title('Prior data (black) and observed data (red)\n%s (black)\n%s (red)' % (os.path.splitext(f_prior_data_h5)[0],os.path.splitext(f_data_h5)[0]) )
    plt.title('Prior data (black) and observed data (red)')
    
    f_data.close()
    f_prior_data.close()

    # set plot in kwarg to True if not allready set
    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s_id%d_%s.png' % (os.path.splitext(f_data_h5)[0],os.path.splitext(f_prior_data_h5)[0],id,d_str))
    plt.show()
    
    return True

def plot_data_prior_post(f_post_h5, i_plot=-1, nr=200, id=0, ylim=None, Dkey=[], **kwargs):
    """
    Compare prior predictions, posterior predictions, and observed data.

    Creates logarithmic plots showing the evolution from prior to posterior
    predictions compared to observed data. Displays data fit quality and
    sampling results with temperature and evidence information.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    i_plot : int, optional
        Index of specific observation to plot. If -1, plots random selection
        of data points (default is -1).
    nr : int, optional
        Maximum number of realizations to plot for each type (default is 200).
    id : int or list of int, optional
        Data set identifier(s) to plot. If 0, plots all available datasets.
        If list, plots each dataset separately (default is 0).
    ylim : tuple or list, optional
        Y-axis limits as (ymin, ymax). If None, uses automatic scaling (default is None).
    Dkey : list or str, optional
        Explicit data key specification. If empty, automatically detects
        available datasets (default is []).
    **kwargs : dict
        Additional keyword arguments:
        - showInfo : int, level of debug output (0=none, >0=verbose)
        - is_log : bool, use linear instead of logarithmic y-axis (default False)
        - hardcopy : bool, save plot as PNG file (default False)

    Returns
    -------
    None
        Function creates and displays plots but does not return values.

    Notes
    -----
    The plot shows three data types:
    - Prior realizations (wheat/gray lines): model predictions before inversion
    - Posterior realizations (black lines): model predictions after inversion 
    - Observed data (red dots with error bars): actual measurements
    
    For specific observations (i_plot >= 0), temperature and log evidence
    values are displayed. Random subsets are used when more realizations
    are available than requested.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import h5py
    import os
    
    showInfo = kwargs.get('showInfo', 0)
    is_log = kwargs.get('is_log', False)

    ## Check if the data file f_data_h5 exists
    if not os.path.exists(f_post_h5):
        print("plot_data: File %s does not exist" % f_data_h5)
        return


    f_post = h5py.File(f_post_h5,'r')

    f_prior_h5 = f_post['/'].attrs['f5_prior']
    f_data_h5 = f_post['/'].attrs['f5_data']


    # if id is a list of integers, then loop over them and call 
    # plot_data_prior_post for each id
    if isinstance(id, list):
        for i in id:
            plot_data_prior_post(f_post_h5, i_plot=i_plot, nr=nr, id=i, ylim=ylim, **kwargs)
        return

    if id==0:
        # get number of data sets in f_post_h5
        nd = 0
        id_plot = []
        with h5py.File(f_data_h5,'r') as f_data:
            for key in f_data.keys():
                if key[0]=='D':
                    if showInfo>0:
                        print("plot_data_prior_post: Found data set %s" % key)
                    nd += 1
                    id_plot.append(nd)  

        #print(id_plot)
        plot_data_prior_post(f_post_h5, i_plot=i_plot, nr=nr, id=id_plot, ylim=ylim, **kwargs)
        return

    if id>0:
        Dkey = 'D%d' % id

    f_data = h5py.File(f_data_h5,'r')
    f_prior = h5py.File(f_prior_h5,'r')

    cols=['gray','black','red']
    cols=['wheat','black','red']

    if len(Dkey)==0:
        nd = 0
        Dkeys = []
        for key in f_data.keys():
            if key[0]=='D':
                if showInfo>0:
                    print("plot_data_prior_post: Found data set %s" % key)
                Dkeys.append(key)
            nd += 1
        Dkey=Dkeys[0]
        if showInfo>0:
            print("plot_data_prior_post: Using data set %s" % Dkey)

    noise_model = f_data['/%s' % Dkey].attrs['noise_model']
    if noise_model == 'gaussian':
        noise_model = 'Gaussian'
        d_obs = f_data['/%s' % Dkey]['d_obs'][:]
        try:
            d_std = f_data['/%s' % Dkey]['d_std'][:]
        except:
            if 'Cd' in f_data['/%s' % Dkey].keys():
                # if 'Cd' is 3 dim then take the diagonal
                if len(f_data['/%s' % Dkey]['Cd'].shape)==3:
                    d_std = np.sqrt(np.diag(f_data['/%s' % Dkey]['Cd'][i_plot]))
                else:
                    d_std = np.sqrt(f_data['/%s' % Dkey]['Cd'])
            else:
                d_std = np.zeros(d_obs.shape)

        if i_plot==-1:
            # get 400 random unique index of d_obs
            i_use = np.random.choice(d_obs.shape[0], nr, replace=False)
        else:
            nr = np.min([nr,d_obs.shape[0]])
            i_use = f_post['/i_use'][i_plot,0:nr]
            i_use = i_use.flatten()
        nr=len(i_use)
        
        ns,ndata = f_data['/%s' % Dkey]['d_obs'].shape
        d_post = np.zeros((nr,ndata))
        d_prior = np.zeros((nr,ndata))
        
        N = f_prior[Dkey].shape[0]
        # set id_plot to be nr random locagtions in 1:ndata
        i_prior_plot = np.random.randint(0,N,nr)
        for i in range(nr):
            d_prior[i]=f_prior[Dkey][i_prior_plot[i],:]    
            if i_plot>-1:
                d_post[i]=f_prior[Dkey][i_use[i],:]
    
        #i_plot=[]
        fig, ax = plt.subplots(1,1,figsize=(7,7))
        if is_log:
            ax.plot(d_prior.T,'-',linewidth=1.4, label='d_prior', color=cols[0])
            ax.plot(d_post.T,'-',linewidth=.2, label='d_prior', color=cols[1])
        
            print('plot_data_prior_post: Plotting log10(d_prior)')
            print('This is not implemented yet')
            return        
        else:
            ax.semilogy(d_prior.T,'-',linewidth=.2, label='d_prior', color=cols[0])

        if i_plot>-1:            
            ax.semilogy(d_post.T,'-',linewidth=.2, label='d_prior', color=cols[1])
        
            ax.semilogy(d_obs[i_plot,:],'.',markersize=6, label='d_obs', color=cols[2])
            ax.semilogy(d_obs[i_plot,:]-2*d_std[i_plot,:],'-',linewidth=1, label='d_obs', color=cols[2])
            ax.semilogy(d_obs[i_plot,:]+2*d_std[i_plot,:],'-',linewidth=1, label='d_obs', color=cols[2])

            #ax.text(0.1, 0.1, 'Data set %s, Observation # %d' % (Dkey, i_plot+1), transform=ax.transAxes)
            ax.text(0.1, 0.1, 'T = %4.2f.' % (f_post['/T'][i_plot]), transform=ax.transAxes)
            ax.text(0.1, 0.2, 'log(EV) = %4.2f.' % (f_post['/EV'][i_plot]), transform=ax.transAxes)
            plt.title('Data set %s, Observation # %d' % (Dkey, i_plot+1))
        else:   
            # select nr random unqiue index of d_obs
            i_d = np.random.choice(d_obs.shape[0], nr, replace=False)
            if is_log:
                ax.plot(d_obs[i_d,:].T,'-',linewidth=.1, label='d_obs', color=cols[2])
                ax.plot(d_obs[i_d,:].T,'*',linewidth=.1, label='d_obs', color=cols[2])
            else:
                ax.semilogy(d_obs[i_d,:].T,'-',linewidth=1, label='d_obs', color=cols[2])
                ax.semilogy(d_obs[i_d,:].T,'*',linewidth=1, label='d_obs', color=cols[2])

        if ylim is not None:            
            plt.ylim(ylim)

        plt.xlabel('Data #')
        plt.ylabel('dBDt')
        plt.grid()
        #plt.legend()

        # set plot in kwarg to True if not allready set
        if 'hardcopy' not in kwargs:
            kwargs['hardcopy'] = False
        if kwargs['hardcopy']:
            # strip the filename from f_data_h5
            # get filename without extension of f_post_h5
            if i_plot==-1:
                plt.savefig('%s_%s.png' % (os.path.splitext(f_post_h5)[0],Dkey))
            else:
                plt.savefig('%s_%s_id%05d.png' % (os.path.splitext(f_post_h5)[0],Dkey,i_plot))
        plt.show()
    

def plot_prior_stats(f_prior_h5, Mkey=[], nr=100, **kwargs):
    """
    Visualize prior model parameter distributions and sample realizations.

    Creates comprehensive plots showing parameter histograms, logarithmic
    distributions, and spatial/temporal realizations for prior model
    parameters. Useful for validating prior distributions before inversion.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model realizations.
    Mkey : str or list, optional
        Model parameter key(s) to plot (e.g., 'M1', 'M2'). If empty list,
        plots statistics for all available model parameters (default is []).
    nr : int, optional
        Maximum number of realizations to display in realization plots.
        Actual number used is minimum of nr and available realizations (default is 100).
    **kwargs : dict
        Additional keyword arguments:
        - hardcopy : bool, save plots as PNG files (default True)

    Returns
    -------
    None
        Function creates and displays plots but does not return values.

    Notes
    -----
    For continuous parameters, creates a 2x2 subplot layout:
    - Top left: Linear histogram of parameter values
    - Top right: Log10 histogram with scientific notation tick labels
    - Bottom: Realizations plot showing parameter variation
    
    For discrete parameters, creates similar layout but with:
    - Class-based histograms with appropriate colormaps
    - Categorical realizations with class names and colors
    
    Color limits and colormaps are automatically retrieved from file attributes.
    Multi-dimensional parameters show spatial patterns, while single parameters
    show temporal variation.
    """
    from matplotlib.colors import LogNorm
    
    f_prior = h5py.File(f_prior_h5,'r')

    # If Mkey is not set, plot for all M* keys in prior and return 
    if len(Mkey)==0:
        for key in f_prior.keys():
            if (key[0]=='M'):
                plot_prior_stats(f_prior_h5, Mkey=key, nr=nr, **kwargs)
        
        f_prior.close()
        return  

    if Mkey[0]!='/':
        Mkey = '/%s' % Mkey

    # check if Mkey is in the keys of f_prior
    if Mkey not in f_prior.keys():
        print("Mkey=%s not found in %s" % (Mkey, f_prior_h5))
        return

    # check if name is in the attributes of key Mkey
    if 'name' in f_prior['/%s'%Mkey].attrs.keys():
        name = '%s:%s' %  (Mkey[1::],f_prior['/%s'%Mkey].attrs['name'][:])
        #print(name)
    else:
        name = Mkey


    f_prior['/%s'%Mkey].attrs.keys()
    if 'x' in f_prior['/%s'%Mkey].attrs.keys():
        z = f_prior['/%s'%Mkey].attrs['x']
    else:
        z = f_prior['/%s'%Mkey].attrs['z']


    # update nr if it is larger than the number of realizations in f_prior[Mkey]
    if len(f_prior[Mkey][:])<nr:
        nr = np.min([nr, len(f_prior[Mkey][:])])
        print('plot_prior_stats: Using %d realizations' % nr)

    M = f_prior[Mkey][:]
    N, Nm = M.shape
    clim,cmap = ig.get_clim_cmap(f_prior_h5, Mstr=Mkey)

    is_discrete = f_prior['/%s'%Mkey].attrs['is_discrete']    
    
    if not is_discrete:
        # CONTINUOUS
        
        # PLOT Mkey histrogram  and log10 histogram
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        m0 = ax[0,0].hist(M.flatten(),101)
        ax[0,0].set_xlabel(name)
        ax[0,0].set_ylabel('Distribution')

        # Handle log(0) by filtering out zeros and negative values
        M_log = M.flatten()
        M_log = M_log[M_log > 0]  # Remove zeros and negative values
        if len(M_log) > 0:
            m1 = ax[0,1].hist(np.log10(M_log), 101)
        else:
            # If no positive values, create empty histogram
            m1 = ax[0,1].hist([], 101)
        ax[0,1].set_xlabel('log10(%s)' % name)

        # set xtcik labels as 10^x where x i the xtick valye
        ax[0,1].set_xticks(ax[0,1].get_xticks())  # Ensure ticks are set
        ticks = ax[0,1].get_xticks()
        ax[0,1].set_xticks(ticks)
        ax[0,1].set_xticklabels(['$10^{%3.1f}$'%i for i in ticks])
        ax[0,1].set_ylabel('Distribution')

        ax[0, 0].grid()
        ax[0, 1].grid()
        ax[1, 0].axis('off')    
        ax[1, 1].axis('off')
        
        # Plot actual realizatrions
        ax[1, 0] = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        X,Y = np.meshgrid(np.arange(1,nr+1),z)
        ax[1,0].invert_yaxis()
        if Nm>1:
            m2 = ax[1,0].pcolor(X,Y,M[0:nr,:].T, 
                            cmap=cmap, 
                            shading='auto',
                            norm=LogNorm())
            # set clim to clim
            m2.set_clim(clim[0],clim[1])
            #m2.set_clim(clim[0]-.5,clim[1]+.5)      
            fig.colorbar(m2, ax=ax[1,0], label=Mkey[1::])
        else:
            m2 = ax[1,0].plot(np.arange(1,nr+1),M[0:nr,:].flatten()) 
            ax[1,0].set_xlim(1,nr)

        ax[1,0].set_xlabel('Realization #')
        ax[1,0].set_ylabel(name)
        
        tit = '%s - %s ' % (os.path.splitext(f_prior_h5)[0],name) 
        plt.suptitle(tit)

    else:
        # DISCRETE
        
        # get attribute class_name if it exist
        
        if 'class_id' in f_prior[Mkey].attrs.keys():
            class_id = f_prior[Mkey].attrs['class_id'][:].flatten()
        else:   
            print('No class_id found')
        if 'class_name' in f_prior[Mkey].attrs.keys():
            class_name = f_prior[Mkey].attrs['class_name'][:].flatten()
        else:
            class_name = []
        n_class = len(class_name)

        
        # PLOT Mkey histrogram  and log10 histogram
        fig, ax = plt.subplots(2,2,figsize=(10,10))

        m0 = ax[0,0].hist(M.flatten(),101)
        ax[0,0].set_xlabel(name)
        ax[0,0].set_ylabel('Distribution')
        
        m1 = ax[0,1].hist(np.log10(M.flatten()),101)
        ax[0,1].set_xlabel(name)

        # set xtcik labels as 10^x where x i the xtick valye
        ax[0,1].set_xticks(ax[0,1].get_xticks())  # Ensure ticks are set
        ax[0,1].set_xticklabels(['$10^{%3.1f}$'%i for i in ax[0,1].get_xticks()])
        ax[0,1].set_ylabel('Distribution')

        ax[0, 0].grid()
        ax[0, 1].grid()
        ax[1, 0].axis('off')    
        ax[1, 1].axis('off')
        
       # Plot actual realizations
        ax[1, 0] = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        X,Y = np.meshgrid(np.arange(1,nr+1),z)
        ax[1,0].invert_yaxis()
        if Nm>1:
            m2 = ax[1,0].pcolor(X,Y,M[0:nr,:].T, 
                            cmap=cmap, 
                            shading='auto')
            # set clim to clim
            m2.set_clim(clim[0],clim[1])


            #m2.set_clim(clim[0],clim[1])
            #m2.set_clim(clim[0]-.5,clim[1]+.5)      
            #fig.colorbar(m2, ax=ax[1,0], label=Mkey[1::])
            m2.set_clim(clim[0]-.5,clim[1]+.5)      
            #fig.colorbar(m2, ax=ax[1,0], label=Mkey)
            cbar1 = fig.colorbar(m2, ax=ax[1,0], label='label')
            cbar1.set_ticks(np.arange(n_class)+1)
            cbar1.set_ticklabels(class_name)
            cbar1.ax.invert_yaxis()
            

            '''
            im1 = ax[0].pcolormesh(ID[:,i1:i2], ZZ[:,i1:i2], Mode[:,i1:i2], 
                cmap=cmap,            
                shading='auto')
            im1.set_clim(clim[0]-.5,clim[1]+.5)        

            ax[0].set_title('Mode')
            # /fix set the ticks to be 1 to n_class, and use class_name as tick labels
            cbar1 = fig.colorbar(im1, ax=ax[0], label='label')
            cbar1.set_ticks(np.arange(n_class)+1)
            cbar1.set_ticklabels(class_name)
            cbar1.ax.invert_yaxis()
            '''


        else:
            m2 = ax[1,0].plot(np.arange(1,101),M[0:nr,:].flatten()) 
            ax[1,0].set_xlim(1,nr)

        ax[1,0].set_xlabel('Realization #')
        ax[1,0].set_ylabel(name)
        
        tit = '%s - %s ' % (os.path.splitext(f_prior_h5)[0],name) 
        plt.suptitle(tit)

    f_prior.close()

    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s.png' % (os.path.splitext(f_prior_h5)[0],Mkey[1::]))


# function that reads cmap and clim if they are set
def get_clim_cmap(f_prior_h5, Mstr='/M1'):
    """
    Retrieve color scale limits and colormap from prior model attributes.

    Extracts visualization parameters stored in HDF5 file attributes,
    providing default values when attributes are not present. Used by
    plotting functions for consistent color scaling across visualizations.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model data.
    Mstr : str, optional
        Model parameter key string (e.g., '/M1', '/M2', default is '/M1').

    Returns
    -------
    clim : list
        Color scale limits as [min_value, max_value]. Returns [10, 500]
        as default if 'clim' attribute is not found.
    cmap : matplotlib.colors.ListedColormap or str  
        Colormap object created from stored colormap array, or 'jet'
        string as default if 'cmap' attribute is not found.

    Notes
    -----
    The function automatically converts stored colormap arrays to matplotlib
    ListedColormap objects with proper transposition. Color limits and
    colormaps are typically set during prior model generation to ensure
    consistent visualization across different plotting functions.
    """
    with h5py.File(f_prior_h5,'r') as f_prior:
        if 'clim' in f_prior[Mstr].attrs.keys():
            clim = f_prior[Mstr].attrs['clim'][:].flatten()
        else:
            clim = [.1, 2600]
            clim = [10, 500]
        if 'cmap' in f_prior[Mstr].attrs.keys():
            cmap = f_prior[Mstr].attrs['cmap'][:]
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cmap.T)
        else:
            cmap = 'jet'

        return clim, cmap


def plot_cumulative_probability_profile(P_hypothesis, i1=0, i2=0, label=None, colors = None, hardcopy=True, name='cumulative_probability_profile'):
    """
    Plot the cumulative probability profile of different hypotheses.
    
    This function visualizes how the probability of different hypotheses accumulates
    over a sequence of data points, with each hypothesis represented as a colored
    area in a stacked plot.
    
    :param P_hypothesis: A 2D array where each row represents a hypothesis and each column represents a data point. Values should be probabilities
    :type P_hypothesis: numpy.ndarray
    :param i1: Starting index for the x-axis (data points)
    :type i1: int, optional
    :param i2: Ending index for the x-axis (data points). If 0, uses the number of data points in P_hypothesis
    :type i2: int, optional
    :param label: List of labels for each hypothesis. If None, generic labels will be created
    :type label: list of str, optional
    :param colors: List of colors for each hypothesis. If None, colors from the 'hot' colormap will be used
    :type colors: list or numpy.ndarray, optional
    :param hardcopy: If True, saves the figure to a file
    :type hardcopy: bool, optional
    :param name: Base name for the output file when hardcopy is True
    :type name: str, optional
    
    :returns: None (displays the plot and optionally saves it to a file)
    :rtype: None
    
    .. note::
        The plot shows how probabilities accumulate across hypotheses, with each hypothesis
        represented as a colored band. The total height of all bands at any x position equals 1.0
        (or the sum of probabilities for that data point).
    """

    import matplotlib.pyplot as plt
    import numpy as np

    nhypothesis = P_hypothesis.shape[0]

    if i2==0:
        i2 = P_hypothesis.shape[1]

    if label is None:
        # Generate  list of length nhypothesis, with generic label names
        label = [f'Hypothesiss {i+1}' for i in range(nhypothesis)]

    # Define nypothesis colors from the hot colormap
    if colors is None:
        colors = plt.cm.hot(np.linspace(0, 1, nhypothesis))

    ii  = np.arange(i1,i2,1)
    P_hypothesis_plot = P_hypothesis[:,ii]
    fig, ax = plt.subplots(figsize=(12, 6))

    
    # Calculate cumulative probabilities
    cum_probs = np.zeros((P_hypothesis_plot.shape[0] + 1, P_hypothesis_plot.shape[1]))
    for i in range(P_hypothesis_plot.shape[0]):
        cum_probs[i+1] = cum_probs[i] + P_hypothesis_plot[i]

    # Plot filled areas between cumulative probabilities
    for i in range(P_hypothesis_plot.shape[0]):
        ax.fill_between(
            ii, 
            cum_probs[i], 
            cum_probs[i+1], 
            color=colors[i], 
            alpha=0.7,
            label=label[i]
        )

    # Add labels and title
    ax.set_xlabel('Data point index')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Probability of Hypotheses')
    ax.legend(loc='upper right')

    # Optional: Limit x-axis if needed to focus on a specific range
    # ax.set_xlim(0, 1000)  # Uncomment to focus on first 1000 data points

    plt.tight_layout()
    # Save the figure if hardcopy is True
    if hardcopy:
        plt.savefig(name, dpi=300)
        print("Saved figure as %s.png" % (name))
    plt.show()
    

