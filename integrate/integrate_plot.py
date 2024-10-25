import os
import numpy as np
import h5py
import integrate as ig
import matplotlib.pyplot as plt


def plot_posterior_cumulative_thickness(f_post_h5, im=2, icat=[0], property='median', usePrior=False, **kwargs):
    """
    Plots the posterior cumulative thickness.

    :param f_post_h5: The file path to the posterior data in HDF5 format.
    :type f_post_h5: str
    :param im: The index of the image.
    :type im: int
    :param icat: The index or list of indices of the categories.
    :type icat: int or list
    :param property: The property to plot ('median', 'mean', 'std', 'relstd').
    :type property: str
    :param usePrior: Whether to use prior data.
    :type usePrior: bool
    :param kwargs: Additional keyword arguments.
    :returns: fig -- The matplotlib figure object.
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
    Plot a 2D feature from a given HDF5 file.

    :param f_post_h5: Path to the HDF5 file.
    :type f_post_h5: str
    :param key: Key of the feature to plot. If not provided, the first key found in the file will be used.
    :type key: str
    :param i1: Start index of the feature to plot.
    :type i1: int
    :param i2: End index of the feature to plot.
    :type i2: int
    :param im: Index of the feature.
    :type im: int
    :param iz: Index of the z-coordinate.
    :type iz: int
    :param uselog: Flag indicating whether to apply logarithmic scaling to the data.
    :type uselog: int
    :param title_text: Additional text to include in the plot title.
    :type title_text: str
    :param kwargs: Additional keyword arguments to be passed to the scatter plot.
    :returns: int -- 1 if the plot is successful.
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
            #plt.show()

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
            #plt.show()
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

def plot_profile(f_post_h5, i1=1, i2=1e+9, im=0, **kwargs):

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
    Plot discrete profiles from a given HDF5 file.

    Parameters: 
    - f_post_h5 (str): Path to the HDF5 file.
    - i1 (int, optional): Starting index for the profile. Defaults to 1.
    - i2 (int, optional): Ending index for the profile. Defaults to 1e+9.
    - im (int, optional): Index of the profile to plot. Defaults to 1.

    Returns:
    - None
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
    Plot continuous profiles from a given HDF5 file.

    Parameters: 
    - f_post_h5 (str): Path to the HDF5 file.
    - i1 (int, optional): Starting index for the profile. Defaults to 1.
    - i2 (int, optional): Ending index for the profile. Defaults to 1e+9.
    - im (int, optional): Index of the profile to plot. Defaults to 1.

    Returns:
    - None
    """
    from matplotlib.colors import LogNorm

    kwargs.setdefault('hardcopy', False)
    kwargs.setdefault('cmap', 'jet')
    
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
    
    if nm>1:
        # MEAN
        #im1 = ax[0].pcolormesh(ID[:,i1:i2], ZZ[:,i1:i2], Mean[:,i1:i2], 
        im1 = ax[0].pcolormesh(DDc, ZZc, Mean[:,i1:i2], 
                cmap=cmap,            
                shading='auto',
                norm=LogNorm())
        im1.set_clim(clim[0],clim[1])        
        ax[0].set_title('Mean %s' % name)
        fig.colorbar(im1, ax=ax[0], label='%s' % name)
    
    if nm>1:
        # MEDIAN
        im2 = ax[1].pcolormesh(DDc, ZZc, Median[:,i1:i2], 
                cmap=cmap,            
                shading='auto',
                norm=LogNorm())  # Set color scale to logarithmic
        im2.set_clim(clim[0],clim[1])        
        ax[1].set_title('Median %s' % name)
        fig.colorbar(im2, ax=ax[1], label='%s' % name) 

    if nm>1:
        # STD
        import matplotlib
        im3 = ax[2].pcolormesh(DDc, ZZc, Std[:,i1:i2], 
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red"]), 
                    shading='auto')
        im3.set_clim(0,1)
        ax[2].set_title('Std %s' % name)
        fig.colorbar(im3, ax=ax[2], label='Standard deviation (Ohm.m)')
    else:
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

def plot_data_xy(f_data_h5, pl_type='all', **kwargs):
    import integrate as ig
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
    Plot the data from an HDF5 file.

    :param f_data_h5: The path to the HDF5 file.
    :type f_data_h5: str
    :param i_plot: The indices of the data to plot. Default is 0.
    :type i_plot: int or array-like, optional
    :param Dkey: The key(s) of the data set(s) to plot. Default is an empty list.
    :type Dkey: str or list, optional
    :param plType: The type of plot to use ('imshow' or 'plot'). Default is 'imshow'.
    :type plType: str, optional
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :returns: None
    :rtype: None

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
                    d_str='d_obs', 
                    alpha=0.5,
                    ylim=None, 
                    **kwargs):
    """
    Plot the prior data on top of prior data realizations. 
    Usefull for checking the consistency between the choid of prior and observed before runnning an inversion.
    """

    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    f_data = h5py.File(f_data_h5)
    f_prior_data = h5py.File(f_prior_data_h5)
    
    plt.figure(figsize=(7,6))
    # PLOT PRIOR REALIZATIONS
    dh5_str = 'D%d' % (id)
    if dh5_str in f_prior_data:
        f_prior_data['D1']  
        npr = f_prior_data['D1'].shape[0]
        
        nr = np.min([nr,npr])
        # select nr random sample of d_obs
        i_use = np.sort(np.random.choice(npr, nr, replace=False))
        D = f_prior_data[dh5_str][i_use]
        
        plt.semilogy(D.T,'k-',alpha=alpha, linewidth=0.1)
        
    else:   
        print('%s not in f_prior_data' % dh5_str)

    # PLOT OBSERVED DATA
    dh5_str = 'D%d/%s' % (id,d_str)

    # check that dh5_str is in f_data
    if dh5_str in f_data:
        d_obs = f_data[dh5_str][:]
        ns, nd = f_data[dh5_str].shape
        nr = np.min([nr,ns])    
        # select nr random sample of d_obs
        i_use_d = np.sort(np.random.choice(ns, nr, replace=False))

        plt.semilogy(d_obs[i_use_d,:].T,'r-',alpha=alpha, linewidth=0.1,label='d_obs')
        
    else:
        print('%s not in f_data'% dh5_str)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid()
    plt.xlabel('Data #')
    plt.ylabel('Data Value')
    plt.tight_layout()
    plt.title('Prior data (black) and observed data (red)\n%s (black)\n%s (red)' % (os.path.splitext(f_prior_data_h5)[0],os.path.splitext(f_data_h5)[0]) )
    
    f_data.close()
    f_prior_data.close()

    # set plot in kwarg to True if not allready set
    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s_id%d_%s.png' % (os.path.splitext(f_data_h5)[0],os.path.splitext(f_prior_data_h5)[0],id,d_str))


def plot_data_prior_post(f_post_h5, i_plot=-1, nr=200, id=0, Dkey=[], **kwargs):
    """
    Plot the prior and posterior data for a given dataset.

    :param f_post_h5: The path to the post data file.
    :type f_post_h5: str
    :param i_plot: The index of the observation to plot.
    :type i_plot: int
    :param i_d: Data number to plot.
    :type i_d: int
    :param Dkey: String of the hdf5 key for the data set.
    :type Dkey: str
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :returns: None
    :rtype: None
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
            plot_data_prior_post(f_post_h5, i_plot=i_plot, nr=nr, id=i, **kwargs)
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
        plot_data_prior_post(f_post_h5, i_plot=i_plot, nr=nr, id=id_plot, **kwargs)
        return

    if id>0:
        Dkey = 'D%d' % id
    

    f_data = h5py.File(f_data_h5,'r')
    f_prior = h5py.File(f_prior_h5,'r')
    

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
            i_use = f_post['/i_use'][i_plot,:]
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
            ax.plot(d_prior.T,'-',linewidth=.1, label='d_prior', color='gray')
            ax.plot(d_post.T,'-',linewidth=.1, label='d_prior', color='black')
        
            print('plot_data_prior_post: Plotting log10(d_prior)')
            print('This is not implemented yet')
            return        
        else:
            ax.semilogy(d_prior.T,'-',linewidth=.1, label='d_prior', color='gray')

        if i_plot>-1:            
            ax.semilogy(d_post.T,'-',linewidth=.1, label='d_prior', color='black')
        
            ax.semilogy(d_obs[i_plot,:],'r.',markersize=6, label='d_obs')
            ax.semilogy(d_obs[i_plot,:]-2*d_std[i_plot,:],'r.',markersize=3, label='d_obs')
            ax.semilogy(d_obs[i_plot,:]+2*d_std[i_plot,:],'r.',markersize=3, label='d_obs')

            #ax.text(0.1, 0.1, 'Data set %s, Observation # %d' % (Dkey, i_plot+1), transform=ax.transAxes)
            ax.text(0.1, 0.1, 'T = %4.2f.' % (f_post['/T'][i_plot]), transform=ax.transAxes)
            ax.text(0.1, 0.2, 'log(EV) = %4.2f.' % (f_post['/EV'][i_plot]), transform=ax.transAxes)
            plt.title('Data set %s, Observation # %d' % (Dkey, i_plot+1))
        else:   
            # select nr random unqiue index of d_obs
            i_d = np.random.choice(d_obs.shape[0], nr, replace=False)
            if is_log:
                ax.plot(d_obs[i_d,:].T,'r-',linewidth=.1, label='d_obs')
            else:
                ax.semilogy(d_obs[i_d,:].T,'r-',linewidth=.1, label='d_obs')
            
        plt.xlabel('Data #')
        plt.ylabel('Data')
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
            

def plot_prior_stats(f_prior_h5, Mkey=[], nr=100, **kwargs):
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
        m1 = ax[0,1].hist(np.log10(M.flatten()),101)
        ax[0,1].set_xlabel(name)

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
            m2 = ax[1,0].plot(np.arange(1,101),M[0:nr,:].flatten()) 
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
    