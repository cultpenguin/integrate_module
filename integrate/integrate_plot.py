import os
import numpy as np
import h5py
import integrate as ig
import matplotlib.pyplot as plt



def plot_feature_2d(f_post_h5, key='', i1=1, i2=1e+9, im=1, iz=0, uselog=0, title_text='', **kwargs):
    
    
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

    print("Plotting Feature %d from %s/%s" % (iz, dstr,key))

    with h5py.File(f_post_h5,'r') as f_post:

        if dstr in f_post:
            if key in f_post[dstr].keys():
                D = f_post[dstr][key][:,iz][:]
                if uselog==1:
                    D=np.log10(D)
                # plot this KEY
                plt.figure(1, figsize=(20, 10))
                plt.scatter(X[i1:i2],Y[i1:i2],c=D[i1:i2],**kwargs)            
                plt.grid()
                plt.xlabel('X')                
                plt.colorbar()
                plt.title("%s/%s[%d,:] %s %s" %(dstr,key,iz,title_text,name))
                plt.axis('equal')
                if 'clim' in kwargs:
                    plt.clim(kwargs['clim'])
                
                f_png = '%s_%d_%d_%d_%s%02d_feature.png' % (os.path.splitext(f_post_h5)[0],i1,i2,im,key,iz)
                plt.savefig(f_png)
                plt.show()

                
            else:
                print("Key %s not found in %s" % (key, dstr))
    return 1

def plot_T_EV(f_post_h5, i1=1, i2=1e+9, T_min=1, T_max=100, pl='both', hardcopy=False, **kwargs):

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
        plt.scatter(X[i1:i2],Y[i1:i2],c=np.log10(T[i1:i2]),cmap='jet',**kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.clim(np.log10(clim))  
        print(clim)
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
        #if 'vmin' not in kwargs:
        #    kwargs['vmin'] = EV_min
        #if 'vmax' not in kwargs:
        #    kwargs['vmax'] = EV_max
        print('EV_min=%f, EV_max=%f' % (EV_min, EV_max))
        plt.figure(2, figsize=(20, 10))
        plt.scatter(X[i1:i2],Y[i1:i2],c=EV[i1:i2],cmap='jet', vmin = EV_min, vmax=EV_max, **kwargs)            
        plt.grid()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.title('EV')
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
        print(non_nan)

        plt.figure(3, figsize=(20, 10))
        plt.scatter(X[i1:i2],Y[i1:i2],c=non_nan[i1:i2],cmap='jet', **kwargs)            
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
            

    return 1

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

    kwargs.setdefault('hardcopy', True)
    
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im

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
        print(clim)

    if is_discrete:
        print("This is a discrete model. Use plot_profile_discrete instead")

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
        i2=nd

    from matplotlib.colors import LogNorm

    # Create a figure with 3 subplots sharing the same Xaxis!
    fig, ax = plt.subplots(4,1,figsize=(20,10), gridspec_kw={'height_ratios': [3, 3, 3, 1]})

    # MEAN
    im1 = ax[0].pcolormesh(ID[:,i1:i2], ZZ[:,i1:i2], Mean[:,i1:i2], 
            cmap='jet',            
            shading='auto',
            norm=LogNorm())
    im1.set_clim(clim[0],clim[1])        
    ax[0].set_title('Mean')
    fig.colorbar(im1, ax=ax[0], label='Resistivity (Ohm.m)')
    
    # MEDIAN
    im2 = ax[1].pcolormesh(ID[:,i1:i2], ZZ[:,i1:i2], Median[:,i1:i2], 
            cmap='jet',            
            shading='auto',
            norm=LogNorm())  # Set color scale to logarithmic
    im2.set_clim(clim[0],clim[1])        
    ax[1].set_title('Median')
    fig.colorbar(im2, ax=ax[1], label='Resistivity (Ohm.m)')

    # STD
    im3 = ax[2].pcolormesh(ID[:,i1:i2], ZZ[:,i1:i2], Std[:,i1:i2], 
            cmap='hot_r', 
            vmin=0, vmax=0.5, 
            shading='auto')
    im2.set_clim(clim[0],clim[1])        
    ax[2].set_title('std')
    fig.colorbar(im3, ax=ax[2], label='Standard deviation (Ohm.m)')


    ## T and V
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    
    im4 = ax[3].semilogy(ID[0,i1:i2],T[i1:i2], 'k', label='T')
    plt.semilogy(ID[0,i1:i2],-EV[i1:i2], 'r', label='-EV')
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
    f_png = '%s_%d_%d_profile.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
    plt.savefig(f_png)
    plt.show()

def plot_data_xy(f_data_h5, i_plot=[], Dkey=[], **kwargs):
    import integrate as ig
    import matplotlib.pyplot as plt
    
    # Get 'f_prior' and 'f_data' from the selected file 
    # and display them in the sidebar
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    fig, ax = plt.subplots()
    ax.set_title('GEOMETRY')
    cbar1 = plt.colorbar(ax.scatter(X/1000, Y/1000, c=ELEVATION, s=20, cmap='jet'))
    cbar1.set_label('Elevation (m)')
    cbar2 = plt.colorbar(ax.scatter(X/1000, Y/1000, c=LINE, s=1, cmap='gray'))
    cbar2.set_label('LINE')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    return fig
          

def plot_data(f_data_h5, i_plot=[], Dkey=[], **kwargs):
    """
    Plot the data from an HDF5 file.

    Parameters:
    - f_data_h5 (str): The path to the HDF5 file.
    - i_plot (int or array-like, optional): The indices of the data to plot. Default is 0.
    - Dkey (str or list, optional): The key(s) of the data set(s) to plot. Default is an empty list.
    - **kwargs: Additional keyword arguments.

    Returns:
    - None

    Raises:
    - None

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
        im1 = ax[0].imshow(d_obs[i_plot,:].T, aspect='auto', cmap='jet_r', norm=matplotlib.colors.LogNorm(), extent=extent)
        im2 = ax[1].imshow(d_std[i_plot,:].T, aspect='auto', cmap='hot_r', norm=matplotlib.colors.LogNorm(), extent=extent)
        im3 = ax[2].imshow((d_std[i_plot,:]/d_obs[i_plot,:]).T, aspect='auto', vmin = 0.00, vmax = 0.10, extent=extent)
        ax[0].set_title('d_obs: observed data')
        ax[1].set_title('d_std: standard deviation')
        ax[2].set_title('d_std/d_obs: relative standard deviation')
        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])
        fig.colorbar(im3, ax=ax[2])
        ax[0].set_ylabel('Data #')
        ax[1].set_ylabel('Data #')
        ax[2].set_ylabel('Data #')

        im4 = ax[3].plot(i_plot,non_nan[i_plot], 'k.', markersize=.5)
        ax[3].set_ylabel('Number of data')
        ax[3].grid()
        ax[3].set_xlim(xlim)

        # Create an invisible colorbar for the last subplot
        cbar4 = fig.colorbar(im3, ax=ax[3])
        cbar4.solids.set(alpha=0)
        cbar4.outline.set_visible(False)
        cbar4.ax.set_yticks([])  # Hide the colorbar ticks
        cbar4.ax.set_yticklabels([])  # Hide the colorbar ticks labels

        ax[-1].set_xlabel('Index')
        
        plt.suptitle('Data set %s' % Dkey)
        plt.tight_layout()
    else:
        print("plot_data: Unknown noise model: %s" % noise_model)
        
    # set plot in kwarg to True if not allready set
    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s.png' % (os.path.splitext(f_data_h5)[0],Dkey))



def plot_data_prior_post(f_post_h5, i_plot=0, Dkey=[], **kwargs):
    """
    Plot the prior and posterior data for a given dataset.

    Parameters:
    - f_post_h5 (str): The path to the post data file.
    - i_plot (int): The index of the observation to plot..  
    - Dkey (str): String of the hdf5 key for the data set.    
    - **kwargs: Additional keyword arguments.

    Returns:
    - None
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import h5py
    import os
    
    ## Check if the data file f_data_h5 exists
    if not os.path.exists(f_post_h5):
        print("plot_data: File %s does not exist" % f_data_h5)
        return

    f_post = h5py.File(f_post_h5,'r')

    f_prior_h5 = f_post['/'].attrs['f5_prior']
    f_data_h5 = f_post['/'].attrs['f5_data']

    f_data = h5py.File(f_data_h5,'r')
    f_prior = h5py.File(f_prior_h5,'r')
    
    if len(Dkey)==0:
        nd = 0
        Dkeys = []
        for key in f_data.keys():
            if key[0]=='D':
                #print("plot_data: Found data set %s" % key)
                Dkeys.append(key)
            nd += 1
        Dkey=Dkeys[0]
        #print("plot_data: Using data set %s" % Dkey)

    noise_model = f_data['/%s' % Dkey].attrs['noise_model']
    if noise_model == 'gaussian':
        noise_model = 'Gaussian'
        d_obs = f_data['/%s' % Dkey]['d_obs'][:]
        d_std = f_data['/%s' % Dkey]['d_std'][:]

        i_use = f_post['/i_use'][i_plot,:]
        #i_use.sort()
        # flatten i_use
        i_use = i_use.flatten()

        nr=len(i_use)
        ns,ndata = f_data['/%s' % Dkey]['d_obs'].shape
        d_post = np.zeros((nr,ndata))
        d_prior = np.zeros((nr,ndata))

        for i in range(nr):
            d_post[i]=f_prior[Dkey][i_use[i]-1,:]
            d_prior[i]=f_prior[Dkey][i,:]    

        #i_plot=[]
        fig, ax = plt.subplots(1,1,figsize=(7,7))
        ax.semilogy(d_prior.T,'-',linewidth=.1, label='d_prior', color='gray')
        ax.semilogy(d_post.T,'-',linewidth=.1, label='d_prior', color='black')
        
        ax.semilogy(d_obs[i_plot,:],'r.',markersize=6, label='d_obs')
        ax.semilogy(d_obs[i_plot,:]-2*d_std[i_plot,:],'r.',markersize=3, label='d_obs')
        ax.semilogy(d_obs[i_plot,:]+2*d_std[i_plot,:],'r.',markersize=3, label='d_obs')
        
        #ax.text(0.1, 0.1, 'Data set %s, Observation # %d' % (Dkey, i_plot+1), transform=ax.transAxes)
        ax.text(0.1, 0.1, 'T = %4.2f.' % (f_post['/T'][i_plot]), transform=ax.transAxes)
        ax.text(0.1, 0.2, 'EV = %4.2f.' % (f_post['/EV'][i_plot]), transform=ax.transAxes)
        print(f_post['/T'][i_plot])
        plt.title('Data set %s, Observation # %d' % (Dkey, i_plot+1))
        plt.xlabel('Data #')
        plt.ylabel('Data')
        plt.grid()

        # set plot in kwarg to True if not allready set
        if 'hardcopy' not in kwargs:
            kwargs['hardcopy'] = True
        if kwargs['hardcopy']:
            # strip the filename from f_data_h5
            # get filename without extension of f_post_h5
            plt.savefig('%s_%s_id%05d.png' % (os.path.splitext(f_post_h5)[0],Dkey,i_plot+1))



def plot_prior_stats(f_prior_h5, Mkey='M1', **kwargs):

    f_prior = h5py.File(f_prior_h5,'r')
    f_prior['/%s'%Mkey].attrs.keys()
    if 'x' in f_prior['/%s'%Mkey].attrs.keys():
        z = f_prior['/%s'%Mkey].attrs['x']
    else:
        z = f_prior['/%s'%Mkey].attrs['z']

    is_discrete = f_prior['/%s'%Mkey].attrs['is_discrete']    


    if not is_discrete:

        # setup a figure with two suplots in row ONE AND ONE SUBPLOT IN ROW 2
        

        M = f_prior[Mkey][:]
        fig, ax = plt.subplots(2,2,figsize=(10,10))
        m0 = ax[0,0].hist(M.flatten(),101)
        ax[0,0].set_xlabel(Mkey)
        ax[0,0].set_ylabel('Distribution')
        m1 = ax[0,1].hist(np.log10(M.flatten()),101)
        ax[0,1].set_xlabel(Mkey)

        # set xtcik labels as 10^x where x i the xtick valye
        ax[0,1].set_xticklabels(['$10^{%3.1f}$'%i for i in ax[0,1].get_xticks()])
        ax[0,1].set_ylabel('Distribution')


        # use the ax[1,0] and ax[1,1] for one ploit
        nr=100
        # set the extent from 1,nr and z[0],z[-1]
        extent = [1,nr,z[0],z[-1]]

        ax[1, 0].axis('off')    
        ax[1, 1].axis('off')

        ax[1, 0] = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        m2 = ax[1,0].imshow(M[0:nr,:].T, aspect='auto', extent=extent)
        fig.colorbar(m2, ax=ax[1,0], label=Mkey)
        tit = '%s - %s ' % (os.path.splitext(f_prior_h5)[0],Mkey) 
        plt.suptitle(tit)
    else:
        print("is_discrete=%d not yet implemented" % is_discrete)

    f_prior.close()

    if 'hardcopy' not in kwargs:
        kwargs['hardcopy'] = True
    if kwargs['hardcopy']:
        # strip the filename from f_data_h5
        plt.savefig('%s_%s.png' % (os.path.splitext(f_prior_h5)[0],Mkey))

