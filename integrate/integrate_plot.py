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
                plt.show()

                # get filename without extension
                #f_png = '%s_%d_%d_%s_feature.png' % (os.path.splitext(f_post_h5)[0],i1,i2,dstr[0:-1])
                #plt.savefig(f_png)

            else:
                print("Key %s not found in %s" % (key, dstr))
    return 1

def plot_T(f_post_h5, i1=1, i2=1e+9, T_min=0, T_max=100, **kwargs):
    
    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    with h5py.File(f_post_h5,'r') as f_post:
        T=f_post['/T'][:].T
        try:
            T=f_post['/T_mul'][:]
        except:
            a=1

    nd = X.shape[0]
    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd

    if i2<i1:
        i2=i1+1

    plt.figure(1, figsize=(20, 10))
    plt.scatter(X[i1:i2],Y[i1:i2],c=T[i1:i2],cmap='jet',**kwargs)            
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.clim(T_min,T_max)  
    plt.colorbar()
    plt.clim(T_min,T_max)
    plt.title('Temperature')
    plt.axis('equal')
    plt.show()

    # get filename without extension
    f_png = '%s_%d_%d_T.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
    plt.savefig(f_png)

    return 1

def plot_profile_continuous(f_post_h5, i1=1, i2=1e+9, im=1):
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

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']
    
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im

    with h5py.File(f_prior_h5,'r') as f_prior:
        z = f_prior[Mstr].attrs['z'][:].flatten()
        is_discrete = f_prior[Mstr].attrs['is_discrete']
        if 'clim' in f_prior[Mstr].attrs.keys():
            clim = f_prior[Mstr].attrs['clim'][:].flatten()
        else:
            clim = [.1, 2600]
        print(clim)

    if is_discrete:
        print("This is a discrete model. Use plot_profile_discrete instead")

    with h5py.File(f_post_h5,'r') as f_post:
        Mean=f_post[Mstr+'/Mean'][:].T
        Median=f_post[Mstr+'/Median'][:].T
        Std=f_post[Mstr+'/Std'][:].T
        T=f_post['/T'][:].T
        try:
            T=f_post['/T_mul'][:]
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

    # compute the depth from the surface plus the elevation
    for i in range(nd):
        ZZ[:,i] = ELEVATION[i]-ZZ[:,i]


    if i1<1: 
        i1=0
    if i2>nd-1:
        i2=nd

    from matplotlib.colors import LogNorm

    # Create a figure with 3 subplots sharing the same Xaxis!
    plt.figure(1, figsize=(20, 10))
    plt.subplot(4,1,1)
    plt.pcolor(ID[:,i1:i2], ZZ[:,i1:i2], Mean[:,i1:i2], 
            cmap='hsv',            
            shading='auto',
            norm=LogNorm())
    plt.clim(clim[0],clim[1])        
    plt.title('Mean')
    plt.colorbar()

    plt.subplot(4,1,2)
    plt.pcolor(ID[:,i1:i2], ZZ[:,i1:i2], Median[:,i1:i2], 
            cmap='hsv',            
            shading='auto',
            norm=LogNorm())  # Set color scale to logarithmic
    plt.clim(clim[0],clim[1])        
    plt.title('Median')
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.pcolor(ID[:,i1:i2], ZZ[:,i1:i2], Std[:,i1:i2], 
            cmap='jet', 
            vmin=0, vmax=0.5, 
            shading='auto')
    plt.title('Std')
    plt.colorbar()
    plt.subplot(4,1,4)
    plt.plot(ID[0,i1:i2],T[i1:i2], 'k')
    plt.ylabel('Temperature')
    plt.grid()
    plt.xlabel('ID')
    plt.tight_layout()
    plt.show()

    # get filename without extension
    f_png = '%s_%d_%d_profile.png' % (os.path.splitext(f_post_h5)[0],i1,i2)
    plt.savefig(f_png)

            