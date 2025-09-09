
if __name__ == "__main__":
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
    if __name__ == "__main__":
        parallel = True
    else:
        parallel = ig.use_parallel(showInfo=1)

    print('parallel=%s' % parallel )

    hardcopy = True 
    import matplotlib.pyplot as plt
    #plt.ion()
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

    # %%
    # Select how many prior model realizations (N) should be generated
    import integrate as ig
    import numpy as np
    N=6000000

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

    if doPlot:
        ig.plot_prior_stats(f_prior_h5)

    #ig.copy_hdf5_file(f_prior_h5, 'PRIOR_2.h5',showInfo=2)
    #
    M,idx = ig.load_prior_model(f_prior_h5)

    M_RHO = M[0]
    M_ALT = M[im_height-1]

    #%% 
    print(GENERAL['RxCoilPosition'])
    txrx_dx= GENERAL['RxCoilPosition'][0][0]
    txrx_dy= GENERAL['RxCoilPosition'][0][1]
    txrx_dz= -1*GENERAL['RxCoilPosition'][0][2]
    print('txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (txrx_dx, txrx_dy, txrx_dz))
    #%% 
    stmfiles=[file_stm]
    f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5,
                        im_height=im_height, 
                        stmfiles=stmfiles, 
                        showInfo=0, 
                        txrx_dx = txrx_dx,
                        txrx_dy = txrx_dy,
                        txrx_dz = txrx_dz,
                        parallel=parallel)

    f_prior_data_h5 = ig.prior_data_identity(f_prior_data_h5, im=im_height)

    if doPlot:
        ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=1, id_data=1)
        ig.plot_data_prior(f_prior_data_h5, f_data_h5, id=2, id_data=2)

    #%%
    D_obs = ig.load_data(f_data_h5)

    D_prior, idx = ig.load_prior_data(f_prior_data_h5)

    #%% %%%% INVERT
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, 
                                    f_data_h5, 
                                    f_post_h5 = 'POST.h5', 
                                    showInfo=1, 
                                    parallel=parallel,
                                    id_use=[1,2])

    # %%
    if doPlot:
        ig.plot_profile(f_post_h5, i1=1, i2=4000, im=1, hardcopy=hardcopy)
        ig.plot_profile(f_post_h5, i1=1, i2=4000, im=4, hardcopy=hardcopy)
    #%%
    if doPlot:
        ig.plot_T_EV(f_post_h5, pl='LOGL_mean',hardcopy=hardcopy)

    # %%
    if doPlot:
        ig.plot_feature_2d(f_post_h5,im=1,iz=10, key='Median', uselog=1, cmap='jet', s=10,hardcopy=hardcopy)
        plt.show()
    # %%



    # %% [markdown]
    # ## Export results to CSV format
    # Export the posterior results to CSV files for use in GIS software or further analysis.

    # %%
    f_csv, f_point_csv = ig.post_to_csv(f_post_h5)




    # %%
    # Optional: Use PyVista for 3D visualization of X,Y,Z coordinates with median resistivity
    plPyVista = True
    if (plPyVista)&(doPlot):
        # Example filename (actual filename will be generated automatically):
        import pandas as pd
        df = pd.read_csv(f_point_csv)
        df.head()

        import pyvista as pv
        import numpy as np
        #pv.set_jupyter_backend('client')
        #pv.set_jupyter_backend('pythreejs')  # or 'static' if you want static images
        #pv.set_jupyter_backend('trame')  # or 'static' if you want static images
        pv.set_plot_theme("document")
        p = pv.Plotter()
        # Don't filter out the resistivity range you want to map opacity for
        filtered_df = df  # Use all data or filter by other criteria like LINE
        #filtered_df = df[(df['LINE'] > 1000) & (df['LINE'] < 1400) ]
        points = filtered_df[['X', 'Y', 'Z']].values[:]
        std = filtered_df[['Std']].values[:]
        # Scale Z-axis differently
        z_exxageration = 2
        points[:, 2] = points[:, 2] * z_exxageration
        median = np.log10(filtered_df['Mean'].values[:])
        #opacity = np.where(filtered_df['Median'].values[:] > 100, 0.2, 1.0)
        # Linear opacity mapping from resistivity values
        resistivity_values = filtered_df['Median'].values[:]
        min_resistivity = .1;150   # Full opacity (1.0) at this resistivity
        max_resistivity = 200;400  # No opacity (0.0) at this resistivity
        opacity = np.clip((max_resistivity - resistivity_values) / (max_resistivity - min_resistivity), 0.1, 1.0)
        # COmpute opacrity from Std. Everything with Std>1 shgould be transrent. Everything with Std<0.1 should be fully opaque
        min_std = 0.1   # Full opacity (1.0) at this
        max_std = 1.0  # No opacity (0.0) at this
        opacity = np.clip((max_std - std.flatten()) / (max_std - min_std), 0.99, 1.0)
        p.add_points(points, render_points_as_spheres=True, point_size=10, scalars=median, cmap='jet', opacity=opacity)
        #p.add_points(points, render_points_as_spheres=True, point_size=6, scalars=median, cmap='jet', opacity=1)
        p.show_grid()
        p.show()


    

# %%
