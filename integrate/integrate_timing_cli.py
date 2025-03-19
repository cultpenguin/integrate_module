#!/usr/bin/env python
# %% [markdown]
# # INTEGRATE timing
# This notebook compares CPU time using for both forward modeling and inversion 
#
# # Force use of single CPU with numpy
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#


# %% Functions


def allocate_large_page():
    import os
    import ctypes
    """Allocates a 2MB large page if running on Windows."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        
        LARGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
        
        MEM_COMMIT = 0x1000
        MEM_LARGE_PAGES = 0x20000000
        PAGE_READWRITE = 0x04
        
        ptr = kernel32.VirtualAlloc(None, LARGE_PAGE_SIZE, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
        
        if not ptr:
            error_code = ctypes.GetLastError()
            print(f"Failed to allocate large page. Error code: {error_code}")
            return None
        
        print(f"Successfully allocated {LARGE_PAGE_SIZE} bytes at address {hex(ptr)}")
        return ptr
    else:
        print("Large pages are only supported on Windows.")
        return None


## Example usage
#large_page_memory = allocate_large_page()



def timing_compute(N_arr=[], Nproc_arr=[]):

    import integrate as ig
    # check if parallel computations can be performed
    parallel = ig.use_parallel(showInfo=1)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import loglog
    import time
    import h5py
    # get name of CPU
    import psutil

    # Get hostname and number of processors
    import socket
    hostname = socket.gethostname()
    import platform
    hostname = platform.node()
    system = platform.system()

    ## Get number of processors
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    Ncpu = physical_cores

    print("Hostname (system): %s (%s) " % (hostname, system))
    print("Number of processors: %d" % Ncpu)
    
    # SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
    files = ig.get_case_data()
    f_data_h5 = files[0]
    file_gex= ig.get_gex_file_from_data(f_data_h5)

    print("Using data file: %s" % f_data_h5)
    print("Using GEX file: %s" % file_gex)

    with h5py.File(f_data_h5, 'r') as f:
        nobs = f['D1/d_obs'].shape[0]


    ## Setup the timing test

    #### Set the size of the data sets to test
    if len(N_arr)==0:
        N_arr = np.array([100,500,1000,5000,10000,50000,100000, 500000, 1000000, 5000000])

    # Set the number of cores to test
    if len(Nproc_arr)==0:
        Nproc_arr=2**(np.double(np.arange(1+int(np.log2(Ncpu)))))   

    n1 = len(N_arr)
    n2 = len(Nproc_arr)    

    print("Testing on %d data sets of sizes" % len(N_arr))
    print(N_arr)
    print("Testing on %d sets of cores" %  len(Nproc_arr))
    print(Nproc_arr)


    file_out  = 'timing_%s-%s-%dcore_Nproc%d_N%d.npz' % (hostname,system,Ncpu,len(Nproc_arr), len(N_arr))
    print("Writing results to %s " % file_out)

    ## TIMING

    showInfo = 0

    T_prior = np.zeros((n1,n2))*np.nan
    T_forward = np.zeros((n1,n2))*np.nan
    T_rejection = np.zeros((n1,n2))*np.nan
    T_poststat = np.zeros((n1,n2))*np.nan

    testRejection = True
    testPostStat = True  
                
    for j in np.arange(n2):
        Ncpu = int(Nproc_arr[j])

        for i in np.arange(len(N_arr)):
            N=int(N_arr[i])
            Ncpu_min = int(np.floor(2**(np.log10(N)-4)))
            
            print('=====================================================')
            print('TIMING: N=%d, Ncpu=%d, Ncpu_min=%d'%(N,Ncpu,Ncpu_min))
            print('=====================================================')
            
            RHO_min = 1
            RHO_max = 800
            z_max = 50 
            useP = 1
            
            if (Ncpu>=Ncpu_min):
                    
                t0_prior = time.time()
                if useP ==1:
                    ## Layered model    
                    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max, showInfo=showInfo)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                else: 
                    ## N layer model with increasing thickness
                    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max, showInfo=showInfo)
                #t_prior.append(time.time()-t0_prior)
                T_prior[i,j] = time.time()-t0_prior

            
                #ig.plot_prior_stats(f_prior_h5)
                #% A2. Compute prior DATA
                t0_forward = time.time()
                f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=Ncpu, showInfo=showInfo)
                T_forward[i,j]=time.time()-t0_forward

                #% READY FOR INVERSION
                N_use = 1000000
                t0_rejection = time.time()
                if testRejection:
                    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use = N_use, parallel=1, updatePostStat=False,  Ncpu=Ncpu, showInfo=showInfo)
                T_rejection[i,j]=time.time()-t0_rejection

                #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
                t0_poststat = time.time()
                if testPostStat and testRejection:
                    ig.integrate_posterior_stats(f_post_h5,showInfo=showInfo)
                    T_poststat[i,j]=time.time()-t0_poststat

            T_total = T_prior + T_forward + T_rejection + T_poststat
            np.savez(file_out, T_total=T_total, T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr, nobs=nobs)
            
            
    return file_out

def timing_plot(f_timing=''):    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if len(f_timing)==0:
        print('No timing file provided')
        return
    else:
        print('Plotting timing results from %s' % f_timing)

    # file_out is f_timing, without file extension
    file_out = f_timing.split('.')[0]
    
    data = np.load(f_timing)
    T_prior = data['T_prior']
    T_forward = data['T_forward']
    T_rejection = data['T_rejection']
    T_poststat = data['T_poststat']

    N_arr = data['N_arr']
    Nproc_arr = data['Nproc_arr']

    try:
        T_total = data['T_total']
    except:
        T_total = T_prior + T_forward + T_rejection + T_poststat

    try:
        nobs=data['nobs']
    except:
        nobs=11693

    # Plot
    # LSQ, Assumed time, in seconds, for least squares inversion of a single sounding
    t_lsq = 2.0
    # SAMPLING, Assumed time, in seconds, for an McMC inversion of a single sounding
    t_mcmc = 10.0*60.0 

    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    total_mcmc = np.array([nobs*t_mcmc, nobs*t_mcmc/Nproc_arr[-1]])


    # loglog(T_total.T)
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_total.T, 'o-',  label=N_arr)
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_lsq, 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_mcmc, 'r--', label='MCMC')
    plt.legend(loc='upper right')
    plt.xticks(ticks=Nproc_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.ylim(1,1e+8)
    plt.tight_layout()
    plt.savefig('%s_total_sec' % file_out)

    plt.figure(figsize=(6,6)) 
    plt.loglog(N_arr, T_total, 'o-', label=[f'{int(x)}' for x in Nproc_arr])
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('N-prior')
    plt.grid()
    plt.tight_layout()
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_lsq, nobs*t_lsq], 'k--', label='LSQ')
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_mcmc, nobs*t_mcmc], 'r--', label='MCMC')
    plt.legend(loc='upper left')
    #plt.xticks(ticks=N_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.ylim(1,1e+8)
    plt.savefig('%s_total_sec' % file_out)


    #### Plot timing results for forward modeling - GAAEM
    # Average timer per sounding 
    T_forward_sounding = T_forward/N_arr[:,np.newaxis]
    T_forward_sounding_per_sec = N_arr[:,np.newaxis]/T_forward
    T_forward_sounding_per_sec_per_cpu = T_forward_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_forward_sounding_speedup = T_forward_sounding_per_sec/T_forward_sounding_per_sec[0,0]

    ## Forward time per sounding - CPU
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_forward.T, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--', 
                    label='Linear scaling', 
                    linewidth=0.5)   
        except:
            pass
    
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr, loc='upper left')
    plt.ylim(1e-1, 1e+4)
    plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.tight_layout()
    plt.savefig('%s_forward_sec_CPU' % file_out)

    ## Forward time per sounding - Nproc
    plt.figure(figsize=(6,6))    
    plt.loglog(N_arr, T_forward, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            ref_time = T_forward[i,idx]
            ref_N = N_arr[i]
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/ref_N, ref_time*N_arr[-1]/ref_N], 'k--', label='Linear scaling', linewidth=0.5)   
        except:
            pass
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of models')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(Nproc_arr, loc='upper left')
    #plt.ylim(1e-1, 1e+4)
    #plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.tight_layout()
    plt.savefig('%s_forward_sec_N' % file_out)


    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec.T, 'o-')
    # plot line 
    plt.ylabel(r'Forward computations per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.savefig('%s_forward_sounding_per_sec' % file_out)

    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylabel('Forward computations per second per cpu')
    plt.xlabel('Number of processors')
    plt.title('Forward calculation')
    plt.grid()
    # Make yaxis start at 0
    plt.ylim(0, 80)    
    plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.legend(N_arr)
    plt.savefig('%s_forward_sounding_per_sec_per_cpu' % file_out)
    #

    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('gatdaem - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_forward_speedup' % file_out)



    ### STATS FOR REJECTION SAMPLING
    # Average timer per sounding
    T_rejection_sounding = T_rejection/N_arr[:,np.newaxis]
    T_rejection_sounding_per_sec = N_arr[:,np.newaxis]/T_rejection
    T_rejection_sounding_per_sec_per_cpu = T_rejection_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec/T_rejection_sounding_per_sec[0,0]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec*0

    T_rejection_per_data = nobs/T_rejection

    for i in range(len(N_arr)):
        # find index of first value in T_rejection_sounding_per_sec[i,:] that is not nan
        try:
            idx = np.where(~np.isnan(T_rejection_sounding_per_sec[i,:]))[0][0]
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]/(T_rejection_sounding_per_sec[i,idx]/Nproc_arr[idx]) 
        except:
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]*0


    ## Rejection total sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection.T, 'o-')
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.ylim(1e-1, 2e+3)
    plt.savefig('%s_rejection_sec_CPU' % file_out)


    ## Rejection total sec - per process
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection, 'o-')
    for i in range(len(Nproc_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[:,i]))[0][0]
            ref_time = np.abs(T_rejection[idx,i])
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/N_arr[idx], ref_time*N_arr[-1]/N_arr[idx]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.tight_layout()
    plt.ylim(1e-1, 2e+3)
    plt.savefig('%s_rejection_sec_N' % file_out)


    ## Rejection speedup
    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_rejection_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('Rejection sampling - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_speedup' % file_out)


    ## Rejection sound per sec
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_lsq, 1./t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_mcmc, 1./t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - number of soundings per second - $s^{-1}$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.ylim(1e-3, 1e+5)
    plt.savefig('%s_rejection_sound_per_sec' % file_out)

    ## Rejection sec per sounding
    plt.figure(figsize=(6,6))
    plt.semilogy(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    #plt.plot(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_lsq, t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_mcmc, t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - seconds per sounding - $s$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim(1e-5, 1e+3)
    plt.savefig('%s_rejection_sec_per_sound' % file_out)

    ## Rejection sound per sec - N
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.savefig('%s_rejection_sound_per_sec_N' % file_out)

    ## Rejection sound per sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_sounding_per_sec.T, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_sound_per_sec_CPU' % file_out)

    ##  Sound per sec per CPU - N  
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec_per_cpu, 'o-')
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    plt.xlim(90, 5000000*1.1)
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.savefig('%s_rejection_sound_per_sec_per_cpu_N' % file_out)


    ##  Sound per sec per CPU - CPU 
    plt.figure(figsize=(6,6))
    plt.semilogx(Nproc_arr, T_rejection_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylim([0, np.nanmax(T_rejection_sounding_per_sec_per_cpu.T)*1.1])
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.savefig('%s_rejection_sound_per_sec_per_cpu_CPU' % file_out)




    #### STATS FOR POSTERIOR STATISTICS
    # Average timer per sounding
    T_poststat_sounding = T_poststat/N_arr[:,np.newaxis]
    T_poststat_sounding_per_sec = N_arr[:,np.newaxis]/T_poststat
    T_poststat_sounding_per_sec_per_cpu = T_poststat_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_poststat_sounding_speedup = T_poststat_sounding_per_sec/T_poststat_sounding_per_sec[0,0]

    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_poststat_sounding_per_sec.T, 'o-')
    plt.ylabel('Posterior statistics - Soundings per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.savefig('%s_poststat_sounding_per_sec' % file_out)

    # plt.figure(figsize=(6,6))
    # plt.plot(Nproc_arr, T_poststat_sounding_speedup.T, 'o-')
    # # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    # plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # # set xlim to 1, Nproc_arr[-1]
    # plt.xlim(.8, Nproc_arr[-1])
    # plt.ylim(.8, Nproc_arr[-1])
    # plt.ylabel('Posterior statistics - speedup compared to 1 processor')
    # plt.xlabel('Number of processors')
    # plt.grid()
    # plt.legend(N_arr)
    # plt.savefig('%s_poststat_speedup' % file_out)

    #####
    # ## Plot Cumulative Time useage for min and max number of used cores

    i_proc = len(Nproc_arr)-1
    #i_proc= 0

    for i_proc in [0,len(Nproc_arr)-1]:

        T=[T_prior[:,i_proc], T_forward[:,i_proc], T_rejection[:,i_proc], T_poststat[:,i_proc]]

        ### %% Plor cumT as an area plot
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T, labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.plot(N_arr, T_total[:, i_proc], 'k--')
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Time [$s$]')
        plt.title('Cumulative time, using %d processors' % Nproc_arr[i_proc])
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('%s_Ncpu%d_cumT' % (file_out,Nproc_arr[i_proc]))

        # The same as thea area plot but normalized to the total time
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T/np.sum(T, axis=0), labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.xscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Normalized time')
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.title('Normalized time, using %d processors' % Nproc_arr[i_proc])
        plt.savefig('%s_Ncpu%d_cumT_norm' % (file_out,Nproc_arr[i_proc]))



# %% The main function
def main():
    """Entry point for the integrate_timing command."""
    import argparse
    import sys
    import os
    import glob
    import psutil
    import numpy as np

    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set a lower limit for processes to avoid handle limit issues on Windows
    import platform
    if platform.system() == 'Windows':
        # On Windows, limit the max processes to avoid handle limit issues
        multiprocessing.set_start_method('spawn')
        
        # Optional - can help with some multiprocessing issues
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    

    # Create argument parser
    parser = argparse.ArgumentParser(description='INTEGRATE timing benchmark tool')
    
    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot timing results')
    plot_parser.add_argument('file', nargs='?', default='time', help='NPZ file to plot')
    plot_parser.add_argument('--all', action='store_true', help='Plot all NPZ files in the current directory')
    
    # Time command
    time_parser = subparsers.add_parser('time', help='Run timing benchmark')
    time_parser.add_argument('size', choices=['small', 'medium', 'large'], 
                            default='medium', nargs='?', help='Size of the benchmark')
    
    # Add special case handling for '-time' without size argument
    if '-time' in sys.argv and len(sys.argv) == 2:
        print("Please specify a size for the timing benchmark:")
        print("  small  - Quick test with minimal resources")
        print("  medium - Balanced benchmark (default)")
        print("  large  - Comprehensive benchmark (may take hours)")
        print("\nExample: integrate_timing -time medium")
        sys.exit(0)
        
    # Parse arguments
    args = parser.parse_args()
    
    # Set default command if none is provided
    if args.command is None:
        args.command = 'time'
        args.size = 'small'
   
    # Execute command
    if args.command == 'plot':
        if args.all:
            # Plot all NPZ files in the current directory
            files = glob.glob('*.npz')
            for f in files:
                try:
                    timing_plot(f)
                    print(f"Successfully plotted: {f}")
                except Exception as e:
                    print(f"Error plotting {f}: {str(e)}")
        elif args.file:
            # Plot specified file
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            try:
                timing_plot(args.file)
                print(f"Successfully plotted: {args.file}")
            except Exception as e:
                print(f"Error plotting {args.file}: {str(e)}")
        else:
            print("Please specify a file to plot or use --all")
    
    elif args.command == 'time':
        Ncpu = psutil.cpu_count(logical=False)
        import os
        if os.name == 'nt':  # Windows
            # use max 32 Cpus
            Ncpu = min(Ncpu, 32)
                
        k = int(np.floor(np.log2(Ncpu)))
        Nproc_arr = 2**np.linspace(0,k,(k)+1)
        Nproc_arr = np.append(Nproc_arr, Ncpu)
        Nproc_arr = np.unique(Nproc_arr)
        Nproc_arr = np.unique(Nproc_arr)
        
        if args.size == 'small':
            # Small benchmark
            N_arr = np.ceil(np.logspace(2,4,3))
            N_arr = np.array([10000])
            f_timing = timing_compute(
                N_arr = N_arr,
                Nproc_arr = Nproc_arr
            )
        elif args.size == 'medium':
            # Medium benchmark
            N_arr=np.ceil(np.logspace(3,5,9)) 
            Nproc_arr = np.arange(1,Ncpu+1)

            f_timing = timing_compute(
                N_arr=np.ceil(np.logspace(3, 5, 9)), 
                Nproc_arr=Nproc_arr
            )
        elif args.size == 'large':
            # Large benchmark
            N_arr = np.ceil(np.logspace(4,6,7))
            f_timing = timing_compute(                
                N_arr=N_arr,
                Nproc_arr=Nproc_arr
            )
        
        # Always plot the results
        timing_plot(f_timing)

if __name__ == '__main__':
    main()