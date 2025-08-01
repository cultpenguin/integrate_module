#!/usr/bin/env python
"""
INTEGRATE Timing CLI

Command-line interface for timing benchmarks of the INTEGRATE workflow.
This module imports timing functions from the main integrate module.

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

# Import timing functions from integrate module
try:
    # Try relative import first (when run as module)
    from . import integrate as ig
    from .integrate import timing_compute, timing_plot, allocate_large_page
except ImportError:
    try:
        # Try absolute import (when run directly)
        import integrate as ig
        from integrate import timing_compute, timing_plot, allocate_large_page
    except ImportError:
        print("Error: Could not import integrate module. Please ensure it is properly installed.")
        import sys
        sys.exit(1)


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
    parser = argparse.ArgumentParser(
        description='INTEGRATE timing benchmark tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INTEGRATE Timing Benchmark Tool

This tool benchmarks the performance of the complete INTEGRATE workflow including:
1. Prior model generation (layered geological models)
2. Forward modeling using GA-AEM electromagnetic simulation  
3. Rejection sampling for Bayesian inversion
4. Posterior statistics computation

USAGE EXAMPLES:

Basic benchmarks:
  integrate_timing time small                    # Quick test with default settings
  integrate_timing time medium                   # Balanced benchmark  
  integrate_timing time large                    # Comprehensive benchmark

Custom dataset sizes:
  integrate_timing time small --Nmin 5000        # Test with 5000 models
  integrate_timing time small --N 100000         # Test with exactly 100000 models
  integrate_timing time medium --Nmin 10000      # Medium test starting from 10000 models

Custom CPU configurations:
  integrate_timing time small --Ncpu 16          # Test with exactly 16 CPUs
  integrate_timing time medium --cpu-scale linear # Test all CPU counts [1,2,3,...,64]
  integrate_timing time large --cpu-scale log    # Test log scale [1,2,4,8,16,32,64]

Combined options:
  integrate_timing time small --Ncpu 32 --Nmin 50000    # 50k models on 32 CPUs
  integrate_timing time medium --N 25000 --cpu-scale linear  # 25k models, all CPU counts

Plotting results:
  integrate_timing plot timing_results.npz       # Plot specific results file
  integrate_timing plot --all                    # Plot all .npz files in directory

PARAMETER PRIORITY:
- Dataset sizes: --N (highest) > --Nmin > default
- CPU counts: --Ncpu (highest) > --cpu-scale/--Nmin > default

BENCHMARK SIZES:
- small:  ~1,000 models, quick test
- medium: 1,000-100,000 models, balanced test  
- large:  10,000-1,000,000 models, comprehensive test

Results are saved as .npz files and automatically plotted with performance analysis.
        """
    )
    
    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot timing results from benchmark files')
    plot_parser.add_argument('file', nargs='?', default='time', 
                           help='NPZ file containing timing results to plot')
    plot_parser.add_argument('--all', action='store_true', 
                           help='Plot all NPZ timing files in the current directory')
    
    # Time command
    time_parser = subparsers.add_parser('time', help='Run performance benchmark of INTEGRATE workflow')
    time_parser.add_argument('size', choices=['small', 'medium', 'large'], 
                            default='medium', nargs='?', 
                            help='Benchmark size: small (~1k models, quick), medium (1k-100k models), large (10k-1M models)')
    time_parser.add_argument('--cpu-scale', choices=['linear', 'log'], 
                            default='log', 
                            help='CPU scaling method: linear tests [1,2,3,...,Ncpu], log tests [1,2,4,8,...,Ncpu] (default: log)')
    time_parser.add_argument('--Nmin', type=int, default=0,
                            help='Dataset size control: For small benchmark, use exactly this many models. For medium/large, use as starting point in range (default: use benchmark defaults)')
    time_parser.add_argument('--Ncpu', type=int, default=0,
                            help='Use exactly this many CPU cores, overriding all other CPU options (default: 0, use scaling)')
    time_parser.add_argument('--N', type=int, default=0,
                            help='Use exactly this dataset size (number of models), overriding size and Nmin options (default: 0, use size-based defaults)')
    
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
        # Show help when no command is specified
        parser.print_help()
        sys.exit(0)
   
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
        
        # Handle Ncpu option for processors
        if args.Ncpu > 0:
            # Use only the specified number of CPUs
            Nproc_arr = np.array([args.Ncpu])
        else:
            # Determine CPU scaling based on command line option
            if args.cpu_scale == 'linear':
                Nproc_arr = np.arange(1, Ncpu+1)
            else:  # log scaling
                k = int(np.floor(np.log2(Ncpu)))
                Nproc_arr = 2**np.linspace(0,k,(k)+1)
                Nproc_arr = np.append(Nproc_arr, Ncpu)
                Nproc_arr = np.unique(Nproc_arr)

        # Handle dataset sizes
        if args.N > 0:
            # Use only the specified dataset size
            N_arr = np.array([args.N])
        elif args.Nmin > 0 and args.size == 'small':
            # For small benchmark with Nmin: use only that value for dataset size
            N_arr = np.array([args.Nmin])
        elif args.size == 'small':
            # Small benchmark default
            N_arr = np.array([1000])
        elif args.size == 'medium':
            # Medium benchmark
            if args.Nmin > 0:
                # Use Nmin as starting point for medium benchmark
                N_arr = np.ceil(np.logspace(np.log10(args.Nmin), 5, 9))
            else:
                N_arr = np.ceil(np.logspace(3,5,9))
        elif args.size == 'large':
            # Large benchmark
            if args.Nmin > 0:
                # Use Nmin as starting point for large benchmark
                N_arr = np.ceil(np.logspace(np.log10(args.Nmin), 6, 7))
            else:
                N_arr = np.ceil(np.logspace(4,6,7))

        f_timing = timing_compute(
            N_arr=N_arr,
            Nproc_arr=Nproc_arr
        )
        
        # Always plot the results
        timing_plot(f_timing)

if __name__ == '__main__':
    main()