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
        
        k = int(np.floor(np.log2(Ncpu)))
        Nproc_arr = 2**np.linspace(0,k,(k)+1)
        Nproc_arr = np.append(Nproc_arr, Ncpu)
        Nproc_arr = np.unique(Nproc_arr)
        Nproc_arr = Nproc_arr[5::]

        if args.size == 'small':
            # Small benchmark
            N_arr = np.ceil(np.logspace(2,4,3))
            N_arr = np.array([25000])
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