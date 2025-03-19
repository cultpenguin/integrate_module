#!/usr/bin/env python3

import argparse
import os
import sys
from typing import List, Optional

def validate_prior_file(filepath):
    # Check if file exists
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    # Here you would add code to verify it's a PRIOR type HDF5 file
    # For now we just check the extension
    if not filepath.lower().endswith('.h5'):
        raise argparse.ArgumentTypeError(f"File must be an HDF5 file: {filepath}")
    return filepath

def validate_data_file(filepath):
    # Check if file exists
    if not os.path.isfile(filepath):
        raise argparse.ArgumentTypeError(f"File not found: {filepath}")
    # Here you would add code to verify it's a DATA type HDF5 file
    # For now we just check the extension
    if not filepath.lower().endswith('.h5'):
        raise argparse.ArgumentTypeError(f"File must be an HDF5 file: {filepath}")
    return filepath

def main():
    """Command line interface for integrate_rejection function."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian inversion using rejection sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required positional arguments
    parser.add_argument("prior_file_h5", type=validate_prior_file,
                        help="Path to prior model in HDF5 format (PRIOR type)")
    parser.add_argument("data_file_h5", type=validate_data_file, 
                        help="Path to observed data in HDF5 format (DATA type)")
    
    # Optional arguments
    parser.add_argument("--f_post_h5", type=str, default="",
                        help="Path for output posterior file (default: auto-generated)")
    parser.add_argument("--N_use", type=int, default=100000000000,
                        help="Number of samples to use from prior")
    parser.add_argument("--id_use", type=int, nargs="+", default=[],
                        help="Data IDs to use (default: all)")
    parser.add_argument("--nr", type=int, default=400,
                        help="Number of posterior samples to draw")
    parser.add_argument("--autoT", type=int, default=1, choices=[0, 1],
                        help="Auto-compute temperature (1) or use fixed (0)")
    parser.add_argument("--T_base", type=float, default=1.0,
                        help="Base temperature when not using autoT")
    parser.add_argument("--Nchunks", type=int, default=0,
                        help="Number of chunks to divide workload (0 = auto)")
    parser.add_argument("--Ncpu", type=int, default=0,
                        help="Number of CPUs to use (0 = auto)")
    parser.add_argument("--no_parallel", action="store_true",
                        help="Disable parallel processing")
    parser.add_argument("--use_N_best", type=int, default=0,
                        help="Use only the N best samples")
    parser.add_argument("--showInfo", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=normal, 2=verbose)")
    parser.add_argument("--updatePostStat", action="store_true", default=True,
                        help="Update posterior statistics after inversion")
    parser.add_argument("--no_updatePostStat", action="store_false", dest="updatePostStat",
                        help="Skip updating posterior statistics")
    parser.add_argument("--post_dir", type=str, default="",
                        help="Directory for posterior output (default: current directory)")
    
    args = parser.parse_args()
    
    # Import the integrate module (done here to avoid import overhead if help is requested)
    try:
        import integrate
    except ImportError:
        print("ERROR: Could not import integrate module. Please ensure it is installed.")
        sys.exit(1)
    
    # Convert argument list to arrays
    if not args.id_use:
        id_use = []
    else:
        id_use = args.id_use
        
    # Set up output directory
    post_dir = args.post_dir if args.post_dir else "."
    if post_dir and not os.path.exists(post_dir):
        os.makedirs(post_dir)
    
    # Print parameters if verbose
    if args.showInfo > 0:
        print(f"Running integrate_rejection with:")
        print(f"  Prior file: {args.prior_file_h5}")
        print(f"  Data file: {args.data_file_h5}")
        print(f"  Posterior file: {args.f_post_h5 or '(auto-generated)'}")
        print(f"  Samples: {args.N_use}")
        print(f"  Posterior samples: {args.nr}")
        print(f"  Auto temperature: {'Yes' if args.autoT else 'No'}")
        print(f"  Parallel processing: {'No' if args.no_parallel else 'Yes'}")
        if args.Ncpu > 0:
            print(f"  CPUs: {args.Ncpu}")
    
    try:
        # Run the function
        result = integrate.integrate_rejection(
            f_prior_h5=args.prior_file_h5,
            f_data_h5=args.data_file_h5,
            f_post_h5=args.f_post_h5,
            N_use=args.N_use,
            id_use=id_use,
            nr=args.nr,
            autoT=args.autoT,
            T_base=args.T_base,
            Nchunks=args.Nchunks,
            Ncpu=args.Ncpu,
            parallel=not args.no_parallel,
            use_N_best=args.use_N_best,
            showInfo=args.showInfo,
            updatePostStat=args.updatePostStat,
            post_dir=post_dir
        )
        
        if args.showInfo > 0:
            print(f"Posterior saved to: {result}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        if args.showInfo > 1:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
