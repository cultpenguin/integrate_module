# INTEGRATE Python Module

[![PyPI version](https://badge.fury.io/py/integrate_module.svg)](https://badge.fury.io/py/integrate_module)
[![Test PyPI](https://img.shields.io/badge/Test%20PyPI-integrate__module-orange.svg)](https://test.pypi.org/project/integrate-module/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://cultpenguin.github.io/integrate_module/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository holds the INTEGRATE Python module


## Installation 

Assuming you allready have python 3.10 installed

    pip install -i https://test.pypi.org/simple/ integrate_module

On Windows this will also install the python wrapper for the ga-aem (1D EM forward modeling - GPL v2 code) --> [ga-aem-forward-win](https://pypi.org/project/ga-aem-forward-win/)


### PIP (from pypi)

    # Install python3 venv
    sudo apt install python3-venv
    
    # Create virtual environment
    python3 -m venv ~/integrate
    source ~/integrate/bin/activate
    pip install --upgrade pip
    
    # Install integrate module
    pip install -i https://test.pypi.org/simple/ integrate_module
    
### PIP (from source)

    # Install python3 venv
    sudo apt install python3-venv
    
    # Create virtual environment
    python3 -m venv ~/integrate
    source ~/integrate/bin/activate
    pip install --upgrade pip
    
    # Install integrate module
    cd path/to/integrate module
    pip install -e .

### Conda + PIP (from pypi)

Create a Conda environment (called integrate) and install the required modules, using 

    conda create --name integrate python=3.10 numpy pandas matplotlib scipy tqdm requests h5py psutil
    conda activate integrate
    pip install -i https://test.pypi.org/simple/ integrate_module
    
    
    
### Conda + PIP (from source)

Create a Conda environment (called integrate) and install the required modules, using 

    conda create --name integrate python=3.10 numpy pandas matplotlib scipy tqdm requests h5py psutil
    conda activate integrate
    pip install -e .


## GA-AEM

In order to use GA-AEM for forward EM modeling, the 'gatdaem1d' Python module must be installed. Follow instructions at [https://github.com/GeoscienceAustralia/ga-aem](https://github.com/GeoscienceAustralia/ga-aem) or use the information below.


### pypi paackage for windows

In Windows the [ga-aem-forward-win](https://pypi.org/project/ga-aem-forward-win/) package will be autoamtically installed that provides access to the GA-AEM forward code. It can be installed manually using

    pip install ga-aem-forward-win

### Pre-Compiled Python module in Windows

Download pre-compiled version of GA-AEM for windows through the latest  release from https://github.com/GeoscienceAustralia/ga-aem/releases as GA-AEM.zip

Download precompiled FFTW3 windows dlls from https://www.fftw.org/install/windows.html as fftw-3.3.5-dll64.zip 

unzip GA-AEM.zip to get GA-AEM

unzip fftw-3.3.5-dll64.zip to get fftw-3.3.5-dll64

Copy fftw-3.3.5-dll64/*.dll to GA-AEM/python/gatdaem1d/

    cp fftw-3.3.5-dll64/*.dll GA-AEM/python/gatdaem1d/

Install the python gatdaem1d module

    cd GA-AEM/python/
    pip install -e .

    # test the installaion
    cd examples
    python skytem_example.py



### Compile ga-aem Python module in Ubuntu/Linux


A script that downloads and install GA-AEM is located in 'scripts/cmake_build_script_ubuntu_gatdaem1d.sh'. Be sure to use the appropriate Python environment and then run

    sh scripts/cmake_build_script_ubuntu_gatdaem1d.sh
    cd ga-aem/install-ubuntu/python
    pip install .
    
### Compile ga-aem Python module in OSX/Homebrew

First install homebrew, then run 

    sh ./scripts/cmake_build_script_homebrew_gatdaem1d.sh
    cd ga-aem/install-homebrew/python
    pip install .


## Development

The main branch should be the most stable, and updates, less frequent, but with larger changes.

The develop branch contains the current development code and may be updated frequently, and some functions and examples may be broken.



