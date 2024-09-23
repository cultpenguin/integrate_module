# integrate_module

This repository holds the INTEGRATE Python module

Installation 
------------
Assuming you allready have python 3.10 installed

    pip install --upgrade -r requirements.txt
    pip install -e .

-----------


PIP (Test in a fresh install of Ubuntu 22.04, Ubuntu 22.04 in WSL) 
==================================================================

    # Install python3 venv
    sudo apt install python3-venv
    
    # Create virtual environment
    python3 -m venv ~/integrate
    source ~/integrate/bin/activate
    pip install --upgrade pip
    
    # Install integrate module
    cd path/to/integrate module
    pip install --upgrade -r requirements.txt
    pip install -e .
    
    # install GA-AEM
    sh scripts/cmake_build_script_ubuntu_gatdaem1d.sh
    cd ga-aem/install-ubuntu/python
    pip install .
    

Conda + PIP
===========
Create a Conda environment (called integrate) and install the required modules, using 

    conda create --name integrate python=3.10  
    conda activate integrate
    pip install --upgrade -r requirements.txt
    pip install -e .


GA-AEM
======
In order to use GA-AEM for forward EM modeling, the 'gatdaem1d' Python module must be installed.

Follow instructions at https://github.com/GeoscienceAustralia/ga-aem.


-------------------------------------
Pre-Compiled Python module in Windows
-------------------------------------

Download pre-compiled version of GA-AEM for windows through the latest  release from https://github.com/GeoscienceAustralia/ga-aem/releases as GA-AEM.zip

Download precompiled FFTW3 windows dlls from https://www.fftw.org/install/windows.html as fftw-3.3.5-dll64.zip 

unzip GA-AEM.zip to get GA-AEM

unzip fftw-3.3.5-dll64.zip to get fftw-3.3.5-dll64

cp fftw-3.3.5-dll64/*.dll to GA-AEM/python/gatdaem1d/

    cp fftw-3.3.5-dll64/*.dll GA-AEM/python/gatdaem1d/

Install the python gatdaem1d module

    cd GA-AEM/python/
    pip install -e .

    # test the installaion
    cd examples
    python skytem_example.py



-------------------------------------
Compile Python module in Ubuntu/Linux
-------------------------------------

A script that downloads and install GA-AEM is located in 'scripts/cmake_build_script_ubuntu_gatdaem1d.sh'. Be sure to be usiong the appropriate Python environment and then run

    sh scripts/cmake_build_script_ubuntu_gatdaem1d.sh
    cd ga-aem/install-ubuntu/python
    pip install .
    
-------------------------------------
Compile Python module in OSX/Homebrew
-------------------------------------
First install homebrew, then run 

    sh ./scripts/cmake_build_script_homebrew_gatdaem1d.sh
    cd ga-aem/install-homebrew/python
    pip install .



