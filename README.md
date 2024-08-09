# integrate_module

This repository holds the INTEGRATE Python module

Installation 
------------
Assuming you allready have python 3.10 installed

    pip install --upgrade -r requirements.txt
    pip install -e .

-----------


PIP (Test in a fresh install of Ubuntu 22.04, Ubuntu 22.04 in WSL) 
================

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
A script that downloads and install GA-AEM is located in 'scripts/cmake_build_script_ubuntu_gatdaem1d.sh'. Be sure to be usiong the appropriate Python environment and then run

    sh scripts/cmake_build_script_ubuntu_gatdaem1d.sh
    cd ga-aem/install-ubuntu/python
    pip install .
    
