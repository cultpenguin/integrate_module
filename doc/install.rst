============
Installation
============

Python integrate module
=======================

Once a stable version is released, the module will be available on PyPI (https://test.pypi.org/project/integrate/), and it should be installed using 
::
    
        # pip install integrate_module
        pip install -i https://test.pypi.org/simple/ integrate_module

Until then, the module can be installed from the source code. 
The following steps will install the module, optionally in a virtual environment called 'integrate'

:: 

        # Clone the repository
        git clone git@github.com:cultpenguin/integrate_module.git

        # [otionally] create a virtual environment
        sudo apt-get install python3-venv
        python3 -m venv ~/integrate
        source ~/integrate/bin/activate

        
        # Install the module
        cd /path/to/integrate_module
        pip install --upgrade -r requirements.txt
        pip install .


FORWARD MODELING
================

GA-AEM [GA-AEM]
---------------
GA-AEM can be downloaded from [https://github.com/GeoscienceAustralia/ga-aem].

**On Windows:** 

The forward EM codes from GA-AEM should be installed automatically on Windows when installing the `integrate_module` package, but if you want to install it separately, you can run the following command:

GA-AEM can be installed using on Windows using 
::

    pip install ga-aem-forward-win



if you install it manually, for the GA-AEM source code, you need to:

A: Add `path_to_ga-aem/third_party/fftw3.2.2.dlls/64bit/` to the Windows path `$PATH` under
environment variables. 

B: add `path_to_ga-aem/matlab/gatdaem1d_functions/` and `path_to_ga-aem/matlab/bin/x64/` to the Matlab path
::

    addpath path_to_ga-aem/matlab/gatdaem1d_functions/
    addpath path_to_ga-aem//matlab/bin/x64/


**On Linux/OSX:**

MEX files need to be compiled after downloading the source. The following script will download and compile the GA-AEM source code on Ubuntu 24.04 --> (https://github.com/cultpenguin/integrate_module/blob/main/scripts/cmake_build_script_ubuntu_gatdaem1d.sh).
 

::

    chmod +x cmake_build_script_ubuntu_gatdaem1d.sh
    ./cmake_build_script_ubuntu_gatdaem1d.sh

Then, to install the GA-AEM Python module, navigate to the `ga-aem` directory and run

:: 

    cd path_to_ga-aem/install_ubuntu/python
    pip install .

Then add 
`path_to_ga-aem/install-ubuntu/matlab/gatdaem1d_functions/` and 
`path_to_ga-aem/install-ubuntu/matlab/bin/` and 
to the Matlab path


SimPEG [Python only]
--------------------
[SimPEG]_ can be installed using pip:

::

    pip install simpeg


AarhusInv [Windows only]
------------------------
To use AarhusInv [AarhusInv]_ it must be installed and the path to the executable must be added to the path variable.
In addition a valid license must be associated with the installation.


.. MATLAB
.. ======


.. A Matlab version of the INTEGRATE module is available. It is not guarantied to be up to date with the Python version.

.. The following packages are required using INTEGRATE with MATLAB:

.. - `sippi <https://github.com/cultpenguin/sippi>`_
.. - `mgstat <https://github.com/cultpenguin/mgstat>`_
.. - `sippi-abc <https://github.com/cultpenguin/sippi-abc>`_

.. In addition you will need to install one of the EM forward codes described below. 

..
    Julia
    =====


