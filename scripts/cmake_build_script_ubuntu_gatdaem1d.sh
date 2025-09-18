#!/bin/sh
#
# This script is for compiling the forward code, gatdaem1d, from ga-aem (version >= 2.0.0), on Debian and Ubuntu 
# See more detailed information on https://github.com/GeoscienceAustralia/ga-aem/blob/master/cmake_build_script_ubuntu.sh 
#
# Test this script with the latest Debian using a docker container:
#   docker run -it --rm -v $(pwd):/workspace debian:latest /workspace/cmake_build_script_ubuntu_gatdaem1d.sh
# Test this script with the latest Ubuntu using a docker container:
#	docker run -it --rm -v $(pwd):/workspace ubuntu:latest /workspace/cmake_build_script_ubuntu_gatdaem1d.sh
#
#
set -e  # exit with error if a command fails
set -x  # show commands while executing


# Install necessary packages in Ubuntu (tested on Ubuntu 22.04 LTS and Debian 13)
# Check if sudo is available, if not assume user is root
if command -v sudo >/dev/null 2>&1; then
	sudo sh -c '
		apt-get update &&
		apt-get install -y build-essential libfftw3-dev libfftw3-bin libfftw3-double3 libopenmpi-dev cmake pkg-config git &&
		apt-get autoremove -y'
else
	sh -c '
		apt-get update &&
		apt-get install -y build-essential libfftw3-dev libfftw3-bin libfftw3-double3 libopenmpi-dev cmake pkg-config git &&
		apt-get autoremove -y'
fi

## 1. Clone the ga-aem repository from Github
if ! test -d ga-aem
then
	git clone --recursive https://github.com/GeoscienceAustralia/ga-aem.git
fi
cd ga-aem

## 2. Compile GA-AEM

# INSTALL_DIR is the directory for installing the build package
INSTALL_DIR="$PWD"/install-ubuntu

# BUILD_DIR is a temporary directory for building (compiling and linking)
BUILD_DIR="$PWD"/build-ubuntu

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# compile gatdaem1d
cmake -Wno-dev -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=OFF -DWITH_NETCDF=OFF -DWITH_GDAL=OFF -DWITH_PETSC=OFF ..
cmake --build . --target matlab-bindings --config=Release
cmake --build . --target python-bindings --config=Release
cmake --install . --prefix "$INSTALL_DIR"


## CHECK FOR shared library dependencies
cd ..
readelf -d install-ubuntu/python/gatdaem1d/gatdaem1d.so  | grep 'Shared'

## 3. Install python module
echo  "installing from --> $INSTALL_DIR"
cd "$INSTALL_DIR"/python

# Check if pip is available
if command -v pip >/dev/null 2>&1; then
	if pip install -e .; then
		echo "Python module installed successfully"
		PYTHON_INSTALL_SUCCESS=true
	else
		echo "Python module installation failed"
		PYTHON_INSTALL_SUCCESS=false
	fi
else
	echo "ERROR: pip is not installed. Please install Python and pip to complete the installation."
	echo "On Ubuntu/Debian, you can install with:"
	echo "  apt-get install python3 python3-pip"
	echo "Or use your system's package manager to install Python and pip."
	PYTHON_INSTALL_SUCCESS=false
fi

## 4. Test the installation
if [ "$PYTHON_INSTALL_SUCCESS" = true ]; then
	echo "Running installation test..."
	python "$INSTALL_DIR"/python/examples/skytem_example.py
else
	echo "Skipping installation test due to failed python module installation"
fi
