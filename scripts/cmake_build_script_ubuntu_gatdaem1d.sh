#!/bin/sh
set -e  # exit with error if a command fails
set -x  # show commands while executing

# This script is for compiling the forward code, gatdaem1d, from ga-aem, on Ubuntu (version >= 2.0.0)
# See more detailed information on https://github.com/GeoscienceAustralia/ga-aem/blob/master/cmake_build_script_ubuntu.sh 

# Install necessary packages in Ubuntu (tested on Ubuntu 22.04 LTS and Debian 13)
sudo sh -c '
	apt-get update &&
	apt-get install -y build-essential libfftw3-dev libfftw3-bin libfftw3-double3 libopenmpi-dev cmake pkg-config &&
	apt-get autoremove -y'

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
pip install -e .

## 4. Test the installation
python "$INSTALL_DIR"/python/examples/skytem_example.py
