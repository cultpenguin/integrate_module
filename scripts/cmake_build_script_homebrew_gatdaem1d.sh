#!/bin/sh

# This script is for compiling the forward code, gatdaem1d, from ga-aem, on Ubuntu (version >= 2.0.0)
# See more detailed information on https://github.com/GeoscienceAustralia/ga-aem/blob/master/cmake_build_script_ubuntu.sh 

# # Install necessary packages in Homebrew (tested on macOS XXX)
brew update
brew install gcc
brew install fftw
brew install open-mpi
brew install cmake

## 1. Clone the ga-aem repository from Github
git clone --recursive --depth 1 https://github.com/GeoscienceAustralia/ga-aem.git
cd ga-aem

## 2. Compile GA-AEM

# INSTALL_DIR is the directory for installing the build package
export INSTALL_DIR=$PWD/install-ubuntu

# BUILD_DIR is a temporary directory for building (compiling and linking)
export BUILD_DIR=$PWD/build-ubuntu

mkdir $BUILD_DIR
cd $BUILD_DIR

# compile gatdaem1d
cmake -Wno-dev -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=OFF -DWITH_NETCDF=OFF -DWITH_GDAL=OFF -DWITH_PETSC=OFF ..
#cmake --build . --target matlab-bindings --config=Release
cmake --build . --target python-bindings --config=Release
cmake --install . --prefix $INSTALL_DIR

## 3. Install python module 
echo  $INSTALL_DIR
cp $INSTALL_DIR/python/gatdaem1d/gatdaem1d.dylib $INSTALL_DIR/python/gatdaem1d/gatdaem1d.so 

## Install the Python module
cd $INSTALL_DIR/python
pip install -e .
