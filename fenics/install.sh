#!/bin/bash
conda create --name fenics
source activate fenics
echo "channels:\n - conda-forge\n - defaults\n" > ~/anaconda3/envs/fenics/.condarc
conda install numpy fenics scipy matplotlib
