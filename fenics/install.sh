#!/bin/bash
conda update conda
conda create --name fenics
printf "channels:\n - conda-forge\n - defaults\n" > ~/anaconda3/envs/fenics/.condarc
exit 0
source activate fenics
conda install numpy fenics scipy matplotlib
