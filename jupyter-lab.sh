#!/bin/bash

###
# Taken from CS236781 - Deep Learning: https://vistalab-technion.github.io/cs236781/assignments/hpc-servers
# jupyter-lab.sh
#
# This script is intended to help you run jupyter lab on the course servers.
#
# Make sure to change CONDA_ENV to your desired environment name!
#
#
# Example usage:
#
# To run on the gateway machine (limited resources, no GPU):
# ./jupyter-lab.sh
#
# To run on a compute node:
# srun -c 2 --gres=gpu:1 --pty jupyter-lab.sh
#

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=py37

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

jupyter notebook --no-browser --ip=$(hostname -I) --port-retries=100

