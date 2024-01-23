#!/bin/bash
#SBATCH -o log/%j-testgpu.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:2

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate

python test_gpu.py