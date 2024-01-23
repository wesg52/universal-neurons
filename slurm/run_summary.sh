#!/bin/bash
#SBATCH -o log/%j-summary.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/summary_data
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate


python -u summary.py \
    --model $1 \
    --token_dataset $2 \
    --batch_size $3 \
    --output_dir $RESULTS_DIR
