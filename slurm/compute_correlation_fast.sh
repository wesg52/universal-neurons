#!/bin/bash
#SBATCH -o log/%j-correlation.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/correlation_results
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate


python -u correlations_fast.py \
    --model_1_name $1 \
    --model_2_name $2 \
    --token_dataset $3 \
    --similarity_type $4 \
    --baseline $5 \
    --batch_size 24 \
    --model_1_device cuda \
    --model_2_device cuda \
    --correlation_device cpu \
    --save_full_correlation_matrix