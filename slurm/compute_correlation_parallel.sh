#!/bin/bash
#SBATCH -o log/parallel_log/%j-correlation.log
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/correlation_results_parallel
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate


python -u correlations_parallel.py \
    --model_1_name $1 \
    --model_2_name $2 \
    --token_dataset $3 \
    --similarity_type $4 \
    --baseline $5 \
    --batch_size 32 \
    --model_1_device cuda:0 \
    --model_2_device cuda:1 \
    --correlation_device cpu


