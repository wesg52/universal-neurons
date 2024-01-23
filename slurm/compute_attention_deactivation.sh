#!/bin/bash
#SBATCH -o log/attention_deactivation/%j-attention_deactivation.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/attention_deactivation_results
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/nlp/bin/activate


python -u attention_deactivation.py \
    --model_name $1 \
    --token_dataset $2 \
    --min_neuron $3 \
    --max_neuron $4 \
    --batch_size 32 \
    --device cuda \
    --context_length 256 \
    --after_pos 64 \