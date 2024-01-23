#!/bin/bash
#SBATCH -o log/%A_%a-entropy_intervention.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-7

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/intervention_results
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate

INTERVENTION_PARAMS=('-0.17' '0.0' '1.0' '2.0' '3.0' '4.0' '5.0' '6.0')
intervention_param=${INTERVENTION_PARAMS[$SLURM_ARRAY_TASK_ID]}


python -u entropy_intervention.py \
    --model $1 \
    --token_dataset $2 \
    --output_dir $RESULTS_DIR \
    --batch_size 24 \
    --neuron_subset $3 \
    --intervention_param $intervention_param