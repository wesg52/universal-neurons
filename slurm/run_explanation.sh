#!/bin/bash
#SBATCH -o log/%A_%a-run_explanation.log
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --array=0-23

# set the above array to be the number of layers in the model

# set environment variables
export PATH=$NEURON_STATS_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/neuron_stats/results
export DATASET_DIR=/home/gridsan/groups/maia_mechint/token_datasets
export TRANSFORMERS_CACHE=/home/gridsan/groups/maia_mechint/models

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $NEURON_STATS_ROOT/stats/bin/activate

python -u explain.py --layer $SLURM_ARRAY_TASK_ID --feature_type $1 --model $2 --neuron_df_path $3