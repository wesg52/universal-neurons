#!/bin/bash
#SBATCH -o log/%j-weight-summary.log
#SBATCH -c 48
#SBATCH -N 1

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

#PYTHIA_MODELS=('pythia-70m' 'pythia-70m-deduped' 'pythia-160m' 'pythia-160m-deduped' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

GPT2_SMALL_MODELS=('stanford-gpt2-small-a' 'stanford-gpt2-small-b' 'stanford-gpt2-small-c' 'stanford-gpt2-small-d' 'stanford-gpt2-small-e')

GPT2_MEDIUM_MODELS=('stanford-gpt2-medium-a' 'stanford-gpt2-medium-b' 'stanford-gpt2-medium-c' 'stanford-gpt2-medium-d' 'stanford-gpt2-medium-e')

PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-160m-seed1', 'pythia-160m-seed2', 'pythia-160m-seed3')

PYTHIA_LARGE_MODELS=('pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

for model in "${PYTHIA_LARGE_MODELS[@]}"
do
    python -u weights.py \
    --model $model \
    --save_path $RESULTS_DIR
done