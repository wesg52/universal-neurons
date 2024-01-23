#!/bin/bash

PILE_DATASETS=('pile.test.all-100k.512' 'pile.test.all-1m.512' 'pile.test.all-5m.512' 'pile.test.all-10m.512' 'pile.test.all-25m.512' 'pile.test.all-50m.512' 'pile.test.all-100m.512')

for DATASET in "${PILE_DATASETS[@]}"
do
    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-b $DATASET pearson none

    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-c $DATASET pearson none

    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-d $DATASET pearson none

    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-e $DATASET pearson none
done