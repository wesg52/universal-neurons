#!/bin/bash

# gpt2-medium
# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-b pile.test.all-100m.512 pearson none

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-c pile.test.all-100m.512 pearson none

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-d pile.test.all-100m.512 pearson none

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-e pile.test.all-100m.512 pearson none

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-b pile.test.all-100m.512 pearson rotation

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-c pile.test.all-100m.512 pearson rotation

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-d pile.test.all-100m.512 pearson rotation

# sbatch slurm/compute_correlation_fast.sh stanford-gpt2-medium-a stanford-gpt2-medium-e pile.test.all-100m.512 pearson rotation

# # gpt2-small
sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-b pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-c pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-d pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-e pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-b pile.test.all-100m.512 pearson rotation

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-c pile.test.all-100m.512 pearson rotation

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-d pile.test.all-100m.512 pearson rotation

sbatch slurm/compute_correlation_fast.sh stanford-gpt2-small-a stanford-gpt2-small-e pile.test.all-100m.512 pearson rotation

#pythia
sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed1 pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed2 pile.test.all-100m.512 pearson none

sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed3 pile.test.all-100m.512 pearson none


sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed1 pile.test.all-100m.512 pearson rotation

sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed2 pile.test.all-100m.512 pearson rotation

sbatch slurm/compute_correlation_fast.sh pythia-160m pythia-160m-seed3 pile.test.all-100m.512 pearson rotation