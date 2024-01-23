#!/bin/bash


sbatch slurm/compute_correlation.sh pythia-70m pythia-70m-v0 pile.test.all-1m.512 pearson rotation

# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-b pile.test.all-100m.512 pearson none
# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-b pile.test.all-100m.512 pearson rotation

# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-c pile.test.all-100m.512 pearson none
# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-c pile.test.all-100m.512 pearson rotation

# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-d pile.test.all.512 pearson none
# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-d pile.test.all.512 pearson rotation

# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-e pile.test.all.512 pearson none
# sbatch slurm/compute_correlation.sh stanford-gpt2-medium-a stanford-gpt2-medium-e pile.test.all.512 pearson rotation


# sbatch slurm/compute_correlation.sh pythia-70m pythia-70m-v0 pile.test.all.512 pearson none
# sbatch slurm/compute_correlation.sh pythia-70m pythia-70m-v0 pile.test.all.512 pearson gaussian
# sbatch slurm/compute_correlation.sh pythia-70m pythia-70m-v0 pile.test.all.512 pearson permutation
# sbatch slurm/compute_correlation.sh pythia-70m pythia-70m-v0 pile.test.all.512 pearson rotation