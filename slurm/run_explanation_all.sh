#!/bin/bash


sbatch ./slurm/run_explanation.sh token stanford-gpt2-small-a dataframes/stanford-gpt2-small-a/universal.csv

sbatch ./slurm/run_explanation.sh sequence stanford-gpt2-small-a dataframes/stanford-gpt2-small-a/universal.csv

sbatch ./slurm/run_explanation.sh token stanford-gpt2-medium-a dataframes/stanford-gpt2-medium-a/universal.csv

sbatch ./slurm/run_explanation.sh sequence stanford-gpt2-medium-a dataframes/stanford-gpt2-medium-a/universal.csv

sbatch ./slurm/run_explanation.sh token pythia-160m dataframes/stanford-gpt2-small-a/universal.csv
