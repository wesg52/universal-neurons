#!/bin/bash

sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-small-a pile.test.all-100k.512 0 3072
sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-medium-a pile.test.all-100k.512 0 4096

sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-small-a pile.test.all-1m.512 0 1536
sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-small-a pile.test.all-1m.512 1536 3072

sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-medium-a pile.test.all-1m.512 0 1024
sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-medium-a pile.test.all-1m.512 1024 2048
sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-medium-a pile.test.all-1m.512 2048 3072
sbatch slurm/compute_attention_deactivation.sh stanford-gpt2-medium-a pile.test.all-1m.512 3072 4096