#!/bin/bash

DATASETS=('pile.test.enron_emails.512' 'pile.test.nih_exporter.512' 'pile.test.philpapers.512' 'pile.test.bookcorpus2.512' 'pile.test.hackernews.512' 'pile.test.dm_mathematics.512' 'pile.test.pubmed_abstracts.512' 'pile.test.youtubesubtitles.512' 'pile.test.books3.512' 'pile.test.arxiv.512' 'pile.test.ubuntu_irc.512' 'pile.test.openwebtext2.512' 'pile.test.pile_cc.512' 'pile.test.stackexchange.512' 'pile.test.freelaw.512' 'pile.test.europarl.512' 'pile.test.opensubtitles.512' 'pile.test.pubmed_central.512' 'pile.test.github.512' 'pile.test.wikipedia.512' 'pile.test.gutenberg.512' 'pile.test.uspto_backgrounds.512')

for dataset in "${DATASETS[@]}"
do
    sbatch slurm/run_summary.sh pythia-160m-seed1 $dataset 48
    sbatch slurm/run_summary.sh pythia-160m-seed2 $dataset 48
    sbatch slurm/run_summary.sh pythia-160m-seed3 $dataset 48
done

# for dataset in "${DATASETS[@]}"
# do
#     sbatch slurm/run_summary.sh pythia-410m $dataset 32
# done

# for dataset in "${DATASETS[@]}"
# do
#     sbatch slurm/run_summary.sh pythia-1b $dataset 24
# done

# sbatch slurm/run_summary.sh pythia-70m
# sbatch slurm/run_summary.sh pythia-70m-deduped
# sbatch slurm/run_summary.sh pythia-70m-v0
# sbatch slurm/run_summary.sh pythia-70m-deduped-v0

# sbatch slurm/run_summary.sh pythia-160m
# sbatch slurm/run_summary.sh pythia-160m-deduped
# sbatch slurm/run_summary.sh pythia-160m-v0
# sbatch slurm/run_summary.sh pythia-160m-deduped-v0

# sbatch slurm/run_summary.sh pythia-410m
# sbatch slurm/run_summary.sh pythia-1b
# sbatch slurm/run_summary.sh pythia-1.4b
