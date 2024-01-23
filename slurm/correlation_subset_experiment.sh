#!/bin/bash

DATASETS=('pile.test.enron_emails.512' 'pile.test.nih_exporter.512' 'pile.test.philpapers.512' 'pile.test.bookcorpus2.512' 'pile.test.hackernews.512' 'pile.test.dm_mathematics.512' 'pile.test.pubmed_abstracts.512' 'pile.test.youtubesubtitles.512' 'pile.test.books3.512' 'pile.test.arxiv.512' 'pile.test.ubuntu_irc.512' 'pile.test.openwebtext2.512' 'pile.test.pile_cc.512' 'pile.test.stackexchange.512' 'pile.test.freelaw.512' 'pile.test.europarl.512' 'pile.test.opensubtitles.512' 'pile.test.pubmed_central.512' 'pile.test.github.512' 'pile.test.wikipedia.512' 'pile.test.gutenberg.512' 'pile.test.uspto_backgrounds.512')

for dataset in "${DATASETS[@]}"
do
#    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-b $dataset pearson none
#    sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-b $dataset pearson rotation

 #   sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-c $dataset pearson none
 #   sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-c $dataset pearson rotation

     sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-d $dataset pearson none
     sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-d $dataset pearson rotation

     sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-e $dataset pearson none
     sbatch slurm/compute_correlation.sh stanford-gpt2-small-a stanford-gpt2-small-e $dataset pearson rotation
done
