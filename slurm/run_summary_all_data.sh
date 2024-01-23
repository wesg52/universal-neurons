#!/bin/bash
#SBATCH -o log/%j-summary.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

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

DATASETS=('pile.test.enron_emails.512' 'pile.test.nih_exporter.512' 'pile.test.philpapers.512' 'pile.test.bookcorpus2.512' 'pile.test.hackernews.512' 'pile.test.dm_mathematics.512' 'pile.test.pubmed_abstracts.512' 'pile.test.youtubesubtitles.512' 'pile.test.books3.512' 'pile.test.arxiv.512' 'pile.test.ubuntu_irc.512' 'pile.test.openwebtext2.512' 'pile.test.pile_cc.512' 'pile.test.stackexchange.512' 'pile.test.freelaw.512' 'pile.test.europarl.512' 'pile.test.opensubtitles.512' 'pile.test.pubmed_central.512' 'pile.test.github.512' 'pile.test.wikipedia.512' 'pile.test.gutenberg.512' 'pile.test.uspto_backgrounds.512')

for dataset in "${DATASETS[@]}"
do
    python -u summary.py \
        --model $1 \
        --token_dataset $dataset \
        --batch_size 32 \
        --output_dir $RESULTS_DIR
done