#!/bin/bash

GPT2_SMALL_ENTROPY_NEURONS=('11.3030' '11.2859')
GPT2_MEDIUM_ENTROPY_NEURONS=('23.945' '22.2882')
PYTHIA_160M_ENTROPY_NEURONS=('11.1070' '10.1205' '10.2385')

# sbatch slurm/save_neuron_acts.sh stanford-gpt2-small-a pile.test.all-10m.512 24 "${GPT2_SMALL_ENTROPY_NEURONS[@]}"
# sbatch slurm/save_neuron_acts.sh stanford-gpt2-medium-a pile.test.all-10m.512 24 "${GPT2_MEDIUM_ENTROPY_NEURONS[@]}"
# sbatch slurm/save_neuron_acts.sh pythia-160m pile.test.all-10m.512 24 "${PYTHIA_160M_ENTROPY_NEURONS[@]}"

# for NEURON in "${GPT2_SMALL_ENTROPY_NEURONS[@]}"
# do
#     sbatch slurm/entropy_intervention.sh stanford-gpt2-small-a pile.test.all-10m.512 $NEURON
# done


# for NEURON in "${GPT2_MEDIUM_ENTROPY_NEURONS[@]}"
# do
#     sbatch slurm/entropy_intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON
# done

# for NEURON in "${PYTHIA_160M_ENTROPY_NEURONS[@]}"
# do
#     sbatch slurm/entropy_intervention.sh pythia-160m pile.test.all-10m.512 $NEURON
# done


# Random baselines
GPT2_SMALL_RANDOM_NEURONS=('11.2652' '11.1602' '10.2129' '10.906' '10.2944' '10.2783' '11.779' '10.2314' '11.1821' '10.1220' '10.2166' '11.1974' '11.2443' '11.2028' '11.127' '10.1971' '11.148' '11.682' '10.2121' '10.684')
GPT2_MEDIUM_RANDOM_NEURONS=('22.3440' '22.2781' '23.3788' '23.3475' '22.1464' '22.2228' '22.987' '23.3974' '23.529' '22.669' '23.1075' '23.3938' '23.1404' '23.53' '22.125' '22.253' '22.2078' '23.188' '22.2548' '22.401')
PYTHIA_160M_RANDOM_NEURONS=('11.1406' '10.316' '10.2884' '11.2874' '10.1714' '10.351' '11.592' '11.541' '10.2471' '11.1052' '11.1667' '10.2684' '10.97' '10.2004' '10.1371' '11.2835' '11.2470' '11.1351' '10.1137' '10.742')

for NEURON in "${GPT2_SMALL_RANDOM_NEURONS[@]}"
do
    sbatch slurm/entropy_intervention.sh stanford-gpt2-small-a pile.test.all-10m.512 $NEURON
done


# for NEURON in "${GPT2_MEDIUM_RANDOM_NEURONS[@]}"
# do
#     sbatch slurm/entropy_intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON
# done

# for NEURON in "${PYTHIA_160M_RANDOM_NEURONS[@]}"
# do
#     sbatch slurm/entropy_intervention.sh pythia-160m pile.test.all-10m.512 $NEURON
# done
