#!/bin/bash

ALL_CAP_NEURONS=('20.13' '23.3440' '21.2148' '15.591' '19.1121' '18.2336' '15.84' '17.2559')
END_W_ING_NEURONS=('22.1585' '22.3534' '19.2871' '20.1867' '19.1647' '22.904' '19.3984' '15.1699' '23.3844' '16.3122' '18.1984' '21.118' '16.3346' '14.2046' '14.4048')
SECOND_PERSON_NEURONS=('23.2205' '20.2583' '17.774' '18.638' '14.2719' '18.1932' '18.1532' '18.1631' '18.3930')
NEUTRAL_PRONOUN_NEURONS=('22.73' '22.1732' '21.1274' '17.797' '20.603' '16.4092' '16.2529' '19.2820' '18.3690' '15.1552')


# for NEURON in "${SECOND_PERSON_NEURONS[@]}"
# do
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON zero_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON threshold_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON relu_ablation 0

# done


# for NEURON in "${ALL_CAP_NEURONS[@]}"
# do
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON zero_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON threshold_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON relu_ablation 0
# done


# for NEURON in "${NEUTRAL_PRONOUN_NEURONS[@]}"
# do
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON zero_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON threshold_ablation 0
#     sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON relu_ablation 0
# done


for NEURON in "${END_W_ING_NEURONS[@]}"
do
    sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON zero_ablation 0
    sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON threshold_ablation 0
    sbatch slurm/intervention.sh stanford-gpt2-medium-a pile.test.all-10m.512 $NEURON relu_ablation 0
done
