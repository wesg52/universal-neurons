# Universal Neurons
All supporting data and code for Universal Neurons in GPT2 Language Models by Gurnee et al. (2024).

## Contents
* `dataframes/neuron_dfs` contains dataframes with neuron statistics for all neurons for the main models studies.
* `paper_notebooks` contains much of the plotting code to generate the figures in the paper.
* `correlations_fast.py` contains the script to compute neuron correlations.
* `summary.py` and `weights.py` contain scripts to compute neuron activation and weight statistic summaries for use of our summary viewer (contained in `summary_viewer.py`). See next section for more information on the data generated.
* `activations.py` contains scripts to cache neuron activations.
* `explain.py` contains script to compute our reduction in variance explanations.
* `attention_deactivation.py`, `entropy_intervention.py`, and `intervention.py` contain scripts for our functional neuron experiments.
* The `analysis` directory contains further plotting and analysis code.
* The `slurm` directory contains the shell scripts used to run the experiments on the cluster. These are not necessary to run the code, but may be useful to reference if replicating our experiments in a different environment.


## Summary Viewer
For this project, we leveraged precomputed activation data to explore neurons with our neuron viewer.

This data can either be recomputed using `summary.py` and `weights.py` or by downloading the data from our [box](TODO:add link) link. Add this to the top level of the directory. It is organized as follows:

```python
# Summary data for neuron weights in each model
summary_data/model_name/weights/data_file

# Summary data for activations of each model within different datasets
summary_data/model_name/activations/dataset_name/data_file

# This data can be loaded with via the following functions
from summary_viewer import load_all_summaries, load_weights_summary
dataset_summaries = load_all_summaries(model_name)
weight_summaries = load_weights_summary(model_name)
```

A common pattern in the summary data is "compressing" a distribution by binning it while saving the tails. In particular, we compute the following:
- `bin_counts`: a histogram of the distribution (where there is either a corresponding `bin_edges` or some standard bin size, look at the code in `weights.py` for details)
- `max_ix`: the indices of the top k elements of the distribution
- `max_vals`: the values of the top k elements of the distribution
- `min_ix`: the indices of the bottom k elements of the distribution
- `min_vals`: the values of the bottom k elements of the distribution
Though note the naming convention is not always consistent.

In particular, within the `weights` directory there is
- `neuron_comps.pt` which is a dictionary with keys, `in_in`, `in_out`, `out_in`, `out_out` with values also being a dictionary with 'top_neuron_value', 'top_neuron_ix', 'bottom_neuron_value', 'bottom_neuron_ix', 'comp_hist' corresponding to the summary format described above. The distribution here is the cosine similarity between the {input, output} and {input, output} weight vectors of every pair of neurons in the model. Hence each of these will be a `n_layers x d_mlp x {k or n_bins}` tensor.
- `vocab_comps.pt` which is a dict with keys 'E_in', 'U_in', 'E_out', 'U_out' corresponding to the cosine similarities betweein the {Embedding, Unembedding} and the neuron {input, output} weight vectors. Similar to the above, the values of these keys are also a dictionary with a summary data structures above, named 'top_vocab_value', 'top_vocab_ix', 'bottom_vocab_value', 'bottom_vocab_ix', 'comp_hist' again with shape  `n_layers x d_mlp x {k or n_bins}` tensor.
- `{k, q, v, o}_comps.pt`: each of these are tensors with shape `n_layers x d_mlp x n_layers x n_heads_per_layer`. They give the composition scores for each combination of neuron and attention head for the {key, query, value, output} vectors. For example, ||W_QK @ n_out|| / (||W_QK|| ||n_out||) for k_comp.

Activations are similar, with binned activations histograms for each distribution, as well as mean and max vocab activations for each neuron, and max activating dataset examples.


## Cite us
```
@article{gurnee2024universal,
  title={Universal neurons in gpt2 language models},
  author={Gurnee, Wes and Horsley, Theo and Guo, Zifan Carl and Kheirkhah, Tara Rezaei and Sun, Qinyi and Hathaway, Will and Nanda, Neel and Bertsimas, Dimitris},
  journal={arXiv preprint arXiv:2401.12181},
  year={2024}
}
```
