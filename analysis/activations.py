import torch
import einops
import numpy as np
import pandas as pd


def make_dataset_df(ds, decoded_vocab):
    tokens = ds['tokens']
    subset = ds['subset']
    n, d = tokens.shape

    sequence_subset = einops.repeat(np.array(subset), 'n -> n d', d=d)
    sequence_ix = einops.repeat(np.arange(n), 'n -> n d', d=d)
    position = einops.repeat(np.arange(d), 'd -> n d', n=n)

    prev_tokens = torch.concat(
        [torch.zeros(n, 1, dtype=int) - 1, tokens[:, :-1]], dim=1)

    dataset_df = pd.DataFrame({
        'token': tokens.flatten().numpy(),
        'prev_token': prev_tokens.flatten().numpy(),
        'token_str': [decoded_vocab[t] for t in tokens.flatten().numpy()],
        'subset': sequence_subset.flatten(),
        'sequence_ix': sequence_ix.flatten(),
        'position': position.flatten(),
    })
    return dataset_df


def compute_moments_from_binned_data(bin_edges, bin_counts):

    bin_edges = torch.tensor(np.concatenate([
        np.array([bin_edges[0] + bin_edges[0] - bin_edges[1]]),
        bin_edges,
        np.array([bin_edges[-1] + bin_edges[-1] - bin_edges[-2]])
    ]))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    total_frequency = bin_counts.sum(axis=-1)
    mean = (bin_centers * bin_counts).sum(axis=-1) / total_frequency
    variance = (((-mean[:, :, None] + bin_centers) ** 2)
                * bin_counts).sum(axis=-1) / total_frequency
    skewness = (((-mean[:, :, None] + bin_centers) ** 3) *
                bin_counts).sum(axis=-1) / (total_frequency * (variance ** 1.5))
    kurtosis = (((-mean[:, :, None] + bin_centers) ** 4) *
                bin_counts).sum(axis=-1) / (total_frequency * (variance ** 2))
    return mean, variance, skewness, kurtosis


def make_pile_subset_distribution_activation_summary_df(dataset_summaries, bin_edges, include_all=False):
    if include_all:
        full_distribution_bin_count = sum(
            dataset_summaries[k]['neuron_bin_counts']
            for k in dataset_summaries.keys()
        )
        dataset_summaries['all'] = {}
        dataset_summaries['all']['neuron_bin_counts'] = full_distribution_bin_count

    neuron_moment_dict = {}
    for distr in dataset_summaries:
        bin_counts = dataset_summaries[distr]['neuron_bin_counts']
        mean, variance, skewness, kurtosis = compute_moments_from_binned_data(
            bin_edges, bin_counts)
        neuron_moment_dict[('mean', distr)] = mean.flatten()
        neuron_moment_dict[('var', distr)] = variance.flatten()
        neuron_moment_dict[('skew', distr)] = skewness.flatten()
        neuron_moment_dict[('kurt', distr)] = kurtosis.flatten()

    n_layers, d_mlp = mean.shape

    neuron_moment_df = pd.DataFrame(
        neuron_moment_dict,
        index=pd.MultiIndex.from_product([range(n_layers), range(d_mlp)])
    )
    neuron_moment_df.index.names = ['layer', 'neuron']

    if include_all:  # clean up
        del dataset_summaries['all']

    return neuron_moment_df[['mean', 'var', 'skew', 'kurt']]


def get_activation_sparsity_df(dataset_summaries, bin_edges):
    zero_bin = np.argmax(bin_edges >= 0).item()
    subset_bin_counts = {}
    subset_sparsity = {}
    for dataset_name, dataset_summary in dataset_summaries.items():
        subset_bin_counts[dataset_name] = dataset_summary['neuron_bin_counts']
        sparsity = dataset_summary['neuron_bin_counts'][:, :, zero_bin:].sum(
            axis=-1) / dataset_summary['neuron_bin_counts'].sum(axis=-1)
        subset_sparsity[dataset_name] = sparsity.numpy()

    total_bin_counts = sum([v for v in subset_bin_counts.values()])
    total_sparsity = total_bin_counts[:, :, zero_bin:].sum(
        axis=-1) / total_bin_counts.sum(axis=-1)
    subset_sparsity['all'] = total_sparsity

    n_layer, n_neuron = subset_sparsity[list(subset_sparsity.keys())[0]].shape
    sparsity_df = pd.DataFrame({
        k: sparsity_tensor.flatten() for k, sparsity_tensor in subset_sparsity.items()
    }, index=pd.MultiIndex.from_product([range(n_layer), range(n_neuron)]))
    sparsity_df.index.names = ['layer', 'neuron']
    return sparsity_df


def make_full_distribution_activation_summary_df(dataset_summaries, bin_edges):
    pass
