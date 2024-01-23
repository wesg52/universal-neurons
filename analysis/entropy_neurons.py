import os
import datasets
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .activations import make_dataset_df


def make_entropy_intervention_rdf(model_name, dataset_name, neuron_subset, interventions):
    rdfs = []
    for ix, intervention in enumerate(interventions):
        save_path = os.path.join(
            'intervention_results', model_name, dataset_name, neuron_subset, intervention)
        rank_tensor = torch.load(os.path.join(save_path, 'rank.pt'))
        loss_tensor = torch.load(os.path.join(save_path, 'loss.pt'))
        entropy_tensor = torch.load(os.path.join(save_path, 'entropy.pt'))
        scale_tensor = torch.load(os.path.join(save_path, 'scale.pt'))
        idf = pd.DataFrame({
            f'{ix}_rank': rank_tensor.numpy().flatten(),
            f'{ix}_recip_rank': 1 / (rank_tensor.numpy().flatten() + 1),
            f'{ix}_loss': loss_tensor.numpy().flatten(),
            f'{ix}_entropy': entropy_tensor.numpy().flatten(),
            f'{ix}_scale': scale_tensor.numpy().flatten(),
        })
        rdfs.append(idf)

    idf = pd.concat(rdfs, axis=1)

    return idf


def get_nominal_metrics(dataset_name, model_name, decoded_vocab):
    model_family = 'gpt2' if 'gpt2' in model_name else 'pythia'
    ds = datasets.load_from_disk(
        f'token_datasets/{model_family}/{dataset_name}')

    ds_df = make_dataset_df(ds, decoded_vocab)

    if model_family == 'gpt2':
        valid_tokens = ds_df.query(
            'position != 511 and token != 50256').index.values
    else:
        valid_tokens = ds_df.query(
            'position != 511 and token > 1').index.values

    act_save_path = os.path.join(
        'cached_activations', model_name, dataset_name)
    act_df = pd.DataFrame({
        # 'nominal_scale': torch.load(os.path.join(act_save_path, 'scale.pt')).numpy().flatten(),
        'nominal_loss': torch.load(os.path.join(act_save_path, 'loss.pt')).numpy().flatten(),
        'nominal_entropy': torch.load(os.path.join(act_save_path, 'entropy.pt')).numpy().flatten(),
        'nominal_rank': torch.load(os.path.join(act_save_path, 'rank.pt')).numpy().flatten(),
        'nominal_recip_rank': 1 / (torch.load(os.path.join(act_save_path, 'rank.pt')).numpy().flatten() + 1),
    })

    nominal_metrics = act_df.loc[valid_tokens].astype(
        np.float64).mean(axis=0).to_dict()

    return nominal_metrics, valid_tokens


def sample_baseline_neurons(df, k=20, last_l_layers=2, max_norm_percentile=0.9, min_vocab_var_percentile=0.1):
    min_layer = df['layer'].max() - last_l_layers
    max_norm = df['l2_penalty'].quantile(max_norm_percentile)
    min_vocab_var = df['vocab_var'].quantile(min_vocab_var_percentile)
    candidates = df.query(
        'layer > @min_layer and l2_penalty <= @max_norm and vocab_var >= @min_vocab_var')
    return candidates.sample(k)[['layer', 'neuron']].values


def print_baseline_neurons(neuron_arr):
    strings = [f"'{layer}.{neuron}'" for layer, neuron in neuron_arr]
    print(' '.join(strings))


def get_plot_data(neuron_data, model_name, dataset_name, interventions, valid_tokens):
    plot_data = {
        'neuron_entropies': {},
        'neuron_ranks': {},
        'neuron_recip_ranks': {},
        'neuron_losses': {},
        'neuron_scales': {},
    }

    for neuron in neuron_data[model_name]:
        idf = make_entropy_intervention_rdf(
            model_name, dataset_name, neuron, interventions)

        entropy = idf[[c for c in idf.columns if 'entropy' in c]].astype(
            np.float64).loc[valid_tokens].mean().values
        rank = idf[[c for c in idf.columns if 'rank' in c and 'recip' not in c]].astype(
            np.float64).loc[valid_tokens].mean().values
        recip_rank = idf[[c for c in idf.columns if 'recip_rank' in c]].astype(
            np.float64).loc[valid_tokens].mean().values
        loss = idf[[c for c in idf.columns if 'loss' in c]].astype(
            np.float64).loc[valid_tokens].mean().values
        scale = idf[[c for c in idf.columns if 'scale' in c]].astype(
            np.float64).loc[valid_tokens].mean().values

        plot_data['neuron_entropies'][neuron] = entropy
        plot_data['neuron_ranks'][neuron] = rank
        plot_data['neuron_recip_ranks'][neuron] = recip_rank
        plot_data['neuron_losses'][neuron] = loss
        plot_data['neuron_scales'][neuron] = scale

    return plot_data


def plot_entropy_neuron_weight_info(
    main_plot_data, baseline_plot_data, composition_counts, weight_decay_penalty, 
    bins, activation_values, model_name, neuron_data, baseline_neuron_data, clip_norm=5
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    colors = ['tab:blue', 'tab:green']

    ax = axs[0]
    ax.hist(np.minimum(weight_decay_penalty.flatten(), clip_norm), bins=100, alpha=0.5, color='gray')

    for ix, neuron in enumerate(neuron_data[model_name]):
        l, n  = neuron.split('.')
        weight_decay = weight_decay_penalty[int(l), int(n)]
        ax.axvline(weight_decay, alpha=1, color=colors[ix], label=neuron)

    for ix, neuron in enumerate(baseline_neuron_data[model_name]):
        l, n  = neuron.split('.')
        weight_decay = weight_decay_penalty[int(l), int(n)] + (int(n) - 2500) / 20000
        ax.axvline(weight_decay, alpha=1, color='red', linewidth=0.2)

    ax.plot([1, 2], [-1, -1], color='red', linewidth=0.7, label='random')
    # ax.set_xticklabels(['', '0', '1', '2', '3', '4', 'min(5, x)'])
    ax.set_yscale('log')
    ax.set_ylabel('neuron count')
    ax.set_xlabel('$\|W_{in}\|^2 + \|W_{out}\|^2$')
    ax.set_title('(a) Neuron weight norms')
    ax.set_ylim(bottom=0.7)
    ax.legend(title='Neuron', bbox_to_anchor=(1.02, 1), loc='upper right').get_frame().set_alpha(0.3)

    ax = axs[1]
    bin_mids = (bins[:-1] + bins[1:]) / 2

    for ix, neuron in enumerate(neuron_data[model_name]):
        counts = composition_counts[neuron]
        ax.plot(bin_mids, counts, color=colors[ix])

    for ix, neuron in enumerate(baseline_neuron_data[model_name]):
        counts = composition_counts[neuron]
        ax.plot(bin_mids, counts, color='red', linewidth=0.2)

    ax.set_yscale('log')
    ax.set_ylim(bottom=1)

    ax.set_ylabel('vocab count')
    ax.set_xlabel('$\cos(W_U, W_{out})$')
    ax.set_title('(b) Neuron composition w/ unembed')

        # reduce padding between x label and plot
    axs[0].xaxis.labelpad = -1
        
    ax = axs[2]

    for ix, neuron in enumerate(neuron_data[model_name]):
        scales = main_plot_data['neuron_scales'][neuron]
        ax.plot(activation_values, scales, label='L'+neuron, color=colors[ix])

    for ix, neuron in enumerate(baseline_neuron_data[model_name]):
        scales = baseline_plot_data['neuron_scales'][neuron]
        ax.plot(activation_values, scales, color='red', linewidth=0.2)
    ax.plot([0, 0.1], [scales[1], scales[1]], color='red', linewidth=0.5, label='random')
    ax.set_title('(c) Final layer norm scale')
    ax.set_xlabel('fixed activation value')
    ax.set_ylabel('mean scale')

    for ax in axs:
        # turn off top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--')

        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)

    plt.tight_layout()

    return fig, axs


def plot_entropy_neuron_intervention(
    nominal_metrics, main_plot_data, baseline_plot_data, activation_values,
    neuron_data, baseline_neuron_data, model_name
):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    colors = ['tab:blue', 'tab:green', 'tab:purple']

    for ix, neuron in enumerate(neuron_data[model_name]):
        entropy = main_plot_data['neuron_entropies'][neuron]
        rank = main_plot_data['neuron_ranks'][neuron]
        recip_rank = main_plot_data['neuron_recip_ranks'][neuron]
        loss = main_plot_data['neuron_losses'][neuron]

        axs[0].plot(activation_values, recip_rank, color=colors[ix])
        axs[1].plot(activation_values, entropy, color=colors[ix])
        axs[2].plot(activation_values, loss, color=colors[ix])
        # axs[2].plot(activation_values, rank,  color=colors[ix])


    # axs[0].plot([0, 1], [nominal_metrics['nominal_entropy'], nominal_metrics['nominal_entropy']],
    #            color='red', linewidth=0.5, label='random neuron')
    for neuron in baseline_neuron_data[model_name]:
        entropy = baseline_plot_data['neuron_entropies'][neuron]
        rank = baseline_plot_data['neuron_ranks'][neuron]
        recip_rank = baseline_plot_data['neuron_recip_ranks'][neuron]
        loss = baseline_plot_data['neuron_losses'][neuron]

        axs[0].plot(activation_values, recip_rank, color='red', linewidth=0.2)
        axs[1].plot(activation_values, entropy, color='red', linewidth=0.2)
        axs[2].plot(activation_values, loss, color='red', linewidth=0.2)
        # axs[2].plot(activation_values, rank, color='red', linewidth=0.2)

    axs[1].axhline(nominal_metrics['nominal_entropy'], color='black',
                  linestyle='--', label='no intervention', linewidth=1)
    axs[2].axhline(nominal_metrics['nominal_loss'], color='black',
                  linestyle='--', label='no intervention', linewidth=1)
    # axs[2].axhline(nominal_metrics['nominal_rank'], color='black', linestyle='--', label='no intervention', linewidth=1)
    axs[0].axhline(nominal_metrics['nominal_recip_rank'],
                  color='black', linestyle='--', label='no intervention', linewidth=1)

    axs[0].set_ylabel('mean reciprocal rank')
    axs[1].set_ylabel('mean entropy')
    axs[2].set_ylabel('mean loss')

    axs[0].set_title('(d) Next token prediction recip rank')
    axs[1].set_title('(e) Next token prediction entropy')
    axs[2].set_title('(f) Next token prediction loss')

    for ax in axs:
        # turn off top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlabel('fixed activation value')
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)

    axs[0].legend(loc = 'center left').get_frame().set_alpha(0.3)

    plt.tight_layout()

    return fig, axs
