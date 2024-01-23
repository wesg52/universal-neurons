import os
import einops
import torch
import datasets
import numpy as np
import pandas as pd
from utils import *
from transformer_lens import HookedTransformer
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objs as go


### LOADING FUNCTIONS ###

def load_dataset_summary(model_name, dataset_name):
    path = os.path.join(
        'summary_data', model_name, 'activations', dataset_name)
    summary_dict = {}
    for data_file in os.listdir(path):
        if data_file.endswith('.pt'):
            summary_data = torch.load(os.path.join(path, data_file))
            data_name = data_file[:-3]

        elif data_file.endswith('.npy'):
            summary_data = np.load(os.path.join(path, data_file))
            data_name = data_file[:-8]

        summary_dict[data_name] = summary_data

    return summary_dict


def load_all_summaries(model_name):
    dataset_summaries = {}
    for dataset_name in PILE_DATASETS:
        dataset_short_name = dataset_name.split('.')[2]
        dataset_summaries[dataset_short_name] = load_dataset_summary(
            model_name, dataset_name)
    return dataset_summaries


def load_weights_summary(model_name):
    data_dir = os.path.join('summary_data', model_name, 'weights')
    weights_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.pt'):
            weights_data[filename[:-3]
                         ] = torch.load(os.path.join(data_dir, filename))
        elif filename.endswith('.npz'):
            weights_data[filename[:-4]
                         ] = np.load(os.path.join(data_dir, filename))
        elif filename.endswith('.csv'):
            weights_data[filename[:-4]
                         ] = pd.read_csv(os.path.join(data_dir, filename))
    return weights_data


def load_all_token_datasets(model_name):
    token_datasets = {}
    model_family = get_model_family(model_name)
    for dataset_name in PILE_DATASETS:
        dataset_short_name = dataset_name.split('.')[2]
        ds_path = os.path.join('token_datasets', model_family, dataset_name)
        ds = datasets.load_from_disk(ds_path)
        token_datasets[dataset_short_name] = ds['tokens']
    return token_datasets


def get_tokenizer_and_decoded_vocab(model_name='pythia-70m'):
    # use smallest model in model family
    model = HookedTransformer.from_pretrained(model_name)
    decoded_vocab = {
        tix: model.tokenizer.decode(tix)
        for tix in model.tokenizer.get_vocab().values()
    }
    return model.tokenizer, decoded_vocab


### ACTIVATION SUMMARY PLOTTING FUNCTIONS ###

def plot_activation_boxplot_by_datasubset(subset_bin_counts, bin_edges):
    fig, ax = plt.subplots(1, 1, figsize=(len(subset_bin_counts), 5))
    # TODO: add box for all data
    stat_dicts = []
    distrs = []
    for i, (dataset_name, bin_counts) in enumerate(subset_bin_counts.items()):
        total = np.sum(bin_counts)

        # Edges of bins are midpoints
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Calculate cumulative sum of counts
        cumulative_counts = np.cumsum(bin_counts)

        base_percentiles = np.array([0, 25, 50, 75, 100])
        percentiles = np.array([0.01, 0.1, 1, 5, 10, 90, 95, 99, 99.9, 99.99])

        stats = {}
        # Calculate whiskers, medians, boxes, and caps
        for percentile in np.concatenate([base_percentiles, percentiles]):
            idx = np.searchsorted(
                cumulative_counts, percentile / 100.0 * total)

            value = bin_mids[idx]

            if percentile == 0:
                stats['whislo'] = value
            elif percentile == 100:
                stats['whishi'] = value
            elif percentile == 50:
                stats['med'] = value
            elif percentile == 25:
                stats['q1'] = value
            elif percentile == 75:
                stats['q3'] = value
            else:
                stats[str(percentile)] = value

        stat_dicts.append(stats)
        distrs.append(dataset_name)
        # Use bxp to create the boxplot
    ax.bxp(stat_dicts, showfliers=False, medianprops={
           'color': 'black', 'linestyle': '--'})

    seismic = plt.get_cmap('seismic')
    colors = seismic(np.arange(len(percentiles)) / (len(percentiles)-1))

    for ix, p in enumerate(percentiles):
        ax.scatter(
            np.arange(len(stat_dicts))+1, [d[str(p)] for d in stat_dicts],
            color=colors[ix], label=str(p) + '%', marker='_', s=250
        )

    ax.set_xticklabels(distrs)
    ax.set_ylim(bottom=-5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=1,
              title='Percentile', bbox_to_anchor=(1, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    # Turn off top and right splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('pre-activation')
    ax.set_title('Activation by Data Distribution', fontsize=18)

    # rotate and align the tick labels so they look better
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')
    plt.show()
    return ax


def plot_activation_distributions(subset_bin_counts, neuron_bin_edges):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # TODO: figure out if there is OBOE
    dummy_bin = np.array(
        [neuron_bin_edges[-1] + neuron_bin_edges[-1] - neuron_bin_edges[-2]])
    bin_edges = np.concatenate([neuron_bin_edges, dummy_bin])

    total_bin_counts = sum([v for v in subset_bin_counts.values()])
    nonzero_bins = np.nonzero(total_bin_counts)[0]
    min_bin = nonzero_bins[0] - 1
    max_bin = nonzero_bins[-1] + 1

    # TODO: turn into plotly plots
    # Add cdf line to histogram with right axis

    ax = axs[0]
    ax.hist(bin_edges,  weights=total_bin_counts,
            bins=len(bin_edges), log=True)
    ax.set_ylabel('empirical density')

    ax = axs[1]
    for dataset_name, bin_counts in subset_bin_counts.items():
        ax.plot(bin_edges, bin_counts / bin_counts.sum(), label=dataset_name)
        ax.set_ylabel('empirical density')

    ax = axs[2]
    for dataset_name, bin_counts in subset_bin_counts.items():
        ax.plot(bin_edges, bin_counts / bin_counts.sum(), label=dataset_name)
        ax.set_yscale('log')
        ax.set_ylabel('log empirical density')
        ax.set_ylim(bottom=1e-6)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    for ax in axs:
        ax.set_xlim(bin_edges[min_bin], bin_edges[max_bin])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    # show the plot
    display(fig)


def plot_activation_distributions_plotly(subset_bin_counts, neuron_bin_edges):
    fig = sp.make_subplots(rows=1, cols=3)

    dummy_bin = np.array(
        [neuron_bin_edges[-1] + neuron_bin_edges[-1] - neuron_bin_edges[-2]])
    bin_edges = np.concatenate([neuron_bin_edges, dummy_bin])

    total_bin_counts = sum([v for v in subset_bin_counts.values()])
    nonzero_bins = np.nonzero(total_bin_counts)[0]
    min_bin = nonzero_bins[0] - 1
    max_bin = nonzero_bins[-1] + 1

    # To ensure bar chart appears as a histogram, the width of each bar is set to the difference between successive bin edges
    bar_widths = bin_edges[1:] - bin_edges[:-1]

    fig.add_trace(go.Bar(x=bin_edges,
                         y=total_bin_counts,
                         width=bar_widths,
                         name='empirical density',
                         marker=dict(line=dict(width=0))),
                  row=1, col=1)

    for dataset_name, bin_counts in subset_bin_counts.items():
        fig.add_trace(go.Scatter(x=bin_edges,
                                 y=bin_counts / bin_counts.sum(),
                                 name=dataset_name),
                      row=1, col=2)

    for dataset_name, bin_counts in subset_bin_counts.items():
        fig.add_trace(go.Scatter(x=bin_edges,
                                 y=bin_counts / bin_counts.sum(),
                                 name=dataset_name,
                                 line=dict(shape='hv')),
                      row=1, col=3)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=3)

    fig.update_xaxes(
        range=[bin_edges[min_bin], bin_edges[max_bin]], row=1, col=1)
    fig.update_xaxes(
        range=[bin_edges[min_bin], bin_edges[max_bin]], row=1, col=2)
    fig.update_xaxes(
        range=[bin_edges[min_bin], bin_edges[max_bin]], row=1, col=3)

    fig.update_layout(showlegend=True, title_text="Activation Distributions")

    fig.show()


def get_vocab_summary_dfs(dataset_summaries, decoded_vocab, layer, neuron):
    vocab_col_names = ['neuron_vocab_max', 'neuron_vocab_max_ixs',
                       'neuron_vocab_mean', 'neuron_vocab_mean_ixs']

    df_dict = {col: pd.DataFrame({}) for col in vocab_col_names}

    for ds, summary_dict in dataset_summaries.items():
        for col in vocab_col_names:
            try:
                neuron_col = summary_dict[col][layer, neuron].numpy()
            except AttributeError:  # already numpy
                neuron_col = summary_dict[col][layer, neuron]
            if 'ixs' in col:
                neuron_col = [decoded_vocab[t] for t in neuron_col]
            else:
                neuron_col = neuron_col.astype(np.float32)
            df_dict[col][ds] = neuron_col

    max_value_df, max_token_df, mean_value_df, mean_token_df = df_dict.values()
    return max_value_df, max_token_df, mean_value_df, mean_token_df


def vocab_heatmap(value_df, token_df, display_top_k=25, max=True):
    fig = go.Figure(
        data=go.Heatmap(
            z=value_df.head(display_top_k).values,
            text=token_df.head(display_top_k).values,
            texttemplate="%{text}",
            textfont={"size": 10}))

    # reverse the order to make largest first
    fig.update_layout(
        title=f'{"Max" if max else "Mean"} activating vocab by dataset',
        xaxis_title='dataset',
        yaxis_title='token',
        yaxis_autorange='reversed')

    # add tick labels with rotation
    fig.update_xaxes(ticktext=value_df.columns, tickvals=np.arange(
        len(value_df.columns)), tickangle=-45)
    fig.show()
    return fig


def make_vocab_line_plot(max_value_df, max_token_df, mean_value_df, mean_token_df):

    # TODO: add dataset vocab counts to hover template
    # TODO: make legend control both plots

    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Max Activating Vocab by Dataset",
                        "Mean Activating Vocab by Dataset")
    )

    # Loop through each column for max
    for i, col in enumerate(max_value_df.columns):
        fig.add_trace(
            go.Scatter(
                y=max_value_df[col],  # Use max value here
                mode='lines',
                name=col,
                text=max_token_df[col],
                hovertemplate='Value: %{y}<br>Token: %{text}',
                showlegend=False,
            ),
            row=1, col=1  # This goes to the left subplot
        )

    # Loop through each column for mean
    for i, col in enumerate(mean_value_df.columns):
        fig.add_trace(
            go.Scatter(
                y=mean_value_df[col],  # Use mean value here
                mode='lines',
                name=col,
                text=mean_token_df[col],
                hovertemplate='Value: %{y}<br>Token: %{text}',
                # xaxis_title='token rank'
            ),
            row=1, col=2  # This goes to the right subplot
        )
    # Update layout
    fig.update_layout(
        xaxis_title='token rank',
        yaxis_title='Token Activation'
    )
    fig.show()
    return fig


def display_max_activating_examples(
    dataset_summaries, decoded_vocab, token_datasets, layer, neuron,
    display_k_per_dataset=5, tokens_to_display_before=20, tokens_to_display_after=5
):

    for dataset_name, tokenized_dataset in token_datasets.items():
        display(Markdown('### ' + dataset_name))
        neuron_max_ix = dataset_summaries[dataset_name][
            'neuron_max_activating_index'][layer, neuron].numpy()
        neuron_max_value = dataset_summaries[dataset_name][
            'neuron_max_activating_value'][layer, neuron].numpy()

        n_seq, ctx_len = tokenized_dataset.shape

        neuron_max_seq_ix = neuron_max_ix // ctx_len
        neuron_max_token_ix = neuron_max_ix % ctx_len

        # TODO: be more clever about collapsing identical sequences
        # TODO: integrate with circuitsviz
        # (challenge bc we would need to rerun the model to get activations; otherwise just color most activating token)
        for i in range(display_k_per_dataset):
            display_min = neuron_max_token_ix[i] - tokens_to_display_before
            display_max = neuron_max_token_ix[i] + tokens_to_display_after
            display_tokens = tokenized_dataset[neuron_max_seq_ix[i],
                                               display_min:display_max].numpy()
            display_str = ''.join([
                decoded_vocab[t] if ix != tokens_to_display_before
                else '||' + decoded_vocab[t] + '||'
                for ix, t in enumerate(display_tokens)
            ])
            print(f'{i+1}) {round(neuron_max_value[i], 2)} | {display_str}')

### WEIGHTS SUMMARY PLOTTING FUNCTIONS ###


def get_neuron_summary_dfs(neuron_comp_data, layer, neuron):
    comp_type_display_names = {
        'in_in': '(n_in, W_in)',
        'in_out': '(W_out, n_in)',
        'out_in': '(n_out, W_in)',
        'out_out': '(n_out, W_out)'
    }
    comp_col_names = ['top_neuron_value', 'top_neuron_ix',
                      'bottom_neuron_value', 'bottom_neuron_ix']
    n_layers, n_neurons, _ = neuron_comp_data['in_in']['top_neuron_value'].shape
    df_dict = {col: pd.DataFrame({}) for col in comp_col_names}
    for comp_type, summary_dict in neuron_comp_data.items():
        for col in comp_col_names:
            data_col = summary_dict[col][layer, neuron].numpy()
            if col[-2:] == 'ix':
                data_col = np.unravel_index(data_col, (n_layers, n_neurons))
                data_col = [f'L{l}.{n}' for l, n in zip(*data_col)]
            df_dict[col][comp_type_display_names[comp_type]] = data_col
    return df_dict


def get_vocab_composition_summary_dfs(vocab_comp_data, decoded_vocab, layer, neuron):
    comp_type_display_names = {
        'U_out': '(n_out, W_U)',
        'E_in': '(W_E, n_in)',
        'U_in': '(n_in, W_U)',
        'E_out': '(W_E, n_out)',
    }
    comp_col_names = ['top_vocab_value', 'top_vocab_ix',
                      'bottom_vocab_value', 'bottom_vocab_ix']
    vocab_df_dict = {col: pd.DataFrame({}) for col in comp_col_names}
    for comp_type in comp_type_display_names.keys():
        summary_dict = vocab_comp_data[comp_type]
        for col in comp_col_names:
            data_col = summary_dict[col][layer, neuron].numpy()
            if col[-2:] == 'ix':
                data_col = [decoded_vocab.get(t, f'N/A {t}') for t in data_col]
            vocab_df_dict[col][comp_type_display_names[comp_type]] = data_col
    return vocab_df_dict


def neuron_or_vocab_composition_heatmap(top_value_df, top_ix_df, bottom_value_df, bottom_ix_df, display_top_k=10, neuron=True):
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("max similarity", "min similarity")
    )

    heatmap_max = go.Heatmap(
        z=top_value_df.head(display_top_k).values,
        text=top_ix_df.head(display_top_k).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='Inferno',
        colorbar=dict(len=0.5, yanchor="bottom",
                      y=0.5, title="Max composition")
    )

    heatmap_min = go.Heatmap(
        z=bottom_value_df.head(display_top_k).values,
        text=bottom_ix_df.head(display_top_k).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='Inferno_r',
        colorbar=dict(len=0.5, yanchor="top", y=0.5, title="Min composition")
    )

    fig.add_trace(heatmap_max, row=1, col=1)
    fig.add_trace(heatmap_min, row=1, col=2)

    entity = "Neuron input/output" if neuron else "Token (un)embed"
    # reverse the order to make largest first
    fig.update_layout(
        title=f'{entity} weight cosine similarities',
        xaxis_title='composition',
        yaxis_title=entity.lower().split(' ')[0],
        yaxis_autorange='reversed')

    # add tick labels with rotation
    fig.update_xaxes(ticktext=top_value_df.columns, tickvals=np.arange(
        len(top_value_df.columns)), tickangle=-45)

    fig.show()

    return fig


def neuron_and_vocab_density_plots(weights_data, layer, neuron):
    neuron_comp_type_display_names = {
        'in_in': '(n_in, W_in)',
        'in_out': '(W_out, n_in)',
        'out_in': '(n_out, W_in)',
        'out_out': '(n_out, W_out)'
    }
    vocab_comp_type_display_names = {
        'U_out': '(n_out, W_U)',
        'E_in': '(W_E, n_in)',
        'U_in': '(n_in, W_U)',
        'E_out': '(W_E, n_out)',
    }
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    bin_edges = torch.linspace(-1, 1, 100)
    dummy_bin = np.array(
        [bin_edges[-1] + bin_edges[-1] - bin_edges[-2]])
    bin_edges = np.concatenate([bin_edges, dummy_bin])

    vocab_min_bin = len(bin_edges) - 1
    vocab_max_bin = 0
    for comp_type, legend_key in vocab_comp_type_display_names.items():
        bin_counts = weights_data['vocab_comps'][comp_type]['comp_hist'][layer, neuron].numpy(
        )
        comp_bins = np.nonzero(bin_counts)[0]
        vocab_min_bin = min(vocab_min_bin, comp_bins.min())
        vocab_max_bin = max(vocab_max_bin, comp_bins.max())
        axs[0].plot(bin_edges, bin_counts, label=legend_key)
        axs[1].plot(bin_edges, bin_counts, label=legend_key)

    axs[0].set_xlim(bin_edges[vocab_min_bin] - 0.025,
                    bin_edges[vocab_max_bin] + 0.025)
    axs[1].set_xlim(bin_edges[vocab_min_bin] - 0.025,
                    bin_edges[vocab_max_bin] + 0.025)
    axs[1].set_yscale('log')

    neuron_min_bin = len(bin_edges) - 1
    neuron_max_bin = 0
    for comp_type, legend_key in neuron_comp_type_display_names.items():
        bin_counts = weights_data['neuron_comps'][comp_type]['comp_hist'][layer, neuron].numpy(
        )
        comp_bins = np.nonzero(bin_counts)[0]
        neuron_min_bin = min(neuron_min_bin, comp_bins.min())
        neuron_max_bin = max(neuron_max_bin, comp_bins.max())
        axs[2].plot(bin_edges, bin_counts, label=legend_key)
        axs[3].plot(bin_edges, bin_counts, label=legend_key)

    axs[2].set_xlim(bin_edges[neuron_min_bin] - 0.025,
                    bin_edges[neuron_max_bin] + 0.025)
    axs[3].set_xlim(bin_edges[neuron_min_bin] - 0.025,
                    bin_edges[neuron_max_bin] + 0.025)
    axs[3].set_yscale('log')

    ax_titles = [
        'vocab composition', 'vocab composition (log)',
        'neuron composition', 'neuron composition (log)'
    ]
    axs[0].set_ylabel('count')
    for ix, ax in enumerate(axs):
        ax.set_title(ax_titles[ix])
        ax.legend(loc='upper right')
        ax.set_xlabel('cosine similarity')
        # turn off top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.show()
    return axs


def plot_neuron_attn_composition(weights_data, layer, neuron):
    ATTN_COMPS = ['v_comps', 'o_comps', 'q_comps', 'k_comps']
    TITLES = [
        'V comp (W_OV @ n_out)',
        'O Comp (n_in @ W_OV)',
        'Q comp (n_out @ W_QK)',
        'K comp (W_QK @ n_out)'
    ]

    fig = sp.make_subplots(
        rows=1, cols=4, subplot_titles=TITLES, shared_yaxes=True)

    for i, attn_comp in enumerate(ATTN_COMPS):
        fig.add_trace(
            go.Heatmap(
                z=weights_data[attn_comp][layer, neuron],
                coloraxis="coloraxis"
            ),
            row=1,
            col=i+1
        )
        fig.update_xaxes(title_text="head", row=1, col=i+1)

    fig.update_yaxes(title_text="layer", row=1, col=1)
    fig.update_layout(coloraxis=dict(colorscale='Viridis'),
                      width=1000, height=350)
    fig.show()


### FULL SUMMARY ###

def display_summary(
        dataset_summaries, weight_summaries, token_datasets, decoded_vocab, layer, neuron,
        display_k_per_dataset=0):
    # TODO: don't hardcode bin edges (need to save from summary.py)
    bin_edges = torch.linspace(-10, 15, 256)
    padded_bin_edges = np.concatenate([
        np.array([bin_edges[0] + bin_edges[0] - bin_edges[1]]),
        bin_edges,
        np.array([bin_edges[-1] + bin_edges[-1] - bin_edges[-2]])
    ])
    subset_bin_counts = {}
    for dataset_name, dataset_summary in dataset_summaries.items():
        subset_bin_counts[dataset_name] = dataset_summary['neuron_bin_counts'][layer, neuron].numpy()

    plot_activation_boxplot_by_datasubset(subset_bin_counts, padded_bin_edges)

    plot_activation_distributions_plotly(subset_bin_counts, padded_bin_edges)

    max_value_df, max_token_df, mean_value_df, mean_token_df = get_vocab_summary_dfs(
        dataset_summaries, decoded_vocab, layer, neuron)

    vocab_heatmap(max_value_df, max_token_df, display_top_k=25, max=True)
    vocab_heatmap(mean_value_df, mean_token_df, display_top_k=25, max=False)

    make_vocab_line_plot(
        max_value_df, max_token_df, mean_value_df, mean_token_df)

    # weights
    neuron_and_vocab_density_plots(weight_summaries, layer, neuron)

    vocab_df_dict = get_vocab_composition_summary_dfs(
        weight_summaries['vocab_comps'], decoded_vocab, layer, neuron)
    neuron_df_dict = get_neuron_summary_dfs(
        weight_summaries['neuron_comps'], layer, neuron)

    neuron_or_vocab_composition_heatmap(
        vocab_df_dict['top_vocab_value'],
        vocab_df_dict['top_vocab_ix'],
        vocab_df_dict['bottom_vocab_value'],
        vocab_df_dict['bottom_vocab_ix'],
        neuron=False
    )

    neuron_or_vocab_composition_heatmap(
        neuron_df_dict['top_neuron_value'],
        neuron_df_dict['top_neuron_ix'],
        neuron_df_dict['bottom_neuron_value'],
        neuron_df_dict['bottom_neuron_ix'],
        neuron=True
    )

    plot_neuron_attn_composition(weight_summaries, layer, neuron)

    if display_k_per_dataset > 0:
        display_max_activating_examples(
            dataset_summaries, decoded_vocab, token_datasets, layer, neuron,
            display_k_per_dataset=display_k_per_dataset
        )
