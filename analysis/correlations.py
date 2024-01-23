import os
import pickle
import einops
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from utils import vector_histogram


def load_correlation_results(
        model_1_name, model_2_name, dataset, correlation_computation,
        return_np=False, result_dir='correlation_results'):
    file_path = os.path.join(
        result_dir,
        f'{model_1_name}+{model_2_name}',
        dataset,
        correlation_computation,
        'correlation.pt'
    )

    correlation_data = torch.load(file_path, map_location='cpu')
    if return_np:
        correlation_data = correlation_data.numpy()
    return correlation_data


def flatten_layers(correlation_data):
    return einops.rearrange(correlation_data, 'l1 n1 l2 n2 -> (l1 n1) (l2 n2)')


def unflatten_layers(correlation_data, m1_layers, m2_layers=None):
    if m2_layers is None:
        m2_layers = m1_layers
    return einops.rearrange(
        correlation_data, '(l1 n1) (l2 n2) -> l1 n1 l2 n2',
        l1=m1_layers, l2=m2_layers
    )


def summarize_correlation_matrix(correlation_matrix):
    # compute distribution summary
    bin_edges = torch.linspace(-1, 1, 100)

    bin_counts = vector_histogram(correlation_matrix, bin_edges)

    # compute left and right tails
    max_tail_v, max_tail_ix = torch.topk(
        correlation_matrix, 50, dim=1, largest=True)
    min_tail_v, min_tail_ix = torch.topk(
        correlation_matrix, 50, dim=1, largest=False)

    max_v, max_ix = torch.max(correlation_matrix, dim=1)
    min_v, min_ix = torch.min(correlation_matrix, dim=1)

    # compute corr distribution moments
    corr_mean = correlation_matrix.mean(dim=1)
    corr_diffs = correlation_matrix - corr_mean[:, None]
    corr_var = torch.mean(torch.pow(corr_diffs, 2.0), dim=1)
    corr_std = torch.pow(corr_var, 0.5)
    corr_zscore = corr_diffs / corr_std[:, None]
    corr_skew = torch.mean(torch.pow(corr_zscore, 3.0), dim=1)
    corr_kurt = torch.mean(torch.pow(corr_zscore, 4.0), dim=1)

    correlation_summary = {
        'diag_corr': correlation_matrix.diagonal().to(torch.float16),
        'obo_corr': torch.diag(correlation_matrix, diagonal=1).to(torch.float16),
        'bin_counts': bin_counts.to(torch.int32),
        'max_corr': max_v.to(torch.float16),
        'max_corr_ix': max_ix.to(torch.int32),
        'min_corr': min_v.to(torch.float16),
        'min_corr_ix': min_ix.to(torch.int32),
        'max_tail_corr': max_tail_v.to(torch.float16),
        'max_tail_corr_ix': max_tail_ix.to(torch.int32),
        'min_tail_corr': min_tail_v.to(torch.float16),
        'min_tail_corr_ix': min_tail_ix.to(torch.int32),
        'corr_mean': corr_mean.to(torch.float16),
        'corr_var': corr_var.to(torch.float16),
        'corr_skew': corr_skew.to(torch.float16),
        'corr_kurt': corr_kurt.to(torch.float16)
    }
    return correlation_summary


def make_correlation_result_df(model_a, model_b, dataset, metric, baseline_metric, result_dir='correlation_results'):
    corr_data = load_correlation_results(
        model_a, model_b, dataset, metric, return_np=True, result_dir=result_dir)
    n_layers_m1, n_neurons_m1, n_layers_m2, n_neurons_m2 = corr_data.shape
    corr_data = flatten_layers(corr_data)
    if np.isnan(corr_data).any():
        print(f'Warning: setting {np.isnan(corr_data).sum()} nans to zero')
        corr_data = np.nan_to_num(corr_data, nan=0.0)
    max_corr = corr_data.max(axis=1)
    max_corr_ix = corr_data.argmax(axis=1)
    corr_data_diag = np.diag(corr_data)
    del corr_data

    baseline_corr_data = load_correlation_results(
        model_a, model_b, dataset, baseline_metric,
        return_np=True, result_dir=result_dir
    )
    baseline_corr_data = flatten_layers(baseline_corr_data)
    baseline_max_corr = baseline_corr_data.max(axis=1)
    baseline_max_corr_ix = baseline_corr_data.argmax(axis=1)
    del baseline_corr_data

    max_sim = np.unravel_index(max_corr_ix, (n_layers_m2, n_neurons_m2))
    baseline_max_sim = np.unravel_index(
        baseline_max_corr_ix, (n_layers_m2, n_neurons_m2))

    corr_df = pd.DataFrame({
        'max_corr': max_corr,
        'max_sim_layer': max_sim[0],
        'max_sim_neuron': max_sim[1],
        'diag_corr': corr_data_diag,
        'baseline': baseline_max_corr,
        'baseline_layer': baseline_max_sim[0]
    }, index=pd.MultiIndex.from_product([range(n_layers_m1), range(n_neurons_m1)]))
    corr_df.index.names = ['layer', 'neuron']

    return corr_df


def plot_correlation_vs_baseline(corr_df, plot_type='scatter', n_cols=6, title=''):
    n_cols = 6
    n_rows = 4 if corr_df.reset_index().layer.max() == 23 else 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10 if n_rows == 4 else 5))
    for ix, (layer, layer_df) in enumerate(corr_df.reset_index().groupby('layer')):
        ax = axs[ix//n_cols, ix % n_cols]
        if plot_type == 'scatter':
            ax.scatter(layer_df.baseline.values,
                       layer_df.max_corr.values, alpha=0.25, s=2)
        elif plot_type == 'histplot':
            sns.histplot(data=layer_df, x='baseline', y='max_corr', ax=ax)
        ax.set_title(f'Layer {layer}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # diag line
        ax.plot([0, 1], [0, 1], transform=ax.transAxes,
                ls='--', c='red', alpha=0.5)
    if title:
        fig.suptitle(title)
    plt.tight_layout()


def plotly_scatter_corr_by_layer(corr_df):
    data = []
    for layer, lg in corr_df.reset_index().groupby('layer'):
        trace = go.Scatter(
            x=lg['baseline'],
            y=lg['max_corr'],
            mode='markers',
            name=layer,
            marker=dict(
                size=3,
                opacity=0.3
            ),
            text=lg['neuron'],  # Hover text
            hoverinfo=['x', 'y', 'text']  # Only show the hover text
        )
        data.append(trace)

    layout = go.Layout(
        title='Plot',
        xaxis=dict(
            title='baseline'
        ),
        yaxis=dict(
            title='max_corr'
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
