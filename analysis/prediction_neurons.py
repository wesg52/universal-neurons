import math
import torch
import einops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from scipy.stats import gaussian_kde, ttest_ind


def plot_combined_prediction_neuron_skew(combined_df, models, colors, ax=None, layers=(20, 21, 22, 23)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    skews = combined_df['vocab_skew'].values
    x_vals = np.linspace(min(skews), max(skews), 1000)
    for i, l in enumerate(layers):
        model_kdes = []
        model_pred_neurons = []
        layer_df = combined_df.query('layer == @l').query('vocab_kurt > 10')
        for model in models:
            model_skew = layer_df.query('model == @model')['vocab_skew'].values
            kernel = gaussian_kde(model_skew, bw_method=0.5)
            kde_vals = kernel(x_vals)
            model_kdes.append(kde_vals)
            model_pred_neurons.append(len(model_skew))

        model_kdes = np.array(model_kdes)
        # plot mean with fill between min and max
        mean_kde = np.mean(model_kdes, axis=0)
        min_kde = np.min(model_kdes, axis=0)
        max_kde = np.max(model_kdes, axis=0)

        ax.plot(x_vals, mean_kde, label=f'{l}', color=colors[i])
        ax.fill_between(x_vals, min_kde, max_kde, alpha=0.4, color=colors[i])

        mean_pred_neurons = np.mean(model_pred_neurons)
        # algin text to the right
        ax.annotate(f'L{l}: {mean_pred_neurons:.1f}', xy=(
            0.96, 0.90-i*0.055), xycoords='axes fraction', ha='right', va='top')

    ax.annotate(
        f'Mean # pred neurons', xy=(0.96, 0.96),
        xycoords='axes fraction', ha='right', va='top', fontsize=11
    )

    ax.legend(title='layer', loc='upper left')
    ax.set_xlim(-7, 7)

    return ax


def plot_prediction_neurons(top_pred_neurons, top_pred_neuron_class, composition_dict, vocab_df, n_cols=5):
    n_rows = math.ceil(len(top_pred_neurons) / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for ix, ((layer, neuron_ix), class_label) in enumerate(zip(top_pred_neurons, top_pred_neuron_class)):
        ax = axs[ix // n_cols, ix % n_cols]
        comp_scores = composition_dict[layer, neuron_ix]
        bin_range = np.min(comp_scores), np.max(comp_scores)
        ax.hist(comp_scores[vocab_df[class_label]], bins=100,
                log=True, range=bin_range, label='True', alpha=0.5)
        ax.hist(comp_scores[~vocab_df[class_label]], bins=100,
                log=True, range=bin_range, label='False', alpha=0.5)
        ax.set_title(f'L{layer}.{neuron_ix}')
        ax.legend(title=class_label)

        ax.grid(alpha=0.25, linestyle='--')

        if ix % n_cols == 0:
            ax.set_ylabel('token count')

        if ix // n_cols == n_rows - 1:
            ax.set_xlabel('$cos(W_U, w_{out})$')

    plt.tight_layout()


def make_composition_dict(model, neuron_df, use_cos=True):
    if use_cos:
        W_U = model.W_U / model.W_U.norm(dim=0, keepdim=True)
    else:
        W_U = model.W_U
    neuron_col = 'neuron_ix' if 'neuron_ix' in neuron_df.columns else 'neuron'
    composition_dict = {}
    for ix, (layer, neuron) in enumerate(neuron_df[['layer', neuron_col]].values):
        if ix % 100 == 0:
            print(f'{ix} | layer {layer} | neuron {neuron}')
        w_out = model.W_out[layer, neuron]
        if use_cos:
            w_out = w_out / w_out.norm(dim=0, keepdim=True)
        vocab_dist = (w_out @ W_U).cpu().numpy()
        composition_dict[layer, neuron] = vocab_dist
    return composition_dict

# different methods for finding best explanation for prediction neurons


def make_mean_dif_df(vocab_df, composition_dict):
    bool_cols = vocab_df.select_dtypes(include='bool').columns
    mean_dif_dict = {(layer, neuron): {}
                     for layer, neuron in composition_dict.keys()}
    for (layer, neuron), vocab_dist in composition_dict.items():
        for col in bool_cols:
            pos_mean = vocab_dist[vocab_df[col].values].mean()
            neg_mean = vocab_dist[~vocab_df[col].values].mean()
            mean_dif_dict[layer, neuron][col] = pos_mean - neg_mean
    mean_dif_df = pd.DataFrame(mean_dif_dict).T
    return mean_dif_df


def make_welsh_t_df(vocab_df, composition_dict):
    bool_cols = vocab_df.select_dtypes(include='bool').columns
    welsh_t_dict = {(layer, neuron): {}
                    for layer, neuron in composition_dict.keys()}
    for (layer, neuron), vocab_dist in composition_dict.items():
        for col in bool_cols:
            pos_class = vocab_dist[vocab_df[col].values]
            neg_class = vocab_dist[~vocab_df[col].values]
            welsh_t, p_val = ttest_ind(pos_class, neg_class, equal_var=False)
            welsh_t_dict[layer, neuron][col] = welsh_t
    welsh_df = pd.DataFrame(welsh_t_dict).T
    return welsh_df


def make_variance_reduction_df(vocab_df, composition_dict):
    bool_cols = vocab_df.select_dtypes(include='bool').columns
    var_red_dict = {(layer, neuron): {}
                    for layer, neuron in composition_dict.keys()}
    for (layer, neuron), vocab_dist in composition_dict.items():
        var = np.var(vocab_dist)
        for col in bool_cols:
            pos_vocab = vocab_df[col].values
            n1 = pos_vocab.sum()
            n2 = (~pos_vocab).sum()
            pos_class = vocab_dist[pos_vocab]
            neg_class = vocab_dist[~pos_vocab]
            var_red = var - (n1*np.var(pos_class) + n2 *
                             np.var(neg_class))/(n1+n2)
            var_red_dict[layer, neuron][col] = var_red / var
    var_red_df = pd.DataFrame(var_red_dict).T
    return var_red_df


def make_skewness_reduction_df(vocab_df, composition_dict):
    bool_cols = vocab_df.select_dtypes(include='bool').columns
    skew_red_dict = {(layer, neuron): {}
                     for layer, neuron in composition_dict.keys()}
    for (layer, neuron), vocab_dist in composition_dict.items():
        skew = skewness(vocab_dist)
        for col in bool_cols:
            pos_vocab = vocab_df[col].values
            n1 = pos_vocab.sum()
            n2 = (~pos_vocab).sum()
            pos_class = vocab_dist[pos_vocab]
            neg_class = vocab_dist[~pos_vocab]
            skew_red = skew - (n1 * skewness(pos_class) +
                               n2 * skewness(neg_class))/(n1+n2)
            skew_red_dict[layer, neuron][col] = skew_red
    skew_red_df = pd.DataFrame(skew_red_dict).T
    return skew_red_df


def skewness(arr, absolute=True):
    if absolute:
        return abs(np.mean((arr - np.mean(arr) / np.std(arr))**3))
    else:
        return np.mean((arr - np.mean(arr) / np.std(arr))**3)


def kurtosis(arr):
    return np.mean((arr - np.mean(arr) / np.std(arr))**4)


def make_kurtosis_reduction_df(vocab_df, composition_dict):
    bool_cols = vocab_df.select_dtypes(include='bool').columns
    kurt_red_dict = {(layer, neuron): {}
                     for layer, neuron in composition_dict.keys()}
    for (layer, neuron), vocab_dist in composition_dict.items():
        kurt = kurtosis(vocab_dist)
        for col in bool_cols:
            pos_vocab = vocab_df[col].values
            n1 = pos_vocab.sum()
            n2 = (~pos_vocab).sum()
            pos_class = vocab_dist[pos_vocab]
            neg_class = vocab_dist[~pos_vocab]
            kurt_red = kurt - (n1 * kurtosis(pos_class) +
                               n2 * kurtosis(neg_class))/(n1+n2)
            kurt_red_dict[layer, neuron][col] = kurt_red / kurt
    kurt_red_df = pd.DataFrame(kurt_red_dict).T
    return kurt_red_df


PAPER_EXAMPLES = [
    (18, 3482, 'starts_w_space', False),
    (19, 1169, 'is_year', False),
    (23, 2042, 'contains_open_paren', True),
]


PRED_NEURONS = [
    (20, 13, 'all_caps', False),
    (23, 3440, 'all_caps', True),
    (21, 2148, 'all_caps', True),
    (15, 591, 'all_caps', True),
    (19, 1121, 'all_caps', False),
    (18, 2336, 'all_caps', False),
    (15, 84, 'all_caps', False),
    (17, 2559, 'all_caps', False),
    (22, 1585, 'end_w_ing', False),
    (22, 3534, 'end_w_ing', False),
    (19, 2871, 'end_w_ing', True),
    (20, 1867, 'end_w_ing', False),
    (19, 1647, 'end_w_ing', False),
    (22, 904, 'end_w_ing', True),
    (19, 3984, 'end_w_ing', False),
    (15, 1699, 'end_w_ing', False),
    (23, 3844, 'end_w_ing', True),
    (16, 3122, 'end_w_ing', True),
    (18, 1984, 'end_w_ing', False),
    (21, 118, 'end_w_ing', False),
    (16, 3346, 'end_w_ing', False),
    (14, 2046, 'end_w_ing', False),
    (14, 4048, 'end_w_ing', True),
    (21, 2102, 'is_year', True),
    (17, 332, 'is_year', False),
    (19, 1169, 'is_year', False),
    (23, 2260, 'is_year', True),
    (18, 1982, 'is_year', False),
    (18, 836, 'is_intensive_pronoun', False),
    (23, 2205, 'is_second_person_pronoun', True),
    (20, 2583, 'is_second_person_pronoun', True),
    (17, 774, 'is_second_person_pronoun', True),
    (18, 638, 'is_second_person_pronoun', False),
    (14, 2719, 'is_second_person_pronoun', False),
    (18, 1932, 'is_second_person_pronoun', True),
    (18, 1532, 'is_second_person_pronoun', False),
    (18, 1631, 'is_second_person_pronoun', False),
    (18, 3930, 'is_second_person_pronoun', True),
    (23, 2330, 'is_female_pronoun', True),
    (21, 1165, 'is_female_pronoun', True),
    (18, 1927, 'is_female_pronoun', False),
    (22, 73, 'is_neutral_pronoun', True),
    (22, 1732, 'is_neutral_pronoun', True),
    (21, 1274, 'is_neutral_pronoun', True),
    (17, 797, 'is_neutral_pronoun', False),
    (20, 603, 'is_neutral_pronoun', True),
    (16, 4092, 'is_neutral_pronoun', True),
    (16, 2529, 'is_neutral_pronoun', False),
    (19, 2820, 'is_neutral_pronoun', False),
    (18, 3690, 'is_neutral_pronoun', False),
    (15, 1552, 'is_neutral_pronoun', False),
    (23, 2774, 'is_male_pronoun', True),
    (21, 1592, 'is_male_pronoun', True),
    (21, 1961, 'is_male_pronoun', True),
    (19, 1946, 'is_male_pronoun', False),
    (20, 1477, 'is_male_pronoun', True),
    (20, 1601, 'contains_question', True),
    (22, 371, 'contains_question', True),
    (17, 985, 'contains_question', True),
    (17, 1407, 'contains_question', False),
    (10, 3946, 'contains_question', False),
    (23, 2110, 'is_one_digit', True),
    (20, 3025, 'is_one_digit', False),
    (21, 1920, 'contains_exclamation', True),
    (23, 219, 'contains_exclamation', False),
    (22, 1693, 'is_relative_pronoun', True),
    (20, 627, 'is_relative_pronoun', False),
    (23, 1538, 'is_relative_pronoun', True),
    (23, 1796, 'is_relative_pronoun', True),
    (15, 3252, 'is_relative_pronoun', False),
    (23, 2652, 'start_w_no_space_and_digit', False),
    (20, 369, 'start_w_no_space_and_digit', False),
    (22, 1285, 'start_w_no_space_and_digit', False),
    (17, 2194, 'start_w_no_space_and_digit', False),
    (20, 2965, 'contains_close_paren', False),
    (20, 2814, 'contains_close_paren', True),
    (22, 1772, 'contains_close_paren', True),
    (15, 1814, 'contains_close_paren', False),
    (10, 3852, 'contains_close_paren', False),
    (22, 2800, 'contains_quotation', True),
    (23, 2349, 'contains_quotation', True),
    (23, 2957, 'contains_quotation', True),
    (20, 2739, 'contains_quotation', False),
    (22, 1693, 'is_interrogative_pronoun', True),
    (23, 1538, 'is_interrogative_pronoun', True),
    (20, 627, 'is_interrogative_pronoun', False),
    (18, 1510, 'is_interrogative_pronoun', False),
    (23, 1796, 'is_interrogative_pronoun', True),
    (20, 1501, 'is_month', False),
    (22, 206, 'is_month', True),
    (17, 3095, 'is_month', False),
    (19, 1810, 'is_month', False),
    (23, 3151, 'contains_semicolon', True),
    (19, 3691, 'contains_semicolon', True),
    (23, 1697, 'contains_semicolon', False),
    (21, 880, 'contains_open_bracket', True),
    (22, 3319, 'contains_open_bracket', True),
    (23, 5, 'contains_open_bracket', True),
    (16, 2262, 'contains_open_bracket', False),
    (16, 1541, 'contains_open_bracket', False),
    (21, 1131, 'contains_open_paren', True),
    (23, 2042, 'contains_open_paren', True),
    (20, 1561, 'contains_open_paren', False),
    (18, 1387, 'contains_open_paren', False),
    (21, 1325, 'contains_open_paren', False),
    (22, 811, 'contains_open_paren', False),
    (23, 5, 'contains_open_paren', True),
    (17, 398, 'contains_open_paren', False),
    (23, 2926, 'is_demonstrative_pronoun', True),
    (23, 3282, 'is_demonstrative_pronoun', True),
    (21, 3480, 'is_demonstrative_pronoun', True),
    (15, 3365, 'is_demonstrative_pronoun', False),
    (22, 2958, 'is_demonstrative_pronoun', False),
    (18, 121, 'is_demonstrative_pronoun', False),
    (16, 2905, 'is_demonstrative_pronoun', False),
    (21, 2417, 'is_demonstrative_pronoun', False),
    (23, 2652, 'contains_digit', False),
    (23, 3383, 'contains_digit', True),
    (20, 3385, 'contains_digit', False),
    (19, 3120, 'contains_digit', False),
    (18, 732, 'contains_digit', False),
    (23, 1033, 'contains_colon', True),
    (21, 3165, 'contains_colon', True),
    (19, 3488, 'contains_colon', False),
    (20, 3390, 'contains_colon', False),
    (19, 747, 'contains_colon', False),
    (19, 1317, 'contains_colon', False),
    (23, 3649, 'contains_colon', False),
    (22, 2225, 'contains_colon', False),
    (19, 3794, 'is_state', False),
    (19, 3610, 'is_state', False),
    (19, 3282, 'is_state', True),
    (23, 2182, 'is_possessive_pronoun', False),
    (23, 227, 'is_possessive_pronoun', True),
    (19, 177, 'is_possessive_pronoun', False),
    (20, 3314, 'is_possessive_pronoun', False),
    # var
    (21, 3399, 'start_w_no_space', False),
    (23, 680, 'start_w_no_space', False),
    (18, 3483, 'start_w_no_space', True),

    (23, 2772, 'starts_w_cap', True),
    (22, 451, 'starts_w_cap', False),
    (18, 687, 'starts_w_cap', False),
    (18, 693, 'starts_w_cap', True),
    (22, 1609, 'starts_w_cap', False),
]


def plot_percentiles(df, col='vocab_kurt', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    layers = np.arange(0, df.layer.values.max() + 1)
    ps = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99]
    percentiles = np.vstack(df.groupby('layer')[col].apply(
        np.quantile, q=ps, axis=0).values).T
    mean = df.groupby('layer')[col].mean().values

    ultra_light = '#ede1e1'
    light = "#DCBCBC"
    light_highlight = "#C79999"
    mid = "#B97C7C"
    mid_highlight = "#A25050"
    dark = "#8F2727"
    dark_highlight = "#7C0000"

    ax.fill_between(layers, percentiles[0], percentiles[10],
                    facecolor=light, color=ultra_light, label="1-99%")
    ax.fill_between(layers, percentiles[1], percentiles[9],
                    facecolor=light, color=light, label="5-95%")
    ax.fill_between(layers, percentiles[2], percentiles[8],
                    facecolor=light_highlight, color=light_highlight, label="10-90%")
    ax.fill_between(
        layers, percentiles[3], percentiles[7], facecolor=mid, color=mid, label="15-85%")
    ax.fill_between(layers, percentiles[4], percentiles[6],
                    facecolor=mid_highlight, color=mid_highlight, label="25-75%")

    ax.plot(layers, percentiles[5], color=dark, label="median")
    ax.plot(layers, mean, color='blue', label="mean", lw=1)

    ax.legend(loc='upper left' if col == 'vocab_kurt' else 'lower left',
              title='layer percentile', framealpha=0.2)
    ax.set_xlabel('layer')
    ax.set_ylabel('$\cos(W_U, W_{out})$ ' +
                  ('kurtosis' if col == 'vocab_kurt' else 'skew'))


def plot_skew_low_kurt_ps_by_kurtosis(df, kurtosis_split=10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    df_copy = df.copy()
    for layer in df.layer.unique():
        df_copy = df_copy.append({'layer': int(
            layer), 'vocab_kurt': kurtosis_split+1, 'vocab_skew': 0}, ignore_index=True)
    df_copy.layer = df_copy.layer.astype(int)

    layers = np.arange(0, df.layer.values.max() + 1)
    ps = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99]

    high_kurt_df = df_copy.query(f'vocab_kurt > {kurtosis_split}')
    low_kurt_df = df_copy.query(f'vocab_kurt <= {kurtosis_split}')

    low_kurt_ps = np.vstack(low_kurt_df.groupby(
        'layer').vocab_skew.apply(np.quantile, q=ps, axis=0).values).T
    high_kurt_ps = np.vstack(high_kurt_df.groupby(
        'layer').vocab_skew.apply(np.quantile, q=ps, axis=0).values).T

    # high kurtosis
    ultra_light_blue = '#e1e1ed'
    light_blue = "#BCBCDC"
    light_highlight_blue = "#9999C7"
    mid_blue = "#7C7CB9"
    mid_highlight_blue = "#5050A2"
    dark_blue = "#27278F"
    dark_highlight_blue = "#00007C"

    ax.fill_between(layers, high_kurt_ps[0], high_kurt_ps[10],
                    facecolor=light_blue, color=ultra_light_blue, label="1-99%")
    ax.fill_between(layers, high_kurt_ps[1], high_kurt_ps[9],
                    facecolor=light_blue, color=light_blue, label="5-95%")
    ax.fill_between(layers, high_kurt_ps[2], high_kurt_ps[8],
                    facecolor=light_highlight_blue, color=light_highlight_blue, label="10-90%")
    ax.fill_between(layers, high_kurt_ps[3], high_kurt_ps[7],
                    facecolor=mid_blue, color=mid_blue, label="20-80%")
    ax.fill_between(layers, high_kurt_ps[4], high_kurt_ps[6],
                    facecolor=mid_highlight_blue, color=mid_highlight_blue, label="35-65%")
    ax.plot(layers, high_kurt_ps[5], color=dark_blue, label="median")

    # low kurtosis
    ultra_light = '#ede1e1'
    light = "#DCBCBC"
    light_highlight = "#C79999"
    mid = "#B97C7C"
    mid_highlight = "#A25050"
    dark = "#8F2727"
    dark_highlight = "#7C0000"

    ax.fill_between(layers, low_kurt_ps[0], low_kurt_ps[10],
                    facecolor=light, color=ultra_light, label="1-99%")
    ax.fill_between(layers, low_kurt_ps[1], low_kurt_ps[9],
                    facecolor=light, color=light, label="5-95%")
    ax.fill_between(layers, low_kurt_ps[2], low_kurt_ps[8],
                    facecolor=light_highlight, color=light_highlight, label="10-90%")
    ax.fill_between(
        layers, low_kurt_ps[3], low_kurt_ps[7], facecolor=mid, color=mid, label="20-80%")
    ax.fill_between(layers, low_kurt_ps[4], low_kurt_ps[6],
                    facecolor=mid_highlight, color=mid_highlight, label="35-65%")
    ax.plot(layers, low_kurt_ps[5], color=dark, label="median")

    # ax.legend(loc='upper left', title='Percentile')
    ax.set_xlabel('layer')
    ax.set_ylabel('$\cos(W_U, W_{out})$ skew')

    handles_high_kurt = [
        Patch(facecolor=ultra_light_blue, label="1-99%"),
        Patch(facecolor=ultra_light, label="1-99%"),
        Patch(facecolor=light_blue, label="5-95%"),
        Patch(facecolor=light, label="5-95%"),
        Patch(facecolor=light_highlight_blue, label="10-90%"),
        Patch(facecolor=light_highlight, label="10-90%"),
        Patch(facecolor=mid_blue, label="15-85%"),
        Patch(facecolor=mid, label="15-85%"),
        Patch(facecolor=mid_highlight_blue, label="25-75%"),
        Patch(facecolor=mid_highlight, label="25-75%"),
        plt.Line2D([0], [0], color=dark_blue, label="median", linewidth=2),
        plt.Line2D([0], [0], color=dark, label="median", linewidth=2)
    ]
    legend_high_kurt = ax.legend(
        handles=handles_high_kurt,
        loc='lower left',
        title='layer percentile | blue: kurtosis >= 10 | red: kurtosis < 10',
        ncols=6, columnspacing=0.4, fontsize=8, framealpha=0.2
    )
    # Add the legend manually to the current Axes.
    ax.add_artist(legend_high_kurt)


def make_dataset_df(ds):
    n, d = ds['tokens'].shape
    current_token = ds['tokens'].flatten().numpy()
    next_token = torch.concat([ds['tokens'][:, 1:], torch.zeros(n, 1, dtype=int) - 1], dim=1).flatten().numpy()
    position = einops.repeat(np.arange(d), 'd -> n d', n=n).flatten()
    seq_ix = einops.repeat(np.arange(n), 'n -> n d', d=d).flatten()

    ds_df = pd.DataFrame({
        'current_token': current_token.astype(np.int32), 
        'next_token': next_token.astype(np.int32), 
        'position': position.astype(np.int16),
        'seq_ix': seq_ix.astype(np.int32)
    })
    return ds_df